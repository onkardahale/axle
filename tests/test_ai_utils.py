import unittest
from unittest.mock import patch, MagicMock
import os
import torch
import json
from pathlib import Path
from tests import BaseAxleTestCase
from axle.ai_utils import (
    get_model_and_tokenizer,
    generate_commit_message,
    get_cache_dir,
    get_model_config,
    save_model_config,
    ModelLoadError,
    GenerationError,
    CommitMessage
)

class TestAIUtils(BaseAxleTestCase):
    def setUp(self):
        super().setUp()
        
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        
        # Mock tokenizer behavior for chat template
        self.mock_tokenizer.apply_chat_template.return_value = "test prompt"
        
        # Mock tokenizer behavior for encoding
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model behavior - generate returns token tensors
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_model.device = torch.device('cpu')  # Add device attribute
        
        # Mock tokenizer decode to return valid JSON
        self.mock_tokenizer.decode.return_value = '{"type": "feat", "scope": "api", "description": "add new endpoint", "body": "This is the body"}'
        
        # Mock tokenizer attributes
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.pad_token = None

    def tearDown(self):
        super().tearDown()

    def test_get_cache_dir(self):
        cache_dir = get_cache_dir()
        self.assertTrue(isinstance(cache_dir, Path))
        self.assertTrue(cache_dir.exists())

    def test_get_model_config_default(self):
        config = get_model_config()
        self.assertIsInstance(config, dict)
        self.assertIn("model_name", config)
        self.assertIn("temperature", config)
        

    def test_save_and_load_model_config(self):
        test_config = {
            "model_name": "test-model",
            "temperature": 0.8,
            "top_p": 0.9,
            "num_return_sequences": 1
        }
        save_model_config(test_config)
        loaded_config = get_model_config()
        self.assertEqual(loaded_config, test_config)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_get_model_and_tokenizer_success(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_model_from_pretrained.return_value = self.mock_model

        model, tokenizer = get_model_and_tokenizer()

        mock_tokenizer_from_pretrained.assert_called_once()
        mock_model_from_pretrained.assert_called_once()
        self.assertEqual(model, self.mock_model)
        self.assertEqual(tokenizer, self.mock_tokenizer)

    def test_get_model_and_tokenizer_cache_dir_error(self):
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with self.assertRaises(ModelLoadError) as cm:
                get_model_and_tokenizer.cache_clear()
                get_model_and_tokenizer()
            self.assertIn("Failed to create cache directory", str(cm.exception))

    def test_get_model_and_tokenizer_load_error(self):
        with patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Download failed")):
            with self.assertRaises(ModelLoadError) as cm:
                get_model_and_tokenizer.cache_clear()
                get_model_and_tokenizer()
            self.assertIn("Failed to load model or tokenizer", str(cm.exception))

    @patch('axle.ai_utils._render_prompt')
    def test_generate_commit_message_success(self, mock_render_prompt):
        mock_render_prompt.return_value = "test prompt"
        with patch('axle.ai_utils.get_model_and_tokenizer', return_value=(self.mock_model, self.mock_tokenizer)):
            result = generate_commit_message("test diff", "test")
            self.assertEqual(result, "feat(api): add new endpoint\n\nThis is the body")

    @patch('axle.ai_utils._render_prompt')
    def test_generate_commit_message_empty_output(self, mock_render_prompt):
        mock_render_prompt.return_value = "test prompt"
        with patch('axle.ai_utils.get_model_and_tokenizer', return_value=(self.mock_model, self.mock_tokenizer)):
            # Mock decode to return JSON with empty description
            self.mock_tokenizer.decode.return_value = '{"type": "feat", "description": "", "scope": null, "body": null}'
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "test")
            self.assertIn("empty description", str(cm.exception))

    @patch('axle.ai_utils._render_prompt')
    def test_generate_commit_message_generation_error(self, mock_render_prompt):
        mock_render_prompt.return_value = "test prompt"
        with patch('axle.ai_utils.get_model_and_tokenizer', return_value=(self.mock_model, self.mock_tokenizer)):
            self.mock_model.generate.side_effect = Exception("Generation failed")
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api")
            self.assertIn("An unexpected error occurred during commit message generation", str(cm.exception))

    def test_generate_commit_message_empty_diff(self):
        with self.assertRaises(ValueError) as cm:
            generate_commit_message("", "api")
        self.assertEqual(str(cm.exception), "Diff content cannot be empty")

    def test_generate_commit_message_model_load_error(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', side_effect=ModelLoadError("Model not found")):
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api")
            self.assertIn("Model initialization failed", str(cm.exception))

if __name__ == '__main__':
    unittest.main() 