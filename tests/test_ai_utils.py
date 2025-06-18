import unittest
from unittest.mock import patch, MagicMock
import os
import torch
import json
from pathlib import Path
from axle.ai_utils import (
    get_model_and_tokenizer,
    generate_commit_message,
    get_cache_dir,
    get_model_config,
    save_model_config,
    ModelLoadError,
    GenerationError
)

class TestAIUtils(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        
        # Mock tokenizer behavior
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.decode.return_value = "JSON:{\"type\": \"feat\", \"scope\": \"api\", \"description\": \"add new endpoint\"}"
        
        # Mock model behavior
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        self.mock_model.device = torch.device('cpu')  # Add device attribute

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
            "num_return_sequences": 2
        }
        save_model_config(test_config)
        loaded_config = get_model_config()
        self.assertEqual(loaded_config, test_config)

    def test_get_model_and_tokenizer_success(self):
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            mock_tokenizer.return_value = self.mock_tokenizer
            mock_model.return_value = self.mock_model
            
            model, tokenizer = get_model_and_tokenizer()
            
            # Verify model and tokenizer were loaded
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()

    def test_get_model_and_tokenizer_cache_dir_error(self):
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")
            
            with self.assertRaises(ModelLoadError) as cm:
                get_model_and_tokenizer()
            self.assertIn("Failed to create cache directory", str(cm.exception))

    def test_get_model_and_tokenizer_load_error(self):
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Download failed")
            
            with self.assertRaises(ModelLoadError) as cm:
                get_model_and_tokenizer()
            self.assertIn("Failed to load model or tokenizer", str(cm.exception))

    def test_generate_commit_message_with_regeneration(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            # Get initial temperature
            initial_config = get_model_config()
            initial_temp = initial_config["temperature"]
            
            # First generation
            result1 = generate_commit_message("test diff", "test")
            self.assertEqual(result1, "feat(api): add new endpoint")
            
            # Regenerate with higher temperature
            result2 = generate_commit_message("test diff", "test", regenerate=True)
            self.assertEqual(result2, "feat(api): add new endpoint")
            
            # Verify config was updated - temperature should be increased by 0.05
            config = get_model_config()
            expected_temp = min(0.6, initial_temp + 0.05)  # Capped at 0.6
            self.assertEqual(config["temperature"], expected_temp)

    def test_generate_commit_message_empty_output(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            self.mock_tokenizer.decode.return_value = ""
            with self.assertRaises(GenerationError):
                generate_commit_message("test diff", "test")

    def test_generate_commit_message_extraction_error(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            self.mock_tokenizer.decode.return_value = "Invalid output without commit message"
            with self.assertRaises(GenerationError):
                generate_commit_message("test diff", "test")

    def test_generate_commit_message_empty_diff(self):
        with self.assertRaises(ValueError) as cm:
            generate_commit_message("", "api")
        self.assertEqual(str(cm.exception), "Diff content cannot be empty")

    def test_generate_commit_message_model_load_error(self):
        with patch('axle.ai_utils.get_model_and_tokenizer') as mock_get_model:
            mock_get_model.side_effect = ModelLoadError("Model not found")
            
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api")
            self.assertIn("Model initialization failed", str(cm.exception))

    def test_generate_commit_message_out_of_memory(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            self.mock_model.generate.side_effect = torch.cuda.OutOfMemoryError()
            
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api")
            self.assertIn("GPU out of memory", str(cm.exception))

    def test_generate_commit_message_generation_error(self):
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            self.mock_model.generate.side_effect = Exception("Generation failed")
            
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api")
            self.assertIn("An unexpected error occurred", str(cm.exception))

    def test_generate_commit_message_probability_tensor_error(self):
        """Test that probability tensor errors are properly handled during regeneration"""
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            # Simulate the specific error from issue #6
            self.mock_model.generate.side_effect = Exception(
                "probability tensor contains either `inf`, `nan` or element < 0"
            )
            
            with self.assertRaises(GenerationError) as cm:
                generate_commit_message("test diff", "api", regenerate=True)
            
            # Verify the error message explains the issue and provides guidance
            error_msg = str(cm.exception)
            self.assertIn("Model generation failed due to invalid probability values", error_msg)
            self.assertIn("extreme sampling parameters", error_msg)
            self.assertIn("lower temperature values", error_msg)

    def test_regenerate_temperature_capping(self):
        """Test that regenerate mode caps temperature at a safe maximum"""
        with patch('axle.ai_utils.get_model_and_tokenizer', 
                  return_value=(self.mock_model, self.mock_tokenizer)):
            with patch('axle.ai_utils.get_model_config') as mock_get_config:
                with patch('axle.ai_utils.save_model_config') as mock_save_config:
                    # Start with high temperature
                    mock_get_config.return_value = {
                        "temperature": 0.75,
                        "top_p": 0.95,
                        "num_return_sequences": 1
                    }
                    
                    generate_commit_message("test diff", "api", regenerate=True)
                    
                    # Verify temperature was capped at 0.6, not 0.8 (0.75 + 0.05 capped)
                    saved_config = mock_save_config.call_args[0][0]
                    self.assertEqual(saved_config["temperature"], 0.6)

if __name__ == '__main__':
    unittest.main() 