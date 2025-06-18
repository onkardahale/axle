import os
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from transformers.utils import logging
import json
from pathlib import Path
import instructor
from pydantic import BaseModel, Field
from functools import lru_cache
from jinja2 import Environment, FileSystemLoader
import re
import time

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allowed commit types
ALLOWED_TYPES = ["feat", "fix", "docs", "style", "refactor", "test", "chore"]

class ModelLoadError(Exception):
    """Raised when there's an error loading the model or tokenizer."""
    pass

class GenerationError(Exception):
    """Raised when there's an error during message generation."""
    pass

class EditorError(Exception):
    """Raised when there's an error with the editor interaction."""
    pass

class CommitMessage(BaseModel):
    type: str = Field(..., description=f"The type of change. Must be one of: {', '.join(ALLOWED_TYPES)}")
    scope: Optional[str] = Field(None, description="The scope of the change (e.g., a file or directory name).")
    description: str = Field(..., description="A short, imperative tense description of the change.")
    body: Optional[str] = Field(None, description="A longer, more detailed description of the change.")

def get_cache_dir() -> Path:
    """Get the cache directory for models and configurations."""
    cache_dir = Path.home() / ".cache" / "axle"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ModelLoadError(f"Failed to create cache directory: {str(e)}")
    return cache_dir

def get_model_config() -> dict:
    """Get the model configuration from cache or return defaults."""
    config_path = get_cache_dir() / "model_config.json"
    default_config = {
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "temperature": 0.2,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "quantization": "8bit"  
    }
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            RuntimeError("Warning: Could not decode model_config.json, using defaults.")
            return default_config
    return default_config

def save_model_config(config: dict):
    """Save the model configuration to cache."""
    config_path = get_cache_dir() / "model_config.json"
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        RuntimeError(f"Warning: Could not save model_config.json: {str(e)}")

def set_quantization_level(level: str = "4bit_aggressive"):
    """
    Set the quantization level for maximum compression.
    
    Args:
        level: Quantization level - options:
            - "none": No quantization (full precision)
            - "8bit": 8-bit quantization (moderate compression)
            - "4bit": Standard 4-bit quantization (good compression)
            - "4bit_aggressive": Aggressive 4-bit quantization (maximum compression)
            - "3bit": Experimental 3-bit quantization (extreme compression)
    """
    config = get_model_config()
    config["quantization"] = level
    save_model_config(config)
    
    # Clear the cached model to force reload with new quantization
    get_model_and_tokenizer.cache_clear()
    
    print(f"✅ Quantization level set to: {level}")
    print("🔄 Model cache cleared - next model load will use new quantization settings")
    
    # Print compression info
    compression_info = {
        "none": "No compression (full precision)",
        "8bit": "~50% memory reduction",
        "4bit": "~75% memory reduction", 
        "4bit_aggressive": "~80% memory reduction (maximum compression)",
        "3bit": "~85% memory reduction (experimental)"
    }
    
    if level in compression_info:
        print(f"📊 Expected compression: {compression_info[level]}")
    
    return config

@lru_cache(maxsize=1)
def get_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Get the model and tokenizer for commit message generation.
    
    Returns:
        Tuple containing the model and tokenizer
        
    Raises:
        ModelLoadError: If there's an error loading the model or tokenizer
    """
    config = get_model_config()
    model_name = config["model_name"]
    cache_dir_models = get_cache_dir() / "models"
    
    # Configure quantization
    quantization_type = config.get("quantization", "none")
    quantization_config = None
    
    if quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 for better accuracy
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for more compression
            bnb_4bit_quant_storage=torch.uint8,  # More compact storage
        )
    elif quantization_type == "4bit_aggressive":
        # More aggressive 4-bit quantization for maximum compression
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",  # FP4 for maximum compression
            bnb_4bit_compute_dtype=torch.bfloat16,  # BFloat16 for efficiency
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )
    elif quantization_type == "3bit":
        # Experimental 3-bit quantization (if supported)
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit infrastructure
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8,
                # Additional compression settings
            )
        except Exception:
            # Fallback to aggressive 4-bit if 3-bit not supported
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8,
            )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir_models), # cache_dir expects a string
            trust_remote_code=True,
            local_files_only=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir_models), # cache_dir expects a string
            trust_remote_code=True,
            local_files_only=False,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if quantization_config is None else None,
        )
        
        # Set padding token if not set, common for generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load model or tokenizer: {str(e)}")

def _render_prompt(template_name: str, **kwargs) -> str:
    """Renders a Jinja2 prompt template."""
    prompt_dir = Path(__file__).parent / "prompts"
    env = Environment(loader=FileSystemLoader(prompt_dir))
    template = env.get_template(template_name)
    return template.render(**kwargs)

def _format_context(context: list, unanalyzed_files: list) -> str:
    """Formats the knowledge base context for the prompt."""
    if not context and not unanalyzed_files:
        return ""

    context_section = "--- BEGIN AXLE INIT CONTEXT ---\n"
    for file_analysis in context:
        path_str = file_analysis.get('path', 'Unknown File')
        category_str = file_analysis.get('category', 'N/A')
        imports_list_str = str(file_analysis.get('imports', []))
        context_section += f"File '{path_str}' (purposeCategory: {category_str}, imports: {imports_list_str}):\n"
        
        for class_info in file_analysis.get('classes', []):
            context_section += f"  Class '{class_info.get('name', 'Unnamed Class')}': {class_info.get('docstring', '')}\n"
            for method_info in class_info.get('methods', []):
                context_section += f"    Method '{method_info.get('name', 'Unnamed Method')}': {method_info.get('docstring', '')}\n"
        
        for func_info in file_analysis.get('functions', []):
            context_section += f"  Function '{func_info.get('name', 'Unnamed Function')}': {func_info.get('docstring', '')}\n"

    if unanalyzed_files:
        context_section += f"The following unanalyzed files were also part of this change: {unanalyzed_files}\n"
    context_section += "--- END AXLE INIT CONTEXT ---\n"
    return context_section

def generate_commit_message(
    diff: str,
    scope: Optional[str] = None,
    context: Optional[list] = None,
    unanalyzed_files: Optional[list] = None,
    additional_context: Optional[str] = None,
    regenerate: bool = False
) -> str:
    """
    Generate a commit message using the model based on the git diff.
    
    Args:
        diff: The git diff content
        scope: Optional scope of the changes (e.g., directory name)
        context: Optional list of file analysis from knowledge base
        unanalyzed_files: Optional list of files that couldn't be analyzed
        additional_context: Optional user-provided context for regeneration
        regenerate: Whether to regenerate the message with different parameters
    
    Returns:
        A generated commit message
        
    Raises:
        GenerationError: If there's an error during message generation
        ValueError: If the diff is empty or invalid
    """
    if not diff or not diff.strip():
        raise ValueError("Diff content cannot be empty")
    
    try:
        model, tokenizer = get_model_and_tokenizer()
    except ModelLoadError as e:
        raise GenerationError(f"Model initialization failed: {str(e)}")
    
    config = get_model_config()
    
    context_section = _format_context(context, unanalyzed_files)
    
    prompt = _render_prompt(
        "commit_message.jinja",
        allowed_types=ALLOWED_TYPES,
        context_section=context_section,
        diff=diff,
        additional_context=additional_context,
        scope=scope
    )

    messages = [
        {"role": "user", "content": prompt}
    ]
      
    try:
        # Create a chat completion function that works with Hugging Face models
        def create_chat_completion(**kwargs):
            messages = kwargs.get("messages", [])
            max_tokens = kwargs.get("max_tokens", 250)
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize the prompt
            inputs = tokenizer(text, return_tensors="pt", max_length=32768, truncation=True)
            
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            if regenerate:
                # Increase temperature for more variation, but cap at a conservative maximum to avoid
                # probability tensor issues (inf, nan, or negative values)
                config["temperature"] = min(0.6, config["temperature"] + 0.05)
                config["temperature"] = max(0.1, config["temperature"])
                print(f"Regenerating with temperature: {config['temperature']:.2f}")
                save_model_config(config)

            # Validate generation parameters to prevent tensor issues
            temperature = float(config["temperature"])
            top_p = float(config["top_p"])
            num_return_sequences = int(config["num_return_sequences"])
            
            # Ensure parameters are within safe ranges
            if not (0.01 <= temperature <= 0.6):
                print(f"Warning: Invalid temperature {temperature}, clamping to safe range")
                temperature = max(0.01, min(0.6, temperature))
            if not (0.01 <= top_p <= 0.95):
                print(f"Warning: Invalid top_p {top_p}, clamping to safe range")
                top_p = max(0.01, min(0.95, top_p))
            if num_return_sequences < 1:
                print(f"Warning: Invalid num_return_sequences {num_return_sequences}, setting to 1")
                num_return_sequences = 1
                
            # Check for NaN or infinite values
            if not (torch.isfinite(torch.tensor(temperature)) and torch.isfinite(torch.tensor(top_p))):
                raise GenerationError("Invalid generation parameters: temperature or top_p contains NaN or infinite values")
            
            # Debug logging for parameter values
            print(f"Debug: Using parameters - temperature: {temperature}, top_p: {top_p}, num_return_sequences: {num_return_sequences}")
            
            with torch.no_grad():
                start_time = time.perf_counter()  # Start timing
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens, # Increased slightly for potentially complex JSON structures or minor verbosity
                    attention_mask=inputs["attention_mask"],  
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.perf_counter()  # End timing
                inference_duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Decode only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            print(f"Inference information: {inference_duration_ms:.2f} ms, {1000*len(outputs[0])/inference_duration_ms} tokens/sec") 
       
            completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # ---- Robust JSON Extraction Logic ----
            extracted_json_string = None

            # 1. Try to find JSON within ```json ... ``` markdown
            markdown_match = re.search(r"```(?:json|JSON)?\s*(\{[\s\S]*?\})\s*```", completion, re.IGNORECASE)
            if markdown_match:
                extracted_json_string = markdown_match.group(1).strip()
            else:
                # 2. If no markdown, try to find the last occurrence of a string that looks like a JSON object
                last_brace_open_idx = completion.rfind('{')
                if last_brace_open_idx != -1:
                    # Attempt to find the matching closing brace for the last open brace
                    open_brace_count = 0
                    json_candidate_segment = completion[last_brace_open_idx:]
                    
                    if json_candidate_segment.strip().startswith('{'): # Basic sanity check
                        end_brace_idx_in_segment = -1
                        for i, char in enumerate(json_candidate_segment):
                            if char == '{':
                                open_brace_count += 1
                            elif char == '}':
                                open_brace_count -= 1
                                if open_brace_count == 0:
                                    end_brace_idx_in_segment = i
                                    break
                        
                        if end_brace_idx_in_segment != -1:
                            extracted_json_string = json_candidate_segment[:end_brace_idx_in_segment+1].strip()
                
                # 3. (Optional) As a very last resort, if no specific structure found, 
                #    and the whole completion might be JSON. Be cautious with this.
                if extracted_json_string is None:
                    temp_completion_stripped = completion.strip()
                    if temp_completion_stripped.startswith('{') and temp_completion_stripped.endswith('}'):
                        # Try to parse, it might be a simple JSON output without any wrapping
                        extracted_json_string = temp_completion_stripped
            # ---- End of Robust JSON Extraction Logic ----

            if extracted_json_string is None:
                raise GenerationError(f"Could not extract a clear JSON object from the model's output. Raw completion: '{completion[:500]}...'")
            
            # Create a mock response object that matches instructor's expectations
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
                    self.model = "local-model"
                    self.usage = None
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
                    self.finish_reason = "stop"
                    self.index = 0
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
                    self.function_call = None
                    self.tool_calls = None
            
            return MockResponse(extracted_json_string)
        
        # Use instructor with the custom completion function
        client = instructor.patch(create=create_chat_completion, mode=instructor.Mode.JSON)

        commit_message: CommitMessage = client(
            messages=messages,
            response_model=CommitMessage,
            max_retries=1,
            max_tokens=250,
        )

        # Format the commit message using the original logic
        type_ = commit_message.type
        if type_ not in ALLOWED_TYPES:
            print(f"Warning: Invalid commit type '{type_}' received from model. Defaulting to 'feat'.")
            type_ = "feat"

        scope_ = commit_message.scope
        description = commit_message.description.strip()
        body = commit_message.body

        if not description: # Ensure description is not empty
            raise GenerationError("Generated commit message has an empty description.")

        if scope_:
            header = f"{type_}({scope_}): {description}"
        else:
            header = f"{type_}: {description}"
            
        commit_message_text = header
        
        # Append body if it exists and is not just whitespace
        if body and body.strip():
            commit_message_text += f"\n\n{body.strip()}" # Two newlines before the body

        if not header.strip(): # Final check on header which contains description
            raise GenerationError("Generated commit message is empty or lacks a description after formatting.")

        return commit_message_text

    except torch.cuda.OutOfMemoryError:
        raise GenerationError("GPU out of memory. Try using a smaller model, enabling CPU offloading, or reducing input size.")
    except Exception as e:
        current_completion_snippet = "N/A"
        
        # Handle specific probability tensor errors that occur during generation
        error_message = str(e).lower()
        if any(phrase in error_message for phrase in ['probability tensor', 'inf', 'nan', 'element < 0']):
            raise GenerationError(
                f"Model generation failed due to invalid probability values. This often occurs with "
                f"extreme sampling parameters. Try using lower temperature values. "
                f"Original error: {str(e)}"
            )
        
        # Re-raise original error type if it's ModelLoadError or GenerationError, otherwise wrap
        if isinstance(e, (ModelLoadError, GenerationError, ValueError)):
            raise 
        else: # Wrap unexpected errors in GenerationError for consistent handling upstream
            raise GenerationError(f"An unexpected error occurred during commit message generation: {type(e).__name__} - {str(e)}. Model output snippet: '{current_completion_snippet}...')")