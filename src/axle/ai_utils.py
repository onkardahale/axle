import os
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.utils import logging
import json
from pathlib import Path
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
        "num_return_sequences": 1
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
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Set padding token if not set, common for generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load model or tokenizer: {str(e)}")

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
    
    # Construct context section if available
    context_section = ""
    if context:
        context_section = "--- BEGIN AXLE INIT CONTEXT ---\n"
        for file_analysis in context: # 'context' is a list of 'file_analysis' dicts
            # Previous fixes in KnowledgeBase ensure 'path', 'category', 'imports', 
            # 'classes', 'functions' keys exist at this top level of file_analysis.
            path_str = file_analysis.get('path', 'Unknown File') 
            category_str = file_analysis.get('category', 'N/A')
            imports_list_str = str(file_analysis.get('imports', [])) 

            context_section += f"File '{path_str}' (purposeCategory: {category_str}, imports: {imports_list_str}):\n"
            
            # Iterate over classes, using .get() for all optional nested fields
            for class_info in file_analysis.get('classes', []): # .get() for safety, though KB ensures it
                class_name = class_info.get('name', 'Unnamed Class') # Use .get() for name
                context_section += f"  Class '{class_name}':\n"
                
                class_docstring = class_info.get('docstring') # Safely get docstring
                if class_docstring: # Check if the retrieved docstring is truthy
                    context_section += f"    Docstring: \"{class_docstring}\"\n" # Use the safe variable
                
                for method_info in class_info.get('methods', []): # Use .get() for methods list
                    method_name = method_info.get('name', 'Unnamed Method') # .get() for name
                    context_section += f"    Method '{method_name}':\n"
                    
                    method_docstring = method_info.get('docstring') # Safely get docstring
                    if method_docstring: # Check if the retrieved docstring is truthy
                        context_section += f"      Docstring: \"{method_docstring}\"\n" # Use the safe variable
                    
                    parameters = method_info.get('parameters', []) # .get() for parameters list
                    if parameters:
                        param_strings = []
                        for p_info in parameters: 
                            p_name = p_info.get('name', 'param') # .get() for param name
                            p_annotation = p_info.get('annotation') # .get() for param annotation
                            if p_annotation:
                                param_strings.append(f"{p_name} ({p_annotation})")
                            else:
                                param_strings.append(p_name)
                        if param_strings:
                             context_section += f"      Parameters: {', '.join(param_strings)}\n"
            
            # Iterate over functions, using .get() for all optional nested fields
            for func_info in file_analysis.get('functions', []): # .get() for safety
                func_name = func_info.get('name', 'Unnamed Function') # .get() for name
                context_section += f"  Function '{func_name}':\n"
                
                func_docstring = func_info.get('docstring') # Safely get docstring
                if func_docstring: # Check if the retrieved docstring is truthy
                    context_section += f"    Docstring: \"{func_docstring}\"\n" # Use the safe variable
                
                parameters = func_info.get('parameters', []) # .get() for parameters list
                if parameters:
                    param_strings = []
                    for p_info in parameters:
                        p_name = p_info.get('name', 'param') # .get() for param name
                        p_annotation = p_info.get('annotation') # .get() for param annotation
                        if p_annotation:
                            param_strings.append(f"{p_name} ({p_annotation})")
                        else:
                            param_strings.append(p_name)
                        if param_strings:
                            context_section += f"    Parameters: {', '.join(param_strings)}\n"
        
        if unanalyzed_files:
            context_section += f"The following unanalyzed files were also part of this change: {unanalyzed_files}\n"
        context_section += "--- END AXLE INIT CONTEXT ---\n\n"
    
    
    # Construct the messages for chat template
    # In your generate_commit_message function, update the 'messages'
    messages = [
        {"role": "system", "content": f"""You are an expert in Conventional Commits. Your task is to generate commit messages strictly in JSON format based on git diffs and any provided code context.
    You MUST output a single, valid JSON object and NOTHING ELSE. Do not include any explanatory text before or after the JSON object itself.
    The 'description' should be a concise high-level summary. If there are multiple distinct changes or important details, elaborate on them in the 'body' field.
    Allowed types for the 'type' field: {', '.join(ALLOWED_TYPES)}"""},
        {"role": "user", "content": f"""Analyze the provided context and git diff, then generate a commit message in JSON format.

    {context_section} 
    --- BEGIN GIT DIFF ---
    {diff}
    --- END GIT DIFF ---

    {additional_context if additional_context else ''} 

    Scope for the commit (use this if relevant, otherwise determine from context or set to null): {scope if scope else 'general'}

    **Output Adherence Rules:**
    1. Your entire response MUST be a single, valid JSON object.
    2. Do NOT include any text, comments, or explanations before or after the JSON object.
    3. The JSON object MUST contain these fields:
    - `type`: string (one of {', '.join(ALLOWED_TYPES)})
    - `scope`: string (short, lowercase noun) or null
    - `description`: string (concise, imperative, high-level summary starting with a verb, NOT empty)
    - `body`: string (detailed explanation, can be multi-line with '\\n' for newlines, or null if not needed. Use this to list multiple distinct changes or provide more context.)
    4. Ensure all string values within the JSON are properly escaped.

    **JSON Schema Example (with a body):**
    {{
        "type": "refactor",
        "scope": "ai_utils",
        "description": "optimize model parameters and prompt structure",
        "body": "Key improvements made to AI generation module:\n- Adjusted model temperature to 0.2 for more predictable output.\n- Increased maximum token limit to support more complex diffs.\n- Updated the system prompt to provide clearer instructions for JSON formatting."
    }}

    Your response (A single, valid JSON object ONLY):"""}
    ]
      
    try:
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
                max_new_tokens=250, # Increased slightly for potentially complex JSON structures or minor verbosity
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
        
        try:
            commit_data = json.loads(extracted_json_string)
        except json.JSONDecodeError as e:
            raise GenerationError(f"Failed to parse JSON from model: {str(e)}. Extracted JSON string was: '{extracted_json_string}'")

        type_ = commit_data.get("type", "feat")
        if type_ not in ALLOWED_TYPES:
            print(f"Warning: Invalid commit type '{type_}' received from model. Defaulting to 'feat'.")
            type_ = "feat"

        scope_ = commit_data.get("scope")
        description = commit_data.get("description", "").strip()
        body = commit_data.get("body")

        if not description: # Ensure description is not empty
            raise GenerationError("Generated commit message has an empty description.")

        if scope_:
            header = f"{type_}({scope_}): {description}"
        else:
            header = f"{type_}: {description}"
            
        commit_message = header
        
        # Append body if it exists and is not just whitespace
        if body and body.strip():
            commit_message += f"\n\n{body.strip()}" # Two newlines before the body

        if not header.strip(): # Final check on header which contains description
            raise GenerationError("Generated commit message is empty or lacks a description after formatting.")

        return commit_message

    except torch.cuda.OutOfMemoryError:
        raise GenerationError("GPU out of memory. Try using a smaller model, enabling CPU offloading, or reducing input size.")
    except Exception as e:
        current_completion_snippet = "N/A"
        if 'completion' in locals():
            current_completion_snippet = completion[:200]
        elif 'extracted_json_string' in locals() and extracted_json_string is not None:
            current_completion_snippet = extracted_json_string[:200]
        
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
            raise GenerationError(f"An unexpected error occurred during commit message generation: {type(e).__name__} - {str(e)}. Model output snippet: '{current_completion_snippet}...'")