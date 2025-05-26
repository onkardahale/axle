import os
import subprocess
import sys
import shutil  # For checking if an executable exists
import shlex   # For safely splitting command strings

def get_default_editor() -> str:
    """
    Determines the default editor to use, checking git config, environment variables,
    and common fallbacks. Returns the command string for the editor.
    """
    editor_candidates_ordered = []
    
    # 1. Git environment variable GIT_EDITOR (used by git itself)
    #    or `git config core.editor` (which `git var GIT_EDITOR` resolves)
    try:
        git_editor_cmd_proc = subprocess.run(
            ["git", "var", "GIT_EDITOR"], 
            capture_output=True, text=True, check=False
        )
        if git_editor_cmd_proc.returncode == 0 and git_editor_cmd_proc.stdout.strip():
            editor_candidates_ordered.append(git_editor_cmd_proc.stdout.strip())
    except FileNotFoundError:
        print("Warning: git command not found. Cannot check GIT_EDITOR configuration.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not query GIT_EDITOR: {e}", file=sys.stderr)

    # 2. EDITOR environment variable
    env_editor = os.environ.get("EDITOR")
    if env_editor:
        editor_candidates_ordered.append(env_editor)

    # 3. VISUAL environment variable (often used, especially for GUI editors)
    visual_editor = os.environ.get("VISUAL")
    if visual_editor:
        editor_candidates_ordered.append(visual_editor)

    # 4. Common fallbacks (order can be preference)
    editor_candidates_ordered.extend(["nano", "vim", "vi"]) # Prioritize nano for ease of use if available

    # Find the first valid and existing editor from the ordered list
    for editor_cmd_str in editor_candidates_ordered:
        if not editor_cmd_str:  # Skip empty entries that might have resulted from unset vars
            continue
        
        try:
            # shlex.split handles if editor_cmd_str is a simple command like "vim"
            # or a command with arguments like "code --wait"
            parts = shlex.split(editor_cmd_str)
            if not parts: # If shlex.split results in an empty list (e.g., editor_cmd_str was just whitespace)
                continue
            
            executable_name = parts[0]
            
            if shutil.which(executable_name):  # Check if the executable exists in PATH
                return editor_cmd_str  # Return the full command string
        except Exception as e:
            # This might happen if shlex.split fails or shutil.which has an issue
            print(f"Warning: Problem encountered while validating editor candidate '{editor_cmd_str}': {e}", file=sys.stderr)
            continue # Try the next candidate in the list
            
    # If the loop completes, no preferred editor was found and verified
    final_fallback = "vi" # A very basic editor, usually available on Unix-like systems
    print(
        f"Warning: No suitable editor found and verified via GIT_EDITOR, EDITOR, VISUAL, or common fallbacks (nano, vim, vi).\n"
        f"Defaulting to '{final_fallback}'. Please ensure it is installed and in your PATH, or configure your preferred editor.",
        file=sys.stderr
    )
    return final_fallback

def open_editor(file_path: str) -> bool:
    """
    Opens the specified file in the user's default editor.
    Returns True if the editor exits successfully (status code 0), False otherwise.
    """
    editor_cmd_str = get_default_editor()
    
    try:
        # Safely split the editor command string in case it contains arguments
        # (e.g., "code --wait" or "subl -w")
        command_parts = shlex.split(editor_cmd_str)
        if not command_parts:
            print(f"Error: Invalid editor command configured (resolved to an empty command: '{editor_cmd_str}').", file=sys.stderr)
            return False
            
        command_to_run = command_parts + [file_path]
        
        # Inform the user which editor is being launched (useful for debugging)
        # print(f"DEBUG: Launching editor with command: {' '.join(command_to_run)}", file=sys.stderr)
        
        # subprocess.run waits for the editor process to complete.
        # We use check=False to manually inspect the return code.
        result = subprocess.run(command_to_run, check=False)

        if result.returncode == 0:
            # Editor exited successfully (status code 0).
            # This typically means the user closed it normally.
            # Note: This does NOT guarantee that the file was saved for most CLI editors,
            # as exiting without saving is often also a status code 0.
            # However, it means the editor itself didn't report an error.
            return True
        else:
            # Editor exited with a non-zero status code, indicating an error or abnormal exit.
            print(f"Editor '{editor_cmd_str}' exited with status code {result.returncode}.", file=sys.stderr)
            return False
            
    except FileNotFoundError:
        # This error occurs if the executable part of editor_cmd_str (e.g., "vim", "nano")
        # was not found on the system's PATH.
        # The improved get_default_editor with shutil.which should minimize this,
        # but this catch is a good fallback.
        executable_name = shlex.split(editor_cmd_str)[0] if editor_cmd_str else "your configured editor"
        print(
            f"Error: Editor command '{executable_name}' not found in your PATH.\n"
            f"Please install it, ensure it's in your PATH, or correctly set your GIT_EDITOR, EDITOR, or VISUAL environment variables.",
            file=sys.stderr
        )
        return False
    except Exception as e:
        # Catch any other unexpected errors during the editor launch process.
        print(f"An unexpected error occurred while trying to launch editor '{editor_cmd_str}': {str(e)}", file=sys.stderr)
        return False