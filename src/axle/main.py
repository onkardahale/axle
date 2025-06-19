#!/usr/bin/env python3

import sys
import os
import warnings
from io import StringIO

# Suppress bitsandbytes errors and warnings for better user experience
# This must be done before any other imports that might trigger bitsandbytes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Create a custom stderr handler that filters out bitsandbytes messages
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        self.bitsandbytes_warning_shown = False
        
    def write(self, text):
        # Check for bitsandbytes related error messages
        if any(keyword in text.lower() for keyword in ["bitsandbytes", "libbitsandbytes", "dlopen"]) and "bitsandbytes" in text.lower():
            # Show a clean warning once, then suppress verbose details
            if not self.bitsandbytes_warning_shown:
                show_compatibility_message(writer=self.original_stderr)
                self.bitsandbytes_warning_shown = True
            return  # Suppress the verbose error details
        
        # For traceback lines, check if they're part of a bitsandbytes error
        if text.strip().startswith(('File "', '  File "', 'Traceback', '    ', 'OSError:', 'ImportError:')):
            # Buffer the text to see if it's part of a bitsandbytes traceback
            self.buffer += text
            if "bitsandbytes" in self.buffer:
                # Show clean message once if not already shown
                if not self.bitsandbytes_warning_shown:
                    show_compatibility_message(writer=self.original_stderr)
                    self.bitsandbytes_warning_shown = True
                self.buffer = ""  # Clear buffer after handling
                return  # Suppress bitsandbytes tracebacks
            elif text.strip() and not text.strip().startswith(('File "', '  File "', '    ', 'Traceback', 'OSError:', 'ImportError:')):
                # Not a traceback continuation, flush buffer
                if self.buffer:
                    self.original_stderr.write(self.buffer)
                    self.buffer = ""
                self.original_stderr.write(text)
        else:
            # Regular message, flush buffer if any and write
            if self.buffer:
                self.original_stderr.write(self.buffer)
                self.buffer = ""
            self.original_stderr.write(text)
    
    def flush(self):
        if self.buffer:
            self.original_stderr.write(self.buffer)
            self.buffer = ""
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Only install the filter if not in verbose mode
if not (len(sys.argv) > 1 and ("--verbose" in sys.argv or "-v" in sys.argv)):
    sys.stderr = FilteredStderr(sys.stderr)
    
    import logging
    # Disable noisy library logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*device_map.*")
    warnings.filterwarnings("ignore", message=".*Loading.*")

import click
import tempfile      
import subprocess 
from pathlib import Path
from . import __version__
from .git_utils import is_git_installed, get_staged_diff, get_staged_files_count, get_staged_file_paths
from .commit_message import derive_scope
from .editor_utils import open_editor
from .ai_utils import generate_commit_message, show_compatibility_message
from .knowledge_base import KnowledgeBase
from .exceptions import AxleError, GitError, AIError, DependencyError

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"axle {__version__}")
    ctx.exit()

@click.group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed error messages and warnings.')
@click.pass_context
def main(ctx, verbose):
    """Generate commit messages using AI based on staged changes."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if not verbose:
        # Further suppress warnings if not in verbose mode
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

@main.command()
def init():
    """Initialize the knowledge base for the current project."""
    if not is_git_installed():
        click.echo("Error: Git is not installed or not found in PATH.", err=True)
        sys.exit(1)

    try:
        # Check if we're in a git repository
        subprocess.run(['git', 'rev-parse', '--git-dir'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        click.echo("Error: Not a git repository. Please run this command from a git repository.", err=True)
        sys.exit(1)

    project_root = Path(os.getcwd())
    kb = KnowledgeBase(project_root)

    click.echo("Building knowledge base...")
    kb.build_knowledge_base()
    click.echo("Knowledge base built successfully!")

def _handle_user_interaction(initial_message: str, diff: str, scope: str, context: list, unanalyzed_files: list) -> None:
    """Handles the user interaction loop for committing, editing, regenerating, or aborting."""
    current_message_content = initial_message

    while True:
        temp_file_this_iteration = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".txt") as tmp_file:
                tmp_file.write(current_message_content)
                temp_file_this_iteration = tmp_file.name
            
            if not open_editor(temp_file_this_iteration):
                click.echo("Editor failed or was closed without saving. Commit aborted.", err=True)
                if temp_file_this_iteration and os.path.exists(temp_file_this_iteration):
                    os.remove(temp_file_this_iteration)
                return

            with open(temp_file_this_iteration, 'r', encoding='utf-8') as f:
                message_from_editor = f.read()

            issue_refs = click.prompt(
                "Link any issues? (e.g., #123, Closes PROJ-45, Refs #678). Leave blank if none.",
                default='',
                show_default=False
            )

            has_breaking_changes = click.confirm("Are there any BREAKING CHANGES in this commit?")
            breaking_change_desc = ""
            if has_breaking_changes:
                click.echo("Describe the BREAKING CHANGE (and migration path, if any). Press Enter twice to finish.")
                lines = []
                while True:
                    line = input()
                    if not line and lines and lines[-1] == "":
                        break
                    lines.append(line)
                breaking_change_desc = "\n".join(lines[:-1])

            if issue_refs or breaking_change_desc:
                message_from_editor = message_from_editor.rstrip() + "\n\n"
                if issue_refs:
                    message_from_editor += f"Fixes: {issue_refs}\n"
                if breaking_change_desc:
                    message_from_editor += f"BREAKING CHANGE: {breaking_change_desc}\n"

            with open(temp_file_this_iteration, 'w', encoding='utf-8') as f_rewrite:
                f_rewrite.write(message_from_editor)

            user_input = click.prompt(
                "\nCommit message saved. What would you like to do?",
                type=click.Choice(['c', 'e', 'r', 'a'], case_sensitive=False),
                default='c',
                show_choices=True
            )

            if user_input == 'c':
                subprocess.run(['git', 'commit', '-F', temp_file_this_iteration], check=True)
                click.echo("Commit successful.")
                return
            
            elif user_input == 'e':
                current_message_content = message_from_editor
                continue
            
            elif user_input == 'r':
                additional_context = click.prompt(
                    "Optional: Provide any additional context or instructions for regenerating this commit message:",
                    default='',
                    show_default=False
                )
                current_message_content = generate_commit_message(
                    diff, scope,
                    context=context,
                    unanalyzed_files=unanalyzed_files,
                    additional_context=additional_context
                )
                continue
            
            elif user_input == 'a':
                click.echo("Aborted.")
                return
        
        finally:
            if temp_file_this_iteration and os.path.exists(temp_file_this_iteration):
                os.remove(temp_file_this_iteration)

def _commit_workflow(regenerate: bool):
    """The main workflow for the commit command."""
    if not is_git_installed():
        raise GitError("Git is not installed or not found in PATH.")

    try:
        diff = get_staged_diff()
    except RuntimeError as e:
        raise GitError(f"Not a git repository or git diff command failed. Details: {str(e)}")

    if not diff:
        click.echo("No staged changes found. Please use `git add` to stage files before running `axle`.")
        return

    staged_files = get_staged_file_paths()
    scope = derive_scope(staged_files)

    project_root = Path(os.getcwd())
    kb = KnowledgeBase(project_root)

    if not kb.kb_dir.exists():
        click.echo("Warning: Knowledge base not found. Run 'axle init' for improved commit messages.", err=True)
    elif kb.is_stale():
        click.echo("Warning: Knowledge base is stale. Consider running 'axle init' again.", err=True)

    context = []
    unanalyzed_files = []
    for file_path in staged_files:
        rel_path = Path(file_path)
        analysis = kb.get_file_analysis(rel_path)
        if analysis:
            context.append(analysis)
        else:
            unanalyzed_files.append(file_path)

    try:
        commit_message_str = generate_commit_message(diff, scope, context=context, unanalyzed_files=unanalyzed_files)
    except Exception as e:
        raise AIError(f"Failed to generate commit message: {str(e)}")

    _handle_user_interaction(commit_message_str, diff, scope, context, unanalyzed_files)

@main.command()
@click.option('--regenerate', is_flag=True, help='Regenerate the commit message with additional context')
@click.pass_context
def commit(ctx, regenerate):
    """Generate a commit message using AI based on staged changes."""
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    
    try:
        _commit_workflow(regenerate)
    except AxleError as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except ImportError as e:
        if "bitsandbytes" in str(e).lower():
            click.echo("⚠️  Some model optimization features are not available on this system.", err=True)
            click.echo("   The tool will continue with reduced performance.", err=True)
            click.echo("   For better performance, consider using a Linux system with CUDA support.", err=True)
        else:
            click.echo(f"Error: Missing dependency - {e}", err=True)
        
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Catch all other exceptions and provide clean error messages
        error_msg = str(e)
        if "bitsandbytes" in error_msg.lower():
            click.echo("⚠️  Model loading encountered compatibility issues.", err=True)
            click.echo("   This is often due to system compatibility with quantization libraries.", err=True)
            click.echo("   Try running: pip install --upgrade transformers torch", err=True)
        elif "torch" in error_msg.lower() or "cuda" in error_msg.lower():
            click.echo("⚠️  GPU/compute library issues detected.", err=True)
            click.echo("   The tool will attempt to use CPU-only mode.", err=True)
        else:
            click.echo(f"Unexpected error: {error_msg}", err=True)
        
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            click.echo("   Run with --verbose for detailed error information.", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()