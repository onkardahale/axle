#!/usr/bin/env python3

import sys
import click
import tempfile 
import os       
import subprocess 
from . import __version__
from .git_utils import is_git_installed, get_staged_diff, get_staged_files_count, get_staged_file_paths
from .commit_message import derive_scope
from .editor_utils import open_editor
from .ai_utils import generate_commit_message

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"axle {__version__}")
    ctx.exit()

@click.command()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit.')
def main():
    """Generate a commit message using AI based on staged changes."""
    if not is_git_installed():
        click.echo("Error: Git is not installed or not found in PATH.", err=True)
        sys.exit(1)

    try:
        diff = get_staged_diff()
    except RuntimeError as e: # Specific exception from get_staged_diff if it's not a git repo or diff fails
        click.echo(f"Error: Not a git repository or git diff command failed. Details: {str(e)}", err=True)
        sys.exit(1)

    if not diff:
        click.echo("No staged changes found. Please use `git add` to stage files before running `axle`.")
        sys.exit(0)

    staged_files = get_staged_file_paths()
    scope = derive_scope(staged_files)

    try:
        commit_message_str = generate_commit_message(diff, scope)
    except Exception as e:
        click.echo(f"Error: Failed to generate commit message: {str(e)}", err=True)
        sys.exit(1)

    current_message_content = commit_message_str

    while True:
        temp_file_this_iteration = None
        try:
            # Create a temporary file to hold the commit message for the editor
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".txt") as tmp_file:
                tmp_file.write(current_message_content)
                temp_file_this_iteration = tmp_file.name
            
            # open_editor is expected to open the editor for temp_file_this_iteration.
            # It should return True if user saves/exits normally, False otherwise.
            if not open_editor(temp_file_this_iteration):
                click.echo("Editor failed or was closed without saving. Commit aborted.", err=True)
                # Cleanup and exit if open_editor failed
                if temp_file_this_iteration and os.path.exists(temp_file_this_iteration):
                    os.remove(temp_file_this_iteration)
                sys.exit(1)

            # After editor, read the potentially modified message from the temp file.
            with open(temp_file_this_iteration, 'r', encoding='utf-8') as f:
                message_from_editor = f.read()

            user_input = click.prompt(
                "\nCommit message saved. What would you like to do?", # Kept original prompt for test compatibility
                type=click.Choice(['c', 'e', 'a'], case_sensitive=False),
                default='c',
                show_choices=True # Good practice
            )

            if user_input == 'c':
                try:
                    # Use the temp_file_path (which contains the final message) for commit
                    subprocess.run(['git', 'commit', '-F', temp_file_this_iteration], check=True)
                    click.echo("Commit successful.")
                    # sys.exit(0) will be called, finally block will clean up.
                    # No need to os.remove here explicitly if finally handles it before exit.
                    sys.exit(0) 
                except subprocess.CalledProcessError as e:
                    click.echo(f"Error: Git commit failed: {str(e)}", err=True)
                    # sys.exit(1) will be called, finally block will clean up.
                    sys.exit(1)
            
            elif user_input == 'e':
                current_message_content = message_from_editor # Update content for the next editor session
                # Loop continues, temp_file_this_iteration will be cleaned by finally.
                # A new temp file will be created in the next iteration.
                continue 
            
            elif user_input == 'a':
                click.echo("Aborted.")
                # sys.exit(0) will be called, finally block will clean up.
                sys.exit(0)
        
        finally:
            # This block will be executed before sys.exit completes or when 'continue' is called.
            if temp_file_this_iteration and os.path.exists(temp_file_this_iteration):
                os.remove(temp_file_this_iteration)

if __name__ == '__main__':
    main()