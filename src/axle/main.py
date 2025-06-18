#!/usr/bin/env python3

import sys
import click
import tempfile 
import os       
import subprocess 
from pathlib import Path
from . import __version__
from .git_utils import is_git_installed, get_staged_diff, get_staged_files_count, get_staged_file_paths
from .commit_message import derive_scope
from .editor_utils import open_editor
from .ai_utils import generate_commit_message
from .knowledge_base import KnowledgeBase
from .exceptions import AxleError, GitError, AIError

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"axle {__version__}")
    ctx.exit()

@click.group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit.')
def main():
    """Generate commit messages using AI based on staged changes."""
    pass

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
def commit(regenerate):
    """Generate a commit message using AI based on staged changes."""
    try:
        _commit_workflow(regenerate)
    except AxleError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()