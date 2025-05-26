import unittest
from unittest.mock import patch, MagicMock
# import sys # Not strictly needed for tests here unless manipulating sys.path etc.
# from io import StringIO # Not strictly needed for these tests
from click.testing import CliRunner
from axle.main import main # Assuming axle.main can be imported this way

class TestMain(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_version_flag(self):
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("axle 0.1.0", result.output) # Make sure version is accurate

    def test_help_flag(self):
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Generate a commit message using AI based on staged changes", result.output)

    # Patching where the names are looked up in axle.main
    @patch('axle.main.get_staged_diff')
    @patch('axle.main.is_git_installed')
    def test_no_git_installed(self, mock_is_git_installed, mock_get_staged_diff):
        mock_is_git_installed.return_value = False
        # get_staged_diff should not be called if git is not installed.
        # Its side_effect is not strictly necessary to set for this test's primary path,
        # but doesn't hurt if it were accidentally called.
        # mock_get_staged_diff.side_effect = RuntimeError("git is not installed or not found in PATH.")
        result = self.runner.invoke(main)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Git is not installed or not found in PATH.", result.output)

    @patch('axle.main.is_git_installed')
    @patch('axle.main.get_staged_diff')
    def test_no_staged_changes(self, mock_get_staged_diff, mock_is_git_installed):
        mock_is_git_installed.return_value = True
        mock_get_staged_diff.return_value = None # Simulate no diff
        result = self.runner.invoke(main)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No staged changes found", result.output)

    @patch('axle.main.is_git_installed')
    @patch('axle.main.get_staged_diff')
    @patch('axle.main.get_staged_files_count') # Patched in axle.main
    @patch('axle.main.get_staged_file_paths') # Patched in axle.main
    @patch('axle.main.generate_commit_message') # Patched in axle.main
    @patch('axle.main.open_editor')           # Patched in axle.main
    @patch('builtins.input', new_callable=MagicMock) # For click.prompt if it falls back
    @patch('click.prompt', new_callable=MagicMock)    # More direct for click.prompt
    def test_successful_diff(self, mock_click_prompt, mock_input, mock_open_editor, 
                           mock_generate_commit_message, mock_get_staged_file_paths, 
                           mock_get_staged_files_count, mock_get_staged_diff, 
                           mock_is_git_installed):
        # Setup mocks
        mock_is_git_installed.return_value = True
        mock_get_staged_diff.return_value = "test diff"
        mock_get_staged_files_count.return_value = 1 # Ensure this is used or remove if not
        mock_get_staged_file_paths.return_value = ["test.py"]
        mock_generate_commit_message.return_value = "feat: test commit"
        mock_open_editor.return_value = True # Editor opens and saves successfully
        # mock_input.return_value = 'c' # For builtins.input
        mock_click_prompt.return_value = 'c' # For click.prompt

        # Mock subprocess.run for git commit
        with patch('axle.main.subprocess.run') as mock_run: # Patch subprocess where it's used in main
            mock_run.return_value = MagicMock(returncode=0)
            result = self.runner.invoke(main)
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Commit successful", result.output)
            mock_run.assert_called_once() # Check that git commit was called
            # Check that it was called with -F and a file path (harder to check specific temp file path)
            args, kwargs = mock_run.call_args
            self.assertIn('-F', args[0])


    @patch('axle.main.is_git_installed')
    @patch('axle.main.get_staged_diff')
    @patch('axle.main.get_staged_files_count')
    @patch('axle.main.get_staged_file_paths')
    @patch('axle.main.generate_commit_message')
    @patch('axle.main.open_editor')
    @patch('click.prompt', new_callable=MagicMock)
    def test_interactive_commit(self, mock_click_prompt, mock_open_editor, 
                                mock_generate_commit_message, mock_get_staged_file_paths, 
                                mock_get_staged_files_count, mock_get_staged_diff, mock_is_git_installed):
        mock_is_git_installed.return_value = True
        mock_get_staged_diff.return_value = "test diff"
        mock_get_staged_files_count.return_value = 1
        mock_get_staged_file_paths.return_value = ["test.py"]
        mock_generate_commit_message.return_value = "feat: test commit"
        mock_open_editor.return_value = True
        mock_click_prompt.return_value = 'c'

        with patch('axle.main.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = self.runner.invoke(main)
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Commit successful", result.output)

    @patch('axle.main.is_git_installed')
    @patch('axle.main.get_staged_diff')
    @patch('axle.main.get_staged_files_count')
    @patch('axle.main.get_staged_file_paths')
    @patch('axle.main.generate_commit_message')
    @patch('axle.main.open_editor')
    @patch('click.prompt', new_callable=MagicMock)
    def test_interactive_edit_again(self, mock_click_prompt, mock_open_editor, 
                                    mock_generate_commit_message, mock_get_staged_file_paths, 
                                    mock_get_staged_files_count, mock_get_staged_diff, mock_is_git_installed):
        mock_is_git_installed.return_value = True
        mock_get_staged_diff.return_value = "test diff"
        mock_get_staged_files_count.return_value = 1
        mock_get_staged_file_paths.return_value = ["test.py"]
        mock_generate_commit_message.return_value = "feat: initial commit"
        # First call to open_editor (initial edit), second call (after choosing 'e')
        mock_open_editor.return_value = True # Assume editor always succeeds for this test path
        # User chooses 'e' then 'c'
        mock_click_prompt.side_effect = ['e', 'c']

        with patch('axle.main.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = self.runner.invoke(main)
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Commit successful", result.output)
            self.assertEqual(mock_open_editor.call_count, 2) # Editor opened twice

    @patch('axle.main.is_git_installed')
    @patch('axle.main.get_staged_diff')
    @patch('axle.main.get_staged_files_count')
    @patch('axle.main.get_staged_file_paths')
    @patch('axle.main.generate_commit_message')
    @patch('axle.main.open_editor')
    @patch('click.prompt', new_callable=MagicMock)
    def test_interactive_abort(self, mock_click_prompt, mock_open_editor, 
                               mock_generate_commit_message, mock_get_staged_file_paths, 
                               mock_get_staged_files_count, mock_get_staged_diff, mock_is_git_installed):
        mock_is_git_installed.return_value = True
        mock_get_staged_diff.return_value = "test diff"
        mock_get_staged_files_count.return_value = 1
        mock_get_staged_file_paths.return_value = ["test.py"]
        mock_generate_commit_message.return_value = "feat: test commit"
        mock_open_editor.return_value = True
        mock_click_prompt.return_value = 'a' # User aborts

        result = self.runner.invoke(main)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Aborted.", result.output)

if __name__ == '__main__':
    unittest.main()