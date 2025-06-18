"""Tests for the main CLI module."""

import unittest
from unittest.mock import patch, MagicMock, mock_open 
import subprocess 

from click.testing import CliRunner
from axle.main import main

class TestMain(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

        self.patchers = {
            'is_git_installed': patch('axle.main.is_git_installed', return_value=True),
            'subprocess_run': patch('axle.main.subprocess.run'),
            'get_staged_diff': patch('axle.main.get_staged_diff', return_value="fake diff data"),
            'get_staged_files_count': patch('axle.main.get_staged_files_count', return_value=1),
            'get_staged_file_paths': patch('axle.main.get_staged_file_paths', return_value=['fake/path.py']),
            'derive_scope': patch('axle.main.derive_scope', return_value='testscope'),
            'KnowledgeBase': patch('axle.main.KnowledgeBase'),
            'generate_commit_message': patch('axle.main.generate_commit_message', return_value="AI Generated Message"),
            'open_editor': patch('axle.main.open_editor', return_value=True),
            'os_remove': patch('axle.main.os.remove'),
            'NamedTemporaryFile': patch('axle.main.tempfile.NamedTemporaryFile'),
            'os_path_exists': patch('axle.main.os.path.exists'),
            'builtins_open': patch('builtins.open', new_callable=mock_open),
        }
        self.mocks = {name: p.start() for name, p in self.patchers.items()}

        self.mock_rev_parse_success = MagicMock(returncode=0, stdout=b'.git')
        self.mocks['subprocess_run'].return_value = self.mock_rev_parse_success # Default for first call

        mock_kb_instance = MagicMock()
        mock_kb_instance.kb_dir.exists.return_value = True
        mock_kb_instance.is_stale.return_value = False
        mock_kb_instance.get_file_analysis.return_value = "File analysis context"
        self.mocks['KnowledgeBase'].return_value = mock_kb_instance
        self.mock_kb_instance = mock_kb_instance

        self.mock_temp_file_obj = MagicMock()
        self.mock_temp_file_obj.name = "fake_temp_commit_msg.txt"
        self.mock_temp_file_obj.__enter__.return_value = self.mock_temp_file_obj
        self.mock_temp_file_obj.__exit__.return_value = None
        self.mocks['NamedTemporaryFile'].return_value = self.mock_temp_file_obj

        # Configure os.path.exists
        def os_path_exists_side_effect(path):
            if path == self.mock_temp_file_obj.name:
                return True
            return False # Default to false for unexpected paths in test scope
        self.mocks['os_path_exists'].side_effect = os_path_exists_side_effect

        # Configure builtins.open and NamedTemporaryFile write interaction
        self.temp_file_content_store = "AI Generated Message" # Initial default
        
        def named_temp_file_write_effect(content_written):
            self.temp_file_content_store = content_written
        self.mock_temp_file_obj.write.side_effect = named_temp_file_write_effect
        
        # Mock for builtins.open().read() and builtins.open().write()
        def builtins_open_write_effect(content_written):
            self.temp_file_content_store = content_written
        
        # When builtins.open is called, its read/write should use the store
        self.mocks['builtins_open'].return_value.read.side_effect = lambda: self.temp_file_content_store
        self.mocks['builtins_open'].return_value.write.side_effect = builtins_open_write_effect


    def tearDown(self):
        patch.stopall()

    def test_version_flag(self):
        """Test version flag."""
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("axle 3.2.0", result.output)

    def test_help_flag(self):
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Generate a commit message using AI based on staged changes", result.output)

    def test_no_staged_changes(self):
        """Test behavior when there are no staged changes."""
        self.mocks['get_staged_diff'].return_value = ""
        result = self.runner.invoke(main, ['commit'])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("No staged changes found. Please use `git add` to stage files before running `axle`.", result.output)

    def test_no_git_installed(self):
        """Test behavior when git is not installed."""
        self.mocks['is_git_installed'].return_value = False
        result = self.runner.invoke(main, ['commit'])
        self.assertEqual(result.exit_code, 1, msg=result.output)
        self.assertIn("Error: Git is not installed or not found in PATH.", result.output)

    def test_successful_diff_becomes_successful_commit(self):
        """Test successful diff leading to a successful commit."""
        mock_git_commit_success = MagicMock(returncode=0)
        
        def subprocess_run_side_effect(*args, **kwargs):
            command_args = args[0]
            if command_args[0] == 'git' and command_args[1] == 'rev-parse':
                return self.mock_rev_parse_success
            elif command_args[0] == 'git' and command_args[1] == 'commit':
                if kwargs.get('check') and mock_git_commit_success.returncode != 0:
                    raise subprocess.CalledProcessError(mock_git_commit_success.returncode, command_args)
                return mock_git_commit_success
            unexpected_mock = MagicMock(returncode=1, stderr=b"Unexpected direct subprocess call in test")
            if kwargs.get('check'):
                raise subprocess.CalledProcessError(1, command_args)
            return unexpected_mock

        self.mocks['subprocess_run'].side_effect = subprocess_run_side_effect
        
        # Simulate initial message generation
        self.mocks['generate_commit_message'].return_value = "Initial Commit Message"
        # Set initial content for temp file write and subsequent read
        self.temp_file_content_store = "Initial Commit Message"


        test_input = "\nn\nc\n" # No issues, No breaking changes, Commit
        result = self.runner.invoke(main, ['commit'], input=test_input)
        
        self.assertEqual(result.exit_code, 0, msg=f"Output: {result.output} | Exception: {result.exception}")
        self.assertIn("Commit successful.", result.output)
        self.mocks['generate_commit_message'].assert_called_once()
        self.mocks['open_editor'].assert_called_once()
        self.mocks['subprocess_run'].assert_any_call(['git', 'commit', '-F', self.mock_temp_file_obj.name], check=True)
        self.mocks['os_remove'].assert_called_with(self.mock_temp_file_obj.name)

    def test_interactive_commit(self):
        """Test interactive commit message generation and actual commit."""
        initial_message = "AI Generated Message for interactive commit"
        self.mocks['generate_commit_message'].return_value = initial_message
        self.temp_file_content_store = initial_message # Set for reads/writes

        mock_git_commit_success = MagicMock(returncode=0)
        def subprocess_run_side_effect(*args, **kwargs):
            command_args = args[0]
            if command_args[0] == 'git' and command_args[1] == 'rev-parse':
                return self.mock_rev_parse_success
            elif command_args[0] == 'git' and command_args[1] == 'commit':
                return mock_git_commit_success
            return MagicMock(returncode=1, stderr=b"Unexpected subprocess call")
        self.mocks['subprocess_run'].side_effect = subprocess_run_side_effect

        test_input = "MYISSUE-1\nn\nc\n" # Issue "MYISSUE-1", No breaking, Commit
        result = self.runner.invoke(main, ['commit'], input=test_input)

        self.assertEqual(result.exit_code, 0, msg=f"Output: {result.output} | Exception: {result.exception}")
        self.assertIn("Commit successful.", result.output)
        self.mocks['generate_commit_message'].assert_called_once()
        self.mocks['open_editor'].assert_called_once()
        self.mocks['subprocess_run'].assert_any_call(['git', 'commit', '-F', self.mock_temp_file_obj.name], check=True)
        self.mocks['os_remove'].assert_called_with(self.mock_temp_file_obj.name)
        
        # To verify content written for commit:
        self.assertIn("Fixes: MYISSUE-1", self.temp_file_content_store) # Check final content

    def test_interactive_edit_again_then_abort(self):
        """Test choosing 'edit', then 'abort'."""
        initial_message = "AI Message for edit then abort"
        self.mocks['generate_commit_message'].return_value = initial_message
        self.temp_file_content_store = initial_message
        
        test_input = "\nn\ne\n\nn\na\n" # Loop1: No issues, No breaking, Edit; Loop2: No issues, No breaking, Abort
        result = self.runner.invoke(main, ['commit'], input=test_input)
        
        self.assertEqual(result.exit_code, 0, msg=f"Output: {result.output} | Exception: {result.exception}")
        self.assertIn("Aborted.", result.output)
        self.mocks['generate_commit_message'].assert_called_once()
        self.assertEqual(self.mocks['open_editor'].call_count, 2)
        self.assertEqual(self.mocks['os_remove'].call_count, 2) # Should be called after each loop iteration's finally block

    def test_interactive_abort(self):
        """Test interactive commit message abort."""
        initial_message = "AI Message for abort"
        self.mocks['generate_commit_message'].return_value = initial_message
        self.temp_file_content_store = initial_message
        
        test_input = "\nn\na\n" # No issues, No breaking changes, Abort action
        result = self.runner.invoke(main, ['commit'], input=test_input)
        
        self.assertEqual(result.exit_code, 0, msg=f"Output: {result.output} | Exception: {result.exception}")
        self.assertIn("Aborted.", result.output)
        self.mocks['generate_commit_message'].assert_called_once()
        self.mocks['open_editor'].assert_called_once()
        self.mocks['os_remove'].assert_called_with(self.mock_temp_file_obj.name)

if __name__ == '__main__':
    unittest.main()