import unittest
from unittest.mock import patch, MagicMock, call
import os
from axle.editor_utils import get_default_editor, open_editor

class TestEditorUtils(unittest.TestCase):

    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get') # Keep this patch to control environment
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_git_config_success(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test that GIT_EDITOR is preferred and used if valid."""
        # Simulate 'git var GIT_EDITOR' returning 'git-editor --wait'
        mock_subprocess_run.return_value = MagicMock(stdout="git-editor --wait\n", returncode=0, stderr="")
        # Simulate shutil.which finding 'git-editor'
        mock_shutil_which.return_value = "/usr/bin/git-editor" # Path doesn't matter, just needs to be truthy
        # Simulate that os.environ.get returns None for EDITOR and VISUAL for this specific test
        # to ensure that even if called, they don't interfere with GIT_EDITOR taking precedence.
        mock_os_environ_get.return_value = None

        editor = get_default_editor()
        self.assertEqual(editor, "git-editor --wait")
        mock_subprocess_run.assert_called_once_with(
            ["git", "var", "GIT_EDITOR"], capture_output=True, text=True, check=False
        )
        # Assert that os.environ.get was called for EDITOR and VISUAL, as per current code structure
        # This is not strictly necessary to assert, but confirms understanding of current code flow.
        # If you refactor get_default_editor to conditional-check these, this assertion would change.
        mock_os_environ_get.assert_any_call("EDITOR")
        mock_os_environ_get.assert_any_call("VISUAL")

        # This is crucial: shutil.which should only be called for "git-editor"
        # because it's found first and the function should return.
        mock_shutil_which.assert_called_once_with("git-editor")

    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get')
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_env_editor(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test that EDITOR env var is used if GIT_EDITOR fails or is not set."""
        mock_subprocess_run.return_value = MagicMock(stdout="", returncode=0, stderr="")
        mock_os_environ_get.side_effect = lambda key, default=None: "env_editor_cmd" if key == "EDITOR" else None
        mock_shutil_which.side_effect = lambda cmd: "/usr/bin/env_editor_cmd" if cmd == "env_editor_cmd" else None

        editor = get_default_editor()
        self.assertEqual(editor, "env_editor_cmd")
        mock_subprocess_run.assert_called_once_with(
            ["git", "var", "GIT_EDITOR"], capture_output=True, text=True, check=False
        )
        mock_os_environ_get.assert_any_call("EDITOR")
        # shutil.which would be called for any candidate from GIT_EDITOR (if it existed and was non-empty)
        # and then for "env_editor_cmd".
        # If GIT_EDITOR was empty, then "env_editor_cmd" is the first one checked by shutil.which.
        mock_shutil_which.assert_called_once_with("env_editor_cmd")


    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get')
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_visual_editor(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test that VISUAL env var is used if GIT_EDITOR and EDITOR are not set."""
        mock_subprocess_run.return_value = MagicMock(stdout="", returncode=0)
        mock_os_environ_get.side_effect = lambda key, default=None: "visual_cmd" if key == "VISUAL" else (None if key == "EDITOR" else default)
        mock_shutil_which.side_effect = lambda cmd: "/usr/bin/visual_cmd" if cmd == "visual_cmd" else None


        editor = get_default_editor()
        self.assertEqual(editor, "visual_cmd")
        expected_env_calls = [call("EDITOR"), call("VISUAL")] # EDITOR is checked first, then VISUAL
        mock_os_environ_get.assert_has_calls(expected_env_calls, any_order=False)

        # shutil.which called for 'visual_cmd' (assuming GIT_EDITOR and EDITOR candidates were empty or not found by shutil.which)
        mock_shutil_which.assert_called_once_with("visual_cmd")


    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get')
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_fallback_to_nano(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test fallback to 'nano' if others are not found or not set."""
        mock_subprocess_run.return_value = MagicMock(stdout="", returncode=1, stderr="git error")
        mock_os_environ_get.return_value = None
        mock_shutil_which.side_effect = lambda cmd: "/usr/bin/nano" if cmd == "nano" else None

        editor = get_default_editor()
        self.assertEqual(editor, 'nano')
        # shutil.which is called for candidates in order until one is found.
        # If GIT_EDITOR, EDITOR, VISUAL candidates were empty (due to mocks), 'nano' is the first fallback checked.
        mock_shutil_which.assert_called_once_with('nano')


    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get')
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_fallback_to_vim(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test fallback to 'vim' if nano is not found."""
        mock_subprocess_run.return_value = MagicMock(stdout="", returncode=1, stderr="git error")
        mock_os_environ_get.return_value = None
        mock_shutil_which.side_effect = lambda cmd: "/usr/bin/vim" if cmd == "vim" else (None if cmd == "nano" else None)

        editor = get_default_editor()
        self.assertEqual(editor, 'vim')
        expected_shutil_calls = [call('nano'), call('vim')]
        mock_shutil_which.assert_has_calls(expected_shutil_calls, any_order=False)

    @patch('axle.editor_utils.shutil.which')
    @patch('axle.editor_utils.os.environ.get')
    @patch('axle.editor_utils.subprocess.run')
    def test_get_default_editor_ultimate_fallback_to_vi(self, mock_subprocess_run, mock_os_environ_get, mock_shutil_which):
        """Test ultimate fallback to 'vi' if nothing else is found."""
        mock_subprocess_run.return_value = MagicMock(stdout="", returncode=1, stderr="git error")
        mock_os_environ_get.return_value = None
        mock_shutil_which.side_effect = lambda cmd: "/usr/bin/vi" if cmd == "vi" else None # Only vi is "found"

        editor = get_default_editor()
        self.assertEqual(editor, 'vi')
        expected_shutil_calls = [call('nano'), call('vim'), call('vi')]
        mock_shutil_which.assert_has_calls(expected_shutil_calls, any_order=False)


    @patch('axle.editor_utils.subprocess.run')
    @patch('axle.editor_utils.get_default_editor', return_value='vim')
    def test_open_editor_success(self, mock_get_default_editor, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        result = open_editor('file.txt')
        self.assertTrue(result)
        mock_get_default_editor.assert_called_once()
        mock_subprocess_run.assert_called_once_with(['vim', 'file.txt'], check=False)

    @patch('axle.editor_utils.subprocess.run')
    @patch('axle.editor_utils.get_default_editor', return_value='nano')
    def test_open_editor_failure_command_error(self, mock_get_default_editor, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=1)
        result = open_editor('file.txt')
        self.assertFalse(result)
        mock_get_default_editor.assert_called_once()
        mock_subprocess_run.assert_called_once_with(['nano', 'file.txt'], check=False)

    @patch('axle.editor_utils.subprocess.run')
    @patch('axle.editor_utils.get_default_editor', return_value='invalid-editor-cmd')
    def test_open_editor_failure_command_not_found(self, mock_get_default_editor, mock_subprocess_run):
        mock_subprocess_run.side_effect = FileNotFoundError("Editor binary not found")
        result = open_editor('file.txt')
        self.assertFalse(result)
        mock_get_default_editor.assert_called_once()
        mock_subprocess_run.assert_called_once_with(['invalid-editor-cmd', 'file.txt'], check=False)

    @patch('axle.editor_utils.subprocess.run')
    @patch('axle.editor_utils.get_default_editor', return_value='editor-raising-exception')
    def test_open_editor_failure_other_exception(self, mock_get_default_editor, mock_subprocess_run):
        mock_subprocess_run.side_effect = Exception("Some other editor failure")
        result = open_editor('file.txt')
        self.assertFalse(result)
        mock_get_default_editor.assert_called_once()
        mock_subprocess_run.assert_called_once_with(['editor-raising-exception', 'file.txt'], check=False)

if __name__ == '__main__':
    unittest.main()