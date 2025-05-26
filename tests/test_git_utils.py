import unittest
from unittest.mock import patch, MagicMock
import subprocess
from axle import git_utils

class TestGitUtils(unittest.TestCase):
    def test_is_git_installed_found(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            self.assertTrue(git_utils.is_git_installed())

    def test_is_git_installed_not_found(self):
        with patch('subprocess.run', side_effect=FileNotFoundError):
            self.assertFalse(git_utils.is_git_installed())

    def test_get_staged_diff_with_changes(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='diff content\n')
            self.assertEqual(git_utils.get_staged_diff(), 'diff content\n')

    def test_get_staged_diff_no_changes(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='')
            self.assertIsNone(git_utils.get_staged_diff())

    def test_get_staged_diff_git_fails(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout='', stderr='error')
            with self.assertRaises(RuntimeError):
                git_utils.get_staged_diff()

    def test_get_staged_diff_git_not_found(self):
        with patch('subprocess.run', side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError):
                git_utils.get_staged_diff()

    def test_get_staged_files_count_zero(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='')
            self.assertEqual(git_utils.get_staged_files_count(), 0)

    def test_get_staged_files_count_one(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='file1.py\n')
            self.assertEqual(git_utils.get_staged_files_count(), 1)

    def test_get_staged_files_count_multiple(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='file1.py\nfile2.py\nfile3.py\n')
            self.assertEqual(git_utils.get_staged_files_count(), 3)

    def test_get_staged_files_count_git_fails(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout='')
            self.assertEqual(git_utils.get_staged_files_count(), 0)

    def test_get_staged_files_count_exception(self):
        with patch('subprocess.run', side_effect=Exception):
            self.assertEqual(git_utils.get_staged_files_count(), 0)

if __name__ == '__main__':
    unittest.main() 