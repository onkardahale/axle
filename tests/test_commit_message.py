import unittest
from axle.commit_message import generate_mock_commit_message, derive_scope

class TestCommitMessage(unittest.TestCase):
    def test_one_file(self):
        msg = generate_mock_commit_message('diff', 1)
        self.assertEqual(msg, "feat: Implement a new feature")

    def test_zero_files(self):
        msg = generate_mock_commit_message('diff', 0)
        self.assertEqual(msg, "feat: Implement a new feature")

    def test_two_files(self):
        msg = generate_mock_commit_message('diff', 2)
        self.assertTrue(msg.startswith("feat: Implement a new feature\n\n"))
        self.assertIn("This change introduces a new capability to the system.", msg)
        self.assertIn("Further details about the implementation.", msg)

    def test_many_files(self):
        msg = generate_mock_commit_message('diff', 5)
        self.assertTrue(msg.startswith("feat: Implement a new feature\n\n"))
        self.assertIn("This change introduces a new capability to the system.", msg)
        self.assertIn("Further details about the implementation.", msg)

    def test_derive_scope_empty(self):
        self.assertEqual(derive_scope([]), "")

    def test_derive_scope_single_file(self):
        self.assertEqual(derive_scope(["file1.py"]), "file1.py")

    def test_derive_scope_multiple_files_same_dir(self):
        self.assertEqual(derive_scope(["dir/file1.py", "dir/file2.py"]), "dir")

    def test_derive_scope_multiple_files_root(self):
        self.assertEqual(derive_scope(["/file1.py", "/file2.py"]), "")

if __name__ == '__main__':
    unittest.main() 