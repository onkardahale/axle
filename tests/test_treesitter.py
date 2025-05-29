"""Tests for Tree-sitter integration."""

import json
import unittest
from pathlib import Path
import tempfile
import os
from typing import Optional 

from axle.treesitter import TreeSitterParser
from axle.treesitter.exceptions import TreeSitterError, GrammarError

# Check if the new tree-sitter-language-pack is available
_NEW_TSLP_AVAILABLE = True
_NEW_TSLP_IMPORT_ERROR_MSG = ""
try:
    from tree_sitter_language_pack import get_parser as get_new_tslp_parser
    if get_new_tslp_parser is None:
        _NEW_TSLP_AVAILABLE = False
        _NEW_TSLP_IMPORT_ERROR_MSG = "get_parser is None after successful import of tree_sitter_language_pack"
except ImportError as e:
    _NEW_TSLP_AVAILABLE = False
    _NEW_TSLP_IMPORT_ERROR_MSG = f"Failed to import from tree_sitter_language_pack: {e}. Ensure it's installed."


class TestTreeSitter(unittest.TestCase):
    """Test cases for Tree-sitter integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def create_sample_python_file(self, content: Optional[str] = None) -> Path:
        """Create a sample Python file for testing."""
        if content is None:
            content = '''"""Test module docstring."""

import os
from typing import List, Optional
from abc import *
import sys

MAX_RETRIES = 3

class BaseService:
    """Base service class."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        """Get the service name."""
        return self.name

class UserService(BaseService):
    """Service for user operations."""

    def __init__(self, db_url: str):
        super().__init__("user")
        self.db_url = db_url

    def create_user(self, username: str, email: Optional[str] = None) -> bool:
        """Create a new user."""
        # Implementation would go here
        return True

def initialize_services() -> List[BaseService]:
    """Initialize all services."""
    return [UserService("sqlite:///users.db")]
'''
        file_path = self.temp_path / "test_service.py"
        file_path.write_text(content)
        return file_path

    @unittest.skipIf(not _NEW_TSLP_AVAILABLE, f"Skipping Python analysis test: new tree-sitter-language-pack not available. Error: {_NEW_TSLP_IMPORT_ERROR_MSG}")
    def test_python_analyzer(self):
        """Test that the Python analyzer correctly processes a sample file."""
        sample_file = self.create_sample_python_file()
        result = self.parser.analyze_file(sample_file)

        if not hasattr(result, "model_dump"):
            self.fail(f"Analysis result is not a recognized model type: {type(result)}")

        # Check if it's FailedAnalysis (which it shouldn't be for valid code)
        # Assuming FailedAnalysis model has a 'reason' field
        if "reason" in result.model_dump(exclude_none=True) and result.model_dump(exclude_none=True).get("reason") is not None:
             self.fail(f"Analysis unexpectedly failed: {result.model_dump().get('reason')}")
        
        data = result.model_dump(exclude_none=True)

        self.assertEqual(data["file_path"], str(sample_file))
        self.assertEqual(data["analyzer"], "treesitter_python")
        self.assertIn("imports", data, "Field 'imports' missing from analysis data.")
        self.assertEqual(len(data["imports"]), 4)
        imports = {imp["name"]: imp for imp in data["imports"]}
        self.assertIn("os", imports)
        self.assertIn("typing", imports)
        self.assertIn("sys", imports)
        self.assertEqual(imports["typing"]["items"], ["List", "Optional"])
        self.assertIn("classes", data, "Field 'classes' missing from analysis data.")
        self.assertEqual(len(data["classes"]), 2)
        classes = {cls["name"]: cls for cls in data["classes"]}
        self.assertIn("BaseService", classes)
        base = classes["BaseService"]
        self.assertEqual(base["docstring"], "Base service class.")
        self.assertEqual(len(base["methods"]), 2)
        methods_base = {m["name"]: m for m in base["methods"]}
        self.assertIn("get_name", methods_base)
        self.assertEqual(methods_base["get_name"]["docstring"], "Get the service name.")
        self.assertIn("__init__", methods_base)
        self.assertIn("UserService", classes)
        user = classes["UserService"]
        self.assertEqual(user["docstring"], "Service for user operations.")
        self.assertEqual(len(user["bases"]), 1)
        self.assertEqual(user["bases"][0]["name"], "BaseService")
        self.assertEqual(len(user["methods"]), 2)
        methods_user = {m["name"]: m for m in user["methods"]}
        self.assertIn("create_user", methods_user)
        self.assertEqual(methods_user["create_user"]["docstring"], "Create a new user.")
        self.assertEqual(len(methods_user["create_user"]["parameters"]), 3)
        self.assertIn("__init__", methods_user)
        self.assertIn("functions", data, "Field 'functions' missing from analysis data.")
        self.assertEqual(len(data["functions"]), 1)
        func = data["functions"][0]
        self.assertEqual(func["name"], "initialize_services")
        self.assertEqual(func["docstring"], "Initialize all services.")
        self.assertIn("variables", data, "Field 'variables' missing from analysis data.")
        self.assertEqual(len(data["variables"]), 1)
        var = data["variables"][0]
        self.assertEqual(var["name"], "MAX_RETRIES")
        self.assertEqual(var["kind"], "constant")
        self.assertEqual(var["value"], "3")

    def test_unsupported_file(self):
        """Test that analyzing an unsupported file type raises TreeSitterError."""
        file_path = self.temp_path / "test.txt"
        file_path.write_text("This is not a source file")
        with self.assertRaises(TreeSitterError) as context:
            self.parser.analyze_file(file_path)
        self.assertIn("Unsupported file type", str(context.exception))
        self.assertIn(".txt", str(context.exception))

    def test_nonexistent_file(self):
        """Test that analyzing a nonexistent file raises TreeSitterError."""
        non_existent_path = self.temp_path / "nonexistent_unique_name.py"
        with self.assertRaises(TreeSitterError) as context:
            self.parser.analyze_file(non_existent_path)
        self.assertIn("File not found", str(context.exception))
        self.assertIn(str(non_existent_path), str(context.exception))

    @unittest.skipIf(not _NEW_TSLP_AVAILABLE, f"Skipping Python syntax error test: new tree-sitter-language-pack not available. Error: {_NEW_TSLP_IMPORT_ERROR_MSG}")
    def test_python_file_with_syntax_error(self):
        """Test that a Python file with syntax errors results in FailedAnalysis."""
        invalid_python_code = """
def valid_func():
    pass

def invalid_func(
    print("oops" # Syntax error here
"""
        sample_file = self.create_sample_python_file(content=invalid_python_code)
        result = self.parser.analyze_file(sample_file)
        self.assertTrue(hasattr(result, "model_dump"), "Result object does not have model_dump method.")
        data = result.model_dump(exclude_none=True)
        self.assertEqual(data["file_path"], str(sample_file))
        self.assertEqual(data["analyzer"], "treesitter_python")
        self.assertIn("reason", data, "FailedAnalysis should have a reason for syntax errors.")
        self.assertIn("syntax error", data["reason"].lower(), "Reason should indicate a syntax error.")

if __name__ == '__main__':
    unittest.main()
