import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.axle.knowledge_base import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        """Set up a temporary project directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create a simple Python file with classes and functions
        test_file = self.project_root / "test_module.py"
        test_file.write_text('''
"""
Test module for knowledge base testing.
"""
from typing import List, Optional

class TestClass:
    """A test class with methods."""
    
    def __init__(self, name: str):
        """Initialize the test class.
        
        Args:
            name: The name of the test instance
        """
        self.name = name
    
    def test_method(self, param: int) -> str:
        """A test method.
        
        Args:
            param: A test parameter
            
        Returns:
            A test string
        """
        return f"Test {param}"

def test_function(value: Optional[List[str]] = None) -> bool:
    """A test function.
    
    Args:
        value: Optional list of strings
        
    Returns:
        A boolean value
    """
    return bool(value)
''')

        # Create a test file in a subdirectory
        subdir = self.project_root / "subdir"
        subdir.mkdir()
        test_file2 = subdir / "test_util.py"
        test_file2.write_text('''
"""
Utility module for testing.
"""
import os
import sys

def util_function():
    """A utility function."""
    pass
''')

        # Create a test file with framework imports
        test_file3 = self.project_root / "test_web.py"
        test_file3.write_text('''
"""
Web framework test module.
"""
from flask import Flask
from django.http import HttpResponse

app = Flask(__name__)

def handle_request():
    """Handle web request."""
    pass
''')

    def tearDown(self):
        """Clean up the temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization."""
        kb = KnowledgeBase(self.project_root)
        self.assertEqual(kb.project_root, self.project_root)
        self.assertEqual(kb.kb_dir, self.project_root / '.axle')
        self.assertTrue(kb.kb_dir.exists())

    def test_file_analysis(self):
        """Test file analysis functionality."""
        kb = KnowledgeBase(self.project_root)
        test_file = self.project_root / "test_module.py"
        
        analysis = kb.analyze_file(test_file)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['path'], "test_module.py")
        self.assertIn("typing", analysis['imports'])
        self.assertEqual(len(analysis['classes']), 1)
        self.assertEqual(len(analysis['functions']), 1)
        
        # Check class analysis
        test_class = analysis['classes'][0]
        self.assertEqual(test_class['name'], "TestClass")
        self.assertEqual(test_class['docstring'], "A test class with methods.")
        self.assertEqual(len(test_class['methods']), 2)  # __init__ and test_method
        # Check __init__ method docstring
        init_method = next((m for m in test_class['methods'] if m['name'] == '__init__'), None)
        self.assertIsNotNone(init_method)
        self.assertIn("Initialize the test class", init_method['docstring'])
        
        # Check function analysis
        test_func = analysis['functions'][0]
        self.assertEqual(test_func['name'], "test_function")
        self.assertIn("A test function", test_func['docstring'])
        self.assertEqual(len(test_func['parameters']), 1)

    def test_file_category_detection(self):
        """Test file category detection."""
        kb = KnowledgeBase(self.project_root)
        
        # Test utility file
        util_file = self.project_root / "subdir" / "test_util.py"
        analysis = kb.analyze_file(util_file)
        self.assertEqual(analysis['category'], "util")
        
        # Test web framework file
        web_file = self.project_root / "test_web.py"
        analysis = kb.analyze_file(web_file)
        self.assertEqual(analysis['category'], "web_framework")

    def test_build_knowledge_base(self):
        """Test building the complete knowledge base."""
        kb = KnowledgeBase(self.project_root)
        kb.build_knowledge_base()
        
        # Check that knowledge base files were created
        self.assertTrue((kb.kb_dir / "test_module.json").exists())
        self.assertTrue((kb.kb_dir / "subdir" / "test_util.json").exists())
        self.assertTrue((kb.kb_dir / "test_web.json").exists())
        
        # Check content of a knowledge base file
        with open(kb.kb_dir / "test_module.json") as f:
            data = json.load(f)
            self.assertEqual(data['path'], "test_module.py")
            self.assertEqual(len(data['classes']), 1)
            self.assertEqual(len(data['functions']), 1)

    def test_get_file_analysis(self):
        """Test retrieving file analysis from knowledge base."""
        kb = KnowledgeBase(self.project_root)
        kb.build_knowledge_base()
        
        # Test existing file
        analysis = kb.get_file_analysis(Path("test_module.py"))
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['path'], "test_module.py")
        
        # Test non-existent file
        analysis = kb.get_file_analysis(Path("nonexistent.py"))
        self.assertIsNone(analysis)

    def test_knowledge_base_staleness(self):
        """Test knowledge base staleness detection."""
        kb = KnowledgeBase(self.project_root)
        kb.build_knowledge_base()
        
        # Initially should not be stale
        self.assertFalse(kb.is_stale())
        
        # Create a new commit to make it stale
        os.chdir(self.project_root)
        os.system('git init')
        os.system('git add .')
        os.system('git commit -m "Initial commit"')
        os.system('git commit --allow-empty -m "Empty commit"')
        os.system('git commit --allow-empty -m "Another empty commit"')
        
        # Should be stale after multiple commits
        self.assertTrue(kb.is_stale())

    def test_skip_invalid_files(self):
        """Test handling of invalid files."""
        kb = KnowledgeBase(self.project_root)
        
        # Create an invalid Python file
        invalid_file = self.project_root / "invalid.py"
        invalid_file.write_text('''
def invalid_function(
    # Missing closing parenthesis
''')
        
        # Create a binary file
        binary_file = self.project_root / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')
        
        kb.build_knowledge_base()
        
        # Check that init.log was created
        self.assertTrue((kb.kb_dir / "init.log").exists())
        
        # Check that invalid files were logged
        with open(kb.kb_dir / "init.log") as f:
            log_content = f.read()
            self.assertIn("invalid.py", log_content)
            self.assertIn("binary.bin", log_content)

if __name__ == '__main__':
    unittest.main() 