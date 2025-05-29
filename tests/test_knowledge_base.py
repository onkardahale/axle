"""Tests for the knowledge base module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from axle.knowledge_base import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        """Set up a temporary test directory with some Python files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
        # Create a test Python file
        test_file = self.test_dir / "test_file.py"
        test_file.write_text("""
import os
from typing import List, Dict

class TestClass:
    def __init__(self, name: str):
        self.name = name
    
    def test_method(self, param: int) -> str:
        return f"Hello {self.name}"

def test_function(arg: List[str]) -> Dict[str, int]:
    return {"test": 1}
""")
        
        # Create a test directory structure
        (self.test_dir / "src" / "test_package").mkdir(parents=True)
        package_file = self.test_dir / "src" / "test_package" / "__init__.py"
        package_file.write_text("""
from typing import Optional

def package_function() -> Optional[str]:
    return None
""")
    
    def tearDown(self):
        """Clean up the temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_knowledge_base_initialization(self):
        """Test that the knowledge base initializes correctly."""
        kb = KnowledgeBase(self.test_dir)
        self.assertEqual(kb.project_root, self.test_dir)
        self.assertEqual(kb.kb_dir, self.test_dir / '.axle')
        self.assertTrue(kb.kb_dir.exists())

    def test_analyze_file(self):
        """Test that file analysis works correctly."""
        kb = KnowledgeBase(self.test_dir)
        test_file = self.test_dir / "test_file.py"
        
        analysis = kb.analyze_file(test_file)
        self.assertIsNotNone(analysis)
        
        # Check basic structure
        self.assertIn('imports', analysis)
        self.assertIn('classes', analysis)
        self.assertIn('functions', analysis)
        self.assertIn('category', analysis)
        
        # Check imports
        imports = analysis['imports']
        self.assertTrue(any(imp.get('name') == 'os' for imp in imports))
        self.assertTrue(any(imp.get('name') == 'typing' for imp in imports))
        
        # Check class analysis
        classes = analysis['classes']
        self.assertEqual(len(classes), 1)
        test_class = classes[0]
        self.assertEqual(test_class['name'], 'TestClass')
        self.assertEqual(len(test_class['methods']), 2)  # __init__ and test_method
        
        # Check function analysis
        functions = analysis['functions']
        self.assertEqual(len(functions), 1)
        test_function = functions[0]
        self.assertEqual(test_function['name'], 'test_function')

    def test_build_knowledge_base(self):
        """Test that building the knowledge base works correctly."""
        kb = KnowledgeBase(self.test_dir)
        kb.build_knowledge_base()
        
        # Check that analysis files were created
        test_file_analysis = kb.kb_dir / "test_file.json"
        self.assertTrue(test_file_analysis.exists())
        
        # Check package file analysis
        package_analysis = kb.kb_dir / "src" / "test_package" / "__init__.json"
        self.assertTrue(package_analysis.exists())
        
        # Verify analysis content
        with open(test_file_analysis, 'r') as f:
            analysis = json.load(f)
            self.assertIn('imports', analysis)
            self.assertIn('classes', analysis)
            self.assertIn('functions', analysis)

    def test_get_file_analysis(self):
        """Test retrieving file analysis."""
        kb = KnowledgeBase(self.test_dir)
        kb.build_knowledge_base()
        
        # Test with .py extension
        analysis = kb.get_file_analysis(Path("test_file.py"))
        self.assertIsNotNone(analysis)
        self.assertIn('imports', analysis)
        
        # Test without extension
        analysis = kb.get_file_analysis(Path("test_file"))
        self.assertIsNotNone(analysis)
        self.assertIn('imports', analysis)
        
        # Test non-existent file
        analysis = kb.get_file_analysis(Path("nonexistent.py"))
        self.assertIsNone(analysis)

    def test_file_category_determination(self):
        """Test that file categories are determined correctly."""
        kb = KnowledgeBase(self.test_dir)
        
        # Test utility file
        util_file = self.test_dir / "utils" / "helper.py"
        util_file.parent.mkdir()
        util_file.write_text("import os")
        analysis = kb.analyze_file(util_file)
        self.assertEqual(analysis['category'], 'util')
        
        # Test test file
        test_file = self.test_dir / "tests" / "test_something.py"
        test_file.parent.mkdir()
        test_file.write_text("import pytest")
        analysis = kb.analyze_file(test_file)
        self.assertEqual(analysis['category'], 'test')

if __name__ == '__main__':
    unittest.main() 