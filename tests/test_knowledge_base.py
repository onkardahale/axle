"""Tests for the knowledge base module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from axle.knowledge_base import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        """Set up a temporary test directory with some Python and JS files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
        # Create a test Python file
        self.py_test_file_path = self.test_dir / "test_file.py"
        self.py_test_file_path.write_text("""
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
        
        # Create a test JavaScript file
        self.js_test_file_path = self.test_dir / "test_js_file.js"
        self.js_test_file_path.write_text("""
import fs from 'fs'; // For import testing

class JsTestClass {
    constructor(name) {
        this.name = name;
    }
    greet() {
        return `Hello ${this.name}`;
    }
}

function jsTestFunction(a, b) {
    return a + b;
}

const MY_CONSTANT = 123;
""")
        
        # Create a test directory structure (Python package)
        (self.test_dir / "src" / "test_package").mkdir(parents=True)
        self.py_package_file_path = self.test_dir / "src" / "test_package" / "__init__.py"
        self.py_package_file_path.write_text("""
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
        """Test that file analysis works correctly for Python and JavaScript."""
        kb = KnowledgeBase(self.test_dir)
        
        # Test Python file analysis
        py_analysis = kb.analyze_file(self.py_test_file_path)
        self.assertIsNotNone(py_analysis, "Python file analysis should not be None")
        
        self.assertIn('imports', py_analysis)
        self.assertIn('classes', py_analysis)
        self.assertIn('functions', py_analysis)
        self.assertIn('category', py_analysis)
        
        py_imports = py_analysis['imports']
        # Handle both string and dict formats (due to optimization)
        self.assertTrue(any(
            imp == 'os' or (isinstance(imp, dict) and imp.get('name') == 'os') 
            for imp in py_imports
        ))
        self.assertTrue(any(
            imp == 'typing' or (isinstance(imp, dict) and imp.get('name') == 'typing')
            for imp in py_imports
        ))
        
        py_classes = py_analysis['classes']
        self.assertEqual(len(py_classes), 1)
        py_test_class = py_classes[0]
        self.assertEqual(py_test_class['name'], 'TestClass')
        self.assertEqual(len(py_test_class['methods']), 2)  # __init__ and test_method
        
        py_functions = py_analysis['functions']
        self.assertEqual(len(py_functions), 1)
        py_test_function = py_functions[0]
        self.assertEqual(py_test_function['name'], 'test_function')

        # Test JavaScript file analysis
        js_analysis = kb.analyze_file(self.js_test_file_path)
        self.assertIsNotNone(js_analysis, "JavaScript file analysis should not be None")

        self.assertIn('imports', js_analysis)
        self.assertIn('classes', js_analysis)
        self.assertIn('functions', js_analysis)
        self.assertIn('variables', js_analysis)
        self.assertIn('category', js_analysis)

        js_imports = js_analysis['imports']
        # Handle both string and dict formats (due to optimization)
        self.assertTrue(any(
            imp == 'fs' or (isinstance(imp, dict) and imp.get('source') == 'fs')
            for imp in js_imports
        ), "Should find import from 'fs'")

        js_classes = js_analysis['classes']
        self.assertEqual(len(js_classes), 1)
        js_test_class = js_classes[0]
        self.assertEqual(js_test_class['name'], 'JsTestClass')
        self.assertTrue(any(method.get('name') == 'greet' for method in js_test_class['methods']))
        
        js_functions = js_analysis['functions']
        self.assertEqual(len(js_functions), 1)
        self.assertEqual(js_functions[0]['name'], 'jsTestFunction')

        js_variables = js_analysis['variables']
        self.assertEqual(len(js_variables), 1)
        self.assertEqual(js_variables[0]['name'], 'MY_CONSTANT')
        self.assertEqual(js_variables[0]['value'], '123')


    def test_build_knowledge_base(self):
        """Test that building the knowledge base works correctly for multiple languages."""
        kb = KnowledgeBase(self.test_dir)
        kb.build_knowledge_base()
        
        # Check that analysis files were created for Python files
        py_test_file_analysis_path = kb.kb_dir / "test_file.json"
        self.assertTrue(py_test_file_analysis_path.exists())
        
        py_package_analysis_path = kb.kb_dir / "src" / "test_package" / "__init__.json"
        self.assertTrue(py_package_analysis_path.exists())
        
        # Verify Python analysis content
        with open(py_test_file_analysis_path, 'r') as f:
            py_analysis_content = json.load(f)
            self.assertIn('imports', py_analysis_content)
            self.assertIn('classes', py_analysis_content)
            self.assertIn('functions', py_analysis_content)

        # Check that analysis file was created for JavaScript file
        js_test_file_analysis_path = kb.kb_dir / "test_js_file.json"
        self.assertTrue(js_test_file_analysis_path.exists(), 
                        f"JS analysis file missing. KB dir contents: {list(kb.kb_dir.glob('**/*'))}")

        # Verify JavaScript analysis content
        with open(js_test_file_analysis_path, 'r') as f:
            js_analysis_content = json.load(f)
            self.assertIn('imports', js_analysis_content)
            self.assertIn('classes', js_analysis_content)
            self.assertIn('functions', js_analysis_content)
            self.assertIn('variables', js_analysis_content)


    def test_get_file_analysis(self):
        """Test retrieving file analysis for different file types."""
        kb = KnowledgeBase(self.test_dir)
        kb.build_knowledge_base()
        
        # Test Python file with .py extension
        py_analysis_ext = kb.get_file_analysis(Path("test_file.py"))
        self.assertIsNotNone(py_analysis_ext)
        self.assertIn('imports', py_analysis_ext)
        
        # Test Python file without extension
        py_analysis_no_ext = kb.get_file_analysis(Path("test_file"))
        self.assertIsNotNone(py_analysis_no_ext)
        self.assertIn('imports', py_analysis_no_ext)

        # Test JavaScript file with .js extension
        js_analysis_ext = kb.get_file_analysis(Path("test_js_file.js"))
        self.assertIsNotNone(js_analysis_ext, "Failed to get JS file analysis with .js extension")
        self.assertIn('imports', js_analysis_ext)
        
        # Test JavaScript file without extension
        js_analysis_no_ext = kb.get_file_analysis(Path("test_js_file"))
        self.assertIsNotNone(js_analysis_no_ext, "Failed to get JS file analysis without extension")
        self.assertIn('imports', js_analysis_no_ext)
        
        # Test non-existent file
        non_existent_analysis = kb.get_file_analysis(Path("nonexistent.py"))
        self.assertIsNone(non_existent_analysis)

    def test_file_category_determination(self):
        """Test that file categories are determined correctly for different languages."""
        kb = KnowledgeBase(self.test_dir)
        
        # Test Python utility file
        py_util_file = self.test_dir / "utils" / "py_helper.py"
        py_util_file.parent.mkdir(parents=True, exist_ok=True)
        py_util_file.write_text("import os\ndef py_util_func(): pass")
        py_util_analysis = kb.analyze_file(py_util_file)
        self.assertIsNotNone(py_util_analysis, f"Analysis failed for {py_util_file}")
        self.assertEqual(py_util_analysis['category'], 'util')
        
        # Test Python test file
        py_test_file_cat = self.test_dir / "tests" / "test_something.py"
        py_test_file_cat.parent.mkdir(parents=True, exist_ok=True)
        py_test_file_cat.write_text("import pytest\ndef test_example(): assert True")
        py_test_cat_analysis = kb.analyze_file(py_test_file_cat)
        self.assertIsNotNone(py_test_cat_analysis, f"Analysis failed for {py_test_file_cat}")
        self.assertEqual(py_test_cat_analysis['category'], 'test')

        # Test JavaScript utility file
        js_util_file = self.test_dir / "utils" / "js_helper.js"
        # utils directory already created
        js_util_file.write_text("export function jsUtilFunc() {}")
        js_util_analysis = kb.analyze_file(js_util_file)
        self.assertIsNotNone(js_util_analysis, f"Analysis failed for {js_util_file}")
        self.assertEqual(js_util_analysis['category'], 'util')

        # Test JavaScript test file (e.g. spec file with jest import)
        js_test_file_cat = self.test_dir / "tests" / "another_test.spec.js"
        # tests directory already created
        js_test_file_cat.write_text("import { describe, it } from 'jest';\ndescribe('my suite', () => { it('should pass', () => expect(true).toBe(true)); });")
        js_test_cat_analysis = kb.analyze_file(js_test_file_cat)
        self.assertIsNotNone(js_test_cat_analysis, f"Analysis failed for {js_test_file_cat}")
        self.assertEqual(js_test_cat_analysis['category'], 'test')

if __name__ == '__main__':
    unittest.main()