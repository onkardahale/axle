"""Tests for the JavaScript analyzer."""

import unittest
from pathlib import Path
from tree_sitter import Parser as TreeSitterParser
import logging
import shutil
logger = logging.getLogger(__name__)
from tests import BaseAxleTestCase
from src.axle.treesitter.analyzers.javascript_analyzer import JavaScriptAnalyzer
from src.axle.treesitter.models import FileAnalysis, Import, Class, Function, Variable, FailedAnalysis

class TestJavaScriptAnalyzer(BaseAxleTestCase):
    """Test cases for JavaScript analyzer."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.analyzer = JavaScriptAnalyzer()
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)
        # Register for cleanup
        self.register_test_data_dir(self.test_dir)

    def create_test_file(self, content: str) -> Path:
        """Create a temporary test file with the given content."""
        test_file = self.test_dir / "test.js"
        test_file.write_text(content)
        return test_file

    def test_import_statements(self):
        """Test parsing of various import statements."""
        content = """
        import React from 'react';
        import { useState, useEffect } from 'react';
        import * as utils from './utils';
        import { default as ReactDOM } from 'react-dom';
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.imports)
        self.assertEqual(len(result.imports), 4)
        
        # Check default import
        react_import = next(imp for imp in result.imports if imp.name == "React")
        self.assertEqual(react_import.source, "react")
        self.assertIsNone(react_import.items)
        
        # Check named imports
        react_hooks = next(imp for imp in result.imports if imp.source == "react" and imp.items)
        self.assertEqual(set(react_hooks.items), {"useState", "useEffect"})
        
        # Check namespace import
        utils_import = next(imp for imp in result.imports if imp.name == "utils")
        self.assertEqual(utils_import.source, "./utils")
        self.assertEqual(utils_import.items, ["*"])
        
        # Check aliased import
        react_dom = next(imp for imp in result.imports if imp.name == "ReactDOM")
        self.assertEqual(react_dom.source, "react-dom")
        self.assertIsNone(react_dom.items)

    def test_class_declaration(self):
        """Test parsing of class declarations."""
        content = """
        class User {
            constructor(name, age) {
                this.name = name;
                this.age = age;
            }
            
            getName() {
                return this.name;
            }
            
            static create(name, age) {
                return new User(name, age);
            }
        }
        
        class Admin extends User {
            constructor(name, age, role) {
                super(name, age);
                this.role = role;
            }
        }
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 2)
        
        # Check User class
        user_class = next(cls for cls in result.classes if cls.name == "User")
        self.assertIsNone(user_class.bases)
        self.assertIsNotNone(user_class.methods)
        self.assertEqual(len(user_class.methods), 3)  # constructor, getName, create
        
        # Check Admin class
        admin_class = next(cls for cls in result.classes if cls.name == "Admin")
        self.assertIsNotNone(admin_class.bases)
        self.assertEqual(len(admin_class.bases), 1)
        self.assertEqual(admin_class.bases[0].name, "User")
        self.assertIsNotNone(admin_class.methods)
        self.assertEqual(len(admin_class.methods), 1)  # constructor

    def test_function_declaration(self):
        """Test parsing of function declarations."""
        content = """
        function add(a, b) {
            return a + b;
        }

        const multiply = (a, b) => a * b; // Handled by variable processing or expression_statement logic

        function createUser(name, age = 20, ...rest) {
            return { name, age, ...rest };
        }
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)

        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.functions)
        # Arrow functions assigned to const/let/var are processed as variables,
        # or potentially added to functions list via expression_statement logic
        # if assigned to globals without const/let/var.
        self.assertEqual(len(result.functions), 2) 

        add_func = next(func for func in result.functions if func.name == "add")
        self.assertIsNotNone(add_func.parameters)

        self.assertEqual(len(add_func.parameters), 2)
        self.assertEqual(add_func.parameters[0].name, "a")
        self.assertEqual(add_func.parameters[1].name, "b")

        create_user = next(func for func in result.functions if func.name == "createUser")
        self.assertIsNotNone(create_user.parameters)
        self.assertEqual(len(create_user.parameters), 3)
        self.assertEqual(create_user.parameters[0].name, "name")
        self.assertEqual(create_user.parameters[1].name, "age")
        self.assertEqual(create_user.parameters[2].name, "...rest")

    def test_variable_declaration(self):
        """Test parsing of variable declarations."""
        content = """
        const MAX_RETRIES = 3;
        let currentUser = null;
        var oldStyle = 'deprecated';
        
        const config = {
            apiKey: 'secret',
            timeout: 5000
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.variables)
        self.assertEqual(len(result.variables), 4)
        
        # Check constant
        max_retries = next(var for var in result.variables if var.name == "MAX_RETRIES")
        self.assertEqual(max_retries.kind, "constant")
        self.assertEqual(max_retries.value, "3")
        
        # Check let variable
        current_user = next(var for var in result.variables if var.name == "currentUser")
        self.assertEqual(current_user.kind, "external_variable")
        self.assertEqual(current_user.value, "null")
        
        # Check var variable
        old_style = next(var for var in result.variables if var.name == "oldStyle")
        self.assertEqual(old_style.kind, "external_variable")
        self.assertEqual(old_style.value, "deprecated")

    def test_export_statements(self):
        """Test parsing of export statements."""
        content = """
        export const VERSION = '1.0.0';
        
        export function helper() {
            return true;
        }
        
        export class Service {
            constructor() {}
        }
        
        export { helper as util };
        
        export default class MainService {}
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.imports)
        
        # Check named exports
        named_exports = [imp for imp in result.imports if imp.items]
        self.assertTrue(any(imp.items == ["util"] for imp in named_exports))
        
        # Check default export
        default_export = next(imp for imp in result.imports if imp.name == "MainService")
        self.assertEqual(default_export.source, "MainService")

    def test_jsx_support(self):
        """Test basic JSX support."""
        content = """
        import React from 'react';
        
        function App() {
            return (
                <div className="app">
                    <h1>Hello World</h1>
                    <button onClick={() => console.log('clicked')}>
                        Click me
                    </button>
                </div>
            );
        }
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.functions)
        self.assertEqual(len(result.functions), 1)
        
        app_func = result.functions[0]
        self.assertEqual(app_func.name, "App")
        self.assertIsNone(app_func.parameters)

    def test_error_handling(self):
        """Test handling of syntactically valid but simple/edge-case JavaScript."""
        content = """function broken() { if (true) { return; } }"""
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.functions)
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].name, "broken")

if __name__ == '__main__':
    unittest.main() 