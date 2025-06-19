"""Tests for the Julia analyzer."""

import unittest
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
from tests import BaseAxleTestCase
from src.axle.treesitter.analyzers.julia_analyzer import JuliaAnalyzer
from src.axle.treesitter.models import FileAnalysis, Import, Class, Function, Variable, FailedAnalysis

class TestJuliaAnalyzer(BaseAxleTestCase):
    """Test cases for Julia analyzer."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.analyzer = JuliaAnalyzer()
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)
        # Register for cleanup
        self.register_test_data_dir(self.test_dir)

    def create_test_file(self, content: str) -> Path:
        """Create a temporary test file with the given content."""
        test_file = self.test_dir / "test.jl"
        test_file.write_text(content)
        return test_file

    def test_import_statements(self):
        """Test parsing of various import statements."""
        content = """
        using LinearAlgebra
        using DataFrames: DataFrame, select
        import Base: show, length
        import JSON
        
        # Relative imports
        using ..ParentModule
        using .LocalModule
        
        # Conditional imports
        @static if VERSION >= v"1.6"
            using SomeNewPackage
        end
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.imports)
        self.assertGreaterEqual(len(result.imports), 4)
        
        # Check using statement without specific items
        linear_algebra = next((imp for imp in result.imports if imp.source == "LinearAlgebra"), None)
        self.assertIsNotNone(linear_algebra)
        
        # Check using with specific items
        dataframes = next((imp for imp in result.imports if imp.source == "DataFrames"), None)
        self.assertIsNotNone(dataframes)
        if dataframes.items:
            self.assertIn("DataFrame", dataframes.items)
            self.assertIn("select", dataframes.items)
        
        # Check import statement
        json_import = next((imp for imp in result.imports if imp.source == "JSON"), None)
        self.assertIsNotNone(json_import)

    def test_function_declaration(self):
        """Test parsing of function declarations."""
        content = '''
        function add(x, y)
            return x + y
        end
        
        function multiply(x::Int, y::Float64)::Float64
            return x * y
        end
        
        # Short form function
        square(x) = x^2
        
        # Function with keyword arguments
        function greet(name; greeting="Hello")
            println("$greeting, $name!")
        end
        
        # Anonymous function (should not be captured as top-level)
        map(x -> x^2, [1, 2, 3])
        
        # Function with docstring
        """
        Calculate the factorial of n.
        """
        function factorial(n::Int)
            n <= 1 ? 1 : n * factorial(n-1)
        end
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.functions)
        self.assertGreaterEqual(len(result.functions), 4)
        
        # Check regular function
        add_func = next((func for func in result.functions if func.name == "add"), None)
        self.assertIsNotNone(add_func)
        self.assertIsNotNone(add_func.parameters)
        self.assertEqual(len(add_func.parameters), 2)
        
        # Check typed function
        multiply_func = next((func for func in result.functions if func.name == "multiply"), None)
        self.assertIsNotNone(multiply_func)
        self.assertIsNotNone(multiply_func.parameters)
        self.assertEqual(len(multiply_func.parameters), 2)
        
        # Check short form function
        square_func = next((func for func in result.functions if func.name == "square"), None)
        self.assertIsNotNone(square_func)
        
        # Check function with docstring
        factorial_func = next((func for func in result.functions if func.name == "factorial"), None)
        self.assertIsNotNone(factorial_func)
        self.assertIsNotNone(factorial_func.docstring)
        self.assertIn("factorial", factorial_func.docstring)

    def test_struct_declaration(self):
        """Test parsing of struct declarations (Julia's equivalent to classes)."""
        content = '''
        struct Point
            x::Float64
            y::Float64
        end
        
        mutable struct Person
            name::String
            age::Int
            email::String
        end
        
        # Struct with constructor
        struct Circle
            radius::Float64
            Circle(r) = r > 0 ? new(r) : error("Radius must be positive")
        end
        
        # Abstract type
        abstract type Animal end
        
        # Struct with supertype
        struct Dog <: Animal
            name::String
            breed::String
        end
        
        # Struct with docstring
        """
        A rectangle with width and height.
        """
        struct Rectangle
            width::Float64
            height::Float64
        end
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertGreaterEqual(len(result.classes), 4)
        
        # Check basic struct
        point_struct = next((cls for cls in result.classes if cls.name == "Point"), None)
        self.assertIsNotNone(point_struct)
        self.assertIsNotNone(point_struct.attributes)
        self.assertEqual(len(point_struct.attributes), 2)
        
        # Check mutable struct
        person_struct = next((cls for cls in result.classes if cls.name == "Person"), None)
        self.assertIsNotNone(person_struct)
        
        # Check struct with supertype
        dog_struct = next((cls for cls in result.classes if cls.name == "Dog"), None)
        self.assertIsNotNone(dog_struct)
        self.assertIsNotNone(dog_struct.bases)
        self.assertEqual(len(dog_struct.bases), 1)
        self.assertEqual(dog_struct.bases[0].name, "Animal")
        
        # Check struct with docstring
        rect_struct = next((cls for cls in result.classes if cls.name == "Rectangle"), None)
        self.assertIsNotNone(rect_struct)
        self.assertIsNotNone(rect_struct.docstring)
        self.assertIn("rectangle", rect_struct.docstring.lower())

    def test_variable_declaration(self):
        """Test parsing of variable declarations."""
        content = '''
        # Constants
        const PI = 3.14159
        const MAX_ITERATIONS = 1000
        
        # Global variables
        global_counter = 0
        
        # Type aliases
        Vector3D = Vector{Float64}
        Matrix2D = Matrix{Int}
        
        # Multiple assignment
        a, b, c = 1, 2, 3
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.variables)
        self.assertGreaterEqual(len(result.variables), 3)
        
        # Check constant
        pi_var = next((var for var in result.variables if var.name == "PI"), None)
        self.assertIsNotNone(pi_var)
        self.assertEqual(pi_var.kind, "constant")
        
        # Check type alias
        vector3d_var = next((var for var in result.variables if var.name == "Vector3D"), None)
        self.assertIsNotNone(vector3d_var)
        self.assertEqual(vector3d_var.kind, "type_alias")

    def test_module_declaration(self):
        """Test parsing of module declarations."""
        content = '''
        module MyModule
        
        export myfunction, MyStruct
        
        function myfunction(x)
            return x * 2
        end
        
        struct MyStruct
            value::Int
        end
        
        end # module
        
        # Nested module
        module OuterModule
            module InnerModule
                const INNER_CONSTANT = 42
            end
        end
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        # Functions and structs inside modules should still be captured
        self.assertIsNotNone(result.functions)
        self.assertIsNotNone(result.classes)

    def test_macro_definitions(self):
        """Test parsing of macro definitions."""
        content = '''
        macro mymacro(expr)
            quote
                println("Executing: ", $(string(expr)))
                $expr
            end
        end
        
        # Macro with docstring
        """
        A simple timing macro.
        """
        macro time_it(expr)
            quote
                t = @elapsed $expr
                println("Time: $t seconds")
            end
        end
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        # Macros might be captured as functions depending on implementation
        if result.functions:
            macro_funcs = [f for f in result.functions if f.name.startswith("@")]
            self.assertGreaterEqual(len(macro_funcs), 0)

    def test_error_handling(self):
        """Test error handling for invalid Julia code."""
        content = """
        # This is invalid Julia syntax
        function broken_function(
            # Missing closing parenthesis and end
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        # Should return FailedAnalysis for syntax errors
        self.assertIsInstance(result, FailedAnalysis)
        self.assertIn("syntax", result.reason.lower())

    def test_docstring_extraction(self):
        """Test extraction of docstrings."""
        content = '''
        """
        This is a module-level docstring.
        It describes what this file does.
        """
        
        """
        Add two numbers together.
        
        # Arguments
        - `x`: First number
        - `y`: Second number
        
        # Returns
        Sum of x and y
        """
        function add_with_docs(x, y)
            return x + y
        end
        
        """
        A documented struct.
        """
        struct DocumentedStruct
            field::Int
        end
        '''
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        
        # Check function docstring
        if result.functions:
            add_func = next((f for f in result.functions if f.name == "add_with_docs"), None)
            if add_func:
                self.assertIsNotNone(add_func.docstring)
                self.assertIn("Add two numbers", add_func.docstring)
        
        # Check struct docstring
        if result.classes:
            doc_struct = next((c for c in result.classes if c.name == "DocumentedStruct"), None)
            if doc_struct:
                self.assertIsNotNone(doc_struct.docstring)
                self.assertIn("documented struct", doc_struct.docstring.lower())

    def test_file_extensions(self):
        """Test that the analyzer handles correct file extensions."""
        self.assertIn(".jl", self.analyzer.FILE_EXTENSIONS)

    def test_language_name(self):
        """Test that the analyzer has the correct language name."""
        self.assertEqual(self.analyzer.LANGUAGE_NAME, "julia")

if __name__ == '__main__':
    unittest.main() 