"""Tests for the C++ analyzer."""

import unittest
from pathlib import Path
from tree_sitter import Parser as TreeSitterParser
import logging
import shutil
logger = logging.getLogger(__name__)
from tests import BaseAxleTestCase
from src.axle.treesitter.analyzers.cpp_analyzer import CppAnalyzer
from src.axle.treesitter.models import FileAnalysis, Import, Class, Function, Variable, FailedAnalysis, Enum, BaseClass, Method, Parameter, Attribute, EnumMember

class TestCppAnalyzer(BaseAxleTestCase):
    """Test cases for C++ analyzer."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.analyzer = CppAnalyzer()
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)
        # Register for cleanup
        self.register_test_data_dir(self.test_dir)

    def create_test_file(self, content: str, extension: str = ".cpp") -> Path:
        """Create a temporary test file with the given content."""
        test_file = self.test_dir / f"test{extension}"
        test_file.write_text(content)
        return test_file

    def test_include_statements(self):
        """Test parsing of various include statements."""
        content = """
        #include <iostream>
        #include <vector>
        #include "my_header.h"
        #include <string>
        #include "utils/helper.hpp"
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.imports)
        self.assertEqual(len(result.imports), 5)
        
        # Check system includes
        iostream_import = next(imp for imp in result.imports if imp.source == "iostream")
        self.assertEqual(iostream_import.name, "iostream")
        
        vector_import = next(imp for imp in result.imports if imp.source == "vector")
        self.assertEqual(vector_import.name, "vector")
        
        # Check local includes
        my_header = next(imp for imp in result.imports if imp.source == "my_header.h")
        self.assertEqual(my_header.name, "my_header.h")
        
        helper_import = next(imp for imp in result.imports if imp.source == "utils/helper.hpp")
        self.assertEqual(helper_import.name, "utils/helper.hpp")

    def test_class_declaration(self):
        """Test parsing of class declarations."""
        content = """
        class Person {
        private:
            std::string name;
            int age;
            
        public:
            Person(const std::string& n, int a) : name(n), age(a) {}
            
            std::string getName() const {
                return name;
            }
            
            void setAge(int newAge) {
                age = newAge;
            }
            
            static int getMaxAge() {
                return 150;
            }
        };
        
        class Employee : public Person {
        private:
            std::string department;
            
        public:
            Employee(const std::string& n, int a, const std::string& dept) 
                : Person(n, a), department(dept) {}
                
            std::string getDepartment() const {
                return department;
            }
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 2)
        
        # Check Person class
        person_class = next(cls for cls in result.classes if cls.name == "Person")
        self.assertIsNone(person_class.bases)
        self.assertIsNotNone(person_class.methods)
        self.assertEqual(len(person_class.methods), 4)  # constructor, getName, setAge, getMaxAge
        self.assertIsNotNone(person_class.attributes)
        self.assertEqual(len(person_class.attributes), 2)  # name, age
        
        # Check Employee class
        employee_class = next(cls for cls in result.classes if cls.name == "Employee")
        self.assertIsNotNone(employee_class.bases)
        self.assertEqual(len(employee_class.bases), 1)
        self.assertEqual(employee_class.bases[0].name, "Person")
        self.assertEqual(employee_class.bases[0].access, "public")

    def test_function_declaration(self):
        """Test parsing of function declarations."""
        content = """
        int add(int a, int b) {
            return a + b;
        }

        double calculate(double x, double y = 1.0) {
            return x * y;
        }
        
        template<typename T>
        T max(T a, T b) {
            return (a > b) ? a : b;
        }
        
        void processData(const std::vector<int>& data, int& result) {
            result = data.size();
        }
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)

        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.functions)
        self.assertEqual(len(result.functions), 4)

        # Check add function
        add_func = next(func for func in result.functions if func.name == "add")
        self.assertIsNotNone(add_func.parameters)
        self.assertEqual(len(add_func.parameters), 2)
        self.assertEqual(add_func.parameters[0].name, "a")
        self.assertEqual(add_func.parameters[0].type, "int")
        self.assertEqual(add_func.parameters[1].name, "b")
        self.assertEqual(add_func.parameters[1].type, "int")

        # Check calculate function with default parameter
        calc_func = next(func for func in result.functions if func.name == "calculate")
        self.assertIsNotNone(calc_func.parameters)
        self.assertEqual(len(calc_func.parameters), 2)
        self.assertEqual(calc_func.parameters[0].name, "x")
        self.assertEqual(calc_func.parameters[0].type, "double")
        self.assertEqual(calc_func.parameters[1].name, "y")
        self.assertEqual(calc_func.parameters[1].type, "double")

        # Check template function
        max_func = next(func for func in result.functions if func.name == "max")
        self.assertIsNotNone(max_func.parameters)
        self.assertEqual(len(max_func.parameters), 2)

    def test_variable_declaration(self):
        """Test parsing of variable declarations."""
        content = """
        const int MAX_SIZE = 100;
        static int counter = 0;
        extern int global_var;
        
        int numbers[10];
        std::vector<std::string> names;
        auto result = calculateSomething();
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.variables)
        self.assertEqual(len(result.variables), 5)
        
        # Check constant
        max_size = next(var for var in result.variables if var.name == "MAX_SIZE")
        self.assertEqual(max_size.kind, "constant")
        self.assertEqual(max_size.type, "const int")
        self.assertEqual(max_size.value, "100")
        
        # Check static variable
        counter = next(var for var in result.variables if var.name == "counter")
        self.assertEqual(counter.kind, "external_variable")
        self.assertEqual(counter.type, "static int")
        self.assertEqual(counter.value, "0")
        
        # Check extern variable
        global_var = next(var for var in result.variables if var.name == "global_var")
        self.assertEqual(global_var.kind, "external_variable")
        self.assertEqual(global_var.type, "extern int")

    def test_enum_declaration(self):
        """Test parsing of enum declarations."""
        content = """
        enum Color {
            RED,
            GREEN,
            BLUE
        };
        
        enum class Status : int {
            PENDING = 0,
            APPROVED = 1,
            REJECTED = 2
        };
        
        enum Direction {
            NORTH = 1,
            SOUTH,
            EAST = 10,
            WEST
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.enums)
        self.assertEqual(len(result.enums), 3)
        
        # Check Color enum
        color_enum = next(enum for enum in result.enums if enum.name == "Color")
        self.assertIsNotNone(color_enum.members)
        self.assertEqual(len(color_enum.members), 3)
        self.assertEqual(color_enum.members[0].name, "RED")
        self.assertIsNone(color_enum.members[0].value)
        
        # Check Status enum class
        status_enum = next(enum for enum in result.enums if enum.name == "Status")
        self.assertIsNotNone(status_enum.members)
        self.assertEqual(len(status_enum.members), 3)
        self.assertEqual(status_enum.members[0].name, "PENDING")
        self.assertEqual(status_enum.members[0].value, "0")
        self.assertEqual(status_enum.members[1].name, "APPROVED")
        self.assertEqual(status_enum.members[1].value, "1")
        
        # Check Direction enum with mixed values
        direction_enum = next(enum for enum in result.enums if enum.name == "Direction")
        self.assertIsNotNone(direction_enum.members)
        self.assertEqual(len(direction_enum.members), 4)
        self.assertEqual(direction_enum.members[0].name, "NORTH")
        self.assertEqual(direction_enum.members[0].value, "1")
        self.assertEqual(direction_enum.members[1].name, "SOUTH")
        self.assertIsNone(direction_enum.members[1].value)  # implicit increment
        self.assertEqual(direction_enum.members[2].name, "EAST")
        self.assertEqual(direction_enum.members[2].value, "10")

    def test_namespace_and_using(self):
        """Test parsing of namespace and using declarations."""
        content = """
        namespace MyNamespace {
            class MyClass {
            public:
                void doSomething();
            };
            
            void utility_function() {}
        }
        
        using namespace std;
        using MyAlias = std::vector<int>;
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        # Namespace content should be analyzed
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 1)
        self.assertEqual(result.classes[0].name, "MyClass")
        
        self.assertIsNotNone(result.functions)
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(result.functions[0].name, "utility_function")

    def test_struct_declaration(self):
        """Test parsing of struct declarations."""
        content = """
        struct Point {
            int x;
            int y;
            
            Point(int x_val, int y_val) : x(x_val), y(y_val) {}
            
            double distance() const {
                return sqrt(x*x + y*y);
            }
        };
        
        struct Data {
            std::string name;
            int value;
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 2)
        
        # Check Point struct
        point_struct = next(cls for cls in result.classes if cls.name == "Point")
        self.assertIsNotNone(point_struct.attributes)
        self.assertEqual(len(point_struct.attributes), 2)
        self.assertIsNotNone(point_struct.methods)
        self.assertEqual(len(point_struct.methods), 2)  # constructor, distance

    def test_template_class(self):
        """Test parsing of template class declarations."""
        content = """
        template<typename T>
        class Container {
        private:
            T data;
            
        public:
            Container(const T& value) : data(value) {}
            
            T get() const {
                return data;
            }
            
            void set(const T& value) {
                data = value;
            }
        };
        
        template<typename T, int N>
        class Array {
        private:
            T elements[N];
            
        public:
            T& operator[](int index) {
                return elements[index];
            }
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 2)
        
        # Check Container template
        container_class = next(cls for cls in result.classes if cls.name == "Container")
        self.assertIsNotNone(container_class.methods)
        self.assertEqual(len(container_class.methods), 3)  # constructor, get, set

    def test_error_handling(self):
        """Test error handling for malformed C++ code."""
        content = """
        class InvalidClass {
            int x
            // Missing semicolon
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        # Should return FailedAnalysis for malformed code
        self.assertIsInstance(result, FailedAnalysis)
        self.assertIn("syntax error", result.reason.lower())

    def test_comments_and_docstrings(self):
        """Test parsing of comments and documentation."""
        content = """
        /**
         * A utility class for mathematical operations
         */
        class MathUtils {
        public:
            /**
             * Calculates the square of a number
             * @param x The input number
             * @return The square of x
             */
            static int square(int x) {
                return x * x;
            }
            
            // Simple addition function
            static int add(int a, int b) {
                return a + b;
            }
        };
        """
        test_file = self.create_test_file(content)
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 1)
        
        math_class = result.classes[0]
        self.assertEqual(math_class.name, "MathUtils")
        # Check if docstring is captured
        self.assertIsNotNone(math_class.docstring)
        self.assertIn("utility class", math_class.docstring.lower())
        
        # Check method docstrings
        self.assertIsNotNone(math_class.methods)
        square_method = next(method for method in math_class.methods if method.name == "square")
        self.assertIsNotNone(square_method.docstring)
        self.assertIn("square of a number", square_method.docstring.lower())

    def test_header_file_analysis(self):
        """Test analysis of header files."""
        content = """
        #ifndef MYHEADER_H
        #define MYHEADER_H
        
        #include <string>
        
        class MyClass {
        public:
            MyClass();
            ~MyClass();
            void process();
        private:
            std::string data;
        };
        
        void global_function(int param);
        
        extern int global_variable;
        
        #endif // MYHEADER_H
        """
        test_file = self.create_test_file(content, ".h")
        result = self.analyzer.analyze_file(test_file)
        
        self.assertIsInstance(result, FileAnalysis)
        self.assertIsNotNone(result.classes)
        self.assertEqual(len(result.classes), 1)
        self.assertIsNotNone(result.functions)
        self.assertEqual(len(result.functions), 1)
        self.assertIsNotNone(result.variables)
        self.assertEqual(len(result.variables), 1)

if __name__ == '__main__':
    unittest.main() 