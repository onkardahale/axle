"""Julia language analyzer for Tree-sitter."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from tree_sitter import Tree, Node

from .base import BaseAnalyzer
from ..models import (
    FileAnalysis, Import, Class, Function, Variable, Parameter, 
    BaseClass, Method, Attribute
)

logger = logging.getLogger(__name__)

class JuliaAnalyzer(BaseAnalyzer):
    """Analyzer for Julia programming language."""
    
    LANGUAGE_NAME = "julia"
    FILE_EXTENSIONS = (".jl",)
    
    def _analyze_tree(self, tree: Tree, source_code: bytes, file_path: Path) -> FileAnalysis:
        """Analyze the Julia syntax tree and extract structural information."""
        root_node = tree.root_node
        
        imports = []
        classes = []
        functions = []
        variables = []
        
        # Process all top-level nodes
        for child in root_node.children:
            if child.type in ["using_statement", "import_statement"]:
                import_info = self._extract_import(child, source_code)
                if import_info:
                    imports.append(import_info)
            
            elif child.type == "function_definition":
                func_info = self._extract_function(child, source_code)
                if func_info:
                    functions.append(func_info)
            
            elif child.type == "assignment":
                # Handle short-form function definitions: func(x) = x^2
                if self._is_short_form_function(child, source_code):
                    func_info = self._extract_short_form_function(child, source_code)
                    if func_info:
                        functions.append(func_info)
                else:
                    var_info = self._extract_variable(child, source_code)
                    if var_info:
                        variables.append(var_info)
            
            elif child.type == "struct_definition":
                class_info = self._extract_struct(child, source_code)
                if class_info:
                    classes.append(class_info)
            
            elif child.type == "const_statement":
                var_info = self._extract_const_variable(child, source_code)
                if var_info:
                    variables.append(var_info)
            
            elif child.type == "macro_definition":
                # Treat macros as functions
                func_info = self._extract_macro(child, source_code)
                if func_info:
                    functions.append(func_info)
            
            # Handle module contents recursively
            elif child.type == "module_definition":
                module_contents = self._extract_module_contents(child, source_code)
                imports.extend(module_contents.get('imports', []))
                classes.extend(module_contents.get('classes', []))
                functions.extend(module_contents.get('functions', []))
                variables.extend(module_contents.get('variables', []))
        
        # Extract ALL nested functions recursively from the entire tree
        nested_functions = self._extract_all_nested_functions(root_node, source_code)
        functions.extend(nested_functions)
        
        return FileAnalysis(
            file_path=str(file_path),
            analyzer=f"treesitter_{self.LANGUAGE_NAME}",
            imports=imports if imports else None,
            classes=classes if classes else None,
            functions=functions if functions else None,
            variables=variables if variables else None
        )
    
    def _extract_import(self, node: Node, source_code: bytes) -> Optional[Import]:
        """Extract import information from using/import statements."""
        if node.type == "using_statement":
            return self._extract_using_statement(node, source_code)
        elif node.type == "import_statement":
            return self._extract_import_statement(node, source_code)
        return None
    
    def _extract_using_statement(self, node: Node, source_code: bytes) -> Optional[Import]:
        """Extract information from 'using' statements."""
        # using LinearAlgebra
        # using DataFrames: DataFrame, select
        # using ..ParentModule
        
        for child in node.children:
            if child.type == "identifier":
                source_name = self._get_node_text(child, source_code)
                return Import(source=source_name)
            elif child.type == "scoped_identifier":
                source_name = self._get_node_text(child, source_code)
                return Import(source=source_name)
            elif child.type == "selected_import":
                # using Package: item1, item2
                source_name = None
                items = []
                for subchild in child.children:
                    if subchild.type in ["identifier", "scoped_identifier"]:
                        if source_name is None:
                            source_name = self._get_node_text(subchild, source_code)
                        else:
                            items.append(self._get_node_text(subchild, source_code))
                if source_name:
                    return Import(source=source_name, items=items if items else None)
        
        return None
    
    def _extract_import_statement(self, node: Node, source_code: bytes) -> Optional[Import]:
        """Extract information from 'import' statements."""
        # import JSON
        # import Base: show, length
        
        source_name = None
        items = []
        
        for child in node.children:
            if child.type in ["identifier", "scoped_identifier"]:
                if source_name is None:
                    source_name = self._get_node_text(child, source_code)
                else:
                    items.append(self._get_node_text(child, source_code))
            elif child.type == "selected_import":
                for subchild in child.children:
                    if subchild.type in ["identifier", "scoped_identifier"]:
                        if source_name is None:
                            source_name = self._get_node_text(subchild, source_code)
                        else:
                            items.append(self._get_node_text(subchild, source_code))
        
        if source_name:
            return Import(source=source_name, items=items if items else None)
        
        return None
    
    def _extract_function(self, node: Node, source_code: bytes) -> Optional[Function]:
        """Extract function definition information."""
        name = None
        parameters = []
        docstring = None
        
        # Look for preceding docstring
        docstring = self._get_preceding_docstring(node, source_code)
        
        for child in node.children:
            if child.type == "signature":
                # Extract name and parameters from signature
                name, parameters = self._extract_signature(child, source_code)
                break
        
        if name:
            return Function(
                name=name,
                parameters=parameters if parameters else None,
                docstring=docstring
            )
        
        return None
    
    def _extract_short_form_function(self, node: Node, source_code: bytes) -> Optional[Function]:
        """Extract short-form function definitions like: func(x) = x^2"""
        # assignment node where left side is a call expression or where_expression
        left_side = None
        
        for child in node.children:
            if child.type == "call_expression":
                left_side = child
                break
            elif child.type == "where_expression":
                # Handle functions with where clauses: func(x) where T = ...
                for subchild in child.children:
                    if subchild.type == "call_expression":
                        left_side = subchild
                        break
                break
        
        if not left_side:
            return None
        
        name = None
        parameters = []
        
        for child in left_side.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "argument_list":
                # Extract parameters from argument list
                for arg in child.children:
                    if arg.type == "identifier":
                        param_name = self._get_node_text(arg, source_code)
                        parameters.append(Parameter(name=param_name))
                    elif arg.type == "typed_expression":
                        param = self._extract_typed_parameter(arg, source_code)
                        if param:
                            parameters.append(param)
        
        if name:
            return Function(
                name=name,
                parameters=parameters if parameters else None
            )
        
        return None
    
    def _extract_macro(self, node: Node, source_code: bytes) -> Optional[Function]:
        """Extract macro definition information."""
        name = None
        parameters = []
        docstring = None
        
        # Look for preceding docstring
        docstring = self._get_preceding_docstring(node, source_code)
        
        for child in node.children:
            if child.type == "signature":
                # Extract name and parameters from signature
                for subchild in child.children:
                    if subchild.type == "call_expression":
                        for subsubchild in subchild.children:
                            if subsubchild.type == "identifier":
                                macro_name = self._get_node_text(subsubchild, source_code)
                                name = f"@{macro_name}"  # Prefix with @
                            elif subsubchild.type == "argument_list":
                                parameters = self._extract_argument_list(subsubchild, source_code)
        
        if name:
            return Function(
                name=name,
                parameters=parameters if parameters else None,
                docstring=docstring
            )
        
        return None
    
    def _extract_struct(self, node: Node, source_code: bytes) -> Optional[Class]:
        """Extract struct definition information."""
        name = None
        bases = []
        attributes = []
        methods = []
        docstring = None
        
        # Look for preceding docstring
        docstring = self._get_preceding_docstring(node, source_code)
        
        for child in node.children:
            if child.type == "type_head":
                # Extract name and possible inheritance
                name, struct_bases = self._extract_type_head(child, source_code)
                bases.extend(struct_bases)
            elif child.type == "typed_expression":
                # Direct field declaration like x::Float64
                attr = self._extract_struct_field(child, source_code)
                if attr:
                    attributes.append(attr)
            elif child.type == "assignment":
                # Inner constructor like Circle(r) = r > 0 ? new(r) : error(...)
                if self._is_short_form_function(child, source_code):
                    method = self._extract_struct_constructor(child, source_code)
                    if method:
                        methods.append(method)
            elif child.type == "function_definition":
                # Inner constructor function
                method = self._extract_struct_method(child, source_code)
                if method:
                    methods.append(method)
        
        if name:
            return Class(
                name=name,
                bases=bases if bases else None,
                attributes=attributes if attributes else None,
                methods=methods if methods else None,
                docstring=docstring
            )
        
        return None
    
    def _extract_type_head(self, node: Node, source_code: bytes) -> tuple[Optional[str], List[BaseClass]]:
        """Extract struct name and inheritance from type_head."""
        name = None
        bases = []
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "parametrized_type_expression":
                # Handle parametric types like Point{N, T}
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, source_code)
                        break
            elif child.type == "binary_expression":
                # Handle inheritance: Dog <: Animal or Point{N, T} <: Shape
                struct_name = None
                base_name = None
                for subchild in child.children:
                    if subchild.type == "identifier":
                        if struct_name is None:
                            struct_name = self._get_node_text(subchild, source_code)
                        else:
                            base_name = self._get_node_text(subchild, source_code)
                    elif subchild.type == "parametrized_type_expression":
                        # Handle parametric struct name in inheritance
                        for subsubchild in subchild.children:
                            if subsubchild.type == "identifier" and struct_name is None:
                                struct_name = self._get_node_text(subsubchild, source_code)
                                break
                
                if struct_name:
                    name = struct_name
                if base_name:
                    bases.append(BaseClass(name=base_name))
        
        return name, bases
    
    def _extract_struct_field(self, node: Node, source_code: bytes) -> Optional[Attribute]:
        """Extract struct field information from typed_expression."""
        name = None
        type_annotation = None
        
        # For typed_expression: identifier "::" type
        children = list(node.children)
        if len(children) >= 3:
            if children[0].type == "identifier":
                name = self._get_node_text(children[0], source_code)
            if children[2].type == "identifier":
                type_annotation = self._get_node_text(children[2], source_code)
        
        if name:
            return Attribute(name=name, type=type_annotation)
        
        return None
    
    def _extract_struct_constructor(self, node: Node, source_code: bytes) -> Optional[Method]:
        """Extract struct constructor from assignment node."""
        # This is similar to short-form function but for methods
        left_side = None
        
        for child in node.children:
            if child.type == "call_expression":
                left_side = child
                break
        
        if not left_side:
            return None
        
        name = None
        parameters = []
        
        for child in left_side.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "argument_list":
                # Extract parameters from argument list
                for arg in child.children:
                    if arg.type == "identifier":
                        param_name = self._get_node_text(arg, source_code)
                        parameters.append(Parameter(name=param_name))
                    elif arg.type == "typed_expression":
                        param = self._extract_typed_parameter(arg, source_code)
                        if param:
                            parameters.append(param)
        
        if name:
            return Method(name=name, parameters=parameters if parameters else None)
        
        return None
    
    def _extract_struct_method(self, node: Node, source_code: bytes) -> Optional[Method]:
        """Extract method from struct (inner constructor)."""
        name = None
        parameters = []
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "parameter_list":
                parameters = self._extract_parameters(child, source_code)
        
        if name:
            return Method(name=name, parameters=parameters if parameters else None)
        
        return None
    
    def _extract_variable(self, node: Node, source_code: bytes) -> Optional[Variable]:
        """Extract variable assignment information."""
        name = None
        value = None
        type_annotation = None
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type in ["integer_literal", "float_literal", "string_literal", "boolean_literal"]:
                value = self._get_node_text(child, source_code)
            elif child.type == "type_annotation":
                for subchild in child.children:
                    if subchild.type in ["identifier", "parametrized_type"]:
                        type_annotation = self._get_node_text(subchild, source_code)
        
        if name:
            # Determine if this is a type alias
            kind = "type_alias" if self._is_type_alias(node, source_code) else "external_variable"
            
            return Variable(
                name=name,
                kind=kind,
                type=type_annotation,
                value=value
            )
        
        return None
    
    def _extract_const_variable(self, node: Node, source_code: bytes) -> Optional[Variable]:
        """Extract const variable information."""
        name = None
        value = None
        
        for child in node.children:
            if child.type == "assignment":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, source_code)
                    elif subchild.type in ["integer_literal", "float_literal", "string_literal"]:
                        value = self._get_node_text(subchild, source_code)
        
        if name:
            return Variable(name=name, kind="constant", value=value)
        
        return None
    
    def _extract_signature(self, node: Node, source_code: bytes) -> tuple[Optional[str], List[Parameter]]:
        """Extract function name and parameters from signature node."""
        name = None
        parameters = []
        
        for child in node.children:
            if child.type == "call_expression":
                # Extract name and parameters from call expression
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, source_code)
                    elif subchild.type == "argument_list":
                        parameters = self._extract_argument_list(subchild, source_code)
            elif child.type == "typed_expression":
                # Handle typed function signature: func(args)::ReturnType
                for subchild in child.children:
                    if subchild.type == "call_expression":
                        for subsubchild in subchild.children:
                            if subsubchild.type == "identifier":
                                name = self._get_node_text(subsubchild, source_code)
                            elif subsubchild.type == "argument_list":
                                parameters = self._extract_argument_list(subsubchild, source_code)
            elif child.type == "where_expression":
                # Handle signatures with where clauses: func(args) where T
                for subchild in child.children:
                    if subchild.type == "call_expression":
                        for subsubchild in subchild.children:
                            if subsubchild.type == "identifier":
                                name = self._get_node_text(subsubchild, source_code)
                            elif subsubchild.type == "argument_list":
                                parameters = self._extract_argument_list(subsubchild, source_code)
        
        return name, parameters
    
    def _extract_argument_list(self, node: Node, source_code: bytes) -> List[Parameter]:
        """Extract parameters from argument list."""
        parameters = []
        
        for child in node.children:
            if child.type == "identifier":
                param_name = self._get_node_text(child, source_code)
                parameters.append(Parameter(name=param_name))
            elif child.type == "typed_expression":
                # Handle typed parameters like x::Int
                param = self._extract_typed_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "named_argument":
                # Handle keyword arguments like greeting="Hello"
                param = self._extract_named_argument(child, source_code)
                if param:
                    parameters.append(param)
        
        return parameters
    
    def _extract_named_argument(self, node: Node, source_code: bytes) -> Optional[Parameter]:
        """Extract named/keyword argument."""
        name = None
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
                break
        
        if name:
            return Parameter(name=name)
        
        return None

    def _extract_parameters(self, node: Node, source_code: bytes) -> List[Parameter]:
        """Extract function parameters."""
        parameters = []
        
        for child in node.children:
            if child.type == "identifier":
                param_name = self._get_node_text(child, source_code)
                parameters.append(Parameter(name=param_name))
            elif child.type == "typed_parameter":
                param = self._extract_typed_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "optional_parameter":
                param = self._extract_optional_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "slurp_parameter":
                # Varargs parameter like ...args
                param_name = self._get_node_text(child, source_code)
                parameters.append(Parameter(name=param_name))
        
        return parameters
    
    def _extract_typed_parameter(self, node: Node, source_code: bytes) -> Optional[Parameter]:
        """Extract typed parameter like x::Int."""
        name = None
        type_annotation = None
        
        # For typed_expression nodes, we expect: identifier "::" type
        children = list(node.children)
        if len(children) >= 3:
            if children[0].type == "identifier":
                name = self._get_node_text(children[0], source_code)
            if children[2].type == "identifier":
                type_annotation = self._get_node_text(children[2], source_code)
        
        if name:
            return Parameter(name=name, type=type_annotation)
        
        return None
    
    def _extract_optional_parameter(self, node: Node, source_code: bytes) -> Optional[Parameter]:
        """Extract optional parameter with default value."""
        name = None
        type_annotation = None
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
            elif child.type == "assignment":
                # Extract the parameter name from the assignment
                for subchild in child.children:
                    if subchild.type == "identifier" and name is None:
                        name = self._get_node_text(subchild, source_code)
        
        if name:
            return Parameter(name=name, type=type_annotation)
        
        return None
    
    def _extract_module_contents(self, node: Node, source_code: bytes) -> Dict[str, List]:
        """Extract contents from module definitions."""
        imports = []
        classes = []
        functions = []
        variables = []
        
        # Module contents are direct children of module_definition
        for child in node.children:
            if child.type in ["using_statement", "import_statement"]:
                import_info = self._extract_import(child, source_code)
                if import_info:
                    imports.append(import_info)
            elif child.type == "function_definition":
                func_info = self._extract_function(child, source_code)
                if func_info:
                    functions.append(func_info)
            elif child.type == "struct_definition":
                class_info = self._extract_struct(child, source_code)
                if class_info:
                    classes.append(class_info)
            elif child.type == "const_statement":
                var_info = self._extract_const_variable(child, source_code)
                if var_info:
                    variables.append(var_info)
            elif child.type == "assignment":
                # Handle assignments inside modules
                if self._is_short_form_function(child, source_code):
                    func_info = self._extract_short_form_function(child, source_code)
                    if func_info:
                        functions.append(func_info)
                else:
                    var_info = self._extract_variable(child, source_code)
                    if var_info:
                        variables.append(var_info)
        
        return {
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'variables': variables
        }
    
    def _is_short_form_function(self, node: Node, source_code: bytes) -> bool:
        """Check if assignment is a short-form function definition."""
        # Look for pattern: identifier(...) = ...
        for child in node.children:
            if child.type == "call_expression":
                return True
        return False
    
    def _is_type_alias(self, node: Node, source_code: bytes) -> bool:
        """Check if assignment is a type alias."""
        # Simple heuristic: if right side contains type-like patterns
        text = self._get_node_text(node, source_code)
        type_patterns = ["Vector{", "Matrix{", "Array{", "Dict{", "Set{", "Tuple{"]
        return any(pattern in text for pattern in type_patterns)
    
    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring that precedes a node."""
        # Look for string literal in previous sibling or comments
        parent = node.parent
        if not parent:
            return None
        
        node_index = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_index = i
                break
        
        if node_index is None or node_index == 0:
            return None
        
        # Check previous sibling
        prev_sibling = parent.children[node_index - 1]
        if prev_sibling.type == "string_literal":
            docstring = self._get_node_text(prev_sibling, source_code)
            # Remove quotes and clean up
            if docstring.startswith('"""') and docstring.endswith('"""'):
                return docstring[3:-3].strip()
            elif docstring.startswith('"') and docstring.endswith('"'):
                return docstring[1:-1].strip()
        
        return None
    
    def _get_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a node if present."""
        return self._get_preceding_docstring(node, source_code)
    
    def _extract_all_nested_functions(self, node: Node, source_code: bytes) -> List[Function]:
        """Recursively extract all nested function definitions from the AST."""
        nested_functions = []
        
        def find_nested_functions(current_node: Node, parent_function: Optional[str] = None):
            """Recursively find function definitions nested inside other nodes."""
            
            # If this is a function definition and it's nested (has a parent function),
            # extract it
            if current_node.type == "function_definition" and parent_function is not None:
                func_info = self._extract_function(current_node, source_code)
                if func_info:
                    nested_functions.append(func_info)
            
            # If this is an assignment that could be a short-form function and it's nested
            elif current_node.type == "assignment" and parent_function is not None:
                if self._is_short_form_function(current_node, source_code):
                    func_info = self._extract_short_form_function(current_node, source_code)
                    if func_info:
                        nested_functions.append(func_info)
            
            # If this is a macro definition and it's nested (has a parent function),
            # extract it as a function
            elif current_node.type == "macro_definition" and parent_function is not None:
                func_info = self._extract_macro(current_node, source_code)
                if func_info:
                    nested_functions.append(func_info)
            
            # Determine if we're currently inside a function for child processing
            current_function_context = parent_function
            if current_node.type == "function_definition":
                # Extract the function name to use as context for nested functions
                for child in current_node.children:
                    if child.type == "signature":
                        name, _ = self._extract_signature(child, source_code)
                        if name:
                            current_function_context = name
                            break
                
                # If we couldn't extract the function name but we're inside a function,
                # use a generic marker to indicate we're in a function context
                if current_function_context is None and parent_function is not None:
                    current_function_context = "__nested_function__"
                elif current_function_context is None:
                    # This is likely a constructor or complex function, treat it as a function context
                    current_function_context = "__constructor__"
            
            # Recursively process all children
            for child in current_node.children:
                find_nested_functions(child, current_function_context)
        
        # Start the recursive search
        find_nested_functions(node)
        
        return nested_functions 