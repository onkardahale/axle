"""C++-specific Tree-sitter analyzer."""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from tree_sitter import Node, Tree

import logging

logger = logging.getLogger(__name__)

from .base import BaseAnalyzer 
from ..models import (
    FileAnalysis, Import, Class, Method, Function, Variable,
    Parameter, BaseClass, Attribute, Enum, EnumMember
)

class CppAnalyzer(BaseAnalyzer):
    LANGUAGE_NAME = "cpp"
    FILE_EXTENSIONS = (".cpp", ".cc", ".cxx", ".hpp", ".hxx")  # Removed .c and .h (pure C files)

    def _extract_include_path(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract include path from #include statement."""
        text = self._get_node_text(node, source_code)
        
        # Match #include <header> or #include "header"
        include_match = re.match(r'#include\s*[<"](.*?)[>"]', text)
        if include_match:
            return include_match.group(1)
        return None

    def _get_docstring_from_comment(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from C++ comments (// or /* */)."""
        # Look for preceding comment nodes
        parent = node.parent
        if not parent:
            return None
            
        # Find the index of the current node
        node_index = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_index = i
                break
                
        if node_index is None:
            return None
            
        # Look backwards for comments
        comments = []
        for i in range(node_index - 1, -1, -1):
            prev_child = parent.children[i]
            if prev_child.type == "comment":
                comment_text = self._get_node_text(prev_child, source_code)
                # Handle /** */ style comments
                if comment_text.startswith("/**") and comment_text.endswith("*/"):
                    clean_comment = comment_text[3:-2].strip()
                    # Remove leading * from each line
                    lines = clean_comment.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('*'):
                            line = line[1:].strip()
                        cleaned_lines.append(line)
                    return '\n'.join(cleaned_lines).strip()
                # Handle // style comments
                elif comment_text.startswith("//"):
                    comments.insert(0, comment_text[2:].strip())
                else:
                    break
            elif prev_child.type not in ["comment", "\n", " "]:
                break
                
        if comments:
            return '\n'.join(comments).strip()
        return None

    def _process_parameters(self, parameters_node: Node, source_code: bytes) -> List[Parameter]:
        """Process function/method parameters."""
        params = []
        
        for param_node in parameters_node.named_children:
            if param_node.type in ["parameter_declaration", "optional_parameter_declaration"]:
                param_name = None
                param_type = None
                
                # Find type and name
                type_node = param_node.child_by_field_name("type")
                declarator_node = param_node.child_by_field_name("declarator")
                
                if type_node:
                    param_type = self._get_node_text(type_node, source_code)
                    
                if declarator_node:
                    if declarator_node.type == "identifier":
                        param_name = self._get_node_text(declarator_node, source_code)
                    elif declarator_node.type == "reference_declarator":
                        # Handle reference parameters like int& x
                        inner_declarator = declarator_node.children[-1]  # Last child is usually the identifier
                        if inner_declarator.type == "identifier":
                            param_name = self._get_node_text(inner_declarator, source_code)
                            param_type = param_type + "&" if param_type else "&"
                    elif declarator_node.type == "pointer_declarator":
                        # Handle pointer parameters like int* x
                        inner_declarator = declarator_node.children[-1]
                        if inner_declarator.type == "identifier":
                            param_name = self._get_node_text(inner_declarator, source_code)
                            param_type = param_type + "*" if param_type else "*"
                    elif declarator_node.type == "array_declarator":
                        # Handle array parameters like int x[]
                        inner_declarator = declarator_node.child_by_field_name("declarator")
                        if inner_declarator and inner_declarator.type == "identifier":
                            param_name = self._get_node_text(inner_declarator, source_code)
                            param_type = param_type + "[]" if param_type else "[]"
                
                if param_name:
                    params.append(Parameter(name=param_name, type=param_type))
            elif param_node.type == "variadic_parameter":
                params.append(Parameter(name="...", type="..."))
                
        return params

    def _process_class_specifier(self, node: Node, source_code: bytes) -> Optional[Class]:
        """Process class or struct declarations."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
            
        class_name = self._get_node_text(name_node, source_code)
        
        # Extract docstring
        docstring = self._get_docstring_from_comment(node, source_code)
        
        # Extract base classes
        bases = []
        # Search for base_class_clause in named children (it's not a field)
        base_class_clause = None
        for child in node.named_children:
            if child.type == "base_class_clause":
                base_class_clause = child
                break
                
        if base_class_clause:
            access_specifier = None
            base_class_name = None
            
            # The base_class_clause contains access_specifier and type_identifier as direct children
            for child in base_class_clause.named_children:
                if child.type == "access_specifier":
                    access_specifier = self._get_node_text(child, source_code)
                elif child.type == "type_identifier":
                    base_class_name = self._get_node_text(child, source_code)
            
            if base_class_name:
                # Default access if not specified
                if not access_specifier:
                    access_specifier = "private" if node.children[0].type == "class" else "public"
                
                bases.append(BaseClass(name=base_class_name, access=access_specifier))
        
        # Extract members
        methods = []
        attributes = []
        
        # The class body is a field_declaration_list (named child, not field)
        body_node = None
        for child in node.named_children:
            if child.type == "field_declaration_list":
                body_node = child
                break
                
        if body_node:
            current_access = "private" if node.children[0].type == "class" else "public"  # struct vs class default
            
            for member in body_node.named_children:
                if member.type == "access_specifier":
                    # Update current access level
                    access_text = self._get_node_text(member, source_code)
                    if "public" in access_text:
                        current_access = "public"
                    elif "private" in access_text:
                        current_access = "private"
                    elif "protected" in access_text:
                        current_access = "protected"
                elif member.type == "function_definition":
                    method = self._process_method(member, source_code, class_name)
                    if method:
                        methods.append(method)
                elif member.type == "declaration":
                    # Could be method declaration or field
                    # First check if it's a function declaration
                    declarator_node = member.child_by_field_name("declarator")
                    if declarator_node and declarator_node.type == "function_declarator":
                        # It's a method declaration
                        method = self._process_method_declaration(member, source_code, class_name)
                        if method:
                            methods.append(method)
                    else:
                        # It's a field declaration
                        attr = self._process_class_member_declaration(member, source_code)
                        if attr:
                            attributes.append(attr)
                elif member.type == "field_declaration":
                    attr = self._process_field_declaration(member, source_code)
                    if attr:
                        attributes.append(attr)
        
        return Class(
            name=class_name,
            bases=bases if bases else None,
            methods=methods if methods else None,
            attributes=attributes if attributes else None,
            docstring=docstring
        )

    def _process_method(self, node: Node, source_code: bytes, class_name: Optional[str] = None) -> Optional[Method]:
        """Process method definitions."""
        declarator_node = node.child_by_field_name("declarator")
        if not declarator_node:
            return None
            
        method_name = None
        parameters = []
        
        # Handle function_declarator
        if declarator_node.type == "function_declarator":
            declarator_inner = declarator_node.child_by_field_name("declarator")
            if declarator_inner:
                if declarator_inner.type == "identifier":
                    method_name = self._get_node_text(declarator_inner, source_code)
                elif declarator_inner.type == "field_identifier":
                    method_name = self._get_node_text(declarator_inner, source_code)
            
            # Get parameters
            parameters_node = declarator_node.child_by_field_name("parameters")
            if parameters_node:
                parameters = self._process_parameters(parameters_node, source_code)
        
        # Check if this is a constructor (no type field and name matches class)
        type_node = node.child_by_field_name("type")
        if not type_node and method_name == class_name:
            # It's a constructor
            pass  # method_name is already set correctly
        elif not method_name:
            # Try to extract name from declarator directly
            if declarator_node.type == "function_declarator":
                for child in declarator_node.children:
                    if child.type in ["identifier", "field_identifier"]:
                        method_name = self._get_node_text(child, source_code)
                        break
                        
        # Extract docstring
        docstring = self._get_docstring_from_comment(node, source_code)
        
        if method_name:
            return Method(
                name=method_name,
                parameters=parameters if parameters else None,
                docstring=docstring,
                calls=[]
            )
        
        return None

    def _process_method_declaration(self, node: Node, source_code: bytes, class_name: Optional[str] = None) -> Optional[Method]:
        """Process method declarations (not definitions)."""
        declarator_node = node.child_by_field_name("declarator")
        if not declarator_node or declarator_node.type != "function_declarator":
            return None
            
        method_name = None
        parameters = []
        
        # Get the method name
        declarator_inner = declarator_node.child_by_field_name("declarator")
        if declarator_inner:
            if declarator_inner.type == "identifier":
                method_name = self._get_node_text(declarator_inner, source_code)
            elif declarator_inner.type == "field_identifier":
                method_name = self._get_node_text(declarator_inner, source_code)
        
        # Get parameters
        parameters_node = declarator_node.child_by_field_name("parameters")
        if parameters_node:
            parameters = self._process_parameters(parameters_node, source_code)
        
        # Extract docstring
        docstring = self._get_docstring_from_comment(node, source_code)
        
        if method_name:
            return Method(
                name=method_name,
                parameters=parameters if parameters else None,
                docstring=docstring,
                calls=[]
            )
        
        return None

    def _process_class_member_declaration(self, node: Node, source_code: bytes) -> Optional[Attribute]:
        """Process class member declarations."""
        # Look for init_declarator or direct identifier
        type_node = node.child_by_field_name("type")
        attr_type = self._get_node_text(type_node, source_code) if type_node else None
        
        for child in node.named_children:
            if child.type == "init_declarator":
                declarator_node = child.child_by_field_name("declarator")
                if declarator_node and declarator_node.type == "identifier":
                    attr_name = self._get_node_text(declarator_node, source_code)
                    return Attribute(name=attr_name, type=attr_type, static=False)
            elif child.type == "identifier":
                attr_name = self._get_node_text(child, source_code)
                return Attribute(name=attr_name, type=attr_type, static=False)
        
        return None

    def _process_field_declaration(self, node: Node, source_code: bytes) -> Optional[Attribute]:
        """Process field declarations."""
        type_node = node.child_by_field_name("type")
        attr_type = self._get_node_text(type_node, source_code) if type_node else None
        
        # Look for field_identifier or identifier
        for child in node.named_children:
            if child.type in ["field_identifier", "identifier"]:
                attr_name = self._get_node_text(child, source_code)
                return Attribute(name=attr_name, type=attr_type, static=False)
        
        return None

    def _process_function_definition(self, node: Node, source_code: bytes) -> Optional[Function]:
        """Process function definitions."""
        declarator_node = node.child_by_field_name("declarator")
        if not declarator_node:
            return None
            
        function_name = None
        parameters = []
        
        if declarator_node.type == "function_declarator":
            declarator_inner = declarator_node.child_by_field_name("declarator")
            if declarator_inner and declarator_inner.type == "identifier":
                function_name = self._get_node_text(declarator_inner, source_code)
                
            # Get parameters
            parameters_node = declarator_node.child_by_field_name("parameters")
            if parameters_node:
                parameters = self._process_parameters(parameters_node, source_code)
        
        # Extract docstring
        docstring = self._get_docstring_from_comment(node, source_code)
        
        if function_name:
            return Function(
                name=function_name,
                parameters=parameters if parameters else None,
                docstring=docstring,
                calls=[]
            )
        
        return None

    def _process_declaration(self, node: Node, source_code: bytes) -> Optional[Union[Variable, Function]]:
        """Process top-level declarations."""
        # Check if it's a function declaration
        declarator_node = node.child_by_field_name("declarator")
        if declarator_node and declarator_node.type == "function_declarator":
            # It's a function declaration, not definition
            declarator_inner = declarator_node.child_by_field_name("declarator")
            if declarator_inner and declarator_inner.type == "identifier":
                function_name = self._get_node_text(declarator_inner, source_code)
                
                # Get parameters
                parameters_node = declarator_node.child_by_field_name("parameters")
                parameters = []
                if parameters_node:
                    parameters = self._process_parameters(parameters_node, source_code)
                
                return Function(
                    name=function_name,
                    parameters=parameters if parameters else None,
                    docstring=None,
                    calls=[]
                )
        
        # Otherwise, it's a variable declaration
        return self._process_variable_declaration(node, source_code)

    def _process_variable_declaration(self, node: Node, source_code: bytes) -> Optional[Variable]:
        """Process variable declarations."""
        type_node = node.child_by_field_name("type")
        type_text = self._get_node_text(type_node, source_code) if type_node else None
        
        # Check for type qualifiers (const, static, extern, etc.)
        qualifiers = []
        for child in node.children:
            if child.type == "type_qualifier" or child.type == "storage_class_specifier":
                qualifier_text = self._get_node_text(child, source_code)
                qualifiers.append(qualifier_text)
        
        # Combine qualifiers with type
        if qualifiers and type_text:
            type_text = " ".join(qualifiers) + " " + type_text
        elif qualifiers:
            type_text = " ".join(qualifiers)
        
        # Look for init_declarator or identifier
        for child in node.named_children:
            if child.type == "init_declarator":
                declarator_node = child.child_by_field_name("declarator")
                value_node = child.child_by_field_name("value")
                
                if declarator_node:
                    var_name = None
                    if declarator_node.type == "identifier":
                        var_name = self._get_node_text(declarator_node, source_code)
                    elif declarator_node.type == "array_declarator":
                        # Handle array declarations like int arr[10]
                        inner_declarator = declarator_node.child_by_field_name("declarator")
                        if inner_declarator and inner_declarator.type == "identifier":
                            var_name = self._get_node_text(inner_declarator, source_code)
                            type_text = type_text + "[]" if type_text else "[]"
                    
                    if var_name:
                        var_value = self._get_node_text(value_node, source_code) if value_node else None
                        
                        # Determine kind
                        kind = "external_variable"
                        if type_text and "const" in type_text:
                            kind = "constant"
                        elif type_text and ("extern" in type_text or "static" in type_text):
                            kind = "external_variable"
                        
                        return Variable(
                            name=var_name,
                            kind=kind,
                            type=type_text,
                            value=var_value
                        )
            elif child.type == "identifier":
                # Simple declaration without initialization
                var_name = self._get_node_text(child, source_code)
                
                kind = "external_variable"
                if type_text and "const" in type_text:
                    kind = "constant"
                elif type_text and ("extern" in type_text or "static" in type_text):
                    kind = "external_variable"
                
                return Variable(
                    name=var_name,
                    kind=kind,
                    type=type_text,
                    value=None
                )
        
        return None

    def _process_enum_specifier(self, node: Node, source_code: bytes) -> Optional[Enum]:
        """Process enum declarations."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
            
        enum_name = self._get_node_text(name_node, source_code)
        
        # Extract docstring
        docstring = self._get_docstring_from_comment(node, source_code)
        
        # Extract enum members
        members = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for member_node in body_node.named_children:
                if member_node.type == "enumerator":
                    name_field = member_node.child_by_field_name("name")
                    value_field = member_node.child_by_field_name("value")
                    
                    if name_field:
                        member_name = self._get_node_text(name_field, source_code)
                        member_value = None
                        if value_field:
                            member_value = self._get_node_text(value_field, source_code)
                        
                        members.append(EnumMember(name=member_name, value=member_value))
        
        return Enum(
            name=enum_name,
            members=members if members else None,
            docstring=docstring
        )

    def _process_preproc_include(self, node: Node, source_code: bytes) -> Optional[Import]:
        """Process #include statements."""
        include_path = self._extract_include_path(node, source_code)
        if include_path:
            # Keep the full path for the name instead of just filename
            return Import(name=include_path, source=include_path)
        return None

    def _analyze_tree(self, tree: Tree, source_code: bytes, file_path: Path) -> FileAnalysis:
        """Analyze the syntax tree and return structured data."""
        imports = []
        classes = []
        functions = []
        variables = []
        enums = []
        
        def traverse_node(node: Node, namespace_prefix: str = "", inside_class: bool = False):
            """Recursively traverse the syntax tree."""
            if node.type == "preproc_include":
                import_obj = self._process_preproc_include(node, source_code)
                if import_obj:
                    imports.append(import_obj)
            
            elif node.type == "class_specifier" or node.type == "struct_specifier":
                class_obj = self._process_class_specifier(node, source_code)
                if class_obj:
                    # For tests, don't add namespace prefix to keep names simple
                    classes.append(class_obj)
            
            elif node.type == "function_definition":
                if not inside_class:  # Only process functions that are not inside classes
                    func_obj = self._process_function_definition(node, source_code)
                    if func_obj:
                        functions.append(func_obj)
            
            elif node.type == "declaration":
                if not inside_class:  # Only process declarations that are not inside classes
                    # Could be variable or function declaration
                    result = self._process_declaration(node, source_code)
                    if isinstance(result, Function):
                        functions.append(result)
                    elif isinstance(result, Variable):
                        variables.append(result)
            
            elif node.type == "enum_specifier":
                enum_obj = self._process_enum_specifier(node, source_code)
                if enum_obj:
                    enums.append(enum_obj)
            
            elif node.type == "namespace_definition":
                # Extract namespace name and process its contents
                name_node = node.child_by_field_name("name")
                if name_node:
                    namespace_name = self._get_node_text(name_node, source_code)
                    
                    # Process namespace body
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.named_children:
                            traverse_node(child, namespace_name, inside_class)
                    return  # Don't traverse children again
            
            elif node.type == "template_declaration":
                # Process the template's declaration
                # Find the actual declaration within the template
                declaration_node = None
                for child in node.named_children:
                    if child.type in ["class_specifier", "struct_specifier", "function_definition", "declaration"]:
                        declaration_node = child
                        break
                
                if declaration_node:
                    traverse_node(declaration_node, namespace_prefix, inside_class)
                return  # Don't traverse children again
            
            # Determine if we're entering a class for child traversal
            entering_class = node.type in ["class_specifier", "struct_specifier"]
            
            # Recursively process children
            for child in node.named_children:
                traverse_node(child, namespace_prefix, inside_class or entering_class)
        
        # Start traversal from root
        traverse_node(tree.root_node)
        
        return FileAnalysis(
            file_path=str(file_path),
            analyzer=f"treesitter_{self.LANGUAGE_NAME.lower()}",
            imports=imports if imports else None,
            classes=classes if classes else None,
            functions=functions if functions else None,
            variables=variables if variables else None,
            enums=enums if enums else None
        ) 