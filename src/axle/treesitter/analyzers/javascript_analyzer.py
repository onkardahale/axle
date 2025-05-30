"""JavaScript-specific Tree-sitter analyzer."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from tree_sitter import Node, Tree

from .base import BaseAnalyzer
from ..models import (
    FileAnalysis, Import, Class, Method, Function, Variable,
    Parameter, BaseClass, Attribute
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class JavaScriptAnalyzer(BaseAnalyzer):
    LANGUAGE_NAME = "javascript"
    FILE_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs")

    # ... (_get_node_text, _strip_string_quotes, _get_jsdoc_comment remain the same for now) ...
    def _get_node_text(self, node: Optional[Node], source_code: bytes) -> str:
        """Get the text content of a node from the source code."""
        if node is None:
            return ""
        return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _strip_string_quotes(self, s: str) -> str:
        """Strip quotes from string literals, handling template literals."""
        if s.startswith('`') and s.endswith('`'):
            return s[1:-1]
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def _get_jsdoc_comment(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract JSDoc comment from a node if present."""
        # TODO: Implement JSDoc comment extraction
        # This would involve looking at node.prev_named_sibling or similar
        # and checking if it's a comment node matching JSDoc pattern.
        return None

    def _get_identifier_text_from_lhs_expression(self, lhs_node: Node, source_code: bytes) -> Optional[str]:
        """Extracts identifier text if _lhs_expression resolves to a simple identifier."""
        if lhs_node.type == "_lhs_expression":
            # _lhs_expression -> CHOICE [member_expression, subscript_expression, _identifier, _destructuring_pattern]
            # We are interested in _identifier -> identifier
            # It might be a direct child or a named child. Let's check named children first.
            if lhs_node.named_children:
                # Look for an 'identifier' node within _lhs_expression
                for child in lhs_node.named_children:
                    if child.type == "identifier":
                        return self._get_node_text(child, source_code)
                    elif child.type == "_identifier" and child.named_children and child.named_children[0].type == "identifier": # _identifier -> identifier
                        return self._get_node_text(child.named_children[0], source_code)
            # Fallback: if _lhs_expression itself is sometimes aliased to identifier (less common for this rule)
            elif lhs_node.type == "identifier": # Should be caught by the above if _identifier wraps it
                 return self._get_node_text(lhs_node, source_code)

        elif lhs_node.type == "identifier": # Direct identifier case
            return self._get_node_text(lhs_node, source_code)
        return None

    def _extract_name_from_parameter_structure(self, param_def_node: Node, source_code: bytes) -> Optional[str]:
        """Helper to extract parameter name from various pattern structures."""
        node_type = param_def_node.type
        # Using a distinct logger name for this helper for clarity, as seen in logs.
        helper_logger_name = "HelperExtr" 

        logger.debug(f"            [{helper_logger_name}] _extract_name_from_parameter_structure called with node: type='{node_type}', text='{self._get_node_text(param_def_node, source_code)}'")

        if node_type == "identifier":
            return self._get_node_text(param_def_node, source_code)

        elif node_type == "pattern":
            # 'pattern' -> CHOICE [_lhs_expression, rest_pattern, object_pattern, array_pattern]
            # (Grammar for 'pattern' can vary; this covers common cases)
            content_node = None
            if param_def_node.named_children: # Prefer named child if it's the actual content
                content_node = param_def_node.named_children[0]
            elif param_def_node.children: # Else, take the first child as content
                content_node = param_def_node.children[0]
            
            if not content_node:
                logger.debug(f"                [{helper_logger_name}] 'pattern' node has no content_node.")
                return None

            # Delegate to this function again for the content of the pattern
            # This handles cases like pattern -> _lhs_expression -> identifier or pattern -> rest_pattern
            return self._extract_name_from_parameter_structure(content_node, source_code)

        elif node_type == "assignment_pattern": # e.g., age = 20
            left_node = param_def_node.child_by_field_name("left")
            if left_node:
                # The 'left' part is itself a pattern or identifier.
                return self._extract_name_from_parameter_structure(left_node, source_code)
            else:
                logger.debug(f"                [{helper_logger_name}] Assignment pattern without 'left' field.")

        elif node_type == "rest_pattern": # e.g., ...rest
            target_for_name_extraction = None
            if param_def_node.named_children:
                target_for_name_extraction = param_def_node.named_children[0]
                logger.debug(f"                [{helper_logger_name}] rest_pattern: using named child '{target_for_name_extraction.type}' for name extraction.")
            elif param_def_node.children:
                # Typically, rest_pattern has two children: '...' (literal) and then the identifier/pattern.
                # Some grammars might just have the identifier/pattern as the only child if '...' is implicit in the node type.
                if len(param_def_node.children) == 1: # e.g. (rest_pattern (identifier))
                    target_for_name_extraction = param_def_node.children[0]
                    logger.debug(f"                [{helper_logger_name}] rest_pattern: using single child '{target_for_name_extraction.type}' for name extraction.")
                elif len(param_def_node.children) > 1 and param_def_node.children[0].type == "...": # Check for '...' literal
                    target_for_name_extraction = param_def_node.children[1]
                    logger.debug(f"                [{helper_logger_name}] rest_pattern: using child after '...' literal: '{target_for_name_extraction.type}'.")
                elif len(param_def_node.children) > 0 : # Fallback to first child if no '...' found and multiple children
                    target_for_name_extraction = param_def_node.children[0]
                    logger.debug(f"                [{helper_logger_name}] rest_pattern: fallback to first child '{target_for_name_extraction.type}'.")


            if target_for_name_extraction:
                # Recursively call to get the name of the identifier/pattern being "rested"
                name_part = self._extract_name_from_parameter_structure(target_for_name_extraction, source_code)
                if name_part:
                    return "..." + name_part
            
            # If still no name, it indicates an issue with this rest_pattern's structure or handling
            logger.debug(f"            [{helper_logger_name}] Could not extract identifier from within rest_pattern: {str(param_def_node)}")
            # This matches the log: "[HelperExtr] Could not extract name from node type 'rest_pattern'"
            # The above logic should prevent reaching here if structure is (rest_pattern (identifier))

        elif node_type in ("object_pattern", "array_pattern"): # For destructuring parameters
            # For now, returning the textual representation of the pattern.
            # More detailed parsing could extract individual elements if needed.
            return self._get_node_text(param_def_node, source_code)
        
        # If node_type was 'rest_pattern' and it fell through the specific handling above
        if node_type == "rest_pattern":
             logger.warning(f"            [{helper_logger_name}] Failed to extract name from rest_pattern node '{self._get_node_text(param_def_node, source_code)}'. Review rest_pattern handling.")
        else:
             logger.debug(f"            [{helper_logger_name}] _extract_name_from_parameter_structure returning None for unhandled type '{node_type}', text '{self._get_node_text(param_def_node, source_code)}'")
        
        return None
    def _process_parameters(self, parameters_node: Optional[Node], source_code: bytes,
                                m_name: Optional[str] = None,
                                c_name: Optional[str] = None) -> List[Parameter]:
            params = []
            if not parameters_node or parameters_node.type != "formal_parameters":
                logger.debug(f"[{m_name or 'FuncPar'}] No parameters_node or not formal_parameters. Node type: {parameters_node.type if parameters_node else 'None'}")
                return params

            # Log SEXP of the formal_parameters node (using str() as sexp() is deprecated)
            logger.debug(f"[{m_name or 'FuncPar'}] Processing formal_parameters. SEXP: {str(parameters_node)}")

            # Iterate over all children of the formal_parameters node.
            # According to SEXP logs, direct children are 'identifier', 'assignment_pattern', 'rest_pattern', along with punctuation.
            for child_node in parameters_node.children:
                param_name: Optional[str] = None
                current_node_type = child_node.type
                
                # We are interested in nodes that define parameters.
                # The _extract_name_from_parameter_structure helper can handle these types.
                # 'pattern' is also included as the helper can navigate it.
                if current_node_type in ("identifier", "assignment_pattern", "rest_pattern", "object_pattern", "array_pattern", "pattern"):
                    node_text = self._get_node_text(child_node, source_code) # For logging
                    logger.debug(f"  [{m_name or 'FuncPar'}] Potential parameter definition node: type='{current_node_type}', text='{node_text}'")
                    
                    # Call the helper function to extract the parameter name
                    param_name = self._extract_name_from_parameter_structure(child_node, source_code)

                    if param_name:
                        # Ensure type extraction is attempted if param_name is found
                        # For now, type is None as per existing Parameter model instantiation
                        logger.info(f"    [{m_name or 'FuncPar'}] Extracted param name: {param_name}")
                        params.append(Parameter(name=param_name, type=None))
                    else:
                        logger.warning(f"    [{m_name or 'FuncPar'}] Failed to extract param name from node: type='{current_node_type}', text='{node_text}' using _extract_name_from_parameter_structure.")
                
                elif current_node_type in ("(", ")", ",", ";", "comment"): # Known non-parameter nodes / trivia
                    # This branch handles punctuation and comments, no action needed.
                    # logger.debug(f"  [{m_name or 'FuncPar'}] Skipping non-parameter child node: type='{current_node_type}'")
                    pass
                else:
                    # Log unexpected node types found as children of formal_parameters
                    node_text = self._get_node_text(child_node, source_code)
                    logger.warning(f"  [{m_name or 'FuncPar'}] Unhandled child node type in formal_parameters: '{current_node_type}', text='{node_text}'. SEXP: {str(child_node)}")

            logger.debug(f"[{m_name or 'FuncPar'}] Finished _process_parameters for '{m_name}'. Found {len(params)} params: {[p.name for p in params]}")
            return params

    # Helper to correctly find identifier within nested hidden rules
    def _get_name_from_deep_identifier_structure(self, node: Node, source_code: bytes, m_name: Optional[str]=None) -> Optional[str]:
        current = node
        logger.debug(f"    [{m_name or 'Helper'}] _get_name_from_deep_identifier_structure called with node: type='{current.type}', text='{self._get_node_text(current,source_code)}'")
        for i in range(5): # Max depth to prevent infinite loops
            if not current:
                logger.debug(f"      [{m_name or 'Helper'}] Depth {i}: Current node is None, returning None.")
                return None
            logger.debug(f"      [{m_name or 'Helper'}] Depth {i}: current.type='{current.type}'")
            
            if current.type == "identifier":
                name = self._get_node_text(current, source_code)
                logger.debug(f"        [{m_name or 'Helper'}] Found 'identifier', text='{name}'")
                return name
            
            # Relevant wrapper types from grammar: pattern -> _lhs_expression -> _identifier -> identifier
            if current.type in ("_lhs_expression", "_identifier", "pattern"):
                logger.debug(f"        [{m_name or 'Helper'}] Node is wrapper type '{current.type}'. Looking for children.")
                # Iterate ALL children, as hidden rules like _identifier won't be in named_children
                next_node_candidate = None
                for child_node in current.children:
                    # Prioritize going deeper through known structural wrappers or to the target
                    if child_node.type in ("identifier", "_identifier", "_lhs_expression", "pattern"): 
                        next_node_candidate = child_node
                        break
                
                if next_node_candidate:
                    current = next_node_candidate
                    logger.debug(f"          [{m_name or 'Helper'}] Descending into child: type='{current.type}'")
                    continue # Continue to the next level of the loop
                else:
                    logger.debug(f"          [{m_name or 'Helper'}] No relevant child (identifier, _identifier, _lhs_expression, pattern) found to descend further from '{current.type}'.")
                    return None # Cannot go deeper
            else: 
                logger.debug(f"    [{m_name or 'Helper'}] Node type '{current.type}' is not a recognized wrapper for deep identifier search, returning None.")
                return None # Not a structure this helper is designed for
        
        logger.warning(f"    [{m_name or 'Helper'}] Reached max depth in _get_name_from_deep_identifier_structure for initial node type '{node.type}'.")
        return None
    
    def _extract_identifier_from_pattern_path(self, entry_node: Node, source_code: bytes, m_name: Optional[str]=None) -> Optional[str]:
        current = entry_node
        logger.debug(f"    [{m_name or 'Helper'}] _extract_identifier_from_pattern_path with node: type='{current.type}', text='{self._get_node_text(current,source_code)}'")

        for _ in range(5): # Max depth
            if not current:
                logger.debug(f"      [{m_name or 'Helper'}] Current node is None, returning None.")
                return None
            logger.debug(f"      [{m_name or 'Helper'}] current.type='{current.type}'")
            
            if current.type == "identifier":
                name = self._get_node_text(current, source_code)
                logger.debug(f"        [{m_name or 'Helper'}] Found 'identifier', text='{name}'")
                return name
            
            # Navigate common wrappers: pattern -> _lhs_expression -> _identifier -> identifier
            if current.type in ("_lhs_expression", "_identifier", "pattern"):
                next_node_candidate = None
                # Iterate ALL children because hidden rules like _identifier won't be in named_children
                for child_node in current.children:
                    if child_node.type in ("identifier", "_identifier", "_lhs_expression"): # Check for relevant types
                        next_node_candidate = child_node
                        break 
                    # If pattern directly contains object/array_pattern (for destructuring)
                    elif current.type == "pattern" and child_node.type in ("object_pattern", "array_pattern"):
                        # For simple destructuring, we might take its text as param name
                        # but for deep extraction of identifiers within, it needs more logic.
                        # For now, if the test is for simple identifiers, this path is less critical.
                        # Fallback to text of destructuring pattern:
                        # name = self._get_node_text(child_node, source_code)
                        logger.debug(f"        [{m_name or 'Helper'}] Found destructuring pattern, text='{name}'")
                        # return name # This would make the param name "{a,b}" etc.
                        pass 
                if next_node_candidate:
                    current = next_node_candidate
                    logger.debug(f"          [{m_name or 'Helper'}] Descending into child: type='{current.type}'")
                    continue
                else:
                    logger.debug(f"          [{m_name or 'Helper'}] No relevant child found to descend further from '{current.type}'.")
                    return None
            else: 
                logger.debug(f"    [{m_name or 'Helper'}] Node type '{current.type}' not a recognized wrapper for identifier path, returning None.")
                return None
        
        logger.warning(f"    [{m_name or 'Helper'}] Reached max depth for initial node type '{entry_node.type}'.")
        return None


    def _process_method(self, node: Node, source_code: bytes, class_name: Optional[str] = None) -> Method:
        """Process a class method definition."""
        name_node = node.child_by_field_name("name")
        method_name = self._get_node_text(name_node, source_code) if name_node else "UnknownMethod"
        
        parameters_node = node.child_by_field_name("parameters")
        parameters = self._process_parameters(parameters_node, source_code, m_name=method_name, c_name=class_name)
        
        docstring = self._get_jsdoc_comment(node, source_code)

        return Method(
            name=method_name,
            parameters=parameters or None,
            docstring=docstring,
            calls=[] # Placeholder
            # is_static=is_static 
        )

    def _process_class_declaration(self, node: Node, source_code: bytes) -> Class:
        name_node = node.child_by_field_name("name")
        class_name = self._get_node_text(name_node, source_code) if name_node else "UnknownClass"

        bases = []
        # Find 'class_heritage' node, it's not a field of class_declaration but a direct child symbol
        heritage_node: Optional[Node] = None
        for child in node.children: # Iterate children of class_declaration
            if child.type == "class_heritage":
                heritage_node = child
                break

        if heritage_node:
            # class_heritage -> "extends" expression
            # The 'expression' is the superclass. It's the child after "extends" literal.
            # Typically, the literal "extends" is children[0], expression is children[1]
            if len(heritage_node.children) > 1:
                superclass_expr_node = heritage_node.children[1] # The expression node
                # This expression node could be an identifier, member_expression, etc.
                # For 'extends User', it should be an identifier.
                base_name = self._get_node_text(superclass_expr_node, source_code)
                if base_name:
                    bases.append(BaseClass(name=base_name))
        
        methods = []
        attributes = []
        body_node = node.child_by_field_name("body") # This is 'class_body'
        if body_node:
            for member_node in body_node.children: 
                if member_node.type == "method_definition":
                    is_static_method = False
                    # 'method_definition' rule from grammar.json:
                    # SEQ [ REPEAT(decorator), CHOICE(static, ALIAS(static_get_whitespace)), CHOICE(async), CHOICE(get,set,"*"), FIELD name, FIELD params, FIELD body ]
                    # Check for 'static' as one of the early children if they are not fields themselves
                    current_children = list(member_node.children)
                    idx = 0
                    while idx < len(current_children) and current_children[idx].type == 'decorator':
                        idx += 1
                    if idx < len(current_children) and current_children[idx].type == 'static':
                        is_static_method = True

                    method_obj = self._process_method(member_node, source_code, class_name=class_name)
                    # TODO: Update Method model to include is_static and pass it
                    # if method_obj: method_obj.is_static = is_static_method
                    if method_obj: methods.append(method_obj)

                elif member_node.type == "field_definition": # As per grammar.json
                    # field_definition -> SEQ [ REPEAT(decorator), CHOICE(static, BLANK), FIELD property, CHOICE(_initializer, BLANK) ]
                    is_static_attr = False
                    if member_node.children and member_node.children[0].type == 'static': # Check if first non-decorator child is 'static'
                        # Need to skip decorators first if any.
                        actual_member_children = [c for c in member_node.children if c.type != 'decorator']
                        if actual_member_children and actual_member_children[0].type == 'static':
                            is_static_attr = True

                    attr_name_node = member_node.child_by_field_name("property") # grammar.json uses "property"
                    initializer_node = None
                    for child in member_node.children: # _initializer is not a field
                        if child.type == "_initializer":
                            initializer_node = child
                            break

                    attr_value_node = initializer_node.child_by_field_name("value") if initializer_node else None

                    if attr_name_node:
                        name = self._get_node_text(attr_name_node, source_code)
                        value = self._get_node_text(attr_value_node, source_code) if attr_value_node else None
                        attributes.append(Attribute(name=name, type=None, static=is_static_attr, value=value))

        docstring = self._get_jsdoc_comment(node, source_code)

        return Class(
            name=class_name,
            bases=bases or None,
            methods=methods or None,
            attributes=attributes or None,
            docstring=docstring
        )

    def _process_function_declaration(self, node: Node, source_code: bytes) -> Function:
        name_node = node.child_by_field_name("name")
        func_name = self._get_node_text(name_node, source_code) if name_node else "UnknownFunction"
        logger.debug(f"Processing function: {func_name}")

        parameters_node_to_process: Optional[Node] = None

        # Due to "_call_signature" being inlined and defined as a FIELD named "parameters"
        # containing "formal_parameters", we access it directly on the function_declaration node.
        params_field_node = node.child_by_field_name("parameters")

        if params_field_node:
            # The node obtained from child_by_field_name("parameters") should be the "formal_parameters" node.
            if params_field_node.type == "formal_parameters":
                parameters_node_to_process = params_field_node
                logger.debug(
                    f"  Successfully found 'formal_parameters' node (type: {params_field_node.type}) "
                    f"via field 'parameters' for '{func_name}'. Node SEXP: {parameters_node_to_process.sexp()}"
                )
            else:
                # This case would be unexpected if the grammar is consistently applied.
                logger.warning(
                    f"  Node found by field 'parameters' for function '{func_name}' is of type "
                    f"'{params_field_node.type}', NOT 'formal_parameters' as expected from inlined _call_signature. "
                    f"Node SEXP: {params_field_node.sexp()}. Parameters might not be processed correctly."
                )
                # parameters_node_to_process remains None or handle as an error
        else:
            logger.warning(
                f"  Could not find 'formal_parameters' node using field 'parameters' for function '{func_name}'. "
                f"Function node SEXP: {node.sexp()}. This suggests an issue with accessing the inlined field "
                f"or an unexpected tree structure for this function_declaration."
            )

        parameters_list = self._process_parameters(parameters_node_to_process, source_code, m_name=func_name)
        docstring = self._get_jsdoc_comment(node, source_code)

        return Function(
            name=func_name,
            parameters=parameters_list or None, 
            docstring=docstring,
            calls=[]
        )


    