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
        """Extract JSDoc comment from a node if present.
        
        JSDoc comments are block comments that start with /** and are typically
        placed immediately before the declaration they document.
        
        Args:
            node: The AST node to find JSDoc comment for
            source_code: The source code bytes
            
        Returns:
            The JSDoc comment text without the /** */ delimiters, or None if not found
        """
        # Look for comments before the current node
        current = node
        
        # First, try to find the previous sibling that might be a comment
        prev_sibling = current.prev_sibling
        
        # Sometimes comments are not direct siblings, so we need to look more broadly
        # We'll search backwards through siblings and also check the parent's previous siblings
        comment_candidates = []
        
        # Collect potential comment nodes by walking backwards
        while prev_sibling:
            if prev_sibling.type == "comment":
                comment_candidates.append(prev_sibling)
                # Only take the immediately preceding comment to avoid cross-contamination
                break
            elif prev_sibling.type in ("identifier", "}", ";", ")", "export", "async", "class_declaration", "function_declaration", "method_definition"):
                # These are likely part of other declarations, stop searching
                break
            prev_sibling = prev_sibling.prev_sibling
        
        # If no direct siblings found, try parent's previous siblings (for nested declarations like methods)
        if not comment_candidates and current.parent:
            parent_prev = current.parent.prev_sibling
            while parent_prev:
                if parent_prev.type == "comment":
                    comment_candidates.append(parent_prev)
                    # Only take the immediately preceding comment
                    break
                elif parent_prev.type in ("class_declaration", "function_declaration", "method_definition"):
                    # Stop if we hit another declaration
                    break
                parent_prev = parent_prev.prev_sibling
        
        # Process comment candidates to find JSDoc
        for comment_node in comment_candidates:
            comment_text = self._get_node_text(comment_node, source_code)
            
            # Check if it's a JSDoc comment (starts with /** and ends with */)
            if comment_text.startswith("/**") and comment_text.endswith("*/"):
                # Extract the content, removing the /** */ delimiters
                content = comment_text[3:-2].strip()
                
                # Clean up the content by removing leading * from each line
                lines = content.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line = line.strip()
                    # Remove leading * and whitespace
                    if line.startswith('*'):
                        line = line[1:].strip()
                    elif line.startswith('* '):
                        line = line[2:]
                    cleaned_lines.append(line)
                
                # Join lines and clean up extra whitespace
                result = '\n'.join(cleaned_lines).strip()
                
                # Only return non-empty comments
                if result:
                    return result
        
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
                    f"via field 'parameters' for '{func_name}'. Node structure: {str(parameters_node_to_process)}"
                )
            else:
                # This case would be unexpected if the grammar is consistently applied.
                logger.warning(
                    f"  Node found by field 'parameters' for function '{func_name}' is of type "
                    f"'{params_field_node.type}', NOT 'formal_parameters' as expected from inlined _call_signature. "
                    f"Node structure: {str(params_field_node)}. Parameters might not be processed correctly."
                )
                # parameters_node_to_process remains None or handle as an error
        else:
            logger.warning(
                f"  Could not find 'formal_parameters' node using field 'parameters' for function '{func_name}'. "
                f"Function node structure: {str(node)}. This suggests an issue with accessing the inlined field "
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


    def _process_import_statement(self, node: Node, source_code: bytes) -> List[Import]:
        imports: List[Import] = []

        # _from_clause contains FIELD "source" (string).
        # The import_statement can have _from_clause directly, or a "source" field for side-effect imports.
        from_clause_node = None
        for child in node.children: # Find _from_clause among direct children
            if child.type == "_from_clause":
                from_clause_node = child
                break

        source_node = node.child_by_field_name("source") # For side-effect: import "source";
        if from_clause_node: # If from_clause exists, source is inside it
            source_node = from_clause_node.child_by_field_name("source")

        if not source_node:
            return imports # Cannot proceed without a source
        source_str = self._strip_string_quotes(self._get_node_text(source_node, source_code))

        default_import_name: Optional[str] = None
        named_items: List[str] = []
        namespace_alias: Optional[str] = None

        import_clause_node: Optional[Node] = None # Explicitly find import_clause
        for child in node.children:
            if child.type == "import_clause":
                import_clause_node = child
                break

        if import_clause_node:
            # import_clause -> CHOICE [ namespace_import, named_imports, SEQ[identifier, OPTIONAL_SEQ] ]
            clause_content_node = import_clause_node.named_children[0] if import_clause_node.named_children else None
            if not clause_content_node: # Should have one of the choices
                 # Example: import_clause may directly be a 'namespace_import' or 'named_imports' node
                 if import_clause_node.type in ('namespace_import', 'named_imports', 'identifier'): # If clause itself is the content type
                     clause_content_node = import_clause_node
                 # Or, if it's a SEQ, the first element is key for default import
                 elif import_clause_node.children and import_clause_node.children[0].type == 'identifier':
                     clause_content_node = import_clause_node # Treat the SEQ itself for default processing

            if clause_content_node:
                if clause_content_node.type == 'identifier': # Default import: import D from 'M'
                    default_import_name = self._get_node_text(clause_content_node, source_code)
                    # Check for named/namespace imports after default: import D, { A } or import D, * as N
                    # The grammar is import_clause -> SEQ[identifier, OPTIONAL SEQ[",", CHOICE[namespace_import, named_imports]]]
                    # So, if clause_content_node was the SEQ itself (due to above logic)
                    if import_clause_node.children and len(import_clause_node.children) > 1:
                        optional_part_node = import_clause_node.children[1] # This is the CHOICE node after comma
                        if optional_part_node.named_children:
                             additional_import_node = optional_part_node.named_children[0]
                             if additional_import_node.type == 'named_imports':
                                 # Populate named_items (logic as before)
                                 for specifier in additional_import_node.named_children:
                                     if specifier.type == "import_specifier":
                                        # ... extract name/alias into named_items ... (see full method from prev. response)
                                        name_field = specifier.child_by_field_name("name")
                                        alias_field = specifier.child_by_field_name("alias")
                                        item_text = ""
                                        if alias_field: item_text = self._get_node_text(alias_field, source_code)
                                        elif name_field: item_text = self._get_node_text(name_field, source_code)
                                        if item_text: named_items.append(item_text)

                             elif additional_import_node.type == 'namespace_import':
                                 # Populate namespace_alias 
                                 # namespace_import -> "*" "as" identifier
                                 if len(additional_import_node.children) >= 3:
                                     ns_id_node = additional_import_node.children[2] # The identifier part
                                     if ns_id_node.type == 'identifier':
                                          namespace_alias = self._get_node_text(ns_id_node, source_code)


                elif clause_content_node.type == 'named_imports':
                    for specifier in clause_content_node.named_children:
                        if specifier.type == "import_specifier":
                            name_field = specifier.child_by_field_name("name")    # This is _module_export_name (identifier or string)
                            alias_field = specifier.child_by_field_name("alias") # This is identifier

                            imported_as = ""
                            original_name_str = self._get_node_text(name_field, source_code)

                            if alias_field:
                                imported_as = self._get_node_text(alias_field, source_code)
                            else:
                                imported_as = original_name_str

                            # For 'import { default as ReactDOM }'
                            if original_name_str == "default" and alias_field:
                                default_import_name = imported_as # Treat as a default import
                            else:
                                if imported_as: named_items.append(imported_as)

                elif clause_content_node.type == 'namespace_import':
                    # namespace_import -> "*" "as" identifier
                    if len(clause_content_node.children) >= 3 : # children are '*', 'as', identifier
                        ns_id_node = clause_content_node.children[2]
                        if ns_id_node.type == 'identifier':
                            namespace_alias = self._get_node_text(ns_id_node, source_code)

        # Create Import objects
        if default_import_name:
            imports.append(Import(name=default_import_name, source=source_str, items=named_items if named_items else None))
        # Handle cases where named_items were collected but default_import_name was not set (e.g. pure named import)
        elif named_items and not default_import_name and not namespace_alias:
            imports.append(Import(name=None, source=source_str, items=named_items))

        if namespace_alias and not default_import_name: # Ensure not to double count if default also had namespace
            imports.append(Import(name=namespace_alias, source=source_str, items=["*"]))

        # Side-effect import: `import 'module'`
        # Grammar: import_statement -> "import" FIELD "source" string ...
        if not import_clause_node and node.child_by_field_name("source"): # No clause, direct source field
            imports.append(Import(name=None, source=source_str, items=None))

        return imports
    
    def _process_variable_declaration(self, node: Node, source_code: bytes) -> List[Variable]:
        """Process a variable declaration statement (const, let, var)."""
        variables = []
        
        declaration_kind_str = "external_variable" # Default, suitable for 'var' and 'let'
        
        if node.children:
            first_child_node = node.children[0]
            # Check the type of the first child, which is often the keyword node
            first_child_type_str = first_child_node.type 
            if first_child_type_str == "const": # tree-sitter often uses the keyword itself as the type
                declaration_kind_str = "constant"
            # 'let' and 'var' can map to "external_variable" as per current Pydantic Literal
            elif first_child_type_str == "var":
                 declaration_kind_str = "external_variable" # Or a more generic "variable" if model changes
            elif first_child_type_str == "let":
                 declaration_kind_str = "external_variable" # Changed from "lexical_variable"

        # Fallback if the node type itself indicates the kind (e.g., "const_declaration")
        elif node.type.startswith("const"): 
             declaration_kind_str = "constant"
        
        for child_node in node.children:
            if child_node.type == "variable_declarator":
                declarator = child_node 
                name_node = declarator.child_by_field_name("name")
                value_node = declarator.child_by_field_name("value")

                if name_node:
                    name = self._get_node_text(name_node, source_code)
                    value_str = None
                    if value_node:
                        value_str = self._get_node_text(value_node, source_code)
                        if value_node.type in ("string_literal", "string", "template_string"):
                             value_str = self._strip_string_quotes(value_str)

                    variables.append(Variable(name=name, kind=declaration_kind_str, type=None, value=value_str))
        return variables

    def _process_export_statement(self, node: Node, source_code: bytes) -> List[Import]:
        exports: List[Import] = []

        # Grammar for export_statement is a CHOICE of two main SEQ patterns.
        # Pattern 1: "export" (CHOICE ["*", namespace_export, export_clause]) (_from_clause)? _semicolon;
        # Pattern 2: "export" (FIELD "declaration" | SEQ ["default", CHOICE [FIELD "declaration", FIELD "value" _semicolon]])

        # Check children for keywords and structure
        children = list(node.children)
        
        # Skip leading decorators if any for cleaner logic on keyword positions
        start_index = 0
        while start_index < len(children) and children[start_index].type == "decorator":
            start_index += 1
        
        if start_index >= len(children) or children[start_index].type != 'export': # Should always start with "export" (after decorators)
            return exports 

        # Pattern 2: Handles `export default ...` and `export <declaration>`
        # Check for "default" keyword after "export"
        if (start_index + 1) < len(children) and children[start_index + 1].type == 'default': # export default ...
            is_default_export = True
            # The item exported is either a "declaration" field or a "value" field (expression)
            # These fields are on the main export_statement node according to grammar.json
            # for `export default declaration` or `export default value;`
            declaration_node = node.child_by_field_name("declaration") # For `export default function/class Foo(){}`
            value_node = node.child_by_field_name("value")             # For `export default myIdentifierOrExpr;`
            
            item_to_export_node = declaration_node or value_node

            if item_to_export_node:
                default_export_name = "default" 
                source_name_for_default = "self" 

                if item_to_export_node.type in ("function_declaration", "class_declaration"):
                    name_field_node = item_to_export_node.child_by_field_name("name")
                    if name_field_node: 
                        declared_name = self._get_node_text(name_field_node, source_code)
                        default_export_name = declared_name
                        source_name_for_default = declared_name # For 'export default class MainService {}' -> source='MainService'
                elif item_to_export_node.type == "identifier": 
                    declared_name = self._get_node_text(item_to_export_node, source_code)
                    default_export_name = declared_name
                    source_name_for_default = declared_name 
                
                exports.append(Import(name=default_export_name, source=source_name_for_default, items=["default"]))
            return exports

        # Pattern 2 (continued): `export <declaration>` (no "default" keyword)
        declaration_node_non_default = node.child_by_field_name("declaration")
        if declaration_node_non_default:
            if declaration_node_non_default.type in ("lexical_declaration", "variable_declaration"):
                temp_vars = self._process_variable_declaration(declaration_node_non_default, source_code)
                for var_info in temp_vars:
                    exports.append(Import(name=var_info.name, source="self", items=[var_info.name]))
            elif declaration_node_non_default.type in ("function_declaration", "class_declaration"):
                name_node = declaration_node_non_default.child_by_field_name("name")
                if name_node:
                    name_str = self._get_node_text(name_node, source_code)
                    exports.append(Import(name=name_str, source="self", items=[name_str]))
            return exports

        # Pattern 1: Handles `export * ...`, `export namespace ...`, `export { ... }`
        # Structure: "export" (CHOICE) (_from_clause)? _semicolon
        # The CHOICE node is children[start_index + 1]
        
        choice_node_after_export: Optional[Node] = None
        if (start_index + 1) < len(children):
            choice_node_after_export = children[start_index + 1]
        
        from_clause_node: Optional[Node] = None # _from_clause contains FIELD "source" string
        # _from_clause is a sibling to the choice_node_after_export or part of its SEQ members
        # The grammar shows _from_clause can follow '*', namespace_export, or export_clause.
        
        # Try finding _from_clause directly on the export_statement node
        # Grammar: _from_clause -> "from" FIELD "source" string
        # This might not be a field on export_statement itself, but part of a SEQ.

        # Find export_clause and _from_clause by iterating children of the main 'node'
        actual_export_clause_node: Optional[Node] = None
        actual_from_clause_node: Optional[Node] = None
        star_literal_node: Optional[Node] = None
        namespace_export_symbol_node: Optional[Node] = None

        for child in node.children: # Iterate children of the main 'export_statement' node
            if child.type == "export_clause":
                actual_export_clause_node = child
            elif child.type == "_from_clause": # This node contains the source string for re-exports
                actual_from_clause_node = child
            elif child.type == '"*"': # string literal "*"
                star_literal_node = child
            elif child.type == "namespace_export": # For * as ns
                 namespace_export_symbol_node = child


        if actual_export_clause_node:
            # export_clause -> "{" REPEAT(export_specifier) "}"
            # export_specifier -> FIELD "name" _module_export_name [ "as" FIELD "alias" _module_export_name ]
            exported_items_list = []
            for specifier_node in actual_export_clause_node.children:
                if specifier_node.type == "export_specifier":
                    name_field_node = specifier_node.child_by_field_name("name") # _module_export_name
                    alias_field_node = specifier_node.child_by_field_name("alias") # _module_export_name
                    
                    item_name_to_add = ""
                    if alias_field_node: # export { name as alias } -> alias is exported name
                        item_name_to_add = self._get_node_text(alias_field_node, source_code)
                    elif name_field_node: # export { name } -> name is exported name
                        item_name_to_add = self._get_node_text(name_field_node, source_code)
                    
                    if item_name_to_add:
                        exported_items_list.append(item_name_to_add)
            
            if exported_items_list:
                module_source_str = "self"
                if actual_from_clause_node:
                    source_ident_node = actual_from_clause_node.child_by_field_name("source") # This is a 'string' node
                    if source_ident_node:
                         module_source_str = self._strip_string_quotes(self._get_node_text(source_ident_node, source_code))
                
                exports.append(Import(
                    name=None if module_source_str == "self" else module_source_str, 
                    source=module_source_str, 
                    items=exported_items_list
                ))
            return exports

        # Handling `export * from 'module'` or `export * as namespace from 'module'`
        # export_statement -> "export" "*" _from_clause
        # export_statement -> "export" namespace_export _from_clause
        if (star_literal_node or namespace_export_symbol_node) and actual_from_clause_node:
            source_ident_node = actual_from_clause_node.child_by_field_name("source")
            if source_ident_node:
                module_source_str = self._strip_string_quotes(self._get_node_text(source_ident_node, source_code))
                if namespace_export_symbol_node: # export * as ns from 'module'
                    # namespace_export -> "*" "as" _module_export_name (identifier)
                    # The third child of namespace_export is the identifier for 'ns'
                    if len(namespace_export_symbol_node.children) >= 3:
                        ns_alias_node = namespace_export_symbol_node.children[2] # _module_export_name
                        if ns_alias_node: # Check if it's identifier or string
                             ns_name = self._get_node_text(ns_alias_node, source_code)
                             exports.append(Import(name=ns_name, source=module_source_str, items=[f"* as {ns_name}"])) # Or just items=["*"]
                elif star_literal_node: # export * from 'module'
                    exports.append(Import(name=module_source_str, source=module_source_str, items=["*"]))
            return exports
            
        return exports

    def _analyze_tree(self, tree: Tree, source_code: bytes, file_path: Path) -> FileAnalysis:
        """Analyze the JavaScript syntax tree and return structured data."""
        root = tree.root_node
        logger.debug(f"Starting analysis for: {file_path}")
        
        analysis = FileAnalysis(
            file_path=str(file_path),
            analyzer="treesitter_javascript", # Use direct string
            imports=[],
            classes=[],
            functions=[],
            variables=[],
            enums=None 
        )
        
        top_level_nodes = root.children if root else [] # Check if root exists
        for node in top_level_nodes:
            node_type = node.type
            if node_type == "import_statement":
                imports_list = self._process_import_statement(node, source_code)
                if imports_list: analysis.imports.extend(imports_list)
            elif node_type == "export_statement":
                # _process_export_statement should return a list of Import-like objects
                # These represent what's being made available by the module.
                exports_list = self._process_export_statement(node, source_code)
                if exports_list: analysis.imports.extend(exports_list) # Storing exports in the 'imports' field for now
            elif node_type == "class_declaration":
                class_obj = self._process_class_declaration(node, source_code)
                if class_obj: analysis.classes.append(class_obj)
            elif node_type == "function_declaration": # Handles `function foo() {}`
                func_obj = self._process_function_declaration(node, source_code)
                if func_obj: analysis.functions.append(func_obj)
            elif node_type in ("lexical_declaration", "variable_declaration"): # Common for const, let, var
                vars_list = self._process_variable_declaration(node, source_code)
                if vars_list: 
                    for var_obj in vars_list: # Check if any variable is an arrow function
                        # This requires inspecting var_obj.value or the original value_node
                        # For simplicity, this example doesn't convert arrow func vars to Function objects here.
                        # That logic was previously in _analyze_tree's expression_statement part.
                        analysis.variables.append(var_obj)
            elif node_type == "expression_statement":
                # Handles assignments like `myGlobal = () => {}` if not a const/let/var declaration
                if node.named_children: # Or node.children
                    expression_child_node = node.named_children[0] # Or node.children[0]
                    if expression_child_node.type == "assignment_expression": # Or "assignment"
                        target_node = expression_child_node.child_by_field_name("left")
                        value_node = expression_child_node.child_by_field_name("right")
                        
                        if target_node and target_node.type == "identifier" and \
                           value_node and value_node.type == "arrow_function":
                            name = self._get_node_text(target_node, source_code)
                            parameters_node = value_node.child_by_field_name("parameters")
                            parameters = self._process_parameters(parameters_node, source_code, m_name=name)
                            
                            func_obj = Function(
                                name=name,
                                parameters=parameters or None,
                                docstring=self._get_jsdoc_comment(value_node, source_code), # JSDoc on arrow func
                                calls=[]
                            )
                            if func_obj: analysis.functions.append(func_obj)
        
        # Set lists to None if they are empty, as per Pydantic model (Optional fields)
        if not analysis.imports: analysis.imports = None
        if not analysis.classes: analysis.classes = None
        if not analysis.functions: analysis.functions = None
        if not analysis.variables: analysis.variables = None
        
        return analysis