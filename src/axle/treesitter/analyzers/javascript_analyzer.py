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
    