"""Python-specific Tree-sitter analyzer."""

import textwrap
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

class PythonAnalyzer(BaseAnalyzer):
    LANGUAGE_NAME = "python"
    FILE_EXTENSIONS = (".py",)

    def _get_node_text(self, node: Optional[Node], source_code: bytes) -> str:
        if node is None: return ""
        return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _strip_string_quotes(self, s: str) -> str:
        if (s.startswith('"""') and s.endswith('"""')) or \
           (s.startswith("'''") and s.endswith("'''")):
            return s[3:-3]
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def _get_docstring_from_block(self, block_node: Optional[Node], source_code: bytes) -> Optional[str]:
        if not block_node or not block_node.named_children:
            return None
        
        first_statement_candidate = block_node.named_children[0]
        if first_statement_candidate.type == "expression_statement":
            if first_statement_candidate.named_children and \
               first_statement_candidate.named_children[0].type == "string":
                string_node = first_statement_candidate.named_children[0]
                raw_docstring_text = self._get_node_text(string_node, source_code)
                unquoted_docstring = self._strip_string_quotes(raw_docstring_text)
                if '\n' in unquoted_docstring or \
                   (raw_docstring_text.startswith('"""') or raw_docstring_text.startswith("'''")):
                    return textwrap.dedent(unquoted_docstring).strip()
                else:
                    return unquoted_docstring.strip()
        return None

    def _get_identifier_from_typed_node(self, typed_node: Node, source_code: bytes) -> Optional[Node]:
        name_node = typed_node.child_by_field_name("name")
        if name_node: return name_node
        name_node = typed_node.child_by_field_name("parameter")
        if name_node: return name_node
        if typed_node.named_child_count > 0:
            first_named_child = typed_node.named_children[0]
            if first_named_child.type == 'identifier':
                return first_named_child
        if typed_node.child_count > 0:
            first_child = typed_node.children[0]
            if first_child.is_named and first_child.type == 'identifier':
                 return first_child
        return None

    def _process_parameters(self, parameters_node: Node, source_code: bytes,
                            m_name: Optional[str] = None,
                            c_name: Optional[str] = None) -> List[Parameter]:
        params = []
        is_target_method_for_diag = m_name == "create_user" and (c_name is None or c_name == "UserService")

        if is_target_method_for_diag:
            logger.info(f"\n DIAGNOSTIC: Entering _process_parameters for {c_name}.{m_name}")
            if parameters_node:
                node_repr = str(parameters_node) 
                if hasattr(parameters_node, 'sexp') and callable(str(parameters_node)):
                    try:
                        s_expr_result = parameters_node.sexp()
                        if isinstance(s_expr_result, bytes):
                            node_repr = s_expr_result.decode('utf-8', 'replace')
                        elif isinstance(s_expr_result, str):
                            node_repr = s_expr_result
                    except DeprecationWarning: 
                        logger.debug("Node.sexp() is deprecated, using str() for diagnostic output.")
                    except Exception as e:
                        logger.warning(f"Error getting/decoding .sexp(): {e}, falling back to str().")
                
                logger.info(f"  Parameters Node Representation:\n{node_repr}")
                logger.info(f"  Iterating NAMED children of Parameters Node (Count: {len(list(parameters_node.named_children))}):")
            else:
                 logger.info(f"  DIAGNOSTIC: Parameters Node for {c_name}.{m_name} is None itself.")

        for i, param_node in enumerate(parameters_node.named_children if parameters_node else []):
            param_name: Optional[str] = None
            param_type_str: Optional[str] = None
            node_type = param_node.type
            node_text = self._get_node_text(param_node, source_code)

            if is_target_method_for_diag:
                param_node_repr = str(param_node)
                if hasattr(param_node, 'sexp') and callable(str(param_node)):
                    try:
                        s_expr_res = param_node.sexp()
                        if isinstance(s_expr_res, bytes): param_node_repr = s_expr_res.decode('utf-8', 'replace')
                        elif isinstance(s_expr_res, str): param_node_repr = s_expr_res
                    except DeprecationWarning: pass 
                    except Exception: pass 
                logger.info(f"    DIAG PARAM LOOP [{i}]: node_type='{param_node.type}', node_text='{node_text}'\n      Representation: {param_node_repr}")
            
            name_field_node: Optional[Node] = None
            type_field_node: Optional[Node] = None
            if node_type == "identifier": param_name = node_text
            elif node_type == "typed_parameter":
                name_field_node = self._get_identifier_from_typed_node(param_node, source_code)
                type_field_node = param_node.child_by_field_name("type")
                if not type_field_node and name_field_node and param_node.named_child_count > 0:
                    type_field_node = next((child for child in param_node.named_children if child != name_field_node and child.type not in ['identifier', 'ERROR']), None)
                if name_field_node: param_name = self._get_node_text(name_field_node, source_code)
                if type_field_node: param_type_str = self._get_node_text(type_field_node, source_code)
            elif node_type == "default_parameter":
                name_container_node = param_node.child_by_field_name("name")
                if name_container_node:
                    if name_container_node.type == "identifier": param_name = self._get_node_text(name_container_node, source_code)
                    elif name_container_node.type == "typed_parameter":
                        name_field_node = self._get_identifier_from_typed_node(name_container_node, source_code)
                        type_field_node = name_container_node.child_by_field_name("type")
                        if not type_field_node and name_field_node and name_container_node.named_child_count > 0:
                             type_field_node = next((child for child in name_container_node.named_children if child != name_field_node and child.type not in ['identifier', 'ERROR']), None)
                        if name_field_node: param_name = self._get_node_text(name_field_node, source_code)
                        if type_field_node: param_type_str = self._get_node_text(type_field_node, source_code)
            elif node_type == "typed_default_parameter":
                typed_param_part_node: Optional[Node] = None
                for field_name_attempt in ["parameter", "name"]:
                    candidate_node = param_node.child_by_field_name(field_name_attempt)
                    if candidate_node and candidate_node.type == "typed_parameter": typed_param_part_node = candidate_node; break
                if not typed_param_part_node: typed_param_part_node = next((c for c in param_node.named_children if c.type == 'typed_parameter'), None)
                if not typed_param_part_node: typed_param_part_node = next((c for c in param_node.children if c.is_named and c.type == 'typed_parameter'), None)
                if typed_param_part_node:
                    name_field_node = self._get_identifier_from_typed_node(typed_param_part_node, source_code)
                    type_field_node = typed_param_part_node.child_by_field_name("type")
                    if not type_field_node and name_field_node and typed_param_part_node.named_child_count > 0:
                        type_field_node = next((child for child in typed_param_part_node.named_children if child != name_field_node and child.type not in ['identifier', 'ERROR']), None)
                    if name_field_node: param_name = self._get_node_text(name_field_node, source_code)
                    if type_field_node: param_type_str = self._get_node_text(type_field_node, source_code)
                else: 
                    name_field_node = self._get_identifier_from_typed_node(param_node, source_code)
                    type_field_node = param_node.child_by_field_name("type")
                    if not type_field_node and name_field_node and param_node.named_child_count > 0:
                         potential_types = [c for c in param_node.named_children if c != name_field_node and c != param_node.child_by_field_name("value") and c.type not in ['identifier', 'ERROR', 'operator']]
                         if len(potential_types) == 1: type_field_node = potential_types[0]
                    if name_field_node: param_name = self._get_node_text(name_field_node, source_code)
                    if type_field_node: param_type_str = self._get_node_text(type_field_node, source_code)
                    logger.info(f"    TDP: Flat structure attempt for '{node_text}'. Name: '{param_name}', Type: '{param_type_str}'.")
            elif node_type == "list_splat_pattern":
                ident_node = param_node.child_by_field_name("name") or next((c for c in param_node.named_children if c.type == 'identifier'), None) or (param_node.children[1] if param_node.child_count > 1 and param_node.children[0].type == "*" and param_node.children[1].type == 'identifier' else None)
                if ident_node: param_name = self._get_node_text(ident_node, source_code)
                else: logger.warning(f"    Could not find identifier for list_splat_pattern: {node_text}")
            elif node_type == "dictionary_splat_pattern":
                ident_node = param_node.child_by_field_name("name") or next((c for c in param_node.named_children if c.type == 'identifier'), None) or (param_node.children[1] if param_node.child_count > 1 and param_node.children[0].type == "**" and param_node.children[1].type == 'identifier' else None)
                if ident_node: param_name = self._get_node_text(ident_node, source_code)
                else: logger.warning(f"    Could not find identifier for dictionary_splat_pattern: {node_text}")
            elif node_type in ("positional_separator", "keyword_separator", "tuple_pattern"):
                if is_target_method_for_diag: logger.info(f"    DIAG PARAM SKIP: type='{node_type}'")
                continue
            else: logger.warning(f"    Unhandled parameter node type: '{node_type}' with text '{node_text}'")

            if param_name:
                if is_target_method_for_diag: logger.info(f"    DIAG PARAM APPEND: Name='{param_name}', Type='{param_type_str}'")
                params.append(Parameter(name=param_name, type=param_type_str))
            elif node_type not in ("positional_separator", "keyword_separator", "tuple_pattern"):
                if is_target_method_for_diag: logger.info(f"    DIAG PARAM MISSED: Name NOT found for node_type='{node_type}', text='{node_text}'")
                logger.warning(f"    Could not extract name for param node: type='{node_type}', text='{node_text}'.")
        
        if is_target_method_for_diag:
            logger.info(f"DIAGNOSTIC: Exiting _process_parameters for {c_name}.{m_name}. Total parameters extracted: {len(params)}")
            if params:
                 logger.info(f"  Final extracted params for {c_name}.{m_name}:")
                 for p_idx, p_obj in enumerate(params): logger.info(f"    Param [{p_idx}]: Name='{p_obj.name}', Type='{p_obj.type}'")
            else: logger.info(f"  No parameters extracted for {c_name}.{m_name}.")
        return params

    def _process_method(self, node: Node, source_code: bytes, class_name: Optional[str] = None) -> Method:
        name_node = node.child_by_field_name("name")
        method_name_str = self._get_node_text(name_node, source_code) if name_node else "UnknownMethod"
        parameters_node = node.child_by_field_name("parameters")
        current_parameters: List[Parameter] = []
        if parameters_node:
            current_parameters = self._process_parameters(parameters_node, source_code, m_name=method_name_str, c_name=class_name)
        elif method_name_str == "create_user" and (class_name is None or class_name == "UserService"):
            logger.info(f"DIAGNOSTIC: Parameters Node IS NONE for method '{class_name}.{method_name_str}'.")
        docstring = self._get_docstring_from_block(node.child_by_field_name("body"), source_code)
        return Method(name=method_name_str, parameters=current_parameters or None, docstring=docstring, calls=[])

    def _process_class(self, node: Node, source_code: bytes) -> Optional[Union[Class, Enum]]:
        name_node = node.child_by_field_name("name")
        class_name_str = self._get_node_text(name_node, source_code) if name_node else "UnknownClass"
        docstring = self._get_docstring_from_block(node.child_by_field_name("body"), source_code)
        bases: List[BaseClass] = []
        base_class_names: List[str] = []
        superclasses_node = node.child_by_field_name("superclasses")
        if superclasses_node:
            for base_arg_node in superclasses_node.named_children:
                if base_arg_node.type in ("identifier", "dotted_name", "attribute"):
                    base_name = self._get_node_text(base_arg_node, source_code)
                    bases.append(BaseClass(name=base_name)); base_class_names.append(base_name)
        is_enum = any(b_name in ["Enum", "enum.Enum"] for b_name in base_class_names)
        methods: List[Method] = []; attributes: List[Attribute] = []; enum_members: List[EnumMember] = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    methods.append(self._process_method(child, source_code, class_name=class_name_str))
                elif child.type == "expression_statement":
                    first_expr = child.children[0] if child.children else None
                    if first_expr and first_expr.type == "assignment":
                        if is_enum:
                            member = self._process_enum_member(first_expr, source_code)
                            if member: enum_members.append(member)
                        else:
                            attr = self._process_class_attribute(first_expr, source_code)
                            if attr: attributes.append(attr)
        if is_enum: return Enum(name=class_name_str, bases=bases or None, members=enum_members or None, methods=methods or None, docstring=docstring)
        return Class(name=class_name_str, bases=bases or None, methods=methods or None, attributes=attributes or None, docstring=docstring)

    def _process_import(self, node: Node, source_code: bytes) -> List[Import]:
        imports = []
        logger.debug(f"  Processing import_statement: {self._get_node_text(node, source_code)}")
        import_item_container_node = node.child_by_field_name("name")
        item_nodes_to_process = []
        if import_item_container_node:
            if import_item_container_node.type in ("_import_list", "import_list"):
                item_nodes_to_process = list(import_item_container_node.named_children)
            elif import_item_container_node.type in ("dotted_name", "identifier", "aliased_import"):
                item_nodes_to_process = [import_item_container_node]
            else:
                logger.warning(f"    Import: Unexpected type '{import_item_container_node.type}' for 'name' field. Processing its named children.")
                item_nodes_to_process = list(import_item_container_node.named_children)
        else:
            container_by_type = next((child for child in node.named_children if child.type in ("_import_list", "import_list")), None)
            if container_by_type: item_nodes_to_process = list(container_by_type.named_children)
            else:
                logger.warning(f"    Import: No clear container. Processing direct named children (excluding 'import' keyword).")
                item_nodes_to_process = [n for n in node.named_children if n.type != 'import']
        for item_node in item_nodes_to_process:
            if item_node.type == "dotted_name":
                name = self._get_node_text(item_node, source_code)
                imports.append(Import(name=name, source=name, items=None))
            elif item_node.type == "aliased_import":
                name_field = item_node.child_by_field_name("name")
                alias_field = item_node.child_by_field_name("alias")
                original_name = self._get_node_text(name_field, source_code) if name_field else ""
                if alias_field: imports.append(Import(name=self._get_node_text(alias_field, source_code), source=original_name, items=None))
                elif original_name: imports.append(Import(name=original_name, source=original_name, items=None))
        return imports

    def _process_import_from(self, node: Node, source_code: bytes) -> List[Import]:
            logger.debug(f"  Processing import_from_statement: {self._get_node_text(node, source_code)}")
            module_name_str = ""
            imported_items_texts = []
            is_wildcard_import = False

            # Get module name
            module_node_candidate = node.child_by_field_name("module_name") or \
                                    node.child_by_field_name("module") 
            
            module_name_str = self._get_node_text(module_node_candidate, source_code)
            
            # Get all nodes associated with the "name" field, as it can be multiple.
            # These can be import_list, parenthesized_import_list, wildcard_import, 
            # or direct identifiers/dotted_names/aliased_imports.
            all_name_field_nodes = list(node.children_by_field_name("name"))
            
            if not all_name_field_nodes:
                # Check if items are direct children not under 'name' field (older/different grammars)
                potential_item_nodes = [
                    child for child in node.named_children 
                    if child.type in ("wildcard_import", "import_list", "parenthesized_import_list", 
                                    "dotted_name", "identifier", "aliased_import")
                    and child != module_node_candidate # Avoid reprocessing module_name if it's a direct child
                ]
                all_name_field_nodes = potential_item_nodes

            # This list will hold the actual item nodes (identifiers, aliased_imports, etc.)
            item_nodes_to_extract_text_from = [] 
            
            for name_field_node_idx, name_field_node in enumerate(all_name_field_nodes):
                if name_field_node.type == "wildcard_import":
                    is_wildcard_import = True
                    imported_items_texts.append("*")
                    break # Wildcard import usually means no other specific items are listed with it
                elif name_field_node.type == "import_list":
                    item_nodes_to_extract_text_from.extend(list(name_field_node.named_children))
                elif name_field_node.type == "parenthesized_import_list":
                    inner_list_node = next((child for child in name_field_node.named_children if child.type == "import_list"), None)
                    if inner_list_node:
                        item_nodes_to_extract_text_from.extend(list(inner_list_node.named_children))
                    else: # Process named children of parenthesized_import_list directly if no inner import_list
                        item_nodes_to_extract_text_from.extend(list(name_field_node.named_children))
                elif name_field_node.type in ("dotted_name", "identifier", "aliased_import"):
                    item_nodes_to_extract_text_from.append(name_field_node)
                else:
                    # This case handles unexpected structures. If a 'name' field node is not one of the above,
                    # we can try to see if its named children are the actual import items.
                    logger.warning(f"    ImportFrom: Unexpected type '{name_field_node.type}' for a 'name' field node. Text: '{self._get_node_text(name_field_node, source_code)}'. Attempting to process its named children.")
                    item_nodes_to_extract_text_from.extend(list(name_field_node.named_children))

            if not is_wildcard_import:
                logger.debug(f"    ImportFrom: Nodes to extract text from: {len(item_nodes_to_extract_text_from)}")
                for item_idx, item_node in enumerate(item_nodes_to_extract_text_from):
                    item_text = ""
                    logger.debug(f"      Extracting text from item_node [{item_idx}]: type='{item_node.type}', text='{self._get_node_text(item_node, source_code)}'")
                    if item_node.type == "aliased_import":
                        alias_node = item_node.child_by_field_name("alias")
                        if alias_node: # Use the alias as the imported name
                            item_text = self._get_node_text(alias_node, source_code)
                        else: # No alias, use the original name part of the aliased_import
                            name_node = item_node.child_by_field_name("name")
                            if name_node: 
                                item_text = self._get_node_text(name_node, source_code)
                                logger.debug(f"        Aliased import item missing 'alias' field, using 'name' field text: {item_text}")
                            else:
                                logger.warning(f"        Aliased import item missing both 'alias' and 'name' fields: {self._get_node_text(item_node, source_code)}")
                    elif item_node.type in ("dotted_name", "identifier"):
                        item_text = self._get_node_text(item_node, source_code)
                    elif item_node.type == "ERROR": # Skip ERROR nodes that might appear in malformed lists
                        logger.warning(f"        Skipping ERROR item_node in import_from items list: {self._get_node_text(item_node, source_code)}")
                        continue 
                    else:
                        logger.warning(f"        Skipping unexpected item_node type '{item_node.type}' during text extraction from import_from items list.")
                        continue

                    if item_text:
                        logger.debug(f"        Extracted item: '{item_text}'")
                        imported_items_texts.append(item_text)
                    # Only warn if text extraction failed for an expected type
                    elif item_node.type in ("aliased_import", "dotted_name", "identifier"): 
                        logger.warning(f"        Could not extract text for import_from item_node: type='{item_node.type}', text='{self._get_node_text(item_node,source_code)}'")

            # Finalize items list for the Import object
            # If wildcard, items should be ["*"]. If specific items, use them. Otherwise, None.
            items_for_import_object = None
            if is_wildcard_import: # `imported_items_texts` should already contain "*" if wildcard was processed.
                if "*" not in imported_items_texts: imported_items_texts.append("*") # Ensure it's there
                items_for_import_object = ["*"] # Standardize to just ["*"] for wildcard
            elif imported_items_texts:
                items_for_import_object = imported_items_texts
            
            if module_name_str and (items_for_import_object is not None): # Check items_for_import_object not module_name_str and (imported_items_texts or is_wildcard_import)
                logger.debug(f"    ImportFrom: Successfully processed. Module: '{module_name_str}', Items: {items_for_import_object}")
                return [Import(name=module_name_str, source=module_name_str, items=items_for_import_object)]
            
            logger.warning(f"    ImportFrom: Failed to fully process (module or items missing at the end). Node: {self._get_node_text(node, source_code)}. Module: '{module_name_str}', Extracted items: {imported_items_texts}, Wildcard: {is_wildcard_import}")
            return []

    def _process_function(self, node: Node, source_code: bytes) -> Function:
        name_node = node.child_by_field_name("name")
        func_name = self._get_node_text(name_node, source_code) if name_node else "UnknownFunction"
        parameters_node = node.child_by_field_name("parameters")
        parameters = self._process_parameters(parameters_node, source_code, m_name=func_name) if parameters_node else []
        docstring = self._get_docstring_from_block(node.child_by_field_name("body"), source_code)
        return Function(name=func_name, parameters=parameters or None, docstring=docstring, calls=[])

    def _process_expression_statement_for_variable(self, node: Node, source_code: bytes) -> Optional[Variable]:
        if not node.named_children or node.named_children[0].type != "assignment": return None
        assignment_node = node.named_children[0]
        name_str: Optional[str] = None; value_str: Optional[str] = None; type_str: Optional[str] = None
        left_node = assignment_node.child_by_field_name("left")
        right_node = assignment_node.child_by_field_name("right")
        type_annotation_node = assignment_node.child_by_field_name("type")
        if left_node:
            if left_node.type == "identifier":
                name_str = self._get_node_text(left_node, source_code)
                if type_annotation_node: type_str = self._get_node_text(type_annotation_node, source_code)
            elif left_node.type == "typed_identifier":
                name_cand_node = self._get_identifier_from_typed_node(left_node, source_code)
                type_cand_node = left_node.child_by_field_name("type")
                if name_cand_node: name_str = self._get_node_text(name_cand_node, source_code)
                if type_cand_node: type_str = self._get_node_text(type_cand_node, source_code)
        if right_node:
            simple_literals = ("integer", "string", "float", "true", "false", "none")
            if right_node.type in simple_literals:
                value_str = self._get_node_text(right_node, source_code)
                if right_node.type == "string": value_str = self._strip_string_quotes(value_str)
        if name_str:
            is_const = name_str.isupper() and (name_str.isalpha() or ("_" in name_str and name_str.replace("_", "").isalpha()))
            kind = "constant" if is_const else "external_variable"
            return Variable(name=name_str, kind=kind, type=type_str, value=value_str)
        return None

    def _process_class_attribute(self, assignment_node: Node, source_code: bytes) -> Optional[Attribute]:
        name_str: Optional[str] = None; type_str: Optional[str] = None; value_str: Optional[str] = None
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")
        type_annot = assignment_node.child_by_field_name("type") # For PEP 526 var: type = value
        if left:
            if left.type == "identifier": 
                name_str = self._get_node_text(left, source_code)
                # If type annotation is directly on assignment (var: type = value), left is identifier, type is on assignment_node
                if type_annot: type_str = self._get_node_text(type_annot, source_code)
            elif left.type == "typed_identifier": # (var: type) = value
                name_cand = self._get_identifier_from_typed_node(left, source_code)
                type_cand = left.child_by_field_name("type") # type is on typed_identifier
                if name_cand: name_str = self._get_node_text(name_cand, source_code)
                if type_cand: type_str = self._get_node_text(type_cand, source_code)
        if right and right.type in ("integer", "string", "float", "true", "false", "none"):
            value_str = self._get_node_text(right, source_code)
            if right.type == "string": value_str = self._strip_string_quotes(value_str)
        if name_str: return Attribute(name=name_str, type=type_str, static=False, value=value_str)
        return None

    def _process_enum_member(self, assignment_node: Node, source_code: bytes) -> Optional[EnumMember]:
        name_str = None; value_str = None
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")
        if left and left.type == "identifier": name_str = self._get_node_text(left, source_code)
        if right:
            if right.type in ("integer", "string", "float", "true", "false", "none"):
                value_str = self._get_node_text(right, source_code)
                if right.type == "string": value_str = self._strip_string_quotes(value_str)
            elif right.type == "call": value_str = self._get_node_text(right, source_code)
        if name_str: return EnumMember(name=name_str, value=value_str)
        return None

    def _analyze_tree(self, tree: Tree, source_code: bytes, file_path: Path) -> FileAnalysis:
        root = tree.root_node
        logger.debug(f"Starting analysis for: {file_path}")
        analysis = FileAnalysis(file_path=str(file_path), analyzer="treesitter_python", imports=[], classes=[], functions=[], variables=[], enums=[])
        top_level_nodes = root.children if root and root.children else []
        for node in top_level_nodes:
            if node.type == "import_statement": analysis.imports.extend(self._process_import(node, source_code))
            elif node.type == "import_from_statement":
                processed = self._process_import_from(node, source_code)
                if processed: analysis.imports.extend(processed)
            elif node.type == "class_definition":
                obj = self._process_class(node, source_code)
                if isinstance(obj, Class): analysis.classes.append(obj)
                elif isinstance(obj, Enum): analysis.enums.append(obj)
            elif node.type == "function_definition": analysis.functions.append(self._process_function(node, source_code))
            elif node.type == "expression_statement":
                var = self._process_expression_statement_for_variable(node, source_code)
                if var: analysis.variables.append(var)
        if not analysis.imports: analysis.imports = None
        if not analysis.classes: analysis.classes = None
        if not analysis.functions: analysis.functions = None
        if not analysis.variables: analysis.variables = None
        if not analysis.enums: analysis.enums = None
        return analysis