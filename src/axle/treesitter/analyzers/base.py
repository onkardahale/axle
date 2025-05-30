"""Base class for Tree-sitter language analyzers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from tree_sitter import Parser as TreeSitterParser, Tree, Node

try:
    from tree_sitter_language_pack import get_parser as get_tslp_provider_parser
    _TSLP_PROVIDER_IMPORT_ERROR = None
except ImportError as e:
    get_tslp_provider_parser = None
    _TSLP_PROVIDER_IMPORT_ERROR = str(e)

from ..models import FileAnalysis, FailedAnalysis 
from ..exceptions import GrammarError, ParsingError

logger = logging.getLogger(__name__)

class BaseAnalyzer(ABC):
    """
    Base class for all language-specific analyzers,
    using the new 'tree-sitter-language-pack' for parser acquisition.
    """

    LANGUAGE_NAME: str
    FILE_EXTENSIONS: tuple[str, ...]
    parser: TreeSitterParser

    def __init__(self):
        """
        Initialize the analyzer with the appropriate language parser
        using 'tree-sitter-language-pack'.
        """
        if get_tslp_provider_parser is None:
            raise GrammarError(
                "The 'tree-sitter-language-pack' library could not be imported. "
                "Please ensure it's installed correctly. "
                f"Original import error: {_TSLP_PROVIDER_IMPORT_ERROR}"
            )

        try:
            self.parser = get_tslp_provider_parser(self.LANGUAGE_NAME)
            if self.parser is None or self.parser.language is None:
                raise GrammarError(
                    f"Failed to get a valid parser for '{self.LANGUAGE_NAME}' from tree-sitter-language-pack."
                )
            logger.debug(f"Successfully initialized parser for {self.LANGUAGE_NAME} using tree-sitter-language-pack.")
        except Exception as e:
            logger.error(
                f"Failed to initialize parser for {self.LANGUAGE_NAME} using tree-sitter-language-pack: {e}",
                exc_info=True
            )
            if isinstance(e, LookupError):
                 raise GrammarError(f"Language '{self.LANGUAGE_NAME}' not found or supported by tree-sitter-language-pack: {str(e)}")
            raise GrammarError(
                f"Failed to load '{self.LANGUAGE_NAME}' grammar using tree-sitter-language-pack: {str(e)}."
            )

    def analyze_file(self, file_path: Path) -> FileAnalysis | FailedAnalysis:
        """Analyze a source file and return its structure."""
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()

            tree = self.parser.parse(source_code)

            if not tree:
                # The descriptive reason will now go into FailedAnalysis.reason
                raise ParsingError(str(file_path), "Parser returned no tree.")

            if tree.root_node.has_error: # Corrected: has_error is an attribute
                error_nodes = [child for child in tree.root_node.children if child.type == 'ERROR' or child.is_missing]
                error_reason = "Source code contains syntax errors (root node has_error)." # Default
                if error_nodes:
                    first_error_node = error_nodes[0]
                    error_reason = f"Source code contains syntax errors. First error near line {first_error_node.start_point[0] + 1}, column {first_error_node.start_point[1] + 1} (type: {first_error_node.type})."
                # The descriptive reason will now go into FailedAnalysis.reason
                raise ParsingError(str(file_path), error_reason)

            return self._analyze_tree(tree, source_code, file_path)

        except FileNotFoundError:
            error_msg = "File not found."
            logger.warning(f"{error_msg} during analysis: {file_path}")
            return FailedAnalysis(
                file_path=str(file_path),
                analyzer=f"treesitter_{self.LANGUAGE_NAME.lower()}",
                reason=error_msg
            )
        except ParsingError as pe:
            descriptive_reason = pe.args[1] if len(pe.args) > 1 else "syntax error"
            logger.warning(f"Parsing error for {file_path} ({self.LANGUAGE_NAME}): {descriptive_reason}")
            return FailedAnalysis(
                file_path=str(file_path),
                analyzer=f"treesitter_{self.LANGUAGE_NAME.lower()}",
                reason=descriptive_reason
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error analyzing {file_path} with {self.LANGUAGE_NAME} analyzer: {e}", exc_info=True)
            return FailedAnalysis(
                file_path=str(file_path),
                analyzer=f"treesitter_{self.LANGUAGE_NAME.lower()}",
                reason=error_msg
            )

    @abstractmethod
    def _analyze_tree(self, tree: Tree, source_code: bytes, file_path: Path) -> FileAnalysis:
        """Analyze the syntax tree and return structured data."""
        pass

    def _get_node_text(self, node: Node, source_code: bytes) -> str:
        """Get the text content of a node from the source code (decoded as UTF-8)."""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')

    def _get_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a node if present."""
        return None 