"""Tree-sitter language analyzers using the new tree-sitter-language-pack."""

import logging
from pathlib import Path
from typing import Optional

from tree_sitter import Parser as TreeSitterParser, Tree, Node

try:
    from tree_sitter_language_pack import get_parser as get_tslp_provider_parser
    _TSLP_PROVIDER_IMPORT_ERROR = None
except ImportError as e:
    get_tslp_provider_parser = None
    _TSLP_PROVIDER_IMPORT_ERROR = str(e) # Store error for rich exception in __init__

from ..models import FileAnalysis, FailedAnalysis 
from ..exceptions import GrammarError, ParsingError

from .base import BaseAnalyzer
from .python_analyzer import PythonAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer
from .cpp_analyzer import CppAnalyzer

__all__ = ['BaseAnalyzer', 'PythonAnalyzer', 'JavaScriptAnalyzer', 'CppAnalyzer']

logger = logging.getLogger(__name__)
