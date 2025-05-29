"""Custom exceptions for Tree-sitter integration."""

class TreeSitterError(Exception):
    """Base exception for all Tree-sitter related errors."""
    pass

class GrammarError(TreeSitterError):
    """Raised when there are issues with Tree-sitter grammar loading or availability."""
    pass

class ParsingError(TreeSitterError):
    """Raised when Tree-sitter fails to parse a file due to syntax errors."""
    
    def __init__(self, file_path: str, error_message: str):
        self.file_path = file_path
        self.error_message = error_message
        super().__init__(f"Failed to parse {file_path}: {error_message}") 