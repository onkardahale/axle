class AxleError(Exception):
    """Base exception for all axle errors."""
    pass

class GitError(AxleError):
    """Raised for git-related errors."""
    pass

class KnowledgeBaseError(AxleError):
    """Raised for knowledge base-related errors."""
    pass

class AIError(AxleError):
    """Raised for AI-related errors."""
    pass 