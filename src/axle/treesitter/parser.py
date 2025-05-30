"""Main Tree-sitter parser implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, Type, Optional

from .analyzers import BaseAnalyzer
from .analyzers.python_analyzer import PythonAnalyzer
from .analyzers.javascript_analyzer import JavaScriptAnalyzer
from .models import FileAnalysis, FailedAnalysis
from .exceptions import TreeSitterError, GrammarError

logger = logging.getLogger(__name__)

class TreeSitterParser:
    """Main parser class that coordinates language-specific analyzers."""
    
    def __init__(self):
        """Initialize the parser with available language analyzers."""
        self.analyzers: Dict[str, Type[BaseAnalyzer]] = {
            "python": PythonAnalyzer,
            "javascript": JavaScriptAnalyzer,
            # TODO: Add other language analyzers as they are implemented
            # "go": GoAnalyzer,
            # "cpp": CppAnalyzer,
        }
        
        # Map file extensions to analyzer classes
        self.extension_map: Dict[str, Type[BaseAnalyzer]] = {}
        for analyzer_class in self.analyzers.values():
            for ext in analyzer_class.FILE_EXTENSIONS:
                self.extension_map[ext] = analyzer_class
    
    def analyze_file(self, file_path: Path) -> FileAnalysis | FailedAnalysis:
        """Analyze a source file using the appropriate language analyzer.
        
        Args:
            file_path: Path to the source file to analyze.
            
        Returns:
            FileAnalysis or FailedAnalysis object containing the analysis results.
            
        Raises:
            TreeSitterError: If the file type is not supported or if there are
                issues with the Tree-sitter grammar.
        """
        if not file_path.exists():
            raise TreeSitterError(f"File not found: {file_path}")
        
        # Get the appropriate analyzer for this file
        analyzer_class = self.extension_map.get(file_path.suffix.lower())
        if not analyzer_class:
            raise TreeSitterError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            # Create analyzer instance and analyze the file
            analyzer = analyzer_class()
            return analyzer.analyze_file(file_path)
            
        except GrammarError as e:
            logger.error(f"Grammar error analyzing {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return FailedAnalysis(
                file_path=str(file_path),
                analyzer=f"treesitter_{analyzer_class.LANGUAGE_NAME.lower()}"
            )
    
    def analyze_directory(self, directory: Path, output_dir: Optional[Path] = None) -> None:
        """Analyze all supported source files in a directory.
        
        Args:
            directory: Path to the directory to analyze.
            output_dir: Optional directory to write JSON output files to.
                If not provided, will use `.axle/kb` in the analyzed directory.
        """
        if not directory.exists():
            raise TreeSitterError(f"Directory not found: {directory}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = directory / ".axle" / "kb"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all supported source files
        source_files = []
        for ext in self.extension_map:
            source_files.extend(directory.rglob(f"*{ext}"))
        
        # Analyze each file and write results
        for file_path in source_files:
            try:
                result = self.analyze_file(file_path)
                
                # Write analysis to JSON file
                output_file = output_dir / f"{file_path.name}.json"
                with open(output_file, 'w') as f:
                    json.dump(result.model_dump(exclude_none=True), f, indent=2)
                
                logger.info(f"Analyzed {file_path} -> {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                # Write failure record
                output_file = output_dir / f"{file_path.name}.json"
                with open(output_file, 'w') as f:
                    json.dump(FailedAnalysis(
                        file_path=str(file_path),
                        analyzer="unknown"  # We don't know which analyzer failed
                    ).model_dump(), f, indent=2) 