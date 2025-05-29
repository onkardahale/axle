"""
Knowledge base module for axle.

This module handles the creation and management of the local knowledge base
that stores information about the project's codebase structure.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .treesitter import TreeSitterParser
from .treesitter.exceptions import TreeSitterError

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Manages the local knowledge base for a project."""
    
    def __init__(self, project_root: Path):
        """Initialize the knowledge base manager.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.kb_dir = project_root / '.axle'
        self.kb_dir.mkdir(exist_ok=True)
        self.parser = TreeSitterParser()
        
    def analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a Python file and extract structural information.
        
        Args:
            file_path: Absolute path to the Python file to analyze.
            
        Returns:
            Dictionary containing the extracted information with a 'path' key
            containing the relative path, or None if analysis fails.
        """
        try:
            relative_path_str = str(file_path.relative_to(self.project_root))
            result = self.parser.analyze_file(file_path)
            
            if result is None:
                logger.warning(f"Parser returned None for {file_path}")
                return None

            analysis = result.model_dump(exclude_none=True)
            
            # --- Standardize Path ---
            if 'file_path' in analysis:
                del analysis['file_path']
            analysis['path'] = relative_path_str
            
            # --- Ensure Core Structural Keys Exist ---
            # If 'classes' is not in analysis (due to exclude_none=True or parser),
            # add it as an empty list. Same for functions and imports.
            analysis.setdefault('imports', [])
            analysis.setdefault('classes', [])
            analysis.setdefault('functions', [])
            
            # --- Category Determination ---
            # file_path is absolute here for categorization logic
            analysis['category'] = self._determine_file_category(file_path, analysis['imports']) 
           
            return analysis
            
        except TreeSitterError as e: 
            logger.warning(f"TreeSitter parsing failed for {file_path}: {str(e)}")
            return None
        except Exception as e:
            # Log the actual exception type for better debugging
            logger.warning(f"Failed to analyze {file_path} due to {type(e).__name__}: {str(e)}")
            return None
        
    def _determine_file_category(self, file_path: Path, imports: List[Dict[str, Any]]) -> str:
        """Determine the category of a file based on its path and imports.
        
        Args:
            file_path: Path to the file
            imports: List of imports in the file
            
        Returns:
            String representing the file category
        """
        path_str = str(file_path)
        
        # Check import-based heuristics first
        framework_indicators = {
            'django': 'web_framework',
            'flask': 'web_framework',
            'pytest': 'test',
            'unittest': 'test',
            'sqlalchemy': 'database',
            'pandas': 'data_processing',
            'numpy': 'data_processing',
            'tensorflow': 'ml',
            'torch': 'ml',
            'sklearn': 'ml'
        }
        
        # Extract base module names from imports
        import_names = []
        for imp_item in imports: # imp_item is a dict e.g. {'name': 'sys', 'source': 'sys'}
            if isinstance(imp_item, dict):
                name = imp_item.get('name', '')
                if name: # 'name' can be like 'pathlib' or '.git_utils'
                    import_names.append(name.split('.')[0].lstrip('.')) # Get base module, remove leading '.'
              
        for import_name in import_names:
            if not import_name: continue # Skip empty strings from relative imports like '.'
            for indicator, category in framework_indicators.items():
                if indicator in import_name.lower():
                    return category
                    
        # Check path-based heuristics (util/helper before test/spec)
        if 'util' in path_str.lower() or 'helper' in path_str.lower(): return 'util'
        if 'test' in path_str.lower() or 'spec' in path_str.lower(): return 'test'
        if 'controller' in path_str.lower(): return 'controller'
        if 'service' in path_str.lower(): return 'service'
        if Path(file_path).name == '__init__.py': return 'package_init'
        if Path(file_path).name == 'main.py' or Path(file_path).name == 'cli.py' : return 'entrypoint'
        return 'unknown'
    
    def build_knowledge_base(self) -> None:
        """Build the knowledge base by analyzing all Python files in the project."""
        skipped_files = []
        # Walk through all Python files in the project
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Skip files in the .axle directory
                    if '.axle' in file_path.parts:
                        continue
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        # Store the analysis in a JSON file
                        rel_path = file_path.relative_to(self.project_root)
                        kb_file = self.kb_dir / rel_path.with_suffix('.json')
                        kb_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(kb_file, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, indent=2)
                    else:
                        skipped_files.append(str(file_path))
                elif not file.endswith(('.py', '.pyc', '.pyo', '.pyd')):
                    # Log non-Python files
                    skipped_files.append(str(Path(root) / file))
        # Store the list of skipped files
        if skipped_files:
            with open(self.kb_dir / 'init.log', 'w', encoding='utf-8') as f:
                f.write("Skipped files:\n")
                for file in skipped_files:
                    f.write(f"- {file}\n")
        # Store the current HEAD commit hash
        try:
            import subprocess
            # Only try to get HEAD if .git exists
            if (self.project_root / '.git').exists():
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    with open(self.kb_dir / 'head_commit', 'w') as f:
                        f.write(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to store HEAD commit hash: {str(e)}")
    
    def get_file_analysis(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Retrieve the analysis for a specific file.
        
        Args:
            file_path: Path to the file relative to project root
            
        Returns:
            Dictionary containing the file analysis, or None if not found
        """
        # Accept both with and without .py extension
        kb_file = self.kb_dir / file_path.with_suffix('.json')
        if not kb_file.exists():
            # Try as-is (in case .json is already present)
            kb_file = self.kb_dir / file_path
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def is_stale(self) -> bool:
        """Check if the knowledge base is stale.
        
        Returns:
            True if the knowledge base is stale, False otherwise
        """
        try:
            import subprocess
            # Only check staleness if .git exists and head_commit exists
            if not (self.project_root / '.git').exists():
                return False
            head_file = self.kb_dir / 'head_commit'
            if not head_file.exists():
                return True
            # Get current HEAD
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode != 0:
                return True
            current_head = result.stdout.strip()
            with open(head_file, 'r') as f:
                stored_head = f.read().strip()
            # If HEADs don't match, check number of commits between them
            if current_head != stored_head:
                result = subprocess.run(
                    ['git', 'rev-list', '--count', f"{stored_head}..{current_head}"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    num_commits = int(result.stdout.strip())
                    # Consider stale if more than 2 commits have passed
                    return num_commits > 2
            return False
        except Exception as e:
            logger.warning(f"Failed to check knowledge base staleness: {str(e)}")
            return True 