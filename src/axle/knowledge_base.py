"""
Knowledge base module for axle.

This module handles the creation and management of the local knowledge base
that stores information about the project's codebase structure.
"""

import json
import os
import logging
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from .treesitter import TreeSitterParser
from .treesitter.exceptions import TreeSitterError

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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
        self.ignore_patterns = self._load_ignore_patterns()
        
    def _load_ignore_patterns(self) -> List[str]:
        """Load ignore patterns from .axleignore file.
        
        Returns:
            List of ignore patterns
        """
        ignore_file = self.project_root / '.axleignore'
        patterns = []
        
        if ignore_file.exists():
            try:
                with open(ignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except IOError as e:
                logger.warning(f"Failed to read .axleignore file: {e}")
        
        return patterns
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if a path should be ignored based on .axleignore patterns.
        
        Args:
            path: Path to check (can be file or directory)
            
        Returns:
            True if the path should be ignored, False otherwise
        """
        try:
            relative_path = path.relative_to(self.project_root)
        except ValueError:
            # Path is not relative to project root, ignore it
            return True
            
        path_str = str(relative_path)
        path_parts = relative_path.parts
        
        for pattern in self.ignore_patterns:
            # Check if pattern matches the full path
            if fnmatch.fnmatch(path_str, pattern):
                return True
            
            # Check if pattern matches any part of the path (for directory patterns)
            if pattern.endswith('/'):
                # Directory pattern - check if any parent directory matches
                dir_pattern = pattern.rstrip('/')
                for part in path_parts:
                    if fnmatch.fnmatch(part, dir_pattern):
                        return True
            else:
                # Check if any parent directory matches the pattern
                for part in path_parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True
                        
                # Check if filename matches the pattern
                if fnmatch.fnmatch(path.name, pattern):
                    return True
        
        return False
    
    def analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a source file and extract structural information.
        
        Args:
            file_path: Absolute path to the source file to analyze.
            
        Returns:
            Dictionary containing the extracted information with a 'path' key
            containing the relative path, or None if analysis fails.
        """
        try:
            relative_path_str = str(file_path.relative_to(self.project_root))
            # self.parser.analyze_file uses the extension_map internally to find the correct analyzer
            result = self.parser.analyze_file(file_path) 
            
            if result is None: # Should not happen if parser returns FailedAnalysis object on error
                logger.warning(f"Parser returned None for {file_path}")
                return None

            # result can be FileAnalysis or FailedAnalysis
            # model_dump will work for both Pydantic models
            analysis = result.model_dump(exclude_none=True)

            # If result is FailedAnalysis, it means parsing failed for a *supported* file type.
            if 'reason' in analysis and analysis.get('analyzer'): # Heuristic for FailedAnalysis
                 logger.warning(f"TreeSitter parsing failed for {file_path}: {analysis.get('reason')}")
                 return None 

            # --- Standardize Path (only for successful FileAnalysis) ---
            if 'file_path' in analysis: # Original key from FileAnalysis model
                del analysis['file_path']
            analysis['path'] = relative_path_str # Add relative path
            
            # --- Ensure Core Structural Keys Exist (only for successful FileAnalysis) ---
            analysis.setdefault('imports', [])
            analysis.setdefault('classes', [])
            analysis.setdefault('functions', [])
            
            # --- Category Determination (only for successful FileAnalysis) ---
            # file_path is absolute here for categorization logic
            # analysis['imports'] will be a list of dicts due to model_dump
            analysis['category'] = self._determine_file_category(file_path, analysis['imports']) 
           
            # --- Optimize Analysis for Token Efficiency ---
            analysis = self._optimize_analysis(analysis)
           
            return analysis
            
        except TreeSitterError as e: # This catches GrammarError or "Unsupported file type" from self.parser.analyze_file
            logger.warning(f"TreeSitter processing failed for {file_path}: {str(e)}")
            return None
        except Exception as e:
            # Log the actual exception type for better debugging
            logger.warning(f"Failed to analyze {file_path} due to {type(e).__name__}: {str(e)}", exc_info=True)
            return None
        
    def _determine_file_category(self, file_path: Path, imports: List[Dict[str, Any]]) -> str:
        """Determine the category of a file based on its path and imports.
        
        Args:
            file_path: Path to the file
            imports: List of import data (as dicts) in the file
            
        Returns:
            String representing the file category
        """
        path_str = str(file_path).lower() # Use lowercase for path matching
        
        # Check import-based heuristics first
        framework_indicators = {
            'django': 'web_framework',
            'flask': 'web_framework',
            'react': 'web_framework', 
            'angular': 'web_framework', 
            'vue': 'web_framework', 
            'express': 'web_framework', 
            'pytest': 'test',
            'unittest': 'test',
            'jest': 'test', 
            'mocha': 'test', 
            'sqlalchemy': 'database',
            'mongoose': 'database', 
            'sequelize': 'database', 
            'pandas': 'data_processing',
            'numpy': 'data_processing',
            'tensorflow': 'ml',
            'torch': 'ml',
            'sklearn': 'ml'
        }
        
        import_names = []
        if imports: 
            for imp_item in imports: 
                if isinstance(imp_item, dict):
                    # 'source' is often more reliable for library name than 'name' (which can be an alias or specific item)
                    source_module = imp_item.get('source', '')
                    if source_module:
                        source_parts = source_module.split('/')
                        base_name_candidate = source_parts[0]
                        if base_name_candidate and base_name_candidate not in ('.', '..'):
                             base_name_candidate = os.path.splitext(base_name_candidate)[0]
                             import_names.append(base_name_candidate.split('.')[0].lstrip('.'))

                    name_module = imp_item.get('name', '')
                    if name_module and not imp_item.get('items'): 
                        if name_module == source_module or not source_module: 
                            import_names.append(name_module.split('.')[0].lstrip('.'))

        for import_name in list(set(import_names)): # Deduplicate
            if not import_name: continue 
            for indicator, category in framework_indicators.items():
                if indicator in import_name.lower(): 
                    return category
                    
        # Check path-based heuristics
        if 'util' in path_str or 'helper' in path_str or 'lib' in path_str: return 'util'
        if 'test' in path_str or 'spec' in path_str: return 'test'
        if 'config' in path_str or 'conf' in path_str: return 'config'
        if 'setup' in path_str : return 'config' 
        if 'model' in path_str: return 'model'
        if 'schema' in path_str: return 'model' 
        if 'controller' in path_str or 'handler' in path_str or 'router' in path_str or 'route' in path_str: return 'controller'
        if 'service' in path_str: return 'service'
        if 'component' in path_str: return 'ui_component' 
        if 'view' in path_str: return 'ui_component' 
        if Path(file_path).name == '__init__.py': return 'package_init'
        if Path(file_path).name in ('main.js', 'index.js', 'app.js', 'server.js', 'cli.js', 'main.py', 'cli.py'): return 'entrypoint'
        return 'unknown'
    
    def _optimize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize analysis data to reduce token count while preserving information.
        
        Args:
            analysis: Original analysis dictionary
            
        Returns:
            Optimized analysis dictionary
        """
        optimized = {}
        
        # Always include essential fields
        optimized['analyzer'] = analysis.get('analyzer', '')
        optimized['path'] = analysis.get('path', '')
        
        # Optimize imports - remove redundant structure
        if analysis.get('imports'):
            optimized['imports'] = self._optimize_imports(analysis['imports'])
        
        # Optimize classes - remove empty calls arrays and self parameters
        if analysis.get('classes'):
            optimized['classes'] = self._optimize_classes(analysis['classes'])
        
        # Optimize functions
        if analysis.get('functions'):
            optimized['functions'] = self._optimize_functions(analysis['functions'])
        
        # Only include variables if they have meaningful content
        if analysis.get('variables'):
            opt_vars = self._optimize_variables(analysis['variables'])
            if opt_vars:
                optimized['variables'] = opt_vars
        
        # Only include category if it's not 'unknown'
        category = analysis.get('category')
        if category and category != 'unknown':
            optimized['category'] = category
            
        return optimized
    
    def _optimize_imports(self, imports: List[Dict[str, Any]]) -> List[Any]:
        """Optimize import representations."""
        optimized = []
        for imp in imports:
            # For simple imports where name equals source, just store the name
            if (imp.get('name') == imp.get('source') and 
                not imp.get('items')):
                optimized.append(imp['name'])
            else:
                # Keep structured format but remove redundancy
                opt_imp = {}
                if imp.get('name'):
                    opt_imp['name'] = imp['name']
                if imp.get('source') and imp['source'] != imp.get('name'):
                    opt_imp['source'] = imp['source']
                if imp.get('items'):
                    opt_imp['items'] = imp['items']
                optimized.append(opt_imp)
        return optimized
    
    def _optimize_classes(self, classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize class representations."""
        optimized = []
        for cls in classes:
            opt_class = {'name': cls['name']}
            
            # Preserve docstring completely
            if cls.get('docstring'):
                opt_class['docstring'] = cls['docstring']
            
            # Include bases if present
            if cls.get('bases'):
                opt_class['bases'] = cls['bases']
            
            # Optimize methods
            if cls.get('methods'):
                opt_methods = []
                for method in cls['methods']:
                    opt_method = self._optimize_method(method)
                    opt_methods.append(opt_method)
                opt_class['methods'] = opt_methods
            
            optimized.append(opt_class)
        return optimized
    
    def _optimize_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize function representations."""
        optimized = []
        for func in functions:
            opt_func = self._optimize_method(func)  # Same optimization as methods
            optimized.append(opt_func)
        return optimized
    
    def _optimize_method(self, method: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize method/function representation."""
        opt_method = {'name': method['name']}
        
        # Preserve docstring completely
        if method.get('docstring'):
            opt_method['docstring'] = method['docstring']
        
        # Optimize parameters - remove 'self' and empty parameter lists
        if method.get('parameters'):
            opt_params = []
            for param in method['parameters']:
                if param.get('name') != 'self':  # Skip self parameters
                    opt_params.append(param)
            if opt_params:  # Only include if non-empty after filtering
                opt_method['parameters'] = opt_params
        
        # Only include calls if non-empty
        if method.get('calls'):
            opt_method['calls'] = method['calls']
        
        return opt_method
    
    def _optimize_variables(self, variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize variable representations, only keeping meaningful ones."""
        optimized = []
        for var in variables:
            # Only keep variables that have values or non-standard kinds
            if (var.get('value') or 
                var.get('kind') != 'external_variable'):
                optimized.append(var)
        return optimized if optimized else None
    
    def build_knowledge_base(self) -> None:
        """Build the knowledge base by analyzing all supported files in the project."""
        skipped_files_log = []
        analyzed_files_count = 0
        
        supported_extensions = tuple(self.parser.extension_map.keys())
        if not supported_extensions:
            logger.warning("No supported file extensions found in TreeSitterParser. Knowledge base might be empty.")

        common_non_source_extensions = ('.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.json', '.md', '.txt', '.log', '.gz', '.zip', '.tar') 
        
        # Default directories to always ignore
        default_ignore_dirs = {'.git', '.hg', '.svn', '.vscode', 'node_modules', '__pycache__', '.axle'}

        for root, dirs, files in os.walk(self.project_root):
            # Filter out default ignore directories
            dirs[:] = [d for d in dirs if d not in default_ignore_dirs]
            
            # Filter out directories based on .axleignore patterns
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not self._should_ignore_path(root_path / d)]

            for file_name in files:
                file_path = Path(root) / file_name
                
                # Skip files that match .axleignore patterns
                if self._should_ignore_path(file_path):
                    skipped_files_log.append(f"{str(file_path)} (ignored by .axleignore)")
                    continue
                
                if file_name.lower().endswith(supported_extensions):
                    if '.axle' in file_path.parts:
                        continue
                        
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        rel_path = file_path.relative_to(self.project_root)
                        kb_file_path_parts = list(rel_path.parts)
                        kb_file_name = os.path.splitext(kb_file_path_parts[-1])[0] + '.json'
                        kb_file_path_parts[-1] = kb_file_name
                        
                        kb_file = self.kb_dir.joinpath(*kb_file_path_parts)
                        
                        kb_file.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with open(kb_file, 'w', encoding='utf-8') as f:
                                # Use compact JSON formatting to save tokens
                                json.dump(analysis, f, separators=(',', ':'), ensure_ascii=False)
                            analyzed_files_count += 1
                        except IOError as e:
                            logger.error(f"Failed to write knowledge base file {kb_file}: {e}")
                            skipped_files_log.append(f"{str(file_path)} (write error)")
                    else:
                        skipped_files_log.append(f"{str(file_path)} (analysis failed or returned None)")
                else:
                    if not file_name.lower().endswith(common_non_source_extensions) and not file_name.startswith('.'):
                        skipped_files_log.append(f"{str(file_path)} (unsupported extension: {file_path.suffix})")
        
        log_content = f"Knowledge base build summary:\n- Analyzed {analyzed_files_count} files.\n"
        if skipped_files_log:
            log_content += "\nSkipped or failed files/reasons:\n"
            for entry in skipped_files_log:
                log_content += f"- {entry}\n"
        else:
            log_content += "\nNo files were skipped or failed analysis (among potentially relevant files).\n"
            
        try:
            with open(self.kb_dir / 'init.log', 'w', encoding='utf-8') as f:
                f.write(log_content)
        except IOError as e:
             logger.error(f"Failed to write init.log: {e}")

        try:
            import subprocess
            git_dir = self.project_root / '.git'
            if git_dir.exists() and git_dir.is_dir(): 
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    check=False 
                )
                if result.returncode == 0:
                    with open(self.kb_dir / 'head_commit', 'w', encoding='utf-8') as f:
                        f.write(result.stdout.strip())
                else:
                    logger.warning(f"Failed to get HEAD commit hash: git rev-parse HEAD failed with code {result.returncode}. Stderr: {result.stderr.strip()}")
        except FileNotFoundError:
            logger.warning("Failed to store HEAD commit hash: 'git' command not found.")
        except Exception as e: 
            logger.warning(f"Failed to store HEAD commit hash due to an unexpected error: {str(e)}")

    def get_file_analysis(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Retrieve the analysis for a specific file.
        
        Args:
            file_path: Path to the file relative to project root
            
        Returns:
            Dictionary containing the file analysis, or None if not found
        """
        base_name_no_ext = os.path.splitext(file_path.name)[0]
        relative_dir_parts = list(file_path.parent.parts)
        json_file_name = base_name_no_ext + '.json'
        kb_file = self.kb_dir.joinpath(*relative_dir_parts, json_file_name)
        
        if kb_file.exists():
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error reading or parsing knowledge base file {kb_file}: {e}")
                return None
        return None
    
    def is_stale(self) -> bool:
        """Check if the knowledge base is stale.
        
        Returns:
            True if the knowledge base is stale, False otherwise
        """
        try:
            import subprocess
            git_dir = self.project_root / '.git'
            if not (git_dir.exists() and git_dir.is_dir()):
                return False 

            head_file = self.kb_dir / 'head_commit'
            if not head_file.exists():
                return True 

            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=False
            )
            if result.returncode != 0:
                logger.warning(f"is_stale check: 'git rev-parse HEAD' failed. Assuming stale. Error: {result.stderr.strip()}")
                return True 
            current_head = result.stdout.strip()

            with open(head_file, 'r', encoding='utf-8') as f:
                stored_head = f.read().strip()

            if current_head != stored_head:
                verify_stored_head_cmd = ['git', 'rev-parse', '--verify', f"{stored_head}^{{commit}}"]
                verify_result = subprocess.run(verify_stored_head_cmd, capture_output=True, text=True, cwd=self.project_root, check=False)

                if verify_result.returncode != 0:
                    logger.warning(f"is_stale check: Stored HEAD '{stored_head}' is not a valid commit. Assuming stale.")
                    return True 

                count_cmd = ['git', 'rev-list', '--count', f"{stored_head}..{current_head}"]
                result_count = subprocess.run(
                    count_cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    check=False
                )
                if result_count.returncode == 0:
                    try:
                        num_commits = int(result_count.stdout.strip())
                        return num_commits > 5 
                    except ValueError:
                        logger.warning(f"is_stale check: Could not parse commit count output: '{result_count.stdout.strip()}'. Assuming stale.")
                        return True
                else:
                    logger.warning(f"is_stale check: 'git rev-list --count' failed. Assuming stale. Error: {result_count.stderr.strip()}")
                    return True
            return False 
        except FileNotFoundError:
            logger.warning("is_stale check: 'git' command not found. Assuming not stale (cannot verify).")
            return False 
        except Exception as e:
            logger.warning(f"Failed to check knowledge base staleness due to an unexpected error: {str(e)}. Assuming stale.")
            return True