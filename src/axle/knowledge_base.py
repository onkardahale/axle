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
    
    def build_knowledge_base(self) -> None:
        """Build the knowledge base by analyzing all supported files in the project."""
        skipped_files_log = []
        analyzed_files_count = 0
        
        supported_extensions = tuple(self.parser.extension_map.keys())
        if not supported_extensions:
            logger.warning("No supported file extensions found in TreeSitterParser. Knowledge base might be empty.")

        common_non_source_extensions = ('.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.json', '.md', '.txt', '.log', '.gz', '.zip', '.tar') 

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in ['.git', '.hg', '.svn', '.vscode', 'node_modules', '__pycache__', '.axle']]

            for file_name in files:
                file_path = Path(root) / file_name
                
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
                                json.dump(analysis, f, indent=2)
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