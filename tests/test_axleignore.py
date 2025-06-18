"""Test .axleignore functionality."""

import tempfile
import os
from pathlib import Path
import pytest

from src.axle.knowledge_base import KnowledgeBase


class TestAxleIgnore:
    """Test the .axleignore functionality."""
    
    def test_ignore_patterns_loading(self):
        """Test that ignore patterns are loaded correctly from .axleignore file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create a .axleignore file
            ignore_file = project_root / '.axleignore'
            ignore_content = """# Test ignore file
build/
*.log
temp*
# Comment line

dist/"""
            ignore_file.write_text(ignore_content, encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Check that patterns are loaded correctly
            expected_patterns = ['build/', '*.log', 'temp*', 'dist/']
            assert kb.ignore_patterns == expected_patterns
    
    def test_should_ignore_directory_patterns(self):
        """Test that directory patterns are matched correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create a .axleignore file with directory patterns
            ignore_file = project_root / '.axleignore'
            ignore_file.write_text("build/\ndist/\nnode_modules/", encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Test directory patterns
            assert kb._should_ignore_path(project_root / 'build')
            assert kb._should_ignore_path(project_root / 'build' / 'subdir')
            assert kb._should_ignore_path(project_root / 'dist')
            assert kb._should_ignore_path(project_root / 'src' / 'node_modules')
            assert not kb._should_ignore_path(project_root / 'src')
            assert not kb._should_ignore_path(project_root / 'buildsomething')
    
    def test_should_ignore_file_patterns(self):
        """Test that file patterns are matched correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create a .axleignore file with file patterns
            ignore_file = project_root / '.axleignore'
            ignore_file.write_text("*.log\n*.tmp\nconfig.json", encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Test file patterns
            assert kb._should_ignore_path(project_root / 'app.log')
            assert kb._should_ignore_path(project_root / 'src' / 'debug.log')
            assert kb._should_ignore_path(project_root / 'temp.tmp')
            assert kb._should_ignore_path(project_root / 'config.json')
            assert not kb._should_ignore_path(project_root / 'app.py')
            assert not kb._should_ignore_path(project_root / 'logging.py')
    
    def test_should_ignore_wildcard_patterns(self):
        """Test that wildcard patterns work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create a .axleignore file with wildcard patterns
            ignore_file = project_root / '.axleignore'
            ignore_file.write_text("temp*\n*cache*\ntest*", encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Test wildcard patterns
            assert kb._should_ignore_path(project_root / 'temp')
            assert kb._should_ignore_path(project_root / 'temporary')
            assert kb._should_ignore_path(project_root / 'temp.txt')
            assert kb._should_ignore_path(project_root / 'mycache')
            assert kb._should_ignore_path(project_root / 'cache_dir')
            assert kb._should_ignore_path(project_root / 'test_file.py')
            assert kb._should_ignore_path(project_root / 'testing')
            assert not kb._should_ignore_path(project_root / 'src')
            assert not kb._should_ignore_path(project_root / 'main.py')
    
    def test_no_axleignore_file(self):
        """Test that when no .axleignore file exists, no patterns are loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            kb = KnowledgeBase(project_root)
            
            # Check that no patterns are loaded
            assert kb.ignore_patterns == []
            assert not kb._should_ignore_path(project_root / 'any_file.py')
            assert not kb._should_ignore_path(project_root / 'any_dir')
    
    def test_empty_axleignore_file(self):
        """Test that an empty .axleignore file results in no patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create an empty .axleignore file
            ignore_file = project_root / '.axleignore'
            ignore_file.write_text("", encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Check that no patterns are loaded
            assert kb.ignore_patterns == []
            assert not kb._should_ignore_path(project_root / 'any_file.py')
    
    def test_axleignore_with_only_comments(self):
        """Test that .axleignore file with only comments results in no patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create a .axleignore file with only comments
            ignore_file = project_root / '.axleignore'
            ignore_file.write_text("# Comment 1\n# Comment 2\n", encoding='utf-8')
            
            kb = KnowledgeBase(project_root)
            
            # Check that no patterns are loaded
            assert kb.ignore_patterns == []
            assert not kb._should_ignore_path(project_root / 'any_file.py')
    
    def test_path_outside_project_root(self):
        """Test that paths outside project root are always ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / 'project'
            project_root.mkdir()
            
            kb = KnowledgeBase(project_root)
            
            # Test path outside project root
            outside_path = Path(temp_dir) / 'outside_file.py'
            assert kb._should_ignore_path(outside_path) 