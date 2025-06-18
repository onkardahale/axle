"""Test package for Axle."""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class BaseAxleTestCase(unittest.TestCase):
    """Base test case class with common teardown functionality."""
    
    def setUp(self):
        """Set up test environment with config backup."""
        super().setUp()
        
        # Create temporary directory for backups and test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dirs = []
        
        # Get the real config path
        from axle.ai_utils import get_cache_dir
        self.real_cache_dir = get_cache_dir()
        self.config_path = self.real_cache_dir / "model_config.json"
        
        # Create backup of original config if it exists
        self.config_backup_path = self.temp_dir / "original_model_config.json"
        self.had_original_config = False
        
        if self.config_path.exists():
            # Copy original config to backup
            shutil.copy2(self.config_path, self.config_backup_path)
            self.had_original_config = True
        
    def tearDown(self):
        """Clean up and restore original configuration."""
        try:
            # Restore original config
            if self.had_original_config and self.config_backup_path.exists():
                # Restore from backup
                shutil.copy2(self.config_backup_path, self.config_path)
            elif not self.had_original_config and self.config_path.exists():
                # If there was no original config, remove the file
                self.config_path.unlink()
                
            # Clean up temporary directory (includes backup)
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                
            # Clean up any registered test data directories
            for test_dir in self.test_data_dirs:
                if test_dir.exists():
                    shutil.rmtree(test_dir)
                    
        except Exception as e:
            # Log the error but don't fail the test
            print(f"Warning: Failed to clean up test resources: {e}")
        finally:
            super().tearDown()
                
    def register_test_data_dir(self, test_dir: Path):
        """Register a test data directory for cleanup."""
        if test_dir not in self.test_data_dirs:
            self.test_data_dirs.append(test_dir)
            
    def get_test_config(self):
        """Get a test configuration."""
        return {
            "model_name": "test-model-for-tests",
            "temperature": 0.3,
            "top_p": 0.9,
            "num_return_sequences": 1
        } 