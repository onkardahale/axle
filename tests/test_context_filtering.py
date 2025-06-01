from pathlib import Path
import sys
import os

# Add the project root to sys.path to allow importing src.axle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.axle.git_utils import get_changed_files_from_diff
# KnowledgeBase is not strictly needed for this test as we mock its behavior for main.py's logic part.

def run_test_simulation():
    print("Starting test simulation for context filtering...")

    sample_diff_content = """
diff --git a/file1.py b/file1.py
index 0000001..0000002 100644
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-old content
+new content
diff --git a/file2.txt b/file2.txt
index 0000003..0000004 100644
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1 @@
-some text
+other text
"""
    # This file is staged but has no textual changes mentioned in the sample_diff_content
    # (e.g. it might be a mode change, or just not part of this particular diff snippet)
    # get_changed_files_from_diff should only pick up file1.py and file2.txt from the diff.

    print(f"Sample diff content:\n{sample_diff_content}")

    # Simulate the part of main.py's commit function
    context = []
    unanalyzed_files = []

    # This is the list of files identified from the diff content
    changed_files_in_diff_output = get_changed_files_from_diff(sample_diff_content)
    print(f"Output from get_changed_files_from_diff: {changed_files_in_diff_output}")

    # This is the list of all files that are staged (simulating git_utils.get_staged_file_paths())
    all_staged_files = ["file1.py", "file2.txt", "file3_no_changes_in_diff.py"]
    print(f"All simulated staged files: {all_staged_files}")

    # Mock KnowledgeBase setup
    class MockKB:
        def __init__(self, project_root_path): # Renamed to avoid conflict
            self.analyses = {
                "file1.py": {"path": "file1.py", "category": "test1", "summary": "Analysis of file1.py"},
                "file2.txt": {"path": "file2.txt", "category": "test2", "summary": "Analysis of file2.txt"},
                "file3_no_changes_in_diff.py": {"path": "file3_no_changes_in_diff.py", "category": "test3", "summary": "Analysis of file3"},
            }
            self.project_root = project_root_path # Storing for potential use, though not directly in get_file_analysis

        def get_file_analysis(self, rel_path: Path):
            # The KB stores analyses by string paths relative to project root
            return self.analyses.get(str(rel_path))

    # Use a dummy project root, as KB initialization might expect it.
    # Actual file operations for KB are mocked.
    kb = MockKB(Path("."))

    print("\nSimulating context building loop:")
    for file_path_str in all_staged_files:
        # This is the core logic from main.py we are testing:
        if file_path_str not in changed_files_in_diff_output:
            print(f"Skipping '{file_path_str}' as it's not in changed_files_in_diff_output ({changed_files_in_diff_output})")
            continue

        print(f"Processing '{file_path_str}' as it IS in changed_files_in_diff_output.")
        rel_path = Path(file_path_str) # KB expects Path object
        analysis = kb.get_file_analysis(rel_path)
        if analysis:
            context.append(analysis)
        else:
            print(f"No analysis found for '{file_path_str}' (this shouldn't happen with current mock).")
            unanalyzed_files.append(file_path_str)

    print(f"\nFinal simulated context (paths): {[item['path'] for item in context]}")
    print(f"Unanalyzed files: {unanalyzed_files}")

    # Assertions
    context_paths = [item['path'] for item in context]
    assert "file1.py" in context_paths, "file1.py should be in context"
    assert "file2.txt" in context_paths, "file2.txt should be in context"
    assert "file3_no_changes_in_diff.py" not in context_paths, "file3_no_changes_in_diff.py should NOT be in context"

    print("\nTest script simulation completed successfully.")

if __name__ == "__main__":
    # Create dummy files for the KnowledgeBase to potentially "see" if it were a real one.
    # This is not strictly necessary for the mocked KB but good practice for more complex tests.
    # However, for this specific test, MockKB directly uses string keys.
    # Path("file1.py").touch()
    # Path("file2.txt").touch()
    # Path("file3_no_changes_in_diff.py").touch()

    run_test_simulation()

    # Clean up dummy files if created
    # if os.path.exists("file1.py"): os.remove("file1.py")
    # if os.path.exists("file2.txt"): os.remove("file2.txt")
    # if os.path.exists("file3_no_changes_in_diff.py"): os.remove("file3_no_changes_in_diff.py")
