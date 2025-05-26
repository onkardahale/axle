import os

def derive_scope(file_paths: list[str]) -> str:
    if not file_paths:
        return ""
    if len(file_paths) == 1:
        return os.path.basename(file_paths[0])
    # For multiple files, find the longest common parent directory
    common_prefix = os.path.commonpath(file_paths)
    if common_prefix == os.path.sep:
        return ""
    return os.path.basename(common_prefix)

def generate_mock_commit_message(diff_content: str, num_files_changed: int, scope: str = "") -> str:
    type_ = "feat"
    description = "Implement a new feature"
    if scope:
        type_ = f"{type_}({scope})"
    if num_files_changed >= 2:
        body = "This change introduces a new capability to the system.\n\nFurther details about the implementation."
        return f"{type_}: {description}\n\n{body}"
    else:
        return f"{type_}: {description}" 