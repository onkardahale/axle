import subprocess

def is_git_installed():
    try:
        result = subprocess.run(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_staged_diff():
    try:
        result = subprocess.run(["git", "diff", "--staged"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError("git diff --staged failed")
        diff_output = result.stdout
        if not diff_output.strip():
            return None
        return diff_output
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not found in PATH.")


def get_staged_files_count():
    try:
        result = subprocess.run(["git", "diff", "--staged", "--name-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return 0
        files = [line for line in result.stdout.splitlines() if line.strip()]
        return len(files)
    except Exception:
        return 0


def get_staged_file_paths():
    try:
        result = subprocess.run(["git", "diff", "--staged", "--name-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return [] 