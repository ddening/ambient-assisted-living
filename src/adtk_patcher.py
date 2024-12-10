from pathlib import Path
import re
import compileall


def comment_line_in_file(file_path: Path, target_pattern: str) -> bool:
    """
    Comment out a line in a Python file that matches the given pattern.

    Args:
        file_path (Path): Path to the Python file to modify.
        target_pattern (str): A regular expression pattern to identify the target line.

    Returns:
        bool: True if a line was commented, False otherwise.
    """
    try:
        # Read the file content
        lines = file_path.read_text().splitlines()

        modified = False
        new_lines = []

        for line in lines:
            # Check if the line matches the target pattern and is not already commented
            if re.match(rf"^\s*{target_pattern}", line) and not line.strip().startswith("#"):
                new_lines.append("# " + line)  # Add a comment marker to the line
                modified = True
            else:
                new_lines.append(line)

        if modified:
            # Write the modified content back to the file
            file_path.write_text("\n".join(new_lines) + "\n")

        return modified
    except Exception as e:
        print(f"Error: {e}")
        return False


def recompile_python_module(file_path: Path) -> None:
    """
    Recompile the Python file to regenerate the .pyc file.

    Args:
        file_path (Path): Path to the Python file.
    """
    try:
        compileall.compile_file(str(file_path), force=True)
        print(f"Recompiled: {file_path}")
    except Exception as e:
        print(f"Error compiling {file_path}: {e}")


if __name__ == "__main__":
    file_to_patch = Path("/opt/conda/lib/python3.12/site-packages/adtk/visualization/_visualization.py")
    line_to_comment = 'plt.style.use\\("seaborn-whitegrid"\\)'

    if file_to_patch.exists():
        if comment_line_in_file(file_to_patch, line_to_comment):
            print(f"Commented out line matching: {line_to_comment}")
            recompile_python_module(file_to_patch)
        else:
            print(f"No line matching {line_to_comment} found or no changes needed.")
    else:
        print(f"File {file_to_patch} does not exist.")
