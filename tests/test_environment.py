"""Phase 0 Environment and Scaffolding Verification Test.

Checks that the runtime environment meets the architectural prerequisites:
- Python version is 3.11 or newer
- The 'src' package can be imported correctly
- Core standard library components for the pipeline are operational
"""

import sys
from pathlib import Path


def test_python_version():
    """Verify runtime Python meets the minimum requirement (>= 3.11)."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert (major, minor) >= (3, 11), f"Python 3.11+ required, detected: {major}.{minor}"


def test_project_structure():
    """Verify required top-level directories and configuration files exist."""
    root = Path(__file__).resolve().parent.parent
    expected_dirs = ["src", "tests", "sample_docs", "docs", "evaluation"]
    for dir_name in expected_dirs:
        assert (root / dir_name).is_dir(), f"Expected directory '{dir_name}' not found at {root}"

    expected_files = [".gitignore", ".env.example", "requirements.txt", "pyproject.toml", "README.md"]
    for file_name in expected_files:
        assert (root / file_name).is_file(), f"Expected file '{file_name}' not found at {root}"


def test_package_import():
    """Verify hybrid_rag package exposes expected version attribute."""
    # Ensure src is on sys.path
    root = Path(__file__).resolve().parent.parent
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import hybrid_rag
    assert hasattr(hybrid_rag, "__version__"), "hybrid_rag package must define __version__"
    assert hybrid_rag.__version__ == "0.1.0"


if __name__ == "__main__":
    test_python_version()
    test_project_structure()
    test_package_import()
    print("All Phase 0 environment tests passed successfully.")
