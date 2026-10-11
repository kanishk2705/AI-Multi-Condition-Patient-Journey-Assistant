"""Static syntax check for the root Streamlit entry point."""
import ast
from pathlib import Path


def test_streamlit_entrypoint_parses_without_importing_ui_dependencies():
    source = Path("app.py").read_text(encoding="utf-8")
    ast.parse(source, filename="app.py")
