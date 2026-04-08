"""
test_no_sync_blocking.py -- Bug 4: No synchronous blocking in async code.

Uses AST analysis to detect synchronous I/O calls inside async functions.
Verifies that asyncio.to_thread is used for sync operations in the query
service.

Marker: regression
"""

import ast
import pathlib

import pytest


# Sync I/O patterns that should not appear inside async def functions
BLOCKED_CALLS = {
    "time.sleep",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.delete",
    "requests.patch",
    "requests.head",
}

# Built-in function names that indicate sync I/O
BLOCKED_NAMES = {
    "open",  # bare open() inside async
}

# Files to scan for sync blocking
TARGET_FILES = [
    "api/services/query_service.py",
    "api/routers/query.py",
    "api/routers/cases.py",
    "api/routers/documents.py",
    "api/routers/files.py",
    "api/routers/conversations.py",
    "api/routers/summaries.py",
]


class SyncBlockingVisitor(ast.NodeVisitor):
    """AST visitor that detects sync I/O calls inside async functions."""

    def __init__(self):
        self.violations = []
        self._in_async = False
        self._current_func = None

    def visit_AsyncFunctionDef(self, node):
        old_async = self._in_async
        old_func = self._current_func
        self._in_async = True
        self._current_func = node.name
        self.generic_visit(node)
        self._in_async = old_async
        self._current_func = old_func

    def visit_FunctionDef(self, node):
        old_async = self._in_async
        old_func = self._current_func
        self._in_async = False
        self._current_func = node.name
        self.generic_visit(node)
        self._in_async = old_async
        self._current_func = old_func

    def visit_Call(self, node):
        if not self._in_async:
            self.generic_visit(node)
            return

        call_name = self._get_call_name(node)
        if call_name in BLOCKED_CALLS:
            self.violations.append(
                f"Sync blocking call '{call_name}' in async function "
                f"'{self._current_func}' at line {node.lineno}"
            )

        # Check for bare function names
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
            self.violations.append(
                f"Sync blocking call '{node.func.id}()' in async function "
                f"'{self._current_func}' at line {node.lineno}"
            )

        self.generic_visit(node)

    def _get_call_name(self, node):
        """Extract dotted call name from a Call node."""
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ""


@pytest.mark.regression
class TestNoSyncBlocking:
    """Detect synchronous blocking calls inside async functions."""

    @pytest.mark.parametrize("filepath", TARGET_FILES)
    def test_no_sync_io_in_async_functions(self, filepath):
        """Source files must not use sync I/O inside async functions."""
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        path = project_root / filepath

        if not path.exists():
            pytest.skip(f"File {filepath} does not exist")

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath)

        visitor = SyncBlockingVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) == 0, (
            f"Found sync blocking calls in {filepath}:\n"
            + "\n".join(f"  - {v}" for v in visitor.violations)
        )

    def test_query_service_uses_asyncio_to_thread(self):
        """query_service.py should use asyncio.to_thread for sync graph execution."""
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        path = project_root / "api" / "services" / "query_service.py"

        if not path.exists():
            pytest.skip("query_service.py does not exist")

        source = path.read_text(encoding="utf-8")
        assert "asyncio.to_thread" in source or "to_thread" in source, (
            "query_service.py must use asyncio.to_thread to wrap synchronous "
            "graph execution in an async context"
        )
