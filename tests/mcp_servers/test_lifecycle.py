import pytest

from mcp_servers.client import MCPClient
from mcp_servers.lifecycle import _MCP_CLIENTS, get_client, start_mcp_servers
from tests.mcp_servers.conftest import CIVIL_LAW_QUERY

pytestmark = pytest.mark.integration

@pytest.fixture(autouse=True)
def clean_registry():
    saved_modules = {name: c._server_module for name, c in _MCP_CLIENTS.items()}
    for client in list(_MCP_CLIENTS.values()):
        client.shutdown()
    _MCP_CLIENTS.clear()
    yield
    for client in list(_MCP_CLIENTS.values()):
        client.shutdown()
    _MCP_CLIENTS.clear()
    for name, module in saved_modules.items():
        c = MCPClient(module)
        c.start()
        _MCP_CLIENTS[name] = c


def test_start_populates_registry():
    start_mcp_servers()
    assert "legal_rag" in _MCP_CLIENTS
    assert "case_doc_rag" in _MCP_CLIENTS
    assert isinstance(_MCP_CLIENTS["legal_rag"], MCPClient)
    assert isinstance(_MCP_CLIENTS["case_doc_rag"], MCPClient)


def test_start_is_idempotent():
    start_mcp_servers()
    pid_legal = _MCP_CLIENTS["legal_rag"]._proc.pid
    pid_case = _MCP_CLIENTS["case_doc_rag"]._proc.pid
    start_mcp_servers()
    assert _MCP_CLIENTS["legal_rag"]._proc.pid == pid_legal
    assert _MCP_CLIENTS["case_doc_rag"]._proc.pid == pid_case


def test_get_client_before_start_raises():
    with pytest.raises(RuntimeError, match="not started"):
        get_client("legal_rag")


def test_get_client_unknown_name_raises():
    start_mcp_servers()
    with pytest.raises(KeyError):
        get_client("nonexistent")


def test_child_processes_alive_after_start():
    start_mcp_servers()
    assert _MCP_CLIENTS["legal_rag"]._proc.poll() is None
    assert _MCP_CLIENTS["case_doc_rag"]._proc.poll() is None


def test_get_client_returns_live_callable_client():
    start_mcp_servers()
    resp = get_client("legal_rag").call(
        "search_legal_corpus",
        query=CIVIL_LAW_QUERY,
        corpus="civil_law",
    )
    assert resp["answer"]
