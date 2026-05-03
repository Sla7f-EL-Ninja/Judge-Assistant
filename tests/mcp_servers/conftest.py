import os

import pytest

from mcp_servers.client import MCPClient
from mcp_servers.lifecycle import _MCP_CLIENTS, start_mcp_servers

SEEDED_CASE_ID = os.environ.get("TEST_CASE_ID", "test-case-001")
CIVIL_LAW_QUERY = "ما هي أركان عقد البيع في القانون المدني المصري؟"
OFF_TOPIC_QUERY = "ما هو سعر الدولار اليوم في السوق السوداء؟"


@pytest.fixture(scope="session")
def mcp_servers_started():
    _MCP_CLIENTS.clear()
    start_mcp_servers()
    yield
    for client in list(_MCP_CLIENTS.values()):
        client.shutdown()
    _MCP_CLIENTS.clear()


@pytest.fixture(scope="session")
def seeded_case_id():
    return SEEDED_CASE_ID


@pytest.fixture(scope="session")
def civil_law_query():
    return CIVIL_LAW_QUERY


@pytest.fixture(scope="session")
def off_topic_query():
    return OFF_TOPIC_QUERY


@pytest.fixture
def isolated_legal_client():
    client = MCPClient("mcp_servers.legal_rag_server")
    client.start()
    yield client
    client.shutdown()


@pytest.fixture
def isolated_case_doc_client():
    client = MCPClient("mcp_servers.case_doc_server")
    client.start()
    yield client
    client.shutdown()
