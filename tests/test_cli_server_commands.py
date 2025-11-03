"""Tests for new llmring server CLI commands."""

from types import SimpleNamespace

import pytest

from llmring import cli


class DummyClient:
    """Simple stand-in for MCPHttpClient."""

    def __init__(self, *, responses):
        self._responses = responses
        self.calls = []
        self.closed = False

    async def get(self, path, params=None):
        self.calls.append((path, params))
        return self._responses.get(path)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_cmd_server_stats_json(monkeypatch, capsys):
    async def fake_env(**kwargs):
        return ("http://localhost:8000", "test-key")

    response = {
        "summary": {
            "total_requests": 2,
            "total_cost": 1.5,
            "total_tokens": 4000,
            "unique_models": 2,
            "unique_origins": 1,
        },
        "by_day": [],
        "by_model": {},
        "by_origin": {},
        "by_alias": {},
    }

    dummy = DummyClient(responses={"/api/v1/stats": response})
    monkeypatch.setattr(cli, "_ensure_server_env", fake_env)
    monkeypatch.setattr("llmring.mcp.http_client.MCPHttpClient", lambda **kwargs: dummy)

    args = SimpleNamespace(
        server=None,
        api_key=None,
        start_date=None,
        end_date=None,
        group_by="day",
        json=True,
    )

    rc = await cli.cmd_server_stats(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert '"total_requests": 2' in out
    assert dummy.closed
    assert dummy.calls and dummy.calls[0][0] == "/api/v1/stats"


@pytest.mark.asyncio
async def test_cmd_server_logs_table(monkeypatch, capsys):
    async def fake_env(**kwargs):
        return ("http://localhost:8000", "test-key")

    logs = [
        {
            "id": "1",
            "logged_at": "2025-01-01T12:00:00Z",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "alias": "fast",
            "profile": "default",
            "origin": "tests",
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_input_tokens": 0,
            "cost": 0.001,
            "metadata": {"foo": "bar"},
            "conversation_id": None,
            "id_at_origin": None,
        }
    ]

    dummy = DummyClient(responses={"/api/v1/logs": logs})
    monkeypatch.setattr(cli, "_ensure_server_env", fake_env)
    monkeypatch.setattr("llmring.mcp.http_client.MCPHttpClient", lambda **kwargs: dummy)

    args = SimpleNamespace(
        server=None,
        api_key=None,
        limit=10,
        offset=0,
        alias=None,
        model=None,
        origin=None,
        start_date=None,
        end_date=None,
        output="table",
        output_file=None,
        json=False,
    )

    rc = await cli.cmd_server_logs(args)
    assert rc == 0
    output = capsys.readouterr().out
    assert "Recent usage logs" in output
    assert "fast" in output
    assert dummy.closed


@pytest.mark.asyncio
async def test_cmd_server_conversations_detail(monkeypatch, capsys):
    async def fake_env(**kwargs):
        return ("http://localhost:8000", "test-key")

    conversation = {
        "id": "abc",
        "title": "Example",
        "model_alias": "balanced",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:01:00Z",
        "origin": "unit-test",
        "messages": [
            {
                "role": "user",
                "content": "hi",
                "input_tokens": 0,
                "output_tokens": 0,
            }
        ],
    }

    def factory(**kwargs):
        dummy = DummyClient(
            responses={
                "/conversations/abc": conversation,
                "/conversations/abc/messages": conversation["messages"],
            }
        )
        return dummy

    monkeypatch.setattr(cli, "_ensure_server_env", fake_env)
    monkeypatch.setattr("llmring.mcp.http_client.MCPHttpClient", factory)

    args = SimpleNamespace(
        server=None,
        api_key=None,
        limit=20,
        offset=0,
        conversation_id="abc",
        messages=True,
        json=False,
    )

    rc = await cli.cmd_server_conversations(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Conversation abc" in out
    assert "Example" in out
    assert "[user] hi" in out
