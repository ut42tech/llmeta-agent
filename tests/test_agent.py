import asyncio
import json

import pytest

from agent import connection_agent


class _DummyRoom:
    def __init__(self, name: str) -> None:
        self.name = name
        self._handlers: dict[str, list] = {}

    def on(self, event: str, handler):
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str) -> None:
        for handler in self._handlers.get(event, []):
            handler()


class _DummyAgent:
    def __init__(self) -> None:
        self.identity = "agent-test"
        self.metadata: str | None = None

    async def set_metadata(self, metadata: str) -> None:
        self.metadata = metadata


class _DummyJobRoom:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyJob:
    def __init__(self, room_name: str) -> None:
        self.id = "job-123"
        self.room = _DummyJobRoom(room_name)


class _DummyContext:
    def __init__(self, room_name: str) -> None:
        self.room = _DummyRoom(room_name)
        self.job = _DummyJob(room_name)
        self.agent = _DummyAgent()
        self.log_context_fields: dict[str, str] = {}
        self.connected = False
        self._shutdown_callbacks = []

    async def connect(self) -> None:
        self.connected = True

    def add_shutdown_callback(self, callback):
        self._shutdown_callbacks.append(callback)

    async def trigger_shutdown(self, reason: str = "test") -> None:
        await asyncio.gather(
            *(callback(reason) for callback in self._shutdown_callbacks)
        )


@pytest.mark.asyncio
async def test_connection_agent_runs_until_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = _DummyContext(room_name="test-room")

    with caplog.at_level("INFO", logger="agent"):
        task = asyncio.create_task(connection_agent(ctx))

        await asyncio.sleep(0)
        assert ctx.connected is True
        assert ctx.log_context_fields == {"room": "test-room", "job_id": "job-123"}
        assert ctx.agent.metadata == json.dumps({"agent": True})

        await ctx.trigger_shutdown()
        await asyncio.wait_for(task, timeout=0.1)

    assert "Disconnecting from room test-room" in caplog.text
