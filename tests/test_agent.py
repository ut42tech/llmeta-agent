import json

import pytest

import agent as agent_module
from agent import PhaseTwoAssistant, connection_agent


class _DummyRoom:
    def __init__(self, name: str) -> None:
        self.name = name


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
        self.proc = type("Proc", (), {"userdata": {}})()
        self.log_context_fields: dict[str, str] = {}
        self.connected = False

    async def connect(self) -> None:
        self.connected = True


class _DummySession:
    def __init__(self) -> None:
        self.started = False
        self.kwargs: dict | None = None
        self.agent: PhaseTwoAssistant | None = None
        self.room: _DummyRoom | None = None

    async def start(self, *, agent, room):
        self.started = True
        self.agent = agent
        self.room = room


@pytest.mark.asyncio
async def test_connection_agent_configures_voice_pipeline(monkeypatch):
    ctx = _DummyContext(room_name="phase-two")

    dummy_session = _DummySession()
    monkeypatch.setattr(agent_module, "_create_session", lambda: dummy_session)

    await connection_agent(ctx)

    assert ctx.log_context_fields == {"room": "phase-two", "job_id": "job-123"}
    assert ctx.connected is True
    assert dummy_session.started is True
    assert isinstance(dummy_session.agent, PhaseTwoAssistant)
    assert dummy_session.room is ctx.room
    assert json.loads(ctx.agent.metadata or "{}") == {"agent": True, "phase": 2}


def test_create_session_uses_expected_models(monkeypatch):
    captured = {}

    class _FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "AgentSession", _FakeSession)
    monkeypatch.setattr(
        agent_module.inference,
        "STT",
        lambda **kwargs: {"kind": "stt", "config": kwargs},
    )
    monkeypatch.setattr(
        agent_module.inference,
        "LLM",
        lambda **kwargs: {"kind": "llm", "config": kwargs},
    )
    monkeypatch.setattr(
        agent_module.inference,
        "TTS",
        lambda **kwargs: {"kind": "tts", "config": kwargs},
    )

    class _FakeVAD:
        @staticmethod
        def load():
            return "vad"

    monkeypatch.setattr(agent_module.silero, "VAD", _FakeVAD)
    monkeypatch.setattr(agent_module, "MultilingualModel", lambda: "turn-detector")

    session = agent_module._create_session()

    assert isinstance(session, _FakeSession)
    assert captured["stt"] == {
        "kind": "stt",
        "config": {"model": "deepgram/nova-2", "language": "ja"},
    }
    assert captured["llm"] == {
        "kind": "llm",
        "config": {"model": "openai/gpt-4.1-mini"},
    }
    assert captured["tts"] == {
        "kind": "tts",
        "config": {
            "model": "cartesia/sonic-3",
            "voice": "59d4fd2f-f5eb-4410-8105-58db7661144f",
        },
    }
    assert captured["vad"] == "vad"
    assert captured["turn_detection"] == "turn-detector"
