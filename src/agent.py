import json
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RoomInputOptions,
    cli,
    inference,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

server = AgentServer()


class PhaseTwoAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "あなたはテスト用のアシスタントです。"
                "落ち着いた口調で、ユーザーが話しかけた内容に短く反応してください。"
                "専門用語が出てきた場合は噛み砕いて説明してください。"
            ),
            allow_interruptions=True,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="接続が完了したら簡単な自己紹介をしてからユーザーの発話を待ってください。"
        )


def _create_session() -> AgentSession:
    return AgentSession(
        stt=inference.STT(model="deepgram/nova-2", language="ja"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="59d4fd2f-f5eb-4410-8105-58db7661144f",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )


@server.rtc_session()
async def connection_agent(ctx: JobContext):
    room_name = getattr(ctx.room, "name", "unknown")
    ctx.log_context_fields = {"room": room_name, "job_id": ctx.job.id}

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent_identity = getattr(ctx.agent, "identity", "unknown-agent")
    logger.info("Starting Phase 2 pipeline in room %s as %s", room_name, agent_identity)

    try:
        await ctx.agent.set_metadata(json.dumps({"agent": True, "phase": 2}))
    except Exception as exc:
        logger.warning("Unable to set agent metadata: %s", exc)

    session = _create_session()
    agent = PhaseTwoAssistant()

    room_input_options = RoomInputOptions(
        audio_enabled=True,
        close_on_disconnect=False,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_options,
    )


if __name__ == "__main__":
    cli.run_app(server)
