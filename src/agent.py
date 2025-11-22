import asyncio
import json
import logging

from dotenv import load_dotenv
from livekit.agents import AgentServer, JobContext, cli

logger = logging.getLogger("agent")

load_dotenv(".env.local")

server = AgentServer()


@server.rtc_session()
async def connection_agent(ctx: JobContext):
    """Phase 1 agent: connect to the room and expose presence."""
    shutdown_event = asyncio.Event()

    async def _on_shutdown(_: str = "") -> None:
        shutdown_event.set()

    ctx.add_shutdown_callback(_on_shutdown)

    room_name = getattr(ctx.job.room, "name", "unknown")
    ctx.log_context_fields = {"room": room_name, "job_id": ctx.job.id}

    await ctx.connect()

    agent_identity = getattr(ctx.agent, "identity", "unknown-agent")
    logger.info("Connected to room %s as %s", room_name, agent_identity)

    try:
        await ctx.agent.set_metadata(json.dumps({"agent": True}))
    except Exception as exc:
        logger.warning("Unable to set agent metadata: %s", exc)

    ctx.room.on("disconnected", lambda *_: shutdown_event.set())

    await shutdown_event.wait()
    logger.info("Disconnecting from room %s", room_name)


if __name__ == "__main__":
    cli.run_app(server)
