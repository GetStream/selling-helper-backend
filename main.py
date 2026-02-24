"""Vision Agents realtime video backend using Gemini."""

from dotenv import load_dotenv

from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import gemini, getstream

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create a realtime video agent using Gemini Live."""
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Assistant", id="agent"),
        instructions="Describe what you see. Be concise.",
        llm=gemini.Realtime(fps=3),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the Stream call and send one initial prompt."""
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.simple_response("What do you see?")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
