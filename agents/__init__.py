from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

# Core, dependency-light agents (always available).
from .templates.explore_agent import Explore
from .templates.random_agent import Random

# Optional templates that pull heavy/extra deps (langgraph, openai, smolagents,
# etc.). Import defensively so the core agents work without those installed.
import logging as _logging
_log = _logging.getLogger(__name__)


def _opt_import(modpath, names):
    try:
        mod = __import__(modpath, globals(), locals(), names, 0)
        for n in names:
            globals()[n] = getattr(mod, n)
    except Exception as e:  # noqa: BLE001
        _log.debug("Optional agent template %s unavailable: %s", modpath, e)


_opt_import("agents.templates.langgraph_functional_agent", ["LangGraphFunc", "LangGraphTextOnly"])
_opt_import("agents.templates.langgraph_random_agent", ["LangGraphRandom"])
_opt_import("agents.templates.langgraph_thinking", ["LangGraphThinking"])
_opt_import("agents.templates.llm_agents", ["LLM", "FastLLM", "GuidedLLM", "ReasoningLLM"])
_opt_import("agents.templates.multimodal", ["MultiModalLLM"])
_opt_import("agents.templates.openclaw_agent", ["OpenClaw"])
_opt_import("agents.templates.reasoning_agent", ["ReasoningAgent"])
_opt_import("agents.templates.smolagents", ["SmolCodingAgent", "SmolVisionAgent"])

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if "ReasoningAgent" in globals():
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "Explore",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
    "OpenClaw",
]
