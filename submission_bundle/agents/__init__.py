"""Minimal agents package for the ARC-AGI-3 Kaggle submission.

Only the no-LLM agents needed for the explore2 submission are included:
explore (baseline) and explore2 (mask + effective-action ordering). The
langgraph/openai/smolagents templates are intentionally omitted so the bundle
installs and runs fully offline with no heavy dependencies.
"""
from typing import Type, cast

from .agent import Agent, Playback  # noqa: F401
from .templates.explore_agent import Explore
from .templates.explore2_agent import Explore2

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "explore": cast(Type[Agent], Explore),
    "explore2": cast(Type[Agent], Explore2),
}

__all__ = ["Agent", "Explore", "Explore2", "AVAILABLE_AGENTS"]
