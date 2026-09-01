"""
RAG-powered Disaster Intelligence Copilot.

Status: IMPLEMENTED ARCHITECTURE
Requires LLM provider (Ollama, etc.) for actual operation.
"""

from .agent import CopilotAgent

__all__ = ["CopilotAgent"]
