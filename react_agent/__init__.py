"""
ReAct Agent Framework

A Python framework for building ReAct (Reasoning + Acting) agents that can use tools
to solve complex problems through iterative reasoning and action.
"""

from .core.agent import ReActAgent
from .core.tool import Tool, tool, param
from .factory import create_agent, create_simple_agent, create_mcp_agent, create_interactive_agent

__version__ = "0.1.0"
__all__ = [
    "ReActAgent", "Tool", "tool", "param",
    "create_agent", "create_simple_agent", "create_mcp_agent", "create_interactive_agent"
]