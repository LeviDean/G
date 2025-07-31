"""
ReAct Agent Framework

A Python framework for building ReAct (Reasoning + Acting) agents that can use tools
to solve complex problems through iterative reasoning and action.
"""

from .core.agent import ReActAgent
from .core.tool import Tool, tool, param
from .core.workspace import Workspace, TemporaryWorkspace, WorkspaceManager, temporary_workspace
from .core.desk import Desk

__version__ = "0.1.0"
__all__ = [
    "ReActAgent", "Tool", "tool", "param",
    "Workspace", "TemporaryWorkspace", "WorkspaceManager", "temporary_workspace",
    "Desk"
]