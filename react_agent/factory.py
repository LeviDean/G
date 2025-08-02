"""
Simple Agent Factory

Two types of agents:
1. Simple Agent - Local tools only (calculator, dispatch)
2. MCP Agent - Local tools + MCP servers for I/O
"""

import os
from typing import List, Dict, Any, Optional
from .core.agent import ReActAgent
from .tools import CalculatorTool, SubAgentDispatchTool


def create_agent(
    system_prompt: str,
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    **kwargs
) -> ReActAgent:
    """
    Create a ReAct agent.
    
    Args:
        system_prompt: System prompt for the agent
        mcp_servers: Optional MCP servers for I/O operations
        api_key: API key (or set OPENAI_API_KEY env var)
        model: Model to use
        **kwargs: Additional agent parameters
        
    Returns:
        Configured ReActAgent
    """
    agent = ReActAgent(
        system_prompt=system_prompt,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model=model,
        mcp_servers=mcp_servers or [],
        auto_connect_mcp=bool(mcp_servers),
        **kwargs
    )
    
    # Always bind essential local tools
    agent.bind_tools([
        CalculatorTool(),
        SubAgentDispatchTool()
    ])
    
    return agent


# Convenience aliases
def create_simple_agent(system_prompt: str, **kwargs) -> ReActAgent:
    """Create agent with local tools only."""
    return create_agent(system_prompt, mcp_servers=None, **kwargs)


def create_mcp_agent(system_prompt: str, mcp_servers: List[Dict[str, Any]], **kwargs) -> ReActAgent:
    """Create agent with MCP servers for I/O."""
    return create_agent(system_prompt, mcp_servers=mcp_servers, **kwargs)


def create_interactive_agent(system_prompt: str, mcp_servers: Optional[List[Dict[str, Any]]] = None, **kwargs) -> ReActAgent:
    """
    Create an interactive agent with real-time status updates, ESC interruption, and tool permissions.
    
    Args:
        system_prompt: System prompt for the agent
        mcp_servers: Optional MCP servers for I/O operations
        **kwargs: Additional agent parameters
        
    Returns:
        Interactive ReActAgent with full real-time capabilities
    """
    return create_agent(
        system_prompt, 
        mcp_servers=mcp_servers, 
        interactive=True,
        **kwargs
    )