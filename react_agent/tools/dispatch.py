import asyncio
from typing import Optional
from ..core.tool import Tool, tool, param
from ..core.agent import ReActAgent


@tool(name="dispatch", description="Dispatch a task to another agent for specialized processing")
class SubAgentDispatchTool(Tool):
    """
    A general-purpose tool for dispatching tasks to other agents.
    Agents can use this adaptively to delegate specialized work.
    """
    
    @param("task", type="string", description="Task description to dispatch")
    @param("agent_prompt", type="string", description="System prompt for the sub-agent to specialize it for this task")
    @param("tools", type="array", description="List of tool names the sub-agent should have access to", required=False)
    async def _execute(self) -> str:
        """Dispatch a task to a dynamically created sub-agent."""
        task = self.get_param("task")
        agent_prompt = self.get_param("agent_prompt")
        tools = self.get_param("tools", [])
        
        if not task or not isinstance(task, str):
            return "Error: 'task' must be a non-empty string"
        
        if not agent_prompt or not isinstance(agent_prompt, str):
            return "Error: 'agent_prompt' must be a non-empty string"
        
        # Get agent configuration from the parent agent if available
        if not hasattr(self, 'agent') or not self.agent:
            return "Error: Dispatch tool must be bound to an agent"
        
        try:
            # Create sub-agent with same API config as main agent
            sub_agent = ReActAgent(
                system_prompt=agent_prompt,
                api_key=self.agent.api_key,
                base_url=getattr(self.agent.client, 'base_url', None) if hasattr(self.agent, 'client') else None,
                model=self.agent.model,
                temperature=self.agent.temperature,
                max_tokens=self.agent.max_tokens,
                verbose=False,
                debug=False
            )
            
            # Ensure main agent has workspace (create default if needed)
            self.agent._ensure_default_workspace()
            
            # Copy workspace if main agent has one
            if hasattr(self.agent, 'workspace') and self.agent.workspace:
                sub_agent.bind_workspace(str(self.agent.workspace.root_path))
            
            # Bind requested tools from main agent's available tools
            if tools:
                for tool_name in tools:
                    if tool_name in self.agent.tools and tool_name != "dispatch":  # Avoid recursion
                        tool_instance = self.agent.tools[tool_name]
                        tool_class = type(tool_instance)
                        sub_agent.bind_tool(tool_class())
            
            # Execute the task
            result = await asyncio.wait_for(
                sub_agent.run(task),
                timeout=60.0  # Default timeout
            )
            
            # Log successful dispatch
            self._log_info(f"Dispatched task successfully: {task[:50]}...")
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = "Task timed out after 60 seconds"
            self._log_error(f"Dispatch timeout: {task[:50]}...")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Error executing dispatched task: {str(e)}"
            self._log_error(f"Dispatch failed: {str(e)}")
            return f"Error: {error_msg}"
