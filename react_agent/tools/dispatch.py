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
    
    def __init__(self, **kwargs):
        # Sub-agent tasks can take longer, use extended timeout
        super().__init__(timeout=120.0, **kwargs)
    
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
            # Extract configuration safely from parent agent
            api_key = getattr(self.agent, 'api_key', None)
            model = getattr(self.agent, 'model', 'gpt-4')
            temperature = getattr(self.agent, 'temperature', 0.1)
            max_tokens = getattr(self.agent, 'max_tokens', None)
            
            # Extract base_url safely
            base_url = None
            if hasattr(self.agent, 'client') and hasattr(self.agent.client, 'base_url'):
                base_url = self.agent.client.base_url
            
            # Create sub-agent with extracted config
            sub_agent = ReActAgent(
                system_prompt=agent_prompt,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                debug=False,
                interactive=False  # Sub-agents should not be interactive
            )
            
            # Add basic calculator tool to sub-agent (most common need)
            from ..tools.calculator import CalculatorTool
            sub_agent.bind_tool(CalculatorTool())
            
            # Bind requested tools from main agent's available tools
            if tools:
                for tool_name in tools:
                    if tool_name in self.agent.tools and tool_name != "dispatch":  # Avoid recursion
                        tool_instance = self.agent.tools[tool_name]
                        tool_class = type(tool_instance)
                        sub_agent.bind_tool(tool_class())
            
            # Execute the task using the standard execute method
            result = await asyncio.wait_for(
                sub_agent.execute(task),
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
