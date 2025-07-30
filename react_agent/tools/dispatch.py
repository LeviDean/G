import asyncio
import json
from typing import List, Dict, Any, Optional
from ..core.tool import Tool, tool, param
from ..core.agent import ReActAgent
from ..core.hierarchical_logger import create_child_logger


@tool(name="dispatch", description="Dispatch multiple tasks to sub-agents running in parallel")
class SubAgentDispatchTool(Tool):
    """
    A tool that can create sub-agents and dispatch multiple tasks to them in parallel.
    Each sub-agent can have its own set of tools and execute independently.
    """
    
    def __init__(self, 
                 main_agent: Optional[ReActAgent] = None,
                 max_concurrent_agents: int = 5,
                 sub_agent_timeout: float = 60.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_agent = main_agent
        self.max_concurrent_agents = max_concurrent_agents
        self.sub_agent_timeout = sub_agent_timeout
        self.calc_agents = []  # Can be set externally for pre-configured agents
    
    @param("tasks", type="array", description="List of task descriptions to dispatch to sub-agents")
    @param("sub_agents", type="array", description="Optional: List of pre-configured sub-agents to use. If not provided, creates from main_agent", required=False)
    @param("return_format", type="string", description="How to format results", enum=["detailed", "summary", "json", "results"], default="results", required=False)
    async def _execute(self) -> str:
        """Dispatch tasks to multiple sub-agents running in parallel."""
        tasks = self.get_param("tasks")
        sub_agents = self.get_param("sub_agents", [])
        return_format = self.get_param("return_format", "detailed")
        
        if not tasks or not isinstance(tasks, list):
            return "Error: 'tasks' must be a non-empty list of task descriptions"
        
        if len(tasks) > self.max_concurrent_agents:
            return f"Error: Too many tasks ({len(tasks)}). Maximum allowed: {self.max_concurrent_agents}"
        
        # Determine how to get sub-agents
        if sub_agents and isinstance(sub_agents, list) and len(sub_agents) > 0:
            # Use provided sub-agents
            if len(sub_agents) != len(tasks):
                return f"Error: Number of sub_agents ({len(sub_agents)}) must match number of tasks ({len(tasks)})"
            
            # Validate that all items are ReActAgent instances
            for i, agent in enumerate(sub_agents):
                if not isinstance(agent, ReActAgent):
                    return f"Error: sub_agents[{i}] is not a ReActAgent instance"
            
            agents_to_use = sub_agents
        elif self.calc_agents and len(self.calc_agents) > 0:
            # Use pre-configured calculation agents (cycling if needed)
            agents_to_use = []
            for i in range(len(tasks)):
                agent_index = i % len(self.calc_agents)
                base_agent = self.calc_agents[agent_index]
                
                # Create a copy with hierarchical logger if main agent has logging
                if self.main_agent and self.main_agent.logger:
                    # Create child logger for sub-agent
                    sub_logger = create_child_logger(self.main_agent.agent_name, base_agent.agent_name, True)
                    
                    # Create agent copy with child logger
                    agent_copy = ReActAgent(
                        system_prompt=base_agent.system_prompt,
                        api_key=base_agent.api_key,
                        base_url=getattr(base_agent.client, 'base_url', None) if hasattr(base_agent, 'client') else None,
                        model=base_agent.model,
                        temperature=base_agent.temperature,
                        max_tokens=base_agent.max_tokens,
                        verbose=False,
                        debug=False,
                        logger=sub_logger,
                        agent_name=base_agent.agent_name
                    )
                    
                    # Copy tools from base agent
                    if hasattr(base_agent, 'tools'):
                        for tool_name, tool_instance in base_agent.tools.items():
                            tool_class = type(tool_instance)
                            agent_copy.bind_tool(tool_class())
                    
                    agents_to_use.append(agent_copy)
                else:
                    # No logging, use original agent
                    agents_to_use.append(base_agent)
        else:
            # Create sub-agents from main agent template
            if not self.main_agent:
                return "Error: No sub_agents provided and no main_agent configured for creating sub-agents"
            
            agents_to_use = []
            for i in range(len(tasks)):
                # Create child logger if main agent has logging
                sub_logger = None
                agent_name = f"SubAgent_{i+1}"
                if self.main_agent.logger:
                    sub_logger = create_child_logger(self.main_agent.agent_name, agent_name, True)
                
                # Create sub-agent with same config as main agent
                sub_agent_config = {
                    "system_prompt": f"You are a task executor. Complete the given task efficiently and return the result clearly.",
                    "api_key": self.main_agent.api_key,
                    "base_url": getattr(self.main_agent.client, 'base_url', None),
                    "model": self.main_agent.model,
                    "temperature": self.main_agent.temperature,
                    "max_tokens": self.main_agent.max_tokens,
                    "verbose": False,  # Keep sub-agents quiet to avoid spam
                    "debug": False,
                    "logger": sub_logger,
                    "agent_name": agent_name
                }
                
                # Create sub-agent
                sub_agent = ReActAgent(**sub_agent_config)
                
                # Copy all tools from main agent except dispatch to avoid recursion
                for tool_name, tool_instance in self.main_agent.tools.items():
                    if tool_name != "dispatch":
                        tool_class = type(tool_instance)
                        sub_agent.bind_tool(tool_class())
                
                agents_to_use.append(sub_agent)
        
        # Prepare async tasks
        sub_agent_tasks = []
        for i, (task, sub_agent) in enumerate(zip(tasks, agents_to_use)):
            sub_agent_tasks.append(self._run_sub_agent(sub_agent, task, i))
        
        # Execute all sub-agents concurrently
        try:
            results = await asyncio.gather(*sub_agent_tasks, return_exceptions=True)
        except Exception as e:
            return f"Error during parallel execution: {str(e)}"
        
        # Format and return results
        return self._format_results(tasks, results, return_format)
    
    async def _run_sub_agent(self, sub_agent: ReActAgent, task: str, task_index: int) -> Dict[str, Any]:
        """Run a single sub-agent with timeout and error handling."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Run the task with timeout
            result = await asyncio.wait_for(
                sub_agent.run(task),
                timeout=self.sub_agent_timeout
            )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            return {
                "task_index": task_index,
                "task": task,
                "status": "success",
                "result": result,
                "execution_time": round(execution_time, 2),
                "token_usage": sub_agent.get_token_usage(),
                "tools_used": list(sub_agent.tools.keys()) if sub_agent.tools else []
            }
            
        except asyncio.TimeoutError:
            return {
                "task_index": task_index,
                "task": task,
                "status": "timeout",
                "result": f"Task timed out after {self.sub_agent_timeout} seconds",
                "execution_time": self.sub_agent_timeout,
                "token_usage": 0,
                "tools_used": []
            }
        except Exception as e:
            return {
                "task_index": task_index,
                "task": task,
                "status": "error",
                "result": f"Error: {str(e)}",
                "execution_time": 0,
                "token_usage": 0,
                "tools_used": []
            }
    
    def _format_results(self, tasks: List[str], results: List[Any], return_format: str) -> str:
        """Format the results according to the specified format."""
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "task_index": i,
                    "task": tasks[i] if i < len(tasks) else f"Task {i}",
                    "status": "error",
                    "result": f"Exception: {str(result)}",
                    "execution_time": 0,
                    "token_usage": 0,
                    "tools_used": []
                })
            else:
                processed_results.append(result)
        
        if return_format == "json":
            return json.dumps(processed_results, indent=2)
        elif return_format == "summary":
            return self._format_summary(processed_results)
        elif return_format == "detailed":
            return self._format_detailed(processed_results)
        else:  # results (default)
            return self._format_results_only(processed_results, tasks)
    
    def _format_summary(self, results: List[Dict[str, Any]]) -> str:
        """Format results as a summary."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        total_time = sum(r["execution_time"] for r in results)
        total_tokens = sum(r["token_usage"] for r in results)
        
        summary = f"=== Sub-Agent Dispatch Summary ===\n"
        summary += f"Total Tasks: {len(results)}\n"
        summary += f"Successful: {len(successful)}\n"
        summary += f"Failed: {len(failed)}\n"
        summary += f"Total Execution Time: {total_time:.2f}s\n"
        summary += f"Total Token Usage: {total_tokens}\n\n"
        
        if successful:
            summary += "✅ Successful Results:\n"
            for result in successful:
                summary += f"  Task {result['task_index'] + 1}: {result['result'][:100]}{'...' if len(result['result']) > 100 else ''}\n"
        
        if failed:
            summary += "\n❌ Failed Tasks:\n"
            for result in failed:
                summary += f"  Task {result['task_index'] + 1}: {result['result']}\n"
        
        return summary
    
    def _format_results_only(self, results: List[Dict[str, Any]], tasks: List[str]) -> str:
        """Format results optimized for agent consumption - just the task-result pairs."""
        output = ""
        
        for i, result in enumerate(results):
            task_desc = tasks[i] if i < len(tasks) else f"Task {i+1}"
            
            if result["status"] == "success":
                # Extract numerical value from verbose result
                numerical_result = self._extract_numerical_value(result['result'])
                output += f"{task_desc}: {numerical_result}\n"
            else:
                output += f"{task_desc}: Error - {result['result']}\n"
        
        return output.strip()
    
    def _extract_numerical_value(self, result_text: str) -> str:
        """Extract the numerical value from verbose calculation results."""
        import re
        
        # Look for patterns like "**123**", "result is 123", "equals 123", etc.
        patterns = [
            r'\*\*([0-9.+-]+)\*\*',  # **123**
            r'result is[:\s]*([0-9.+-]+)',  # result is: 123
            r'equals?[:\s]*([0-9.+-]+)',  # equals 123
            r'answer is[:\s]*([0-9.+-]+)',  # answer is 123
            r'([0-9.+-]+)$',  # Just a number at the end
            r'^([0-9.+-]+)',  # Just a number at the start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, return the original result
        return result_text
    
    def _format_detailed(self, results: List[Dict[str, Any]]) -> str:
        """Format results with full details."""
        output = "=== Sub-Agent Dispatch Results ===\n\n"
        
        for result in results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            output += f"{status_emoji} Task {result['task_index'] + 1}:\n"
            output += f"   Query: {result['task']}\n"
            output += f"   Status: {result['status']}\n"
            output += f"   Execution Time: {result['execution_time']}s\n"
            output += f"   Token Usage: {result['token_usage']}\n"
            output += f"   Tools Available: {', '.join(result['tools_used'])}\n"
            output += f"   Result: {result['result']}\n"
            output += "\n" + "-"*50 + "\n\n"
        
        # Add summary statistics
        successful = [r for r in results if r["status"] == "success"]
        total_time = sum(r["execution_time"] for r in results)
        total_tokens = sum(r["token_usage"] for r in results)
        
        output += f"Summary: {len(successful)}/{len(results)} tasks successful\n"
        output += f"Total execution time: {total_time:.2f}s\n"
        output += f"Total token usage: {total_tokens}\n"
        
        return output