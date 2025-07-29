# ReAct Agent Framework

A Python framework for building ReAct (Reasoning + Acting) agents that can use tools to solve complex problems through iterative reasoning and action.

## Features

- **Async-first architecture** - All tools and agents are fully asynchronous
- **Parallel tool execution** - Multiple tools can run concurrently
- **Task dispatch system** - Break complex problems into parallel sub-tasks
- **Decorator-based tool definition** - Clean, intuitive tool parameter specification
- **OpenAI-compatible** - Works with any OpenAI-compatible API service

## Quick Start

### 1. Installation

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 2. Basic Usage

```python
import asyncio
from react_agent import ReActAgent
from react_agent.tools import CalculatorTool, ShellTool

async def main():
    # Create agent
    agent = ReActAgent(
        api_key="your-api-key",
        model="gpt-4",
        verbose=True
    )
    
    # Bind tools
    agent.bind_tools([
        CalculatorTool(),
        ShellTool()
    ])
    
    # Run a query
    result = await agent.run("Calculate 15 * 27 and then list files in current directory")
    print(result)

# Run the agent
asyncio.run(main())
```

### 3. Task Dispatch for Parallel Processing

```python
from react_agent.tools import SubAgentDispatchTool

# Create specialized sub-agents
math_agent = ReActAgent(api_key="...", temperature=0.1)
math_agent.bind_tool(CalculatorTool())

system_agent = ReActAgent(api_key="...", model="gpt-3.5-turbo")
system_agent.bind_tool(ShellTool())

# Use dispatch tool
dispatch_tool = SubAgentDispatchTool()
dispatch_tool._current_params = {
    "tasks": [
        "Calculate the compound interest: 5000 * (1.05)^10",
        "Check available disk space"
    ],
    "sub_agents": [math_agent, system_agent],
    "return_format": "detailed"
}

result = await dispatch_tool._execute()
```

## Examples

### Test the Framework
```bash
python examples/test_imports.py
```

### Complex Equation Solver
```bash
python examples/equation_solver.py
```

This example demonstrates:
- Breaking complex equations into parallel sub-calculations
- Solving systems of linear equations using multiple methods
- Monte Carlo simulations with parallel processing

## Architecture

### Core Components

- **ReActAgent**: Main agent class with reasoning and tool execution
- **Tool**: Base class for all tools with async execution
- **SubAgentDispatchTool**: Special tool for parallel task dispatch

### Built-in Tools

- **CalculatorTool**: Mathematical expression evaluation
- **ShellTool**: Shell command execution with security features
- **ReadTool**: File reading with line numbers
- **SubAgentDispatchTool**: Parallel task dispatch to sub-agents

### Creating Custom Tools

```python
from react_agent import Tool, tool, param

@tool(name="custom", description="My custom tool")
class CustomTool(Tool):
    @param("input", type="string", description="Input parameter")
    async def _execute(self) -> str:
        input_value = self.get_param("input")
        # Your async logic here
        return f"Processed: {input_value}"
```

## Key Features

### Async Architecture
- All operations are asynchronous for better performance
- Non-blocking I/O operations
- Concurrent tool execution when multiple tools are called

### Task Dispatch System
- Break complex problems into parallel sub-tasks
- Use specialized agents for different types of work
- Significant performance improvements for concurrent workloads

### Security
- Shell command whitelist and dangerous pattern detection
- Parameter validation and type conversion
- Safe mathematical expression evaluation

## Requirements

- Python 3.8+
- OpenAI API key (or compatible service)
- Optional: `aiofiles` for async file operations

## License

MIT License