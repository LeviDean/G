# ReAct Agent Framework

A Python framework for building ReAct (Reasoning + Acting) agents that can use tools to solve complex problems through iterative reasoning and action.

## Features

- 🤖 **ReAct Pattern**: Combines reasoning and acting for intelligent problem-solving
- 🛠️ **Tool Integration**: Easy-to-use decorator-based tool system
- 🔒 **Secure Workspaces**: Sandboxed file operations with path validation
- 🌊 **Streaming Support**: Real-time responses with progress updates
- 🏢 **Desk Interface**: Simple session management with conversation recording
- 📊 **Detailed Logging**: Complete interaction history and statistics
- 🔄 **Unified API**: Auto-detecting conversation state for seamless interactions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd react-agent

# Install dependencies (if any)
pip install openai aiohttp
```

### Basic Usage

```python
import asyncio
from react_agent import ReActAgent
from react_agent.tools import CalculatorTool

async def main():
    # Create agent
    agent = ReActAgent(
        system_prompt="You are a helpful assistant.",
        api_key="your-openai-api-key",
        model="gpt-4"
    )
    
    # Add tools
    agent.bind_tools([CalculatorTool()])
    
    # Simple conversation
    response = await agent.execute("Calculate 15 * 23 + 100")
    print(response)  # "The result is 445"
    
    # Follow-up (automatically continues conversation)
    response = await agent.execute("What's the square root of that?")
    print(response)  # "The square root of 445 is approximately 21.1"

asyncio.run(main())
```

### Streaming Responses

```python
async def streaming_example():
    # Real-time streaming responses
    async for update in agent.stream("Calculate complex math and explain"):
        if update["type"] == "thinking":
            print(update["content"], end="", flush=True)
        elif update["type"] == "tool_call":
            print(f"\n🔧 {update['content']}")
        elif update["type"] == "tool_result":
            print(f"✅ {update['content']}")

asyncio.run(streaming_example())
```

### Desk Interface (Session Management)

```python
from react_agent import Desk

async def desk_example():
    # Create desk with automatic session recording
    desk = Desk(agent)
    
    # Chat with streaming
    async for update in desk.chat_stream("Hello! Can you help with math?"):
        if update["type"] == "thinking":
            print(update["content"], end="")
    
    # Get session statistics
    stats = desk.get_session_stats()
    print(f"Total messages: {stats['total_messages']}")
    print(f"Agent used {stats['agent_details']['total_tool_calls']} tools")

asyncio.run(desk_example())
```

## Available Tools

- **CalculatorTool**: Mathematical calculations
- **ReadTool**: Read files from workspace
- **WriteTool**: Write files to workspace  
- **EditTool**: Edit files using diff-based operations
- **SubAgentDispatchTool**: Delegate tasks to specialized sub-agents
- **ShellTool**: Execute shell commands (use with caution)
- **APIClientTool**: Make HTTP requests to external APIs

## Creating Custom Tools

```python
from react_agent.core.tool import Tool, tool, param

@tool(name="weather", description="Get current weather for a location")
class WeatherTool(Tool):
    """A weather tool for getting current conditions."""
    
    @param("location", type="string", description="The city to get weather for")
    async def _execute(self) -> str:
        """Get current weather for a location."""
        location = self.get_param("location")
        
        # Your implementation here
        # e.g., call weather API
        
        return f"Weather in {location}: Sunny, 75°F"

# Use the tool
agent.bind_tools([WeatherTool()])
```

## Workspace Security

```python
# Secure workspace for file operations
agent.bind_workspace("./safe_directory")

# Tools automatically validate paths
await agent.execute("Create a file called data.txt with some content")
# ✅ Creates ./safe_directory/data.txt

await agent.execute("Read /etc/passwd") 
# ❌ Blocked - outside workspace
```

## Interactive Demo

```bash
python examples/desk_demo.py
```

Available commands:
- `/help` - Show available commands
- `/stats` - Session statistics  
- `/details` - Agent's detailed history
- `/export` - Export conversation
- `/reset` - Clear history
- `/quit` - Exit

## API Reference

### Agent Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `execute(message)` | ✨ **Recommended** - Auto-detects conversation state | General use |
| `stream(message)` | ✨ **Recommended** - Streaming with auto-detection | Real-time responses |
| `run(query)` | Forces new conversation | When you need fresh start |
| `chat(message)` | Forces continuation | When you need to continue |

### Desk Methods

| Method | Description |
|--------|-------------|
| `chat(message)` | Simple chat (non-streaming) |
| `chat_stream(message)` | Chat with streaming |
| `get_session_stats()` | Session and agent statistics |
| `get_agent_detailed_history()` | Complete interaction history |
| `export_conversation()` | Export as markdown/JSON |

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export OPENROUTE_CLAUDE_KEY="your-key"  # For OpenRouter
```

### Agent Configuration

```python
agent = ReActAgent(
    system_prompt="Custom system prompt",
    api_key="your-key",
    base_url="https://api.openai.com/v1",  # or OpenRouter
    model="gpt-4",
    temperature=0.1,
    max_tokens=4000,
    verbose=True,  # Debug output
    debug=True     # Detailed debugging
)
```
