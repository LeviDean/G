# ReAct Agent Framework

A clean, elegant Python framework for building ReAct (Reasoning + Acting) agents with real-time interaction.

## Quick Start

```bash
pip install openai aiohttp
```

```python
import asyncio
from react_agent import ReActAgent

async def main():
    agent = ReActAgent(
        system_prompt="You are a helpful assistant.",
        interactive=True  # Enable real-time features
    )
    response = await agent.execute_interactive("Calculate 15 * 23 + 100")
    print(response)

asyncio.run(main())
```

## Building Your Agent

The framework provides one `ReActAgent` class that you configure for your needs:

### Basic Agent
```python
from react_agent import ReActAgent

agent = ReActAgent(
    system_prompt="You are a Python coding expert.",
    api_key="your-key",
    model="gpt-4",
    temperature=0.1
)

response = await agent.execute("Write a sorting algorithm")
```

### Interactive Agent (Real-time streaming)
```python
agent = ReActAgent(
    system_prompt="You are a coding assistant.",
    interactive=True  # Enables streaming, permissions, color output
)

# Real-time streaming execution
response = await agent.execute_interactive("Create a web scraper")
```

### Agent with File Access
```python
agent = ReActAgent(
    system_prompt="You are a code assistant with file access.",
    interactive=True,
    mcp_servers=[{
        "name": "filesystem",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "/workspace"]
    }]
)
```

### Adding Custom Tools
```python
from react_agent.core.tool import Tool, tool, param

@tool(name="weather", description="Get weather information")
class WeatherTool(Tool):
    @param("city", type="string", description="City name")
    async def _execute(self) -> str:
        city = self.get_param("city")
        return f"Weather in {city}: Sunny, 75°F"

agent.bind_tool(WeatherTool())
```

## Features

- **⚡ Real-time Streaming** - See agent thinking live with color coding
- **🔐 Smart Permissions** - Interactive tool approval (once/always/deny)
- **🔧 Parallel Execution** - Multiple tools run concurrently
- **📁 File Operations** - Secure sandboxed access via MCP servers
- **🧮 Built-in Tools** - Calculator and agent dispatch included

## Interactive Experience

When `interactive=True`:

- 🟢 **Green** user input
- ⚫ **Gray** working status and tool calls  
- ⚪ **White** final agent responses
- 🔐 **Tool permissions** prompt on first use

## Examples & Demos

We provide examples showing different ways to use the framework:

```bash
export OPENAI_API_KEY="your_key"

# Code Agent Demo - interactive file operations
python examples/code_agent_demo.py

# Quick Start Examples - different configurations
python examples/quick_start.py

# Interactive Features Demo - streaming, permissions
python examples/interactive_demo.py
```

### Convenience Factory Functions

For common patterns, we provide factory functions:

```python
from react_agent import create_interactive_agent, create_simple_agent, create_mcp_agent

# These are shortcuts for common ReActAgent configurations:
agent = create_interactive_agent("prompt")  # interactive=True
agent = create_simple_agent("prompt")       # basic configuration  
agent = create_mcp_agent("prompt", servers) # with MCP servers
```

## Architecture

**Core Class:**
- `ReActAgent` - The main agent class you configure

**Built-in Tools:**
- `calculator` - Math operations
- `dispatch` - Delegate to specialized sub-agents

**External Tools (via MCP):**
- File operations, web search, databases, APIs
- Custom MCP servers

**Real-time System:**
- Streaming responses with live thinking
- Interactive tool permissions
- Parallel tool execution
- Clean color-coded output

## Configuration Options

```python
agent = ReActAgent(
    system_prompt="Your agent's role and personality",
    api_key="your-key",              # or set OPENAI_API_KEY
    model="gpt-4",                   # any OpenAI-compatible model  
    temperature=0.1,
    base_url="https://api.openai.com/v1",  # for other providers
    interactive=True,                # enables real-time features
    mcp_servers=[...],               # optional external tools
    debug=False                      # development mode
)
```

---

**Build your intelligent agent. Your way.**