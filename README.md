# ReAct Agent Framework

A Python framework for building ReAct (Reasoning + Acting) agents that can use tools to solve complex problems through iterative reasoning and action.

## Features

- 🤖 **ReAct Pattern**: Combines reasoning and acting for intelligent problem-solving
- 🛠️ **Tool Integration**: Easy-to-use decorator-based tool system
- 🔌 **MCP Integration**: Connect to external MCP (Model Context Protocol) servers
- 🌐 **Unified Tool Ecosystem**: Seamlessly use local and remote tools together
- 🔒 **Secure File Operations**: Sandboxed via MCP filesystem servers
- 🌊 **Streaming Support**: Real-time responses with progress updates
- 📊 **Detailed Logging**: Complete interaction history and statistics
- 🔄 **Unified API**: Auto-detecting conversation state for seamless interactions
- 🎮 **Interactive Demo**: Code Agent interface with colored output

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd react-agent

# Install Python dependencies
pip install openai aiohttp

# Install MCP support (optional - for file operations)
pip install mcp
npm install -g @modelcontextprotocol/server-filesystem
```

### Quick Start

```python
import asyncio
from react_agent import create_simple_agent, create_mcp_agent

async def main():
    # Simple agent with calculator
    simple_agent = create_simple_agent("You are a helpful calculator assistant.")
    response = await simple_agent.execute("Calculate 15 * 23 + 100")
    print(response)
    
    # MCP agent with file operations
    mcp_agent = create_mcp_agent(
        "You can do calculations and file operations.",
        mcp_servers=[{
            "name": "filesystem", 
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/workspace"]
        }]
    )
    
    response = await mcp_agent.execute("Calculate 25*4 and save result to calc.txt")
    print(response)

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

### Interactive Code Agent Demo

Experience a Claude Code-like interface with the ReAct Agent:

```bash
# Install MCP filesystem server (required for file operations)
npm install -g @modelcontextprotocol/server-filesystem

# Set your API key
export OPENROUTE_CLAUDE_KEY="your_key"  # or OPENAI_API_KEY

# Run the interactive demo (MCP server starts automatically)
python examples/code_agent_demo.py
```

**Features:**
- 🟢 **Colored output**: Green user input, yellow agent responses
- 📁 **File operations**: Create, read, edit, list files in sandboxed workspace
- 🧮 **Calculations**: Built-in calculator for math operations
- 🔧 **Commands**: `/help`, `/ls`, `/clear`, `/quit`
- 🌊 **Real-time streaming**: See the agent think in real-time

**Example commands:**
- "Create a Python script that plots a cosine curve"
- "List all files in the workspace"
- "Calculate the factorial of 10"
- "Debug this Python error: NameError"

### MCP (Model Context Protocol) Integration

Connect to external MCP servers for enhanced functionality:

```python
async def mcp_example():
    # Configure MCP servers
    mcp_servers = [
        {
            "name": "filesystem",
            "command": "mcp-server-filesystem",
            "args": ["/workspace"]
        },
        {
            "name": "web",
            "command": "mcp-server-brave-search",
            "env": {"BRAVE_API_KEY": "your_api_key"}
        }
    ]
    
    # Agent with MCP integration
    agent = ReActAgent(
        system_prompt="You are a helpful assistant with web search and file access.",
        mcp_servers=mcp_servers,     # Auto-connect on startup
        auto_connect_mcp=True        # Enable auto-connection
    )
    
    # The agent now has access to both local and MCP server tools
    response = await agent.execute("Search for recent AI news and save to a file")
    
    # Manual MCP server connection
    weather_server = {
        "name": "weather",
        "command": "mcp-server-weather",
        "args": ["--api-key", "your_weather_key"]
    }
    
    success = await agent.connect_mcp_server(weather_server)
    if success:
        print("Weather server connected!")
    
    # List all available tools (local + MCP)
    print("Available tools:", agent.list_tools())
    print("MCP servers:", agent.list_mcp_servers())
    print("MCP tools:", agent.list_mcp_tools())

asyncio.run(mcp_example())
```

**Installation for MCP support:**
```bash
pip install mcp  # Optional - only needed for MCP integration
```

## Architecture

### **Local Tools** 
Built-in tools that run in the agent process:
- **CalculatorTool**: Mathematical calculations
- **SubAgentDispatchTool**: Delegate tasks to specialized agents

### **MCP Tools (External)**
External capabilities via MCP servers:
- **Filesystem Server**: File operations (read, write, edit, list)
- **Web Search Server**: Internet search 
- **Database Server**: SQLite operations
- **GitHub Server**: Repository operations
- **Custom Servers**: Build your own MCP servers

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

## File Operations

File operations are handled by MCP filesystem servers with built-in security:

```python
# MCP filesystem server with workspace restriction
mcp_servers = [{
    "name": "filesystem",
    "command": "npx", 
    "args": ["@modelcontextprotocol/server-filesystem", "/safe_workspace"]
}]

agent = create_mcp_agent("File assistant", mcp_servers)

# Operations are restricted to /safe_workspace
await agent.execute("Create file data.txt with some content")  # ✅ Allowed
await agent.execute("Read /etc/passwd")  # ❌ Blocked by MCP server
```

## Interactive Demos

```bash
# Quick start with agent types
python examples/quick_start.py

# Interactive Code Agent (like Claude Code)
python examples/code_agent_demo.py
```

## API Reference

### Agent Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `execute(message)` | ✨ **Recommended** - Auto-detects conversation state | General use |
| `stream(message)` | ✨ **Recommended** - Streaming with auto-detection | Real-time responses |
| `run(query)` | Forces new conversation | When you need fresh start |
| `chat(message)` | Forces continuation | When you need to continue |

### MCP Methods

| Method | Description |
|--------|-------------|
| `connect_mcp_server(config)` | Connect to an MCP server manually |
| `disconnect_mcp_server(name)` | Disconnect from an MCP server |
| `list_mcp_servers()` | Get list of connected MCP servers |
| `list_mcp_tools()` | Get list of available MCP tools |


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
    debug=True,    # Detailed debugging
    
    # MCP Integration (optional)
    mcp_servers=[
        {
            "name": "server_name",
            "command": "mcp-server-command",
            "args": ["--option", "value"],
            "env": {"API_KEY": "value"}
        }
    ],
    auto_connect_mcp=True  # Auto-connect on startup
)
```

### MCP Server Configuration

MCP servers are configured with a dictionary containing:
- `name`: Unique identifier for the server
- `command`: Command to start the MCP server process
- `args`: Command line arguments (optional)
- `env`: Environment variables (optional)

**Common MCP Servers:**
```python
mcp_servers = [
    # File system access
    {
        "name": "filesystem",
        "command": "mcp-server-filesystem",
        "args": ["/workspace"]
    },
    
    # Web search with Brave
    {
        "name": "brave_search", 
        "command": "mcp-server-brave-search",
        "env": {"BRAVE_API_KEY": "your_key"}
    },
    
    # GitHub integration
    {
        "name": "github",
        "command": "mcp-server-github",
        "env": {"GITHUB_TOKEN": "your_token"}
    },
    
    # SQLite database
    {
        "name": "sqlite",
        "command": "mcp-server-sqlite",
        "args": ["--db-path", "/path/to/database.db"]
    }
]
```
