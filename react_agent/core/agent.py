import json
import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from openai import AsyncOpenAI
from .tool import Tool
from .hierarchical_logger import get_hierarchical_logger, create_child_logger

# Optional MCP imports
try:
    import mcp
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


class ReActAgent:
    """
    A ReAct (Reasoning + Acting) Agent that can use tools to solve problems.
    
    Features:
    - Automatic tool binding and validation
    - MCP (Model Context Protocol) server integration
    - Unified local and remote tool execution
    - Configurable system prompts and behavior
    - Conversation history management
    - Debug and verbose modes
    - Error handling and retries
    - Token usage tracking
    - Support for OpenAI-compatible APIs
    
    """
    
    def __init__(self, 
                 system_prompt: str,
                 agent_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = 128000,
                 verbose: bool = False,
                 debug: bool = False,
                 logger: Optional[logging.Logger] = None,
                 mcp_servers: Optional[List[Dict[str, Any]]] = None,
                 auto_connect_mcp: bool = True,
                 **client_kwargs):
        
        # API Configuration - all services must follow OpenAI specification
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via parameter or OPENAI_API_KEY environment variable")
        
        # Client configuration
        client_config = {
            "api_key": self.api_key,
            **client_kwargs
        }
        
        # Add base_url if provided (for OpenAI-compatible services)
        if base_url:
            client_config["base_url"] = base_url
        elif os.getenv("OPENAI_BASE_URL"):
            client_config["base_url"] = os.getenv("OPENAI_BASE_URL")
        
        self.client = AsyncOpenAI(**client_config)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Agent Configuration
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt is required and cannot be empty")
        self.system_prompt = system_prompt.strip()
        self.verbose = verbose
        self.debug = debug
        
        # Logging Configuration
        self.agent_name = agent_name or f"Agent_{id(self)}"
        self.logger = logger  # Use provided logger or None
        self.enable_logging = logger is not None
        
        # State
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, Any]] = []  # OpenAI API format
        self.detailed_history: List[Dict[str, Any]] = []     # Full interaction history
        self.total_tokens = 0
        self.operation_count = 0
        self.start_time = datetime.now()
        
        # MCP Integration
        self.mcp_servers: Dict[str, Dict] = {}  # {server_name: {session, tools}}
        self.mcp_tools: Dict[str, Dict] = {}   # {tool_name: {server_name, tool_info}}
        self._mcp_server_configs = mcp_servers or []
        self._auto_connect_mcp = auto_connect_mcp and _MCP_AVAILABLE
        
        # Callbacks
        self.on_tool_call: Optional[Callable[[str, Dict], None]] = None
        self.on_iteration: Optional[Callable[[int, str], None]] = None
        
        # Initialize MCP connections if configured
        if self._mcp_server_configs and not _MCP_AVAILABLE:
            self._log_warning("MCP servers configured but MCP client not available. Install with: pip install mcp")
        elif self._auto_connect_mcp and self._mcp_server_configs:
            # Mark for connection on first tool call - deferred to avoid blocking initialization
            self._mcp_connection_pending = True
        else:
            self._mcp_connection_pending = False
        
        self._log_info(f"🤖 Agent initialized - Model: {self.model}, Temperature: {self.temperature}")
    
    def _log_info(self, message: str):
        """Log an info message if logging is enabled."""
        if self.enable_logging and self.logger:
            self.logger.info(message)
    
    def _log_warning(self, message: str):
        """Log a warning message if logging is enabled."""
        if self.enable_logging and self.logger:
            self.logger.warning(f"⚠️  {message}")
    
    def _log_error(self, message: str):
        """Log an error message if logging is enabled."""
        if self.enable_logging and self.logger:
            self.logger.error(f"❌ {message}")
    
    def _log_operation_start(self, operation: str, details: str = ""):
        """Log the start of an operation."""
        self.operation_count += 1
        self._log_info(f"▶️  OPERATION {self.operation_count}: {operation}")
        if details:
            self._log_info(f"   Details: {details}")
    
    def _log_operation_complete(self, operation: str, execution_time: float = 0, result_preview: str = ""):
        """Log the completion of an operation."""
        self._log_info(f"✅ COMPLETED: {operation}")
        if execution_time > 0:
            self._log_info(f"   Time: {execution_time:.2f}s")
        if result_preview:
            preview = result_preview[:100] + "..." if len(result_preview) > 100 else result_preview
            self._log_info(f"   Result: {preview}")
    
    def _add_detailed_history_entry(self, entry_type: str, content: Dict[str, Any]):
        """Add an entry to the detailed conversation history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,  # "user_message", "tool_call", "tool_result", "agent_response", "iteration_start"
            "content": content
        }
        self.detailed_history.append(entry)
        
        # Log if enabled
        if entry_type == "user_message":
            self._log_info(f"👤 User: {content.get('message', '')[:100]}")
        elif entry_type == "tool_call":
            self._log_info(f"🔧 Tool call: {content.get('tool_name')} -> {content.get('args', {})}")
        elif entry_type == "tool_result":
            result_preview = str(content.get('result', ''))[:50]
            self._log_info(f"🛠️  Tool result: {result_preview}...")
        elif entry_type == "agent_response":
            pass
            # response_preview = content.get('response', '')[:100]
            # self._log_info(f"🤖 Agent response: {response_preview}...")
    
    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """Get the complete detailed interaction history."""
        return self.detailed_history.copy()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation including tool interactions."""
        user_messages = [h for h in self.detailed_history if h["type"] == "user_message"]
        tool_calls = [h for h in self.detailed_history if h["type"] == "tool_call"]
        agent_responses = [h for h in self.detailed_history if h["type"] == "agent_response"]
        
        tools_used = {}
        for tool_call in tool_calls:
            tool_name = tool_call["content"].get("tool_name", "unknown")
            tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
        
        return {
            "total_interactions": len(self.detailed_history),
            "user_messages": len(user_messages),
            "tool_calls": len(tool_calls),
            "agent_responses": len(agent_responses),
            "tools_used": tools_used,
            "conversation_duration": (datetime.now() - self.start_time).total_seconds(),
            "total_tokens": self.total_tokens,
            "operations": self.operation_count
        }
        
    def bind_tool(self, tool: Tool) -> 'ReActAgent':
        """
        Bind a tool to the agent.
        
        Args:
            tool: Tool instance to bind
            
        Returns:
            Self for method chaining
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool)}")
        
        if tool.name in self.tools:
            self._log_warning(f"Overriding existing tool '{tool.name}'")
            if self.verbose:
                print(f"Warning: Overriding existing tool '{tool.name}'")
        
        # Create child logger for the tool if agent has logging enabled
        if self.logger is not None:
            tool.logger = create_child_logger(self.agent_name, f"{self.agent_name}.{tool.name}", True)
            tool.enable_logging = True
            tool.agent_name = self.agent_name
        
        # Set agent reference on the tool
        tool.agent = self
        
        self.tools[tool.name] = tool
        
        self._log_info(f"🔧 Tool bound: {tool.name} - {tool.description}")
        if self.debug:
            print(f"Bound tool: {tool.name} - {tool.description}")
            
        return self
        
    def bind_tools(self, tools: List[Tool]) -> 'ReActAgent':
        """
        Bind multiple tools to the agent.
        
        Args:
            tools: List of Tool instances to bind
            
        Returns:
            Self for method chaining
        """
        for tool in tools:
            self.bind_tool(tool)
        return self
    
    
    def unbind_tool(self, name: str) -> 'ReActAgent':
        """Remove a tool by name."""
        if name in self.tools:
            del self.tools[name]
            if self.debug:
                print(f"Unbound tool: {name}")
        return self
    
    def list_tools(self) -> List[str]:
        """Get list of all tool names (local and MCP)."""
        return list(self.tools.keys()) + list(self.mcp_tools.keys())
    
    def get_tool_info(self) -> Dict[str, str]:
        """Get information about bound tools (local and MCP)."""
        info = {name: tool.description for name, tool in self.tools.items()}
        
        # Add MCP tool info
        for tool_name, tool_data in self.mcp_tools.items():
            server_name = tool_data['server_name']
            tool_info = tool_data['tool_info']
            description = tool_info.get('description', 'MCP tool')
            info[tool_name] = f"[{server_name}] {description}"
        
        return info
    
    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all bound tools (local and MCP)."""
        schemas = [tool.get_schema() for tool in self.tools.values()]
        
        # Add MCP tool schemas
        for tool_name, tool_data in self.mcp_tools.items():
            tool_info = tool_data['tool_info']
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.get('description', 'MCP tool'),
                    "parameters": tool_info.get('inputSchema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            schemas.append(schema)
        
        return schemas
    
    async def _ensure_mcp_connections(self):
        """Ensure MCP connections are established if pending."""
        if self._mcp_connection_pending:
            self._mcp_connection_pending = False
            await self._connect_mcp_servers()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the ReAct agent."""
        # Always use the provided system prompt - no auto-generation
        base_prompt = self.system_prompt
        
        # Optionally append tool information if tools are available
        if self.tools:
            tool_descriptions = "\n\nAvailable tools:\n" + "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tools.values()
            ])
            return f"{base_prompt}{tool_descriptions}"
        
        return base_prompt

    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a function call and return the result (async)."""
        function_name = tool_call.function.name
        operation_start = datetime.now()
        
        try:
            # Handle empty arguments
            arguments_str = tool_call.function.arguments.strip()
            if not arguments_str:
                function_args = {}
            else:
                function_args = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in function arguments: {str(e)}"
            self._log_error(f"Tool call failed: {error_msg}")
            self._log_error(f"Raw arguments: '{tool_call.function.arguments}'")
            return f"Error: {error_msg}"
        
        # Check if it's a local tool or MCP tool
        is_mcp_tool = function_name in self.mcp_tools
        is_local_tool = function_name in self.tools
        
        if not is_local_tool and not is_mcp_tool:
            available_local = list(self.tools.keys())
            available_mcp = list(self.mcp_tools.keys())
            available_tools = ", ".join(available_local + available_mcp)
            error_msg = f"Tool '{function_name}' not found. Available tools: {available_tools}"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
        
        # Log tool call start
        self._log_info(f"🔧 Calling tool: {function_name}")
        self._log_info(f"   Args: {function_args}")
        
        # Track in detailed history
        self._add_detailed_history_entry("tool_call", {
            "tool_name": function_name,
            "args": function_args,
            "tool_call_id": tool_call.id
        })
        
        # Call callback if set
        if self.on_tool_call:
            self.on_tool_call(function_name, function_args)
        
        if self.verbose:
            print(f"🔧 Calling tool: {function_name} with args: {function_args}")
        
        try:
            if is_local_tool:
                result = await self.tools[function_name].execute(**function_args)
            else:  # MCP tool
                result = await self._execute_mcp_tool(function_name, function_args)
            
            execution_time = (datetime.now() - operation_start).total_seconds()
            result_str = str(result)
            
            # Log successful completion
            # self._log_info(f"✅ Tool completed: {function_name}")
            # self._log_info(f"   Time: {execution_time:.2f}s")
            # self._log_info(f"   Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
            
            # Track in detailed history
            self._add_detailed_history_entry("tool_result", {
                "tool_name": function_name,
                "result": result_str,
                "execution_time": execution_time,
                "success": True,
                "tool_call_id": tool_call.id
            })
            
            if self.verbose:
                print(f"✅ Tool result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
            
            return result_str
        except Exception as e:
            execution_time = (datetime.now() - operation_start).total_seconds()
            error_msg = f"Error executing {function_name}: {str(e)}"
            
            self._log_error(f"Tool execution failed: {function_name}")
            self._log_error(f"   Time: {execution_time:.2f}s")
            self._log_error(f"   Error: {str(e)}")
            
            # Track failed tool result in detailed history
            self._add_detailed_history_entry("tool_result", {
                "tool_name": function_name,
                "result": error_msg,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "tool_call_id": tool_call.id
            })
            
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg
    
    async def _execute_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute an MCP tool call."""
        if not _MCP_AVAILABLE:
            return "Error: MCP client not available. Install with: pip install mcp"
        
        tool_data = self.mcp_tools.get(tool_name)
        if not tool_data:
            return f"Error: MCP tool '{tool_name}' not found"
        
        server_name = tool_data['server_name']
        server_data = self.mcp_servers.get(server_name)
        if not server_data:
            return f"Error: MCP server '{server_name}' not connected"
        
        # Check if this is a demo mode connection
        if not server_data.get('connected', False):
            return f"MCP tool '{tool_name}' would be executed on server '{server_name}' with args: {args}. (Demo mode - server not actually running)"
        
        # Create fresh session for tool execution
        server_params = server_data.get('server_params')
        if not server_params:
            return f"Error: No server parameters for MCP server '{server_name}'"
        
        try:
            # Create fresh connection for this tool call
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Call the MCP tool using the session
                    response = await session.call_tool(tool_name, args)
                    
                    if hasattr(response, 'content'):
                        if isinstance(response.content, list):
                            # Handle multiple content items
                            return '\n'.join([str(item.text) if hasattr(item, 'text') else str(item) for item in response.content])
                        else:
                            return str(response.content)
                    else:
                        return str(response)
                        
        except Exception as e:
            self._log_error(f"MCP tool execution failed: {str(e)}")
            return f"Error executing MCP tool '{tool_name}': {str(e)}"
    
    async def _connect_mcp_servers(self):
        """Connect to configured MCP servers."""
        if not _MCP_AVAILABLE:
            self._log_warning("Cannot connect to MCP servers: MCP client not available")
            return
        
        for server_config in self._mcp_server_configs:
            await self._connect_mcp_server(server_config)
    
    async def _connect_mcp_server(self, server_config: Dict[str, Any]):
        """Connect to a single MCP server."""
        server_name = server_config.get('name', 'unknown')
        
        try:
            if 'command' in server_config:
                self._log_info(f"🔌 Attempting to connect to MCP server '{server_name}'...")
                
                if _MCP_AVAILABLE:
                    # Try real MCP connection
                    command = server_config['command']
                    args = server_config.get('args', [])
                    env = server_config.get('env', {})
                    
                    try:
                        # Create server parameters
                        server_params = StdioServerParameters(
                            command=command,
                            args=args,
                            env=env
                        )
                        
                        # Test connection and get tools
                        async with stdio_client(server_params) as (read, write):
                            async with ClientSession(read, write) as session:
                                # Initialize the session
                                init_result = await session.initialize()
                                self._log_info(f"MCP server '{server_name}' initialized: {init_result}")
                                
                                # List available tools
                                tools_response = await session.list_tools()
                                tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                        
                        # Store connection info (create fresh sessions per call)
                        self.mcp_servers[server_name] = {
                            'config': server_config,
                            'tools': tools,
                            'connected': True,
                            'server_params': server_params
                        }
                        
                        # Register tools
                        for tool in tools:
                            tool_name = tool.name
                            self.mcp_tools[tool_name] = {
                                'server_name': server_name,
                                'tool_info': {
                                    'name': tool.name,
                                    'description': getattr(tool, 'description', ''),
                                    'inputSchema': getattr(tool, 'inputSchema', {})
                                }
                            }
                        
                        tool_count = len(tools)
                        self._log_info(f"✅ Connected to MCP server '{server_name}' with {tool_count} tools")
                                
                    except Exception as connection_error:
                        # Fallback to demo mode if real connection fails
                        self._log_warning(f"Real MCP connection failed: {connection_error}")
                        self._log_info(f"Falling back to demo mode for server '{server_name}'")
                        
                        self.mcp_servers[server_name] = {
                            'config': server_config,
                            'tools': [],
                            'connected': False  # Demo mode
                        }
                        
                else:
                    # MCP not available, use demo mode
                    self._log_warning(f"MCP client not available - using demo mode for '{server_name}'")
                    self.mcp_servers[server_name] = {
                        'config': server_config,
                        'tools': [],
                        'connected': False  # Demo mode
                    }
                    
            else:
                self._log_warning(f"MCP server '{server_name}' has no connection method configured")
                
        except Exception as e:
            self._log_error(f"Failed to connect to MCP server '{server_name}': {str(e)}")
    
    async def connect_mcp_server(self, server_config: Dict[str, Any]) -> bool:
        """
        Connect to an MCP server manually.
        
        Args:
            server_config: Server configuration dict with 'name', 'command', optional 'args' and 'env'
            
        Returns:
            True if connected successfully, False otherwise
        """
        if not _MCP_AVAILABLE:
            self._log_warning("Cannot connect to MCP server: MCP client not available")
            return False
        
        await self._connect_mcp_server(server_config)
        
        server_name = server_config.get('name', 'unknown')
        return server_name in self.mcp_servers
    
    def disconnect_mcp_server(self, server_name: str):
        """Disconnect from an MCP server."""
        if server_name in self.mcp_servers:
            # Remove tools from this server
            tools_to_remove = [
                tool_name for tool_name, tool_data in self.mcp_tools.items()
                if tool_data['server_name'] == server_name
            ]
            for tool_name in tools_to_remove:
                del self.mcp_tools[tool_name]
            
            # Remove server
            del self.mcp_servers[server_name]
            self._log_info(f"🔌 Disconnected from MCP server '{server_name}'")
    
    def list_mcp_servers(self) -> List[str]:
        """Get list of connected MCP servers."""
        return list(self.mcp_servers.keys())
    
    def list_mcp_tools(self) -> Dict[str, str]:
        """Get list of MCP tools with their server names."""
        return {
            tool_name: f"[{tool_data['server_name']}] {tool_data['tool_info'].get('description', '')}"
            for tool_name, tool_data in self.mcp_tools.items()
        }
    
    async def run(self, query: str, max_iterations: int = 10, stream: bool = False) -> str:
        """
        Backward compatibility method. Use execute() instead.
        Run the ReAct agent starting a new conversation.
        """
        # Reset conversation to ensure new start
        self.conversation_history = []
        return await self.execute(query, max_iterations)
    
    async def execute(self, message: str, max_iterations: int = 10) -> str:
        """
        Execute the agent with a message. Auto-detects new vs continuing conversation.
        
        Args:
            message: The user's message/question
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            The agent's final response
            
        Raises:
            ValueError: If message is empty
            Exception: If API call fails
        """
        if not message.strip():
            raise ValueError("Message cannot be empty")
        
        run_start = datetime.now()
        is_new_conversation = not self.conversation_history
        
        # Ensure MCP connections are established if needed
        await self._ensure_mcp_connections()
        
        # Log operation
        operation = "Agent Execute (New)" if is_new_conversation else "Agent Execute (Continue)"
        self._log_operation_start(operation, f"Message: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        if is_new_conversation:
            self._log_info(f"📋 Available tools: {list(self.tools.keys())}")
            if self.verbose:
                print(f"🤖 Starting ReAct agent with message: {message}")
                print(f"📋 Available tools: {list(self.tools.keys())}")
        
        # Initialize or continue conversation
        if is_new_conversation:
            # New conversation - initialize with system prompt
            self.conversation_history = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": message}
            ]
            operation_type = "Agent Execute New"
        else:
            # Continue existing conversation
            self.conversation_history.append({
                "role": "user", 
                "content": message
            })
            operation_type = "Agent Execute Continue"
        
        # Track in detailed history
        self._add_detailed_history_entry("user_message", {
            "message": message,
            "operation": operation_type
        })
        
        try:
            for iteration in range(max_iterations):
                self._log_info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                
                if self.verbose:
                    print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
                
                # Call iteration callback if set
                if self.on_iteration:
                    self.on_iteration(iteration + 1, message)
                
                # Get tool schemas for function calling
                tools = self._get_tool_schemas() if self.tools else None
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "temperature": self.temperature
                }
                
                if self.max_tokens:
                    api_params["max_tokens"] = self.max_tokens
                    
                if tools:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = "auto"
                
                # Get response from LLM
                response = await self.client.chat.completions.create(**api_params)
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                    if self.debug:
                        print(f"🔢 Tokens used this call: {response.usage.total_tokens}, Total: {self.total_tokens}")
                
                message_obj = response.choices[0].message
                
                # Add assistant message to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message_obj.content,
                    "tool_calls": message_obj.tool_calls
                })
                
                if self.verbose and message_obj.content:
                    print(f"🧠 Agent thinking: {message_obj.content}")
                
                # Execute tool calls if any - run concurrently for better performance
                if message_obj.tool_calls:
                    if len(message_obj.tool_calls) == 1:
                        # Single tool call
                        tool_call = message_obj.tool_calls[0]
                        result = await self._execute_tool_call(tool_call)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    else:
                        # Multiple tool calls - execute concurrently
                        tasks = [self._execute_tool_call(tool_call) for tool_call in message_obj.tool_calls]
                        results = await asyncio.gather(*tasks)
                        
                        for tool_call, result in zip(message_obj.tool_calls, results):
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                else:
                    # No tool calls, we have a final answer
                    if message_obj.content:
                        final_answer = message_obj.content.strip()
                        execution_time = (datetime.now() - run_start).total_seconds()
                        
                        self._log_operation_complete(operation, execution_time)
                        
                        if self.verbose:
                            print(f"✅ Final answer: {final_answer}")
                        
                        agent_response_preview = final_answer[:100] + "..." if len(final_answer) > 100 else final_answer
                        self._log_info(f"🤖 Agent response: {agent_response_preview}")
                        
                        # Track final response in detailed history
                        self._add_detailed_history_entry("agent_response", {
                            "response": final_answer,
                            "iterations": iteration + 1,
                            "operation": operation_type,
                            "execution_time": execution_time,
                            "final": True
                        })
                        
                        return final_answer
            
            timeout_msg = f"Maximum iterations ({max_iterations}) reached without finding a final answer."
            if self.verbose:
                print(f"⏰ {timeout_msg}")
            return timeout_msg
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            if self.debug:
                import traceback
                print(f"❌ {error_msg}")
                traceback.print_exc()
            raise Exception(error_msg) from e
    
    async def stream(self, message: str, max_iterations: int = 10):
        """
        Stream the agent's reasoning process with real-time updates.
        Automatically detects if this is a new conversation or continuation.
        
        Args:
            message: The user's message/question
            max_iterations: Maximum number of reasoning iterations
            
        Yields:
            Dict with progress updates: {"type": "status|thinking|tool_call|tool_result|final", "content": str}
        """
        if not message.strip():
            raise ValueError("Message cannot be empty")
        
        run_start = datetime.now()
        is_new_conversation = not self.conversation_history
        
        # Ensure MCP connections are established if needed
        await self._ensure_mcp_connections()
        
        # Log operation
        operation = "Agent Stream (New)" if is_new_conversation else "Agent Stream (Continue)"
        self._log_operation_start(operation, f"Message: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        # Initialize or continue conversation
        if is_new_conversation:
            # New conversation - initialize with system prompt
            self.conversation_history = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": message}
            ]
            operation_type = "Agent Stream New"
        else:
            # Continue existing conversation
            self.conversation_history.append({
                "role": "user", 
                "content": message
            })
            operation_type = "Agent Stream Continue"
        
        # Track in detailed history
        self._add_detailed_history_entry("user_message", {
            "message": message,
            "operation": operation_type
        })
        
        try:
            for iteration in range(max_iterations):
                # Get tool schemas for function calling
                tools = self._get_tool_schemas() if self.tools else None
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "temperature": self.temperature,
                    "stream": True  # Enable streaming
                }
                
                if self.max_tokens:
                    api_params["max_tokens"] = self.max_tokens
                    
                if tools:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = "auto"
                
                # Get streaming response from LLM
                stream_response = await self.client.chat.completions.create(**api_params)
                
                collected_content = ""
                collected_tool_calls = []
                
                async for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        # Stream thinking content
                        if delta.content:
                            collected_content += delta.content
                            yield {"type": "thinking", "content": delta.content}
                        
                        # Collect tool calls
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Extend tool calls list if needed
                                while len(collected_tool_calls) <= tool_call_delta.index:
                                    collected_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                tc = collected_tool_calls[tool_call_delta.index]
                                if tool_call_delta.id:
                                    tc["id"] = tool_call_delta.id
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        tc["function"]["name"] = tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        tc["function"]["arguments"] += tool_call_delta.function.arguments
                
                # Add complete message to history
                tool_calls_for_history = None
                if collected_tool_calls:
                    # Convert to dict format that can be serialized
                    tool_calls_for_history = []
                    for tc in collected_tool_calls:
                        if tc["function"]["name"]:  # Only add complete tool calls
                            tool_calls_for_history.append({
                                "id": tc["id"],
                                "type": tc["type"],
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": collected_content,
                    "tool_calls": tool_calls_for_history
                })
                
                # Execute tool calls if any
                if tool_calls_for_history:
                    # Create SimpleNamespace objects for tool execution
                    from types import SimpleNamespace
                    for tc_dict in tool_calls_for_history:
                        # Convert dict back to object for _execute_tool_call
                        tool_call = SimpleNamespace()
                        tool_call.id = tc_dict["id"]
                        tool_call.type = tc_dict["type"]
                        tool_call.function = SimpleNamespace()
                        tool_call.function.name = tc_dict["function"]["name"]
                        tool_call.function.arguments = tc_dict["function"]["arguments"]
                        
                        yield {"type": "tool_call", "content": f"🔧 Using {tool_call.function.name}..."}
                        
                        result = await self._execute_tool_call(tool_call)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                        
                        # Show tool result preview
                        result_preview = result[:200] + "..." if len(result) > 200 else result
                        yield {"type": "tool_result", "content": f"✅ Result: {result_preview}"}
                else:
                    # No tool calls, we have a final answer
                    if collected_content:
                        execution_time = (datetime.now() - run_start).total_seconds()
                        
                        # Track final response in detailed history
                        self._add_detailed_history_entry("agent_response", {
                            "response": collected_content,
                            "iterations": iteration + 1,
                            "operation": operation_type,
                            "execution_time": execution_time,
                            "final": True
                        })
                        
                        yield {"type": "final", "content": collected_content}
                        return
            
            # Max iterations reached
            timeout_msg = f"Maximum iterations ({max_iterations}) reached without finding a final answer."
            yield {"type": "final", "content": timeout_msg}
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            yield {"type": "error", "content": error_msg}
            raise Exception(error_msg) from e
    
    # Backward compatibility methods
    async def run_stream(self, query: str, max_iterations: int = 10):
        """
        Backward compatibility method. Use stream() instead.
        Stream the agent's reasoning process starting a new conversation.
        """
        # Reset conversation to ensure new start
        self.conversation_history = []
        async for update in self.stream(query, max_iterations):
            yield update
    
    async def chat_stream(self, message: str, max_iterations: int = 10):
        """
        Backward compatibility method. Use stream() instead.
        Stream the agent's reasoning process continuing existing conversation.
        """
        async for update in self.stream(message, max_iterations):
            yield update
    
    def run_sync(self, query: str, max_iterations: int = 10, stream: bool = False) -> str:
        """
        Synchronous wrapper for run() method.
        
        Args:
            query: The user's query/question
            max_iterations: Maximum number of reasoning iterations
            stream: Whether to stream intermediate results (future feature)
            
        Returns:
            The agent's final response
        """
        return asyncio.run(self.run(query, max_iterations, stream))
    
    def reset(self) -> 'ReActAgent':
        """
        Reset the conversation history and token count.
        
        Returns:
            Self for method chaining
        """
        self.conversation_history = []
        self.total_tokens = 0
        if self.debug:
            print("🔄 Agent state reset")
        return self
    
    async def chat(self, message: str) -> str:
        """
        Backward compatibility method. Use execute() instead.
        Continue the conversation with existing context.
        """
        return await self.execute(message)
    
    
    def chat_sync(self, message: str) -> str:
        """
        Synchronous wrapper for chat() method.
        
        Args:
            message: Follow-up message
            
        Returns:
            Agent's response
        """
        return asyncio.run(self.chat(message))
    
    async def _continue_conversation(self, max_iterations: int = 5) -> str:
        """Continue the conversation from current state (async)."""
        try:
            for iteration in range(max_iterations):
                tools = self._get_tool_schemas() if self.tools else None
                
                api_params = {
                    "model": self.model,
                    "messages": self.conversation_history,
                    "temperature": self.temperature
                }
                
                if self.max_tokens:
                    api_params["max_tokens"] = self.max_tokens
                    
                if tools:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = "auto"
                
                response = await self.client.chat.completions.create(**api_params)
                
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                message = response.choices[0].message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                if message.tool_calls:
                    # Execute multiple tool calls concurrently
                    if len(message.tool_calls) == 1:
                        tool_call = message.tool_calls[0]
                        result = await self._execute_tool_call(tool_call)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    else:
                        tasks = [self._execute_tool_call(tool_call) for tool_call in message.tool_calls]
                        results = await asyncio.gather(*tasks)
                        
                        for tool_call, result in zip(message.tool_calls, results):
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                else:
                    if message.content:
                        return message.content.strip()
            
            return "Could not generate response in maximum iterations."
            
        except Exception as e:
            raise Exception(f"Error in chat continuation: {str(e)}") from e
    
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def get_token_usage(self) -> int:
        """Get total tokens used in this session."""
        return self.total_tokens
    
    def set_system_prompt(self, prompt: str) -> 'ReActAgent':
        """
        Set a custom system prompt.
        
        Args:
            prompt: Custom system prompt
            
        Returns:
            Self for method chaining
        """
        self.system_prompt = prompt
        return self
    
    def set_callbacks(self, 
                     on_tool_call: Optional[Callable[[str, Dict], None]] = None,
                     on_iteration: Optional[Callable[[int, str], None]] = None) -> 'ReActAgent':
        """
        Set callback functions for monitoring agent behavior.
        
        Args:
            on_tool_call: Called when a tool is executed (tool_name, args)
            on_iteration: Called at start of each iteration (iteration_num, query)
            
        Returns:
            Self for method chaining
        """
        if on_tool_call:
            self.on_tool_call = on_tool_call
        if on_iteration:
            self.on_iteration = on_iteration
        return self
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        tool_count = len(self.tools)
        tool_names = list(self.tools.keys())
        base_url = getattr(self.client, 'base_url', 'https://api.openai.com/v1')
        return f"ReActAgent(model={self.model}, base_url={base_url}, tools={tool_count})"
    
