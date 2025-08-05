import json
import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from openai import AsyncOpenAI
from .tool import Tool
from .hierarchical_logger import get_hierarchical_logger, create_child_logger
from .interactive import (
    MessageManager, ToolPermissionManager, InteractiveStatusReporter,
    MessageType, PermissionChoice, setup_default_handlers
)

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
                 max_tokens: Optional[int] = 128_000,
                 debug: bool = False,
                 logger: Optional[logging.Logger] = None,
                 mcp_servers: Optional[List[Dict[str, Any]]] = None,
                 auto_connect_mcp: bool = True,
                 interactive: bool = False,
                 tool_timeout: Optional[float] = None,
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
        self.debug = debug
        
        # Tool Configuration
        self.tool_timeout = tool_timeout if tool_timeout is not None else 30.0
        
        # Logging Configuration
        self.agent_name = agent_name or f"Agent_{id(self)}"
        self.logger = logger  # Use provided logger or None
        self.enable_logging = logger is not None
        
        # State
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, Any]] = []  # OpenAI API format
        self.total_tokens = 0
        
        # MCP Integration
        self.mcp_servers: Dict[str, Dict] = {}  # {server_name: {session, tools}}
        self.mcp_tools: Dict[str, Dict] = {}   # {tool_name: {server_name, tool_info}}
        self._mcp_server_configs = mcp_servers or []
        self._auto_connect_mcp = auto_connect_mcp and _MCP_AVAILABLE
        
        # Callbacks
        self.on_tool_call: Optional[Callable[[str, Dict], None]] = None
        self.on_iteration: Optional[Callable[[int, str], None]] = None
        
        # Interactive system
        self.interactive = interactive
        self.message_manager: Optional[MessageManager] = None
        self.permission_manager: Optional[ToolPermissionManager] = None
        self.status_reporter: Optional[InteractiveStatusReporter] = None
        
        if self.interactive:
            self.message_manager = MessageManager()
            self.permission_manager = ToolPermissionManager()
            self.status_reporter = InteractiveStatusReporter(self.message_manager)
            setup_default_handlers(self.message_manager)
        
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
    
    # Operation tracking removed - simplified logging
    
    # Detailed history removed - simplified for elegance
        
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
        
        # Create child logger for the tool if agent has logging enabled
        if self.logger is not None:
            tool.logger = create_child_logger(self.agent_name, f"{self.agent_name}.{tool.name}", True)
            tool.enable_logging = True
            tool.agent_name = self.agent_name
        
        # Set timeout from agent only if tool uses the default timeout
        # This preserves custom timeouts set by individual tools
        if not hasattr(tool, '_custom_timeout_set'):
            tool.timeout = self.tool_timeout
        
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
        
        # Check for interruption before tool execution
        if self.interactive and self.message_manager and self.message_manager.is_interrupted():
            return "Task interrupted by user."
        
        # Check tool permissions in interactive mode
        if self.interactive and self.permission_manager:
            permission = await self.permission_manager.request_permission(function_name, self.message_manager)
            if permission == PermissionChoice.DENY:
                return f"Tool '{function_name}' not allowed. Task stopped."
        
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
        
        # Tool call tracking via logging
        
        # Report tool call in interactive mode
        if self.interactive and self.status_reporter:
            await self.status_reporter.report_tool_call(function_name, function_args)
        
        # Call callback if set
        if self.on_tool_call:
            self.on_tool_call(function_name, function_args)
        
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
            
            # Tool result tracking via logging and interactive reporting
            
            # Report tool result in interactive mode
            if self.interactive and self.status_reporter:
                await self.status_reporter.report_tool_result(function_name, result_str, True)
            
            return result_str
        except Exception as e:
            execution_time = (datetime.now() - operation_start).total_seconds()
            error_msg = f"Error executing {function_name}: {str(e)}"
            
            self._log_error(f"Tool execution failed: {function_name}")
            self._log_error(f"   Time: {execution_time:.2f}s")
            self._log_error(f"   Error: {str(e)}")
            
            # Tool error tracking via logging and interactive reporting
            
            # Report tool error in interactive mode
            if self.interactive and self.status_reporter:
                await self.status_reporter.report_tool_result(function_name, error_msg, False)
            
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
    
    async def execute_interactive(self, message: str, max_iterations: int = 10, stream: bool = True) -> str:
        """
        Execute the agent with interactive capabilities - real-time status updates,
        ESC interruption, and tool permissions.
        
        Args:
            message: The user's message/question
            max_iterations: Maximum number of reasoning iterations
            stream: Whether to stream LLM responses in real-time (default: True)
            
        Returns:
            The agent's final response
        """
        if not self.interactive:
            raise ValueError("Agent not configured for interactive mode. Set interactive=True in constructor.")
        
        if not message.strip():
            raise ValueError("Message cannot be empty")
        
        # Start the message manager input loop in background
        input_task = asyncio.create_task(self.message_manager.start_input_loop())
        
        try:
            # Execute the agent with real-time interaction
            result = await self._execute_with_interaction(message, max_iterations, stream)
            
            # Wait a moment for any pending output messages to be processed
            await asyncio.sleep(0.1)
            
            return result
        finally:
            # Clean up input loop
            self.message_manager.stop()
            if not input_task.done():
                input_task.cancel()
                try:
                    await input_task
                except asyncio.CancelledError:
                    pass
    
    async def _execute_with_interaction(self, message: str, max_iterations: int = 10, stream: bool = True) -> str:
        """Internal method for interactive execution."""
        run_start = datetime.now()
        is_new_conversation = not self.conversation_history
        
        # Ensure MCP connections are established if needed
        await self._ensure_mcp_connections()
        
        # Report initial status
        if self.status_reporter:
            await self.status_reporter.report_status("Starting agent execution", f"Message: {message[:50]}...")
        
        # Initialize or continue conversation
        if is_new_conversation:
            self.conversation_history = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": message}
            ]
        else:
            self.conversation_history.append({
                "role": "user", 
                "content": message
            })
        
        # User message tracking via logging
        
        try:
            for iteration in range(max_iterations):
                # Check for interruption at start of each iteration
                if self.message_manager.is_interrupted():
                    return "Task interrupted by user."
                
                if self.status_reporter:
                    await self.status_reporter.report_status(f"Iteration {iteration + 1}/{max_iterations}")
                
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
                
                if stream:
                    # Streaming mode - real-time thinking updates
                    api_params["stream"] = True
                    
                    stream_response = await self.client.chat.completions.create(**api_params)
                    
                    collected_content = ""
                    collected_tool_calls = []
                    first_content_chunk = True
                    has_tool_calls_in_stream = False
                    
                    async for chunk in stream_response:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            
                            # Check if we're getting tool calls (indicates working mode)
                            if delta.tool_calls:
                                has_tool_calls_in_stream = True
                            
                            # Stream content in real-time
                            if delta.content:
                                # Add newline before first content chunk for clean separation
                                if first_content_chunk:
                                    if self.status_reporter:
                                        await self.status_reporter.report_thinking("\n")
                                    first_content_chunk = False
                                
                                collected_content += delta.content
                                
                                # If no tool calls detected, this is likely the final response
                                # Use normal color. If tool calls present, use gray (working)
                                if self.status_reporter:
                                    if has_tool_calls_in_stream:
                                        await self.status_reporter.report_thinking(delta.content)
                                    else:
                                        await self.status_reporter.report_response(delta.content)
                            
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
                    
                    # Convert to message format
                    tool_calls_for_history = None
                    if collected_tool_calls:
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
                    
                    # Create message object for consistency with proper tool call objects
                    from types import SimpleNamespace
                    message_obj = SimpleNamespace()
                    message_obj.content = collected_content
                    
                    # Convert dict tool calls to proper objects for _execute_tool_call
                    if tool_calls_for_history:
                        message_obj.tool_calls = []
                        for tc_dict in tool_calls_for_history:
                            tool_call_obj = SimpleNamespace()
                            tool_call_obj.id = tc_dict["id"]
                            tool_call_obj.type = tc_dict["type"]
                            tool_call_obj.function = SimpleNamespace()
                            tool_call_obj.function.name = tc_dict["function"]["name"]
                            tool_call_obj.function.arguments = tc_dict["function"]["arguments"]
                            message_obj.tool_calls.append(tool_call_obj)
                    else:
                        message_obj.tool_calls = None
                    
                else:
                    # Non-streaming mode
                    response = await self.client.chat.completions.create(**api_params)
                    
                    # Track token usage
                    if hasattr(response, 'usage') and response.usage:
                        self.total_tokens += response.usage.total_tokens
                    
                    message_obj = response.choices[0].message
                    
                    # Report thinking if available
                    if message_obj.content and self.status_reporter:
                        await self.status_reporter.report_thinking(message_obj.content)
                
                # Add assistant message to history (use dicts for JSON serialization)
                if stream and tool_calls_for_history:
                    # Streaming mode - use the dict format we created
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message_obj.content,
                        "tool_calls": tool_calls_for_history
                    })
                else:
                    # Non-streaming mode - convert tool calls to dicts if needed
                    tool_calls_dict = None
                    if hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                        tool_calls_dict = []
                        for tc in message_obj.tool_calls:
                            tool_calls_dict.append({
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message_obj.content,
                        "tool_calls": tool_calls_dict
                    })
                
                # Execute tool calls if any
                if message_obj.tool_calls:
                    if len(message_obj.tool_calls) == 1:
                        # Single tool call - check for interruption
                        tool_call = message_obj.tool_calls[0]
                        if self.message_manager.is_interrupted():
                            return "Task interrupted by user."
                        
                        result = await self._execute_tool_call(tool_call)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                        
                        # If tool was denied, stop execution
                        if "not allowed" in result and "stopped" in result:
                            return result
                    else:
                        # Multiple tool calls - execute in parallel for better performance
                        # Check for interruption before starting
                        if self.message_manager.is_interrupted():
                            return "Task interrupted by user."
                        
                        tasks = [self._execute_tool_call(tool_call) for tool_call in message_obj.tool_calls]
                        results = await asyncio.gather(*tasks)
                        
                        for tool_call, result in zip(message_obj.tool_calls, results):
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                            
                            # If any tool was denied, stop execution
                            if "not allowed" in result and "stopped" in result:
                                return result
                else:
                    # No tool calls, we have a final answer
                    if message_obj.content:
                        final_answer = message_obj.content.strip()
                        execution_time = (datetime.now() - run_start).total_seconds()
                        
                        # Final response tracking via logging
                        
                        # Streaming already showed the response, no need to report again
                        
                        return final_answer
            
            timeout_msg = f"Maximum iterations ({max_iterations}) reached without finding a final answer."
            return timeout_msg
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            raise Exception(error_msg) from e

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
        
        if is_new_conversation:
            self._log_info(f"📋 Available tools: {list(self.tools.keys())}")
        
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
        
        # User message tracking via logging
        
        try:
            for iteration in range(max_iterations):
                self._log_info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                
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
                
                # Verbose mode removed - using interactive system instead
                
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
                        
                        # Operation completed
                        
                        agent_response_preview = final_answer[:100] + "..." if len(final_answer) > 100 else final_answer
                        self._log_info(f"🤖 Agent response: {agent_response_preview}")
                        
                        # Final response tracking via logging
                        
                        return final_answer
            
            timeout_msg = f"Maximum iterations ({max_iterations}) reached without finding a final answer."
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
        
        # User message tracking via logging
        
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
    
    def reset_permissions(self) -> 'ReActAgent':
        """Reset all tool permissions in interactive mode."""
        if self.interactive and self.permission_manager:
            self.permission_manager.reset_permissions()
        return self
    
    def is_interactive(self) -> bool:
        """Check if agent is in interactive mode."""
        return self.interactive
    
    def get_permissions(self) -> Dict[str, str]:
        """Get current tool permissions (interactive mode only)."""
        if self.interactive and self.permission_manager:
            return self.permission_manager.permissions.copy()
        return {}
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        tool_count = len(self.tools)
        tool_names = list(self.tools.keys())
        base_url = getattr(self.client, 'base_url', 'https://api.openai.com/v1')
        return f"ReActAgent(model={self.model}, base_url={base_url}, tools={tool_count})"
    
