import json
import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from openai import AsyncOpenAI
from .tool import Tool


class ReActAgent:
    """
    A ReAct (Reasoning + Acting) Agent that can use tools to solve problems.
    
    Features:
    - Automatic tool binding and validation
    - Configurable system prompts and behavior
    - Conversation history management
    - Debug and verbose modes
    - Error handling and retries
    - Token usage tracking
    - Support for OpenAI-compatible APIs
    
    Supported APIs:
    - OpenAI (default)
    - Azure OpenAI
    - Anthropic Claude (via OpenAI compatibility)
    - Local models (Ollama, LocalAI, etc.)
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = None,
                 system_prompt: Optional[str] = None,
                 verbose: bool = False,
                 debug: bool = False,
                 enable_logging: bool = False,
                 agent_name: Optional[str] = None,
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
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.debug = debug
        
        # Logging Configuration
        self.enable_logging = enable_logging
        self.agent_name = agent_name or f"Agent_{id(self)}"
        self.logger = None
        
        if self.enable_logging:
            self.logger = logging.getLogger(f"ReActAgent.{self.agent_name}")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # State
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_tokens = 0
        self.operation_count = 0
        self.start_time = datetime.now()
        
        # Callbacks
        self.on_tool_call: Optional[Callable[[str, Dict], None]] = None
        self.on_iteration: Optional[Callable[[int, str], None]] = None
        
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
        
        self.tools[tool.name] = tool
        
        # Enable logging for the tool if agent logging is enabled
        if hasattr(tool, 'enable_logging'):
            tool.enable_logging = self.enable_logging
            tool.agent_name = self.agent_name
        
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
        """Get list of bound tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self) -> Dict[str, str]:
        """Get information about bound tools."""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all bound tools."""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the ReAct agent."""
        if self.system_prompt:
            return self.system_prompt
        
        tool_descriptions = ""
        if self.tools:
            tool_descriptions = "\n\nAvailable tools:\n" + "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in self.tools.values()
            ])
        
        return f"""You are a ReAct agent that can reason and act using available tools.

Use the ReAct pattern:
1. Think about what you need to do
2. Use available function calls to gather information or perform actions
3. Observe the results
4. Continue reasoning and acting until you can provide a complete answer

You have access to various tools via function calls. Use them as needed to solve the user's query.{tool_descriptions}"""

    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a function call and return the result (async)."""
        function_name = tool_call.function.name
        operation_start = datetime.now()
        
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in function arguments: {str(e)}"
            self._log_error(f"Tool call failed: {error_msg}")
            return f"Error: {error_msg}"
        
        if function_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            error_msg = f"Tool '{function_name}' not found. Available tools: {available_tools}"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
        
        # Log tool call start
        self._log_info(f"🔧 Calling tool: {function_name}")
        self._log_info(f"   Args: {function_args}")
        
        # Call callback if set
        if self.on_tool_call:
            self.on_tool_call(function_name, function_args)
        
        if self.verbose:
            print(f"🔧 Calling tool: {function_name} with args: {function_args}")
        
        try:
            result = await self.tools[function_name].execute(**function_args)
            
            execution_time = (datetime.now() - operation_start).total_seconds()
            result_str = str(result)
            
            # Log successful completion
            self._log_info(f"✅ Tool completed: {function_name}")
            self._log_info(f"   Time: {execution_time:.2f}s")
            self._log_info(f"   Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
            
            if self.verbose:
                print(f"✅ Tool result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
            
            return result_str
        except Exception as e:
            execution_time = (datetime.now() - operation_start).total_seconds()
            error_msg = f"Error executing {function_name}: {str(e)}"
            
            self._log_error(f"Tool execution failed: {function_name}")
            self._log_error(f"   Time: {execution_time:.2f}s")
            self._log_error(f"   Error: {str(e)}")
            
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg
    
    async def run(self, query: str, max_iterations: int = 10, stream: bool = False) -> str:
        """
        Run the ReAct agent on a query (async).
        
        Args:
            query: The user's query/question
            max_iterations: Maximum number of reasoning iterations
            stream: Whether to stream intermediate results (future feature)
            
        Returns:
            The agent's final response
            
        Raises:
            ValueError: If query is empty or no tools are bound when needed
            Exception: If API call fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        run_start = datetime.now()
        self._log_operation_start("Agent Run", f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        self._log_info(f"📋 Available tools: {list(self.tools.keys())}")
        
        if self.verbose:
            print(f"🤖 Starting ReAct agent with query: {query}")
            print(f"📋 Available tools: {list(self.tools.keys())}")
        
        # Initialize conversation
        self.conversation_history = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        try:
            for iteration in range(max_iterations):
                self._log_info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                
                if self.verbose:
                    print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
                
                # Call iteration callback if set
                if self.on_iteration:
                    self.on_iteration(iteration + 1, query)
                
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
                
                message = response.choices[0].message
                
                # Add assistant message to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                if self.verbose and message.content:
                    print(f"🧠 Agent thinking: {message.content}")
                
                # Execute tool calls if any - run concurrently for better performance
                if message.tool_calls:
                    if len(message.tool_calls) == 1:
                        # Single tool call
                        tool_call = message.tool_calls[0]
                        result = await self._execute_tool_call(tool_call)
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    else:
                        # Multiple tool calls - execute concurrently
                        tasks = [self._execute_tool_call(tool_call) for tool_call in message.tool_calls]
                        results = await asyncio.gather(*tasks)
                        
                        for tool_call, result in zip(message.tool_calls, results):
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                else:
                    # No tool calls, we have a final answer
                    if message.content:
                        final_answer = message.content.strip()
                        execution_time = (datetime.now() - run_start).total_seconds()
                        
                        self._log_operation_complete("Agent Run", execution_time, final_answer[:200])
                        self._log_info(f"🎯 Final answer after {iteration + 1} iterations")
                        
                        if self.verbose:
                            print(f"✨ Final answer: {final_answer}")
                        return final_answer
            
            # Max iterations reached
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
        Continue the conversation with a follow-up message (async).
        Maintains conversation context.
        
        Args:
            message: Follow-up message
            
        Returns:
            Agent's response
        """
        if not self.conversation_history:
            # First message, same as run()
            return await self.run(message)
        
        # Add user message to existing conversation
        self.conversation_history.append({
            "role": "user", 
            "content": message
        })
        
        # Continue from current state
        return await self._continue_conversation()
    
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
    
