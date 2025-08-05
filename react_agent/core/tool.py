from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Awaitable
import functools
import asyncio
import inspect
import logging
from datetime import datetime
from .hierarchical_logger import create_child_logger


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to define a tool with name and description.
    
    Usage:
    @tool(name="calculator", description="Perform mathematical calculations")
    class CalculatorTool(Tool):
        ...
    """
    def decorator(cls):
        cls._tool_name = name
        cls._tool_description = description
        return cls
    return decorator


def param(name: str, 
          type: str = "string",
          description: str = "",
          required: bool = True,
          minimum: Optional[int] = None,
          maximum: Optional[int] = None,
          enum: Optional[List[str]] = None,
          default: Any = None):
    """
    Decorator to define a parameter for a tool method.
    
    Usage:
    @param("expression", type="string", description="Math expression to evaluate")
    @param("precision", type="integer", description="Decimal places", minimum=0, maximum=10, required=False)
    @param("style", type="string", enum=["formal", "casual"], default="casual", required=False)
    def _execute(self):
        expression = self.get_param("expression")
        precision = self.get_param("precision")
        ...
    """
    def decorator(func):
        if not hasattr(func, '_parameters'):
            func._parameters = {}
            func._required = []
        
        # Build OpenAI schema from parameters
        schema = {
            "type": type,
            "description": description
        }
        
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum
        if enum is not None:
            schema["enum"] = enum
        if default is not None:
            schema["default"] = default
        
        func._parameters[name] = schema
        
        if required:
            func._required.append(name)
        
        return func
    return decorator


class ParameterAccessMixin:
    """Mixin to provide parameter access in tool methods."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_params = {}
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value that was passed to the current execution."""
        return self._current_params.get(name, default)
    
    def has_param(self, name: str) -> bool:
        """Check if a parameter was provided."""
        return name in self._current_params
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get all parameters passed to the current execution."""
        return self._current_params.copy()


def validate_and_inject_params(func):
    """
    Decorator that validates parameters and injects them into the tool instance
    before calling the decorated method. All tools must be async.
    """
    @functools.wraps(func)
    async def async_wrapper(self, **kwargs):
        # Get parameter definitions from the function
        parameters = getattr(func, '_parameters', {})
        required = getattr(func, '_required', [])
        
        # Validate parameters
        validated_params = {}
        
        # Check required parameters
        for param_name in required:
            if param_name not in kwargs:
                return f"Error: Missing required parameter '{param_name}'"
        
        # Validate each parameter
        for param_name, param_schema in parameters.items():
            value = kwargs.get(param_name)
            
            # Skip if not provided and not required
            if value is None and param_name not in required:
                if 'default' in param_schema:
                    validated_params[param_name] = param_schema['default']
                continue
                
            validated_value, error = _validate_parameter_value(param_name, value, param_schema, required)
            if error:
                return f"Error: Parameter '{param_name}': {error}"
                
            validated_params[param_name] = validated_value
        
        # Inject validated parameters into the instance
        self._current_params = validated_params
        
        # All tools must be async - no sync detection needed
        return await func(self)
    
    return async_wrapper


def _validate_parameter_value(param_name: str, value: Any, schema: Dict[str, Any], required: List[str]) -> tuple[Any, Optional[str]]:
    """Validate a parameter value against OpenAI schema format."""
    if value is None:
        if param_name in required:
            return None, "Parameter is required"
        return schema.get("default"), None
    
    param_type = schema.get("type", "string")
    
    # Type validation and conversion following OpenAI spec
    if param_type == "string":
        str_val = str(value)
        if "enum" in schema and str_val not in schema["enum"]:
            return None, f"Value must be one of {schema['enum']}"
        return str_val, None
        
    elif param_type == "integer":
        try:
            int_val = int(value)
            if "minimum" in schema and int_val < schema["minimum"]:
                return None, f"Value must be >= {schema['minimum']}"
            if "maximum" in schema and int_val > schema["maximum"]:
                return None, f"Value must be <= {schema['maximum']}"
            if "enum" in schema and int_val not in schema["enum"]:
                return None, f"Value must be one of {schema['enum']}"
            return int_val, None
        except (ValueError, TypeError):
            return None, "Must be an integer"
            
    elif param_type == "number":
        try:
            float_val = float(value)
            if "minimum" in schema and float_val < schema["minimum"]:
                return None, f"Value must be >= {schema['minimum']}"
            if "maximum" in schema and float_val > schema["maximum"]:
                return None, f"Value must be <= {schema['maximum']}"
            return float_val, None
        except (ValueError, TypeError):
            return None, "Must be a number"
            
    elif param_type == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True, None
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False, None
        return None, "Must be a boolean"
        
    elif param_type == "array":
        if not isinstance(value, list):
            return None, "Must be an array"
        return value, None
        
    elif param_type == "object":
        if not isinstance(value, dict):
            return None, "Must be an object"
        return value, None
    
    return value, None


class Tool(ParameterAccessMixin, ABC):
    """Base class for all tools that can be used by the ReAct Agent."""
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 description: Optional[str] = None, 
                 parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                 required: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None,
                 agent_name: Optional[str] = None,
                 timeout: Optional[float] = None):
        
        super().__init__()
        
        # Check for decorator-defined metadata
        if hasattr(self, '_execute') and hasattr(self._execute, '_parameters'):
            # Use decorator-defined parameters
            parsed_params = self._execute._parameters
            parsed_required = getattr(self._execute, '_required', [])
        else:
            parsed_params, parsed_required = {}, []
        
        # Use class-level decorator metadata if available
        class_name = getattr(self.__class__, '_tool_name', None)
        class_desc = getattr(self.__class__, '_tool_description', None)
        
        # Priority: constructor args > decorator metadata > defaults
        self.name = name or class_name or self.__class__.__name__.lower().replace('tool', '')
        self.description = description or class_desc or "A tool for the ReAct agent"
        self.parameters = parameters or parsed_params
        self.required = required or parsed_required
        
        # Logging configuration
        self.agent_name = agent_name or "UnknownAgent"
        self.logger = logger  # Use provided logger or None
        self.enable_logging = logger is not None
        
        # Timeout configuration
        if timeout is not None:
            self.timeout = timeout
            self._custom_timeout_set = True  # Mark as having custom timeout
        else:
            self.timeout = 30.0  # Default timeout
            self._custom_timeout_set = False
        
        # Agent reference for accessing bound workspace
        self.agent = None  # Will be set when tool is bound to agent
    
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
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's schema for the LLM to understand how to use it."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }
    
    async def execute(self, **kwargs) -> str:
        """Execute the tool with parameter validation (async)."""
        start_time = datetime.now()
        
        # Log execution start
        self._log_info(f"🔧 Tool execution started")
        self._log_info(f"   Parameters: {kwargs}")
        
        # Get parameter definitions from the _execute method
        parameters = getattr(self._execute, '_parameters', {})
        required = getattr(self._execute, '_required', [])
        
        # Validate parameters
        validated_params = {}
        
        # Check required parameters
        for param_name in required:
            if param_name not in kwargs:
                error_msg = f"Missing required parameter '{param_name}'"
                self._log_error(f"Parameter validation failed: {error_msg}")
                return f"Error: {error_msg}"
        
        # Validate each parameter
        for param_name, param_schema in parameters.items():
            value = kwargs.get(param_name)
            
            # Skip if not provided and not required
            if value is None and param_name not in required:
                if 'default' in param_schema:
                    validated_params[param_name] = param_schema['default']
                continue
                
            validated_value, error = _validate_parameter_value(param_name, value, param_schema, required)
            if error:
                error_msg = f"Parameter '{param_name}': {error}"
                self._log_error(f"Parameter validation failed: {error_msg}")
                return f"Error: {error_msg}"
                
            validated_params[param_name] = validated_value
        
        # Inject validated parameters into the instance
        self._current_params = validated_params
        
        try:
            # Call the _execute method with timeout
            result = await asyncio.wait_for(self._execute(), timeout=self.timeout)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result_str = str(result)
            
            # Log successful completion
            self._log_info(f"✅ Tool execution completed")
            self._log_info(f"   Time: {execution_time:.2f}s")
            self._log_info(f"   Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            timeout_msg = f"Tool execution timed out after {self.timeout}s"
            self._log_error(timeout_msg)
            self._log_error(f"   Time: {execution_time:.2f}s")
            return f"Error: {timeout_msg}"
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_error(f"Tool execution failed: {str(e)}")
            self._log_error(f"   Time: {execution_time:.2f}s")
            raise
    
    @abstractmethod
    async def _execute(self) -> str:
        """
        Implement the actual tool logic. Override this method in subclasses.
        MUST be async - all tools are required to be asynchronous.
        
        Use self.get_param("param_name") to access parameters inside this method.
        Parameters are automatically validated before this method is called.
        
        Example:
        
        async def _execute(self) -> str:
            # Even simple operations should be async
            value = self.get_param("value")
            # For I/O operations:
            # result = await some_async_operation()
            return f"processed: {value}"
        """
        pass