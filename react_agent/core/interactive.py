"""
Interactive Agent System with real-time communication and tool permissions.

This module provides:
- Async message queue system for real-time interaction
- Tool permission management with user prompts
- ESC-based interruption handling
- Real-time status updates during agent execution
"""

import asyncio
import sys
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Simple Chinese character support
import locale
import os

# Set UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass  # Ignore if locale not available


class MessageType(Enum):
    """Types of messages in the interactive system."""
    AGENT_STATUS = "agent_status"          # Gray: 🔄 status updates
    AGENT_THINKING = "agent_thinking"      # Gray: working content  
    AGENT_RESPONSE = "agent_response"      # Normal: final response content
    TOOL_CALL = "tool_call"               # Gray: 🔧 tool calls
    TOOL_RESULT = "tool_result"           # Gray: ✅ tool results


class PermissionChoice(Enum):
    """Tool permission choices."""
    ONCE = "once"
    ALWAYS = "always"
    DENY = "deny"


@dataclass
class Message:
    """A message in the interactive system."""
    type: MessageType
    content: Any
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ToolPermissionManager:
    """Manages tool permissions with interactive prompts."""
    
    def __init__(self):
        self.permissions: Dict[str, str] = {}  # {tool_name: "always"|None}
        self.pending_permission: Optional[str] = None
        
    async def request_permission(self, tool_name: str, message_manager: 'MessageManager') -> PermissionChoice:
        """
        Request permission for a tool with interactive prompt.
        
        Args:
            tool_name: Name of the tool requiring permission
            message_manager: MessageManager instance for communication
            
        Returns:
            PermissionChoice indicating user's decision
        """
        # Check if we already have permission
        if self.permissions.get(tool_name) == "always":
            return PermissionChoice.ALWAYS
            
        # Simple synchronous permission prompt
        print(f"\n❓ Tool '{tool_name}' requires permission:")
        print("  1. Allow once")
        print("  2. Allow always")
        print("  3. Deny (stop task)")
        
        while True:
            try:
                choice = input("Choice (1-3): ").strip()
                
                if choice == "2":  # Always
                    self.permissions[tool_name] = "always"
                    return PermissionChoice.ALWAYS
                elif choice == "1":  # Once
                    return PermissionChoice.ONCE
                elif choice == "3":  # Deny
                    return PermissionChoice.DENY
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                return PermissionChoice.DENY
            except EOFError:
                print("\nInput ended.")
                return PermissionChoice.DENY
    
    def reset_permissions(self):
        """Reset all stored permissions."""
        self.permissions.clear()


# InputHandler removed - was overly complex for simple permission prompts


class MessageManager:
    """Manages bidirectional async communication between user and agent."""
    
    def __init__(self):
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.running = False
        self.interrupted = False
        
    async def send_message(self, msg_type: MessageType, content: Any, metadata: Optional[Dict] = None):
        """Send a message to the output queue."""
        message = Message(
            type=msg_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        await self.output_queue.put(message)
    
    # Input queue removed - simplified to output-only
    
    def set_message_handler(self, msg_type: MessageType, handler: Callable):
        """Set a handler for a specific message type."""
        self.message_handlers[msg_type] = handler
    
    async def start_input_loop(self):
        """Start the async input handling loop."""
        self.running = True
        self.interrupted = False
        
        while self.running:
            try:
                # Process output messages (status updates, etc.)
                try:
                    while True:
                        message = self.output_queue.get_nowait()
                        if message.type in self.message_handlers:
                            await self.message_handlers[message.type](message)
                except asyncio.QueueEmpty:
                    pass
                    
                await asyncio.sleep(0.1)  # Reasonable delay for output processing
                
            except KeyboardInterrupt:
                self.interrupted = True
                await self.send_message(MessageType.USER_INTERRUPT, {"reason": "ctrl_c"})
                break
            except Exception as e:
                await self.send_message(MessageType.ERROR, {"error": str(e)})
    
    def stop(self):
        """Stop the input loop."""
        self.running = False
    
    def is_interrupted(self) -> bool:
        """Check if user has interrupted the process."""
        return self.interrupted
    
    def reset_interrupt(self):
        """Reset the interrupt flag."""
        self.interrupted = False


class InteractiveStatusReporter:
    """Provides real-time status updates during agent execution."""
    
    def __init__(self, message_manager: MessageManager):
        self.message_manager = message_manager
        
    async def report_status(self, status: str, details: Optional[str] = None):
        """Report a status update."""
        await self.message_manager.send_message(
            MessageType.AGENT_STATUS,
            {"status": status, "details": details}
        )
    
    async def report_thinking(self, content: str):
        """Report agent thinking content (working - gray)."""
        await self.message_manager.send_message(
            MessageType.AGENT_THINKING,
            {"content": content}
        )
    
    async def report_response(self, content: str):
        """Report agent response content (final answer - normal color)."""
        await self.message_manager.send_message(
            MessageType.AGENT_RESPONSE,
            {"content": content}
        )
    
    async def report_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Report a tool call."""
        await self.message_manager.send_message(
            MessageType.TOOL_CALL,
            {"tool_name": tool_name, "args": args}
        )
    
    async def report_tool_result(self, tool_name: str, result: str, success: bool = True):
        """Report a tool result."""
        await self.message_manager.send_message(
            MessageType.TOOL_RESULT,
            {"tool_name": tool_name, "result": result, "success": success}
        )
    
# report_final_response removed - redundant with report_response


# Default message handlers for console output
async def default_status_handler(message: Message):
    """Default handler for status messages."""
    content = message.content
    print(f"\n\033[90m🔄 {content['status']}\033[0m", end="", flush=True)
    if content.get('details'):
        print(f"\033[90m - {content['details']}\033[0m", end="", flush=True)


async def default_thinking_handler(message: Message):
    """Default handler for thinking messages (working - gray)."""
    print(f"\033[90m{message.content['content']}\033[0m", end="", flush=True)


async def default_response_handler(message: Message):
    """Default handler for agent response content (normal color)."""
    print(message.content['content'], end="", flush=True)


async def default_tool_call_handler(message: Message):
    """Default handler for tool call messages."""
    content = message.content
    print(f"\n\033[90m🔧 Calling {content['tool_name']}({content['args']})\033[0m", flush=True)


async def default_tool_result_handler(message: Message):
    """Default handler for tool result messages."""
    content = message.content
    symbol = "✅" if content['success'] else "❌"
    result_preview = content['result'][:100] + "..." if len(content['result']) > 100 else content['result']
    print(f"\033[90m{symbol} {result_preview}\033[0m")


def setup_default_handlers(message_manager: MessageManager):
    """Set up default console-based message handlers."""
    message_manager.set_message_handler(MessageType.AGENT_STATUS, default_status_handler)
    message_manager.set_message_handler(MessageType.AGENT_THINKING, default_thinking_handler)
    message_manager.set_message_handler(MessageType.AGENT_RESPONSE, default_response_handler)
    message_manager.set_message_handler(MessageType.TOOL_CALL, default_tool_call_handler)
    message_manager.set_message_handler(MessageType.TOOL_RESULT, default_tool_result_handler)