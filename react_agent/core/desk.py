"""
Desk - Simple Interface Between User and Agent

A "desk" where users interact with agents. The session records all conversations
automatically in a simple, unified manner.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .agent import ReActAgent


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class Message:
    """A message in the conversation."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime


class Desk:
    """
    A simple desk interface where users interact with agents.
    The desk automatically records all conversations as part of the session.
    """
    
    def __init__(self, agent: ReActAgent, session_id: Optional[str] = None):
        self.agent = agent
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Simple conversation history
        self.messages: List[Message] = []
        
        # Session metadata
        self.user_name = "User"
        self.agent_name = getattr(agent, 'agent_name', 'Agent')
        
    async def chat(self, user_message: str) -> str:
        """
        User sends a message to the agent through the desk.
        Session automatically records the conversation.
        """
        # Record user message
        self._add_message(MessageRole.USER, user_message)
        
        # Update agent's conversation history from our session
        self.agent.conversation_history = self._get_conversation_for_llm()
        
        try:
            # Use the unified execute method - it auto-detects new vs continuing conversation
            response = await self.agent.execute(user_message)
            
            # Record agent response
            self._add_message(MessageRole.AGENT, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Record error
            self._add_message(MessageRole.AGENT, error_msg)
            return error_msg
    
    async def chat_stream(self, user_message: str):
        """
        User sends a message to the agent through the desk with streaming response.
        Session automatically records the conversation.
        """
        # Record user message
        self._add_message(MessageRole.USER, user_message)
        
        # Update agent's conversation history from our session
        self.agent.conversation_history = self._get_conversation_for_llm()
        
        full_response = ""
        
        try:
            # Use the unified stream method - it auto-detects new vs continuing conversation
            async for update in self.agent.stream(user_message):
                # Yield the update to the caller
                yield update
                
                # Collect thinking and final content
                if update["type"] == "thinking":
                    full_response += update["content"]
                elif update["type"] == "final":
                    full_response = update["content"]  # Final response overrides collected thinking
            
            # Record agent response
            if full_response:
                self._add_message(MessageRole.AGENT, full_response)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Record error
            self._add_message(MessageRole.AGENT, error_msg)
            yield {"type": "error", "content": error_msg}
    
    def reset_conversation(self):
        """Clear the conversation history."""
        self.messages = []
        self.agent.reset()
        self.updated_at = datetime.now()
    
    def get_conversation_history(self) -> List[Message]:
        """Get the conversation history."""
        return self.messages.copy()
    
    def get_agent_detailed_history(self) -> List[Dict[str, Any]]:
        """Get the agent's detailed interaction history including tool calls and reasoning."""
        return self.agent.get_detailed_history()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        user_messages = [m for m in self.messages if m.role == MessageRole.USER]
        agent_messages = [m for m in self.messages if m.role == MessageRole.AGENT]
        
        # Get agent's detailed statistics
        agent_stats = self.agent.get_statistics()
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_minutes": (self.updated_at - self.created_at).total_seconds() / 60,
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "agent_messages": len(agent_messages),
            "user_name": self.user_name,
            "agent_name": self.agent_name,
            "agent_details": agent_stats
        }
    
    def export_conversation(self, format: str = "markdown") -> str:
        """Export the conversation."""
        if format == "markdown":
            lines = [
                f"# Conversation Session",
                f"Session ID: {self.session_id}",
                f"Date: {self.created_at}",
                ""
            ]
            
            for message in self.messages:
                if message.role == MessageRole.SYSTEM:
                    lines.append(f"*System: {message.content}*")
                elif message.role == MessageRole.USER:
                    lines.append(f"**{self.user_name}**: {message.content}")
                elif message.role == MessageRole.AGENT:
                    lines.append(f"**{self.agent_name}**: {message.content}")
                lines.append("")
                
            return "\n".join(lines)
            
        elif format == "json":
            import json
            data = {
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "user_name": self.user_name,
                "agent_name": self.agent_name,
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role.value,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat()
                    } for m in self.messages
                ]
            }
            return json.dumps(data, indent=2)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _add_message(self, role: MessageRole, content: str) -> Message:
        """Add a message to the conversation."""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def _get_conversation_for_llm(self) -> List[Dict[str, Any]]:
        """Get conversation history formatted for LLM (OpenAI format)."""
        formatted = []
        for message in self.messages:
            if message.role == MessageRole.SYSTEM:
                formatted.append({"role": "system", "content": message.content})
            elif message.role == MessageRole.USER:
                formatted.append({"role": "user", "content": message.content})
            elif message.role == MessageRole.AGENT:
                formatted.append({"role": "assistant", "content": message.content})
        return formatted