#!/usr/bin/env python3
"""
Interactive Math Chat Agent

A conversational agent that can:
1. Chat naturally with users about mathematical problems
2. Automatically detect when complex equations need parallel solving
3. Break down complex problems into sub-tasks when beneficial
4. Provide step-by-step explanations
5. Handle follow-up questions and maintain conversation context

The agent intelligently decides when to use parallel dispatch for performance.
"""

import os
import sys
import asyncio
import time
import re
from typing import Dict, List, Optional

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import CalculatorTool, SubAgentDispatchTool


base_url = "https://openrouter.ai/api/v1"
api_key = "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" 
model_name = "anthropic/claude-sonnet-4"


class MathChatAgent:
    """
    An intelligent math chat agent that can converse with users and solve
    mathematical problems using parallel computation when beneficial.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str, enable_logging: bool = False):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.enable_logging = enable_logging
        
        # Create the main chat agent
        self.chat_agent = ReActAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.3,  # Slightly higher for natural conversation
            system_prompt=self._create_system_prompt(),
            verbose=False,
            enable_logging=enable_logging,
            agent_name="MathChatAgent"
        )
        
        # Create specialized calculation agents for parallel dispatch
        self.calc_agents = []
        for i in range(4):
            agent = ReActAgent(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.1,  # Lower temperature for precision
                system_prompt=f"You are a precision calculator #{i+1}. Provide exact numerical results.",
                verbose=False,
                enable_logging=enable_logging,
                agent_name=f"Calculator_{i+1}"
            )
            agent.bind_tool(CalculatorTool())
            self.calc_agents.append(agent)
        
        # Bind tools to the main chat agent
        self.chat_agent.bind_tools([
            CalculatorTool(),
            SubAgentDispatchTool(
                main_agent=self.chat_agent,
                max_concurrent_agents=8  # Increased to handle more complex equations
            )
        ])
        
        # Update dispatch tool to use pre-configured agents
        dispatch_tool = self.chat_agent.tools['dispatch']
        dispatch_tool.calc_agents = self.calc_agents
        
        print("🧮 Math Chat Agent initialized!")
        if enable_logging:
            print("🔍 Detailed logging enabled")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the math chat agent."""
        return """You are a friendly and intelligent math tutor assistant. You can:

1. **Chat naturally** about mathematical concepts and problems
2. **Solve equations** step by step with clear explanations  
3. **Use parallel computation** for complex problems with multiple sub-calculations
4. **Provide educational insights** about mathematical concepts
5. **Handle follow-up questions** and maintain conversation context

## When to Use Tools:

**Use the calculator tool for:**
- Simple, single calculations
- Step-by-step verification of your work
- Any numerical computation

**Use the dispatch tool for complex problems that have:**
- Multiple independent sub-calculations (like: (15*27 + sqrt(144)) / (8-3) + log(100))
- Several parts that can be computed in parallel
- Complex expressions with nested operations

## Conversation Style:
- Be friendly and encouraging
- Explain your reasoning clearly
- Ask clarifying questions when needed
- Provide step-by-step solutions
- Use emojis appropriately to make math fun! 😊

## Example Dispatch Usage:
When you see a complex expression like "((15 * 27 + sqrt(144)) / (8 - 3)) + (2^8 / 24)", 
you can use the dispatch tool with tasks like:
- "Calculate exactly: 15 * 27 + sqrt(144)"  
- "Calculate exactly: 8 - 3"
- "Calculate exactly: 2^8"

The dispatch tool will return the individual results for each task, which you can then use 
to compute the final answer. Always use the "results" format (default) so you get 
clear task results you can work with.

Remember: Always explain what you're doing and why!"""

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The agent's response
        """
        if not user_message.strip():
            return "I'm here to help with math! What would you like to work on? 😊"
        
        try:
            # Use the chat method to maintain conversation context
            response = await self.chat_agent.chat(user_message)
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Could you please try rephrasing your question?"
    
    async def start_new_conversation(self, user_message: str) -> str:
        """
        Start a new conversation (resets context).
        
        Args:
            user_message: The user's first message
            
        Returns:
            The agent's response
        """
        try:
            # Reset the agent's conversation history
            self.chat_agent.reset()
            
            # Start with the run method for new conversations
            response = await self.chat_agent.run(user_message)
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Could you please try rephrasing your question?"
    
    def get_conversation_stats(self) -> Dict[str, any]:
        """Get statistics about the current conversation."""
        history = self.chat_agent.get_conversation_history()
        token_usage = self.chat_agent.get_token_usage()
        
        return {
            "messages": len(history),
            "tokens_used": token_usage,
            "tools_available": list(self.chat_agent.tools.keys()),
            "agent_operations": getattr(self.chat_agent, 'operation_count', 0)
        }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.chat_agent.reset()
        print("🔄 Conversation history cleared. Starting fresh!")


async def interactive_chat():
    """Run the interactive chat interface."""
    print("🧮 INTERACTIVE MATH CHAT AGENT")
    print("="*60)
    print("Hi! I'm your friendly math tutor assistant! 😊")
    print("I can help you solve equations, explain concepts, and even")
    print("break down complex problems using parallel computation!")
    print()
    print("Commands:")
    print("  • Type your math questions naturally")
    print("  • 'stats' - Show conversation statistics") 
    print("  • 'reset' - Start a new conversation")
    print("  • 'logging on/off' - Toggle detailed logging")
    print("  • 'quit' - Exit the chat")
    print("="*60)
    
    # Ask about logging preference
    enable_logs = input("Enable detailed agent logging? (y/N): ").strip().lower() in ['y', 'yes']
    print()
    
    # Create the math chat agent
    agent = MathChatAgent(api_key, base_url, model_name, enable_logging=enable_logs)
    
    print("Let's start! What math problem would you like to explore? 🤔")
    print()
    
    conversation_started = False
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\nMath Agent: Thanks for chatting! Keep exploring math! 🎓✨")
                break
            
            elif user_input.lower() == 'stats':
                stats = agent.get_conversation_stats()
                print(f"\n📊 Conversation Stats:")
                print(f"   Messages: {stats['messages']}")
                print(f"   Tokens used: {stats['tokens_used']}")
                print(f"   Operations: {stats['agent_operations']}")
                print(f"   Available tools: {', '.join(stats['tools_available'])}")
                print()
                continue
            
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                conversation_started = False
                continue
            
            elif user_input.lower().startswith('logging '):
                if 'on' in user_input.lower():
                    agent.enable_logging = True
                    agent.chat_agent.enable_logging = True
                    print("🔍 Detailed logging enabled")
                else:
                    agent.enable_logging = False
                    agent.chat_agent.enable_logging = False
                    print("🔇 Logging disabled")
                continue
            
            # Process the math question
            print("\nMath Agent: ", end="", flush=True)
            
            start_time = time.time()
            
            if not conversation_started:
                # First message in conversation
                response = await agent.start_new_conversation(user_input)
                conversation_started = True
            else:
                # Continue existing conversation
                response = await agent.chat(user_input)
            
            response_time = time.time() - start_time
            
            print(response)
            
            # Show response time for performance awareness
            if response_time > 1.0:
                print(f"\n⏱️  Response time: {response_time:.1f}s")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nMath Agent: Thanks for chatting! Keep exploring math! 🎓✨")
            break
        except Exception as e:
            print(f"\nMath Agent: I encountered an error: {e}")
            print("Let's try again! 😊\n")


async def demo_conversation():
    """Demonstrate the chat agent with sample conversations."""
    print("🎭 DEMO: Math Chat Agent Conversation")
    print("="*50)
    print("This demo shows how the agent handles different types of math problems.")
    print()
    
    # Create agent with logging for demo
    agent = MathChatAgent(api_key, base_url, model_name, enable_logging=True)
    
    # Sample conversation flow
    demo_messages = [
        "Hi! Can you help me with some math problems?",
        "What's 15 * 27?", 
        "Now can you solve this complex equation: ((15 * 27 + sqrt(144)) / (8 - 3)) + (2^8 / 24)",
        "Can you explain how you solved that so quickly?",
        "What about this system: 2x + 3y = 7 and x - y = 1. Can you solve for x and y?"
    ]
    
    print("🤖 Starting demo conversation...\n")
    
    for i, message in enumerate(demo_messages, 1):
        print(f"👤 User: {message}")
        print("🧮 Math Agent: ", end="", flush=True)
        
        try:
            if i == 1:
                response = await agent.start_new_conversation(message)
            else:
                response = await agent.chat(message)
            
            print(response)
            print("\n" + "-"*50 + "\n")
            
            # Small delay between messages for readability
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Show final stats
    stats = agent.get_conversation_stats()
    print("📊 Final Conversation Stats:")
    print(f"   Total messages: {stats['messages']}")
    print(f"   Tokens used: {stats['tokens_used']}")
    print(f"   Agent operations: {stats['agent_operations']}")


async def main():
    """Main function to choose demo mode."""
    print("🧮 MATH CHAT AGENT")
    print("="*50)
    print("An intelligent conversational agent for solving math problems!")
    print()
    print("Choose mode:")
    print("1. Interactive chat (default)")
    print("2. Demo conversation")
    
    choice = input("\nEnter choice (1-2) or press Enter for default: ").strip()
    
    if choice == "2":
        await demo_conversation()
    else:
        await interactive_chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy calculating! 🧮✨")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your API configuration and try again.")