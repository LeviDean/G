#!/usr/bin/env python3
"""
Interactive Desk Demo

Shows how to use the desk interface - a simple way for users to interact
with agents where the desk automatically manages the session.
"""

import os
import sys
import asyncio

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.core.hierarchical_logger import get_hierarchical_logger
from react_agent.tools import ReadTool, WriteTool, EditTool, CalculatorTool, SubAgentDispatchTool, ShellTool
from react_agent.core.desk import Desk

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-sonnet-4"


async def main():
    """Interactive Desk Demo."""
    print("🏢 INTERACTIVE DESK DEMO")
    print("=" * 50)
    print("Simple desk interface - just sit down and chat!")
    print("Commands: '/help' for command list, '/quit' to exit")
    print("          '/reset' to clear history, '/stats' for session statistics")
    print("          '/details' for agent's detailed history, '/export' to export conversation")
    print()
    
    logger = get_hierarchical_logger("demo", hierarchy_level=0, enable_logging=True)
    
    # Create agent
    agent = ReActAgent(
        system_prompt="""You are a helpful assistant. You can help with calculations, file operations, and general questions.

Available tools: calculator, read, write, edit

Be helpful and conversational.""",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        logger=logger
    )
    
    # Bind tools and workspace
    agent.bind_tools(
        [
            CalculatorTool(), 
            ReadTool(), 
            WriteTool(), 
            EditTool(), 
            SubAgentDispatchTool(), 
            ShellTool()
        ]
    )
    agent.bind_workspace("./tmp")
    
    # Create desk - this automatically handles session management
    desk = Desk(agent)
    
    print(f"🏢 Desk ready! Session: {desk.session_id}")
    print("💡 Examples:")
    print("   • Ask questions and build on previous answers")
    print("   • Calculate something, then ask about the result")
    print("   • Create files and refer to them later")
    print("   • Watch the agent think and work in real-time!")
    print("   • The desk remembers everything automatically")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/help', 'help']:
                print("\n📋 Available Commands:")
                print("   /help      - Show this help message")
                print("   /quit      - Exit the session")  
                print("   /reset     - Clear conversation history")
                print("   /stats     - Show session statistics")
                print("   /details   - Show agent's detailed history")
                print("   /export    - Export conversation to markdown")
                print("\n💡 Or just type any message to chat with the agent!")
                print()
                continue
                
            if user_input.lower() in ['/quit', '/exit', '/q', 'quit', 'exit', 'q']:
                stats = desk.get_session_stats()
                print(f"\n👋 Session ended! Total messages: {stats['total_messages']}")
                break
                
            if user_input.lower() in ['/reset', 'reset']:
                desk.reset_conversation()
                print("🔄 Conversation history cleared!")
                continue
                
            if user_input.lower() in ['/stats', 'stats']:
                stats = desk.get_session_stats()
                print(f"\n📊 Session Statistics:")
                print(f"   Duration: {stats['duration_minutes']:.1f} minutes")
                print(f"   Total messages: {stats['total_messages']}")
                print(f"   User messages: {stats['user_messages']}")
                print(f"   Agent messages: {stats['agent_messages']}")
                if 'agent_details' in stats:
                    agent_details = stats['agent_details']
                    print("   Agent Details:")
                    print(f"     • Tool calls: {agent_details.get('total_tool_calls', 0)}")
                    print(f"     • Interactions: {agent_details.get('total_interactions', 0)}")
                    print(f"     • Total tokens: {agent_details.get('total_tokens', 0)}")
                print()
                continue
                
            if user_input.lower() in ['/details', 'details']:
                # Show both session conversation and agent detailed history
                session_history = desk.get_conversation_history()
                detailed_history = desk.get_agent_detailed_history()
                
                print(f"\n🏢 Session Conversation History ({len(session_history)} messages):")
                print("-" * 60)
                for i, message in enumerate(session_history[-5:], 1):  # Show last 5 session messages
                    timestamp = message.timestamp.strftime("%H:%M:%S")
                    role = message.role.value.upper()
                    content = message.content
                    if len(content) > 150:
                        content = content[:147] + "..."
                    print(f"{i}. [{timestamp}] {role}: {content}")
                if len(session_history) > 5:
                    print(f"... showing last 5 of {len(session_history)} total messages")
                print()
                
                print(f"🤖 Agent Detailed History ({len(detailed_history)} entries):")
                print("-" * 60)
                for i, entry in enumerate(detailed_history[-8:], 1):  # Show last 8 detailed entries
                    timestamp = entry.get('timestamp', 'Unknown')
                    if 'T' in timestamp:  # ISO format
                        timestamp = timestamp.split('T')[1][:8]  # Just time part
                    entry_type = entry.get('type', 'Unknown')
                    content = entry.get('content', {})
                    print(f"{i}. [{timestamp}] {entry_type.upper()}")
                    
                    if entry_type == 'user_message':
                        message = content.get('message', '')
                        operation = content.get('operation', '')
                        if len(message) > 100:
                            message = message[:97] + "..."
                        print(f"   Message: {message}")
                        print(f"   Operation: {operation}")
                    elif entry_type == 'tool_call':
                        tool_name = content.get('tool_name', 'Unknown')
                        args = content.get('args', {})
                        print(f"   Tool: {tool_name}")
                        if args:
                            print(f"   Parameters: {args}")
                    elif entry_type == 'tool_result':
                        tool_name = content.get('tool_name', 'Unknown')
                        result = content.get('result', '')
                        success = content.get('success', False)
                        execution_time = content.get('execution_time', 0)
                        # Truncate long results
                        if len(str(result)) > 150:
                            result = str(result)[:147] + "..."
                        print(f"   Tool: {tool_name} | Success: {success} | Time: {execution_time:.3f}s")
                        print(f"   Result: {result}")
                    elif entry_type == 'agent_response':
                        response = content.get('response', '')
                        iterations = content.get('iterations', 0)
                        operation = content.get('operation', '')
                        execution_time = content.get('execution_time', 0)
                        if len(str(response)) > 150:
                            response = str(response)[:147] + "..."
                        print(f"   Response: {response}")
                        print(f"   Iterations: {iterations} | Operation: {operation} | Time: {execution_time:.3f}s")
                    print()
                    
                if len(detailed_history) > 8:
                    print(f"... showing last 8 of {len(detailed_history)} total entries")
                print("-" * 60)
                continue
                
            if user_input.lower() in ['/export', 'export']:
                conversation = desk.export_conversation("markdown")
                print(f"\n📄 Conversation Export:")
                print("-" * 50)
                print(conversation)
                print("-" * 50)
                continue
            
            # Chat through the desk with streaming - session is handled automatically
            print("\nAgent: ", end="", flush=True)
            
            try:
                final_response = ""
                async for update in desk.chat_stream(user_input):
                    if update["type"] == "status":
                        # Show status updates in dim color
                        print(f"\n   {update['content']}", flush=True)
                    elif update["type"] == "thinking":
                        # Stream the agent's thinking in real-time
                        print(update["content"], end="", flush=True)
                    elif update["type"] == "tool_call":
                        # Show tool calls
                        print(f"\n   {update['content']}", flush=True)
                    elif update["type"] == "tool_result":
                        # Show tool results
                        print(f"   {update['content']}", flush=True)
                        print("   ", end="", flush=True)  # Indent for continued response
                    elif update["type"] == "final":
                        # Final response - might be different from thinking if tools were used
                        if not final_response:  # Only print if we haven't streamed thinking
                            print(update["content"], end="", flush=True)
                        final_response = update["content"]
                    elif update["type"] == "error":
                        print(f"\n❌ {update['content']}", flush=True)
                        
                print()  # New line after streaming is complete
                print()  # Extra line for spacing
                        
            except Exception as e:
                print(f"\n❌ Streaming error: {e}")
                # Fallback to non-streaming
                response = await desk.chat(user_input)
                print(response)
                print()
            
        except KeyboardInterrupt:
            stats = desk.get_session_stats()
            print(f"\n\n👋 Session interrupted! Total messages: {stats['total_messages']}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try again!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your API configuration and try again.")