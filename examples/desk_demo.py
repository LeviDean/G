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
    print("Commands: 'quit' to exit, 'reset' to clear history")
    print("          'stats' for session statistics")
    print("          'details' for agent's detailed history")
    print("          'export' to export conversation")
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
    print("   • The desk remembers everything automatically")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                stats = desk.get_session_stats()
                print(f"\n👋 Session ended! Total messages: {stats['total_messages']}")
                break
                
            if user_input.lower() == 'reset':
                desk.reset_conversation()
                print("🔄 Conversation history cleared!")
                continue
                
            if user_input.lower() == 'stats':
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
                
            if user_input.lower() == 'details':
                detailed_history = desk.get_agent_detailed_history()
                print(f"\n🔍 Agent Detailed History ({len(detailed_history)} entries):")
                print("-" * 60)
                for i, entry in enumerate(detailed_history[-10:], 1):  # Show last 10 entries
                    timestamp = entry.get('timestamp', 'Unknown')
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
                        tool_call_id = content.get('tool_call_id', '')
                        print(f"   Tool: {tool_name}")
                        if args:
                            print(f"   Parameters: {args}")
                        if tool_call_id:
                            print(f"   Call ID: {tool_call_id}")
                    elif entry_type == 'tool_result':
                        tool_name = content.get('tool_name', 'Unknown')
                        result = content.get('result', '')
                        success = content.get('success', False)
                        execution_time = content.get('execution_time', 0)
                        # Truncate long results
                        if len(str(result)) > 200:
                            result = str(result)[:197] + "..."
                        print(f"   Tool: {tool_name}")
                        print(f"   Success: {success}")
                        print(f"   Time: {execution_time:.3f}s")
                        print(f"   Result: {result}")
                    elif entry_type == 'agent_response':
                        response = content.get('response', '')
                        iterations = content.get('iterations', 0)
                        operation = content.get('operation', '')
                        execution_time = content.get('execution_time', 0)
                        if len(str(response)) > 200:
                            response = str(response)[:197] + "..."
                        print(f"   Response: {response}")
                        print(f"   Iterations: {iterations}")
                        print(f"   Operation: {operation}")
                        print(f"   Time: {execution_time:.3f}s")
                    print()
                    
                if len(detailed_history) > 10:
                    print(f"... showing last 10 of {len(detailed_history)} total entries")
                print("-" * 60)
                continue
                
            if user_input.lower() == 'export':
                conversation = desk.export_conversation("markdown")
                print(f"\n📄 Conversation Export:")
                print("-" * 50)
                print(conversation)
                print("-" * 50)
                continue
            
            # Chat through the desk - session is handled automatically
            print("\nAgent: ", end="", flush=True)
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