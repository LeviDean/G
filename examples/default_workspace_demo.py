#!/usr/bin/env python3
"""
Default Workspace Demo

Shows how agents automatically create a default workspace under the current directory
when no workspace is explicitly bound.
"""

import os
import sys
import asyncio

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import ReadTool, WriteTool, EditTool

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-sonnet-4"


async def main():
    """Default Workspace Demo."""
    print("📁 DEFAULT WORKSPACE DEMO")
    print("=" * 50)
    print("Agent with automatic default workspace!")
    print("Commands: 'quit' to exit, 'reset' for new conversation")
    print()
    
    # Create agent WITHOUT binding workspace
    agent = ReActAgent(
        system_prompt="""You are a helpful file assistant. You can read, write, and edit files.

A default workspace will be automatically created for security when you use file operations.
All file operations are constrained to this workspace.

Use relative paths (e.g., "notes.txt", "project/main.py") for all file operations.

Available tools: read, write, edit""",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        debug=True  # Show workspace creation
    )
    
    # Bind file tools - NO workspace binding
    agent.bind_tools([ReadTool(), WriteTool(), EditTool()])
    
    print("🤖 Agent ready! Default workspace will be created automatically.")
    print("💡 Examples:")
    print("   • Create a simple text file")
    print("   • Write some code and then read it back")
    print("   • Edit files with improvements")
    print("   • (Watch for default workspace creation message)")
    print()
    
    conversation_started = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the default workspace demo! Goodbye!")
                if hasattr(agent, 'workspace') and agent.workspace and agent._default_workspace_created:
                    print(f"💡 Default workspace created at: {agent.workspace.root_path}")
                break
                
            if user_input.lower() == 'reset':
                agent.reset()
                conversation_started = False
                print("🔄 Conversation reset. Starting fresh!")
                continue
            
            print("\nAgent: ", end="", flush=True)
            
            if not conversation_started:
                result = await agent.run(user_input)
                conversation_started = True
            else:
                result = await agent.chat(user_input)
            
            print(result)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the default workspace demo! Goodbye!")
            if hasattr(agent, 'workspace') and agent.workspace and agent._default_workspace_created:
                print(f"💡 Default workspace created at: {agent.workspace.root_path}")
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