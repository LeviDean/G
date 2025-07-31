#!/usr/bin/env python3
"""
Interactive File Agent Demo

A simple interactive file management assistant that can read, write, and edit files
safely within a workspace.
"""

import os
import sys
import asyncio
import tempfile

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import ReadTool, WriteTool, EditTool, ShellTool

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-sonnet-4"


async def main():
    """Interactive File Agent."""
    print("📁 INTERACTIVE FILE AGENT")
    print("=" * 50)
    print("Your personal file management assistant!")
    print("Commands: 'quit' to exit, 'reset' for new conversation")
    print()
    
    # Get workspace path from user
    try:
        workspace_path = input("Enter workspace directory (or press Enter for temp directory): ").strip()
        if not workspace_path:
            workspace_path = tempfile.mkdtemp(prefix="file_agent_")
            print(f"📁 Using temporary workspace: {workspace_path}")
    except EOFError:
        workspace_path = tempfile.mkdtemp(prefix="file_agent_")
        print(f"📁 Using temporary workspace: {workspace_path}")
    
    print()
    
    # Create the agent
    agent = ReActAgent(
        system_prompt="""You are a helpful file management assistant. You can read, write, and edit files safely within a workspace.

All file operations are automatically constrained to the workspace for security.

Guidelines:
- Use relative paths (e.g., "config.txt", "src/main.py")
- Ask for confirmation before overwriting important files
- Show file contents when requested
- Explain what operations you're performing
- Be helpful with file organization and editing tasks

Available tools: read, write, edit""",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1
    )
    
    # Bind workspace and tools
    agent.bind_workspace(workspace_path)
    agent.bind_tools([ReadTool(), WriteTool(), EditTool(), ShellTool()])
    
    print("🤖 File agent ready! What would you like to do with files?")
    print("💡 Examples:")
    print("   • Create a new Python script")
    print("   • Read a configuration file") 
    print("   • Edit an existing document")
    print("   • Create project structure with multiple files")
    print()
    
    conversation_started = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the file agent! Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                agent.reset()
                conversation_started = False
                print("🔄 Conversation reset. Starting fresh!")
                continue
            
            print("\nFile Agent: ", end="", flush=True)
            
            if not conversation_started:
                result = await agent.run(user_input)
                conversation_started = True
            else:
                result = await agent.chat(user_input)
            
            print(result)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the file agent! Goodbye!")
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