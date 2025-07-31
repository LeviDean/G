#!/usr/bin/env python3
"""
Interactive Multi-Project Agent Demo

A development assistant that can switch between different project workspaces
and help you manage files across multiple projects.
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import ReadTool, WriteTool, EditTool, CalculatorTool

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-sonnet-4"


async def main():
    """Interactive Multi-Project Agent."""
    print("🗂️  INTERACTIVE MULTI-PROJECT AGENT")
    print("=" * 50)
    print("Your development assistant for multiple projects!")
    print("Commands: 'quit' to exit, 'reset' for new conversation")
    print("          'switch <project>' to change workspace")
    print()
    
    # Setup project workspaces
    base_dir = tempfile.mkdtemp(prefix="multi_project_")
    projects = {
        "frontend": Path(base_dir) / "frontend",
        "backend": Path(base_dir) / "backend", 
        "shared": Path(base_dir) / "shared"
    }
    
    # Create project directories
    for name, path in projects.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 {name.title()} project: {path}")
    
    print()
    
    # Create the agent
    agent = ReActAgent(
        system_prompt="""You are a helpful development assistant for multiple projects. You can work with files and do calculations.

Current workspace will be shown before each response. All file operations are constrained to the current workspace for security.

Available tools: read, write, edit, calculator

Guidelines:
- Use relative paths (e.g., "src/main.py", "config.json")  
- Help with project organization and file management
- Ask for confirmation before overwriting important files
- Explain what you're doing and which project you're working on""",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1
    )
    
    # Bind tools
    agent.bind_tools([ReadTool(), WriteTool(), EditTool(), CalculatorTool()])
    
    # Start with frontend project
    current_project = "frontend"
    agent.bind_workspace(projects[current_project])
    
    print(f"🤖 Multi-project agent ready! Currently in: {current_project}")
    print("💡 Examples:")
    print("   • Create a React component in frontend")
    print("   • switch backend, then create an API endpoint")
    print("   • Create shared utilities in the shared project")
    print("   • Calculate project metrics")
    print()
    
    conversation_started = False
    
    while True:
        try:
            # Show current project in prompt
            user_input = input(f"You [{current_project}]: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the multi-project agent! Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                agent.reset()
                conversation_started = False
                print("🔄 Conversation reset. Starting fresh!")
                continue
            
            # Handle project switching
            if user_input.lower().startswith('switch '):
                project_name = user_input[7:].strip()
                if project_name in projects:
                    current_project = project_name
                    agent.bind_workspace(projects[current_project])
                    print(f"🔄 Switched to {current_project} project")
                    continue
                else:
                    print(f"❌ Unknown project: {project_name}")
                    print(f"Available projects: {', '.join(projects.keys())}")
                    continue
            
            print(f"\nAgent [{current_project}]: ", end="", flush=True)
            
            if not conversation_started:
                result = await agent.run(user_input)
                conversation_started = True
            else:
                result = await agent.chat(user_input)
            
            print(result)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the multi-project agent! Goodbye!")
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