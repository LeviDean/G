#!/usr/bin/env python3
"""
Interactive Dispatch Demo

Demonstrates the dispatch tool with single-task-per-call pattern.
The main agent can dispatch tasks to specialized sub-agents.
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
from react_agent.core.hierarchical_logger import get_hierarchical_logger
from react_agent.tools import ReadTool, WriteTool, EditTool, CalculatorTool, SubAgentDispatchTool

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model_name = "anthropic/claude-sonnet-4"


async def main():
    """Interactive Dispatch Demo."""
    print("🚀 INTERACTIVE ADAPTIVE DISPATCH DEMO")
    print("=" * 50)
    print("Agent with adaptive dispatch capability!")
    print("Commands: 'quit' to exit, 'reset' for new conversation")
    print()
    
    # Create workspace
    workspace_path = "./tmp"
    print(f"📁 Workspace: {workspace_path}")
    print()
    
    logger = get_hierarchical_logger("demo", hierarchy_level=0, enable_logging=True)
    
    # Create main agent with all tools including dispatch
    main_agent = ReActAgent(
        system_prompt="""You are a helpful assistant with dispatch capability. You can:

1. Handle tasks directly with your tools: read, write, edit, calculator
2. Use the dispatch tool to create specialized sub-agents for complex tasks

The dispatch tool lets you create sub-agents with custom prompts and specific tools.
Use it adaptively when you need specialized processing or want to delegate work.

Example dispatch usage:
- dispatch(task="solve complex math", agent_prompt="You are a math expert", tools=["calculator"])
- dispatch(task="analyze code", agent_prompt="You are a code reviewer", tools=["read"])

Available tools: read, write, edit, calculator, dispatch""",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        logger=logger
    )
    
    # Bind workspace and all tools including dispatch
    main_agent.bind_workspace(workspace_path)
    main_agent.bind_tools([
        ReadTool(), 
        WriteTool(), 
        EditTool(), 
        CalculatorTool(),
        SubAgentDispatchTool()
    ])
    
    print("🤖 Agent ready with adaptive dispatch capability!")
    print("💡 Examples:")
    print("   • Calculate complex math and save results")
    print("   • Analyze code files and create reports")
    print("   • Create specialized agents for specific tasks")
    print("   • Handle tasks directly or dispatch as needed")
    print()
    
    conversation_started = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the dispatch demo! Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                main_agent.reset()
                conversation_started = False
                print("🔄 Conversation reset. Starting fresh!")
                continue
            
            
            
            if not conversation_started:
                result = await main_agent.run(user_input)
                conversation_started = True
            else:
                result = await main_agent.chat(user_input)
            
            print(f"\nAgent: {result}", flush=True)
            
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the dispatch demo! Goodbye!")
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