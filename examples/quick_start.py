#!/usr/bin/env python3
"""
ReAct Agent Quick Start Examples

Demonstrates all three agent architectures with practical examples.
"""

import os
import sys
import asyncio

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import create_simple_agent, create_mcp_agent

# Configuration
api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url = "https://openrouter.ai/api/v1"
model = "anthropic/claude-sonnet-4"


async def demo_simple_agent():
    """Demo: Simple agent with local tools only."""
    print("🚀 SIMPLE AGENT DEMO")
    print("=" * 21)
    print("Local tools: calculator, dispatch")
    
    agent = create_simple_agent(
        "You are a helpful assistant with calculation abilities.",
        api_key=api_key, base_url=base_url, model=model
    )
    
    queries = [
        "Calculate 15 * 23 + 100",
        "What is 2^8?",
        "Calculate the area of a circle with radius 5"
    ]
    
    for query in queries:
        print(f"\n→ {query}")
        response = await agent.execute(query)
        print(f"  {response}")


async def demo_mcp_agent():
    """Demo: MCP agent with file operations."""
    print("\n\n📁 MCP AGENT DEMO")
    print("=" * 18)
    print("Local tools + MCP file operations")
    
    agent = create_mcp_agent(
        "You can do calculations and file operations.",
        mcp_servers=[{
            "name": "filesystem",
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/tmp/demo"]
        }],
        api_key=api_key, base_url=base_url, model=model
    )
    
    queries = [
        "Calculate 25 * 4",
        "Create a file called result.txt with the calculation result",
        "List files in the directory"
    ]
    
    for query in queries:
        print(f"\n→ {query}")
        response = await agent.execute(query)
        print(f"  {response}")


async def interactive_demo():
    """Interactive demo - choose your agent type."""
    print("\n\n🎮 INTERACTIVE MODE")
    print("=" * 20)
    print("Choose agent type:")
    print("1. Simple (calculator only)")
    print("2. MCP (calculator + file operations)")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            agent = create_simple_agent(
                "Simple assistant with calculator", 
                api_key=api_key, base_url=base_url, model=model
            )
        elif choice == "2":
            agent = create_mcp_agent(
                "Assistant with calculator and file operations", 
                mcp_servers=[{"name": "filesystem", "command": "npx", 
                             "args": ["@modelcontextprotocol/server-filesystem", "/tmp/interactive"]}],
                api_key=api_key, base_url=base_url, model=model
            )
        else:
            print("Invalid choice")
            return
        
        print(f"\nAgent ready! Type 'quit' to exit.")
        
        while True:
            query = input("\n→ You: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
                
            try:
                response = await agent.execute(query)
                print(f"  Agent: {response}")
            except Exception as e:
                print(f"  Error: {e}")
                
    except KeyboardInterrupt:
        print("\nExiting...")


async def main():
    """Run all demos."""
    if not api_key:
        print("❌ Set OPENROUTE_CLAUDE_KEY environment variable")
        return
    
    print("🤖 ReAct Agent Quick Start")
    print("=" * 30)
    
    # Run demos
    await demo_simple_agent()
    await demo_mcp_agent()
    
    # Interactive demo
    await interactive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Thanks for trying ReAct Agent!")
    except Exception as e:
        print(f"❌ Error: {e}")