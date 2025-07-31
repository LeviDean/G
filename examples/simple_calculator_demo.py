#!/usr/bin/env python3
"""
Simple Calculator Agent Demo

A minimal example demonstrating the ReAct Agent with just a calculator tool.
Perfect for testing basic agent functionality and understanding the framework.
"""

import os
import sys
import asyncio

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import CalculatorTool
from react_agent.core.hierarchical_logger import get_hierarchical_logger


api_key = os.getenv("OPENROUTE_CLAUDE_KEY")
base_url= "https://openrouter.ai/api/v1"

model_name = "anthropic/claude-sonnet-4"


async def interactive_calculator():
    """Run an interactive calculator session."""
    print("🧮 INTERACTIVE CALCULATOR MODE")
    print("=" * 50)
    print("Type math questions and I'll solve them!")
    print("Commands: 'quit' to exit, 'reset' for new conversation")
    print()
    
    # Ask about logging
    try:
        enable_logs = input("Enable detailed logging? (y/N): ").strip().lower() in ['y', 'yes']
    except EOFError:
        enable_logs = False
    
    print()
    
    # Create logger if requested
    logger = None
    if enable_logs:
        logger = get_hierarchical_logger("InteractiveCalculator", hierarchy_level=0, enable_logging=True)
        print("🔍 Detailed logging enabled!")
        print()
    
    # Create the agent with explicit system prompt
    agent = ReActAgent(
        system_prompt="""You are a helpful calculator assistant. Solve math problems step by step using the calculator tool. 

For word problems:
1. Identify what needs to be calculated
2. Set up the mathematical approach
3. Use the calculator for each calculation step
4. After getting all needed values, provide a clear final answer
5. IMPORTANT: Once you have solved the problem completely, provide the final answer and stop

Be conversational and explain your reasoning clearly. When you have calculated the final answer, state it clearly and conclude.""",
        agent_name="InteractiveCalculator",
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        logger=logger
    )
    
    # Bind calculator tool
    agent.bind_tool(CalculatorTool())
    
    print("🤖 Calculator ready! What would you like to calculate?")
    print()
    
    conversation_started = False
    
    while True:
        try:
            user_input = input("[You]: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the calculator! Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                agent.reset()
                conversation_started = False
                print("🔄 Conversation reset. Starting fresh!")
                continue
            
            # print("\nCalculator: ", end="", flush=True)
            
            if not conversation_started:
                result = await agent.run(user_input)
                conversation_started = True
            else:
                result = await agent.chat(user_input)
            
            print(f"\n[Calculator]: {result}", flush=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the calculator! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try again!\n")
            
    detailed = agent.get_detailed_history()
    for entry in detailed:
        print(entry)


if __name__ == "__main__":
    print("Interactive calculator")
    print()
    try:
        asyncio.run(interactive_calculator())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your API configuration and try again.")