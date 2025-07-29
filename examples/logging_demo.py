#!/usr/bin/env python3
"""
Demo of the integrated logging system in ReActAgent and Tools.
"""

import os
import sys
import asyncio

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import ReActAgent
from react_agent.tools import CalculatorTool, SubAgentDispatchTool


async def demo_logging():
    """Demonstrate the logging capabilities."""
    print("🔍 AGENT LOGGING DEMONSTRATION")
    print("="*60)
    print("This demo shows detailed logging of agent and tool activities.")
    print("You can see exactly which agent is doing what and when.")
    print()
    
    # Create agents with logging enabled
    print("Creating agents with logging enabled...")
    
    # Main coordinator agent
    coordinator = ReActAgent(
        api_key="test-key",  # Using dummy key for demo
        model="gpt-4",
        temperature=0.1,
        enable_logging=True,  # Enable logging!
        agent_name="Coordinator",  # Give it a clear name
        verbose=False  # Turn off verbose to see logging clearly
    )
    
    # Specialized calculation agents
    calc_agents = []
    for i in range(3):
        agent = ReActAgent(
            api_key="test-key",
            model="gpt-4", 
            temperature=0.1,
            enable_logging=True,  # Enable logging!
            agent_name=f"Calculator_{i+1}",  # Unique names
            verbose=False
        )
        agent.bind_tool(CalculatorTool())
        calc_agents.append(agent)
    
    # Bind tools to coordinator
    coordinator.bind_tools([
        CalculatorTool(),
        SubAgentDispatchTool(max_concurrent_agents=3)
    ])
    
    print("\n" + "="*60)
    print("🚀 STARTING LOGGED OPERATIONS")
    print("="*60)
    print("Watch the detailed logs below to see agent activities...")
    print()
    
    # Test individual calculator
    print("1. Testing individual calculator tool...")
    calc_tool = CalculatorTool(enable_logging=True, agent_name="Coordinator")
    
    try:
        result = await calc_tool.execute(expression="15 + 27")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Expected error (no API key): {e}")
    
    print("\n2. Testing agent with tool binding...")
    # This will show the binding process in logs
    test_agent = ReActAgent(
        api_key="test-key",
        enable_logging=True,
        agent_name="TestAgent",
        verbose=False
    )
    test_agent.bind_tool(CalculatorTool())
    
    print("\n3. Testing parallel dispatch logging...")
    
    # Create dispatch tool with logging
    dispatch_tool = SubAgentDispatchTool(
        max_concurrent_agents=3
    )
    
    # Manually set up parameters to show dispatch logging
    dispatch_tool._current_params = {
        "tasks": [
            "Calculate 10 + 20",
            "Calculate 30 * 40", 
            "Calculate sqrt(144)"
        ],
        "sub_agents": calc_agents,
        "return_format": "summary"
    }
    
    try:
        # This will show detailed logging of parallel execution
        result = await dispatch_tool._execute()
        print(f"   Dispatch result preview: {result[:100]}...")
    except Exception as e:
        print(f"   Expected error (no API key): {e}")
    
    print("\n" + "="*60)
    print("✅ LOGGING DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nWhat you saw in the logs above:")
    print("• Agent initialization with configuration details")
    print("• Tool binding operations")
    print("• Tool execution start/completion with timing")
    print("• Parameter validation and processing")
    print("• Error handling and reporting")
    print("• Parallel operation coordination")
    print("\nTo enable logging in your own code:")
    print("  agent = ReActAgent(..., enable_logging=True, agent_name='MyAgent')")


async def demo_logging_control():
    """Show how to control logging levels and enable/disable."""
    print("\n🎛️  LOGGING CONTROL DEMONSTRATION")
    print("="*50)
    
    # Show different logging configurations
    configs = [
        {"enable_logging": False, "name": "SilentAgent"},
        {"enable_logging": True, "name": "VerboseAgent"},
    ]
    
    for config in configs:
        print(f"\nTesting with {config['name']} (logging: {config['enable_logging']})...")
        
        agent = ReActAgent(
            api_key="test-key",
            enable_logging=config["enable_logging"],
            agent_name=config["name"],
            verbose=False
        )
        
        agent.bind_tool(CalculatorTool())
        print(f"  {config['name']} created and tool bound")
    
    print("\n✅ Logging control demo completed!")
    print("\nYou can see the difference:")
    print("• SilentAgent: No detailed logs")
    print("• VerboseAgent: Full logging details")


async def main():
    """Run logging demonstrations."""
    await demo_logging()
    await demo_logging_control()
    
    print("\n" + "="*60)
    print("🎯 KEY TAKEAWAYS")
    print("="*60)
    print("✅ Logging is now built into ReActAgent and Tool classes")
    print("✅ Enable with: enable_logging=True, agent_name='YourAgent'")
    print("✅ Shows detailed execution flow, timing, and results")
    print("✅ Helps debug complex multi-agent workflows")
    print("✅ Can be toggled on/off as needed")
    print("✅ Each agent gets its own logger namespace")


if __name__ == "__main__":
    asyncio.run(main())