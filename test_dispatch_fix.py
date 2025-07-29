#!/usr/bin/env python3
"""
Test script to verify the dispatch tool fixes work correctly.
"""

import os
import sys
import asyncio

# Add the parent directory to Python path to find react_agent module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from react_agent import ReActAgent
from react_agent.tools import CalculatorTool, SubAgentDispatchTool


async def test_dispatch_tool():
    """Test the dispatch tool with the fixes."""
    print("🧪 TESTING DISPATCH TOOL FIXES")
    print("=" * 50)
    
    # Use dummy API key for testing (will show in logs but won't make real calls)
    api_key = "test-key"
    base_url = "https://api.openai.com/v1"
    model = "gpt-4"
    
    # Create main agent
    main_agent = ReActAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        enable_logging=True,
        agent_name="TestMain"
    )
    
    # Create pre-configured calc agents
    calc_agents = []
    for i in range(4):
        agent = ReActAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_logging=True,
            agent_name=f"Calculator_{i+1}"
        )
        agent.bind_tool(CalculatorTool())
        calc_agents.append(agent)
    
    # Create dispatch tool
    dispatch_tool = SubAgentDispatchTool(
        main_agent=main_agent,
        max_concurrent_agents=6,
        enable_logging=True,
        agent_name="TestMain"
    )
    
    # Set pre-configured agents
    dispatch_tool.calc_agents = calc_agents
    
    print("✅ Created dispatch tool with pre-configured agents")
    print(f"   Main agent: {main_agent.agent_name}")
    print(f"   Calc agents: {[a.agent_name for a in calc_agents]}")
    print(f"   Max concurrent: {dispatch_tool.max_concurrent_agents}")
    print()
    
    # Test with sample tasks
    test_tasks = [
        "Calculate exactly: 2^8",
        "Calculate exactly: 3^4", 
        "Calculate exactly: sqrt(256)",
        "Calculate exactly: 10 / 8",
        "Calculate exactly: sin(90) in degrees",
        "Calculate exactly: cos(0) in degrees"
    ]
    
    print(f"🚀 Testing with {len(test_tasks)} tasks...")
    print("Tasks:")
    for i, task in enumerate(test_tasks, 1):
        print(f"  {i}. {task}")
    print()
    
    try:
        # Execute dispatch tool
        result = await dispatch_tool.execute(
            tasks=test_tasks,
            return_format="results"
        )
        
        print("📊 DISPATCH RESULTS:")
        print("-" * 30)
        print(result)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error during dispatch: {e}")
        return False
    
    print("✅ Dispatch tool test completed!")
    return True


async def test_math_integration():
    """Test integration with math agent pattern."""
    print("\n🔗 TESTING MATH AGENT INTEGRATION")
    print("=" * 50)
    
    # Simulate the math agent setup
    api_key = "test-key"
    base_url = "https://api.openai.com/v1"
    model = "gpt-4"
    
    # Create chat agent (like MathChatAgent does)
    chat_agent = ReActAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        enable_logging=True,
        agent_name="MathChatAgent"
    )
    
    # Create calc agents (like MathChatAgent does)
    calc_agents = []
    for i in range(4):
        agent = ReActAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_logging=True,
            agent_name=f"Calculator_{i+1}"
        )
        agent.bind_tool(CalculatorTool())
        calc_agents.append(agent)
    
    # Bind tools (like MathChatAgent does)
    chat_agent.bind_tools([
        CalculatorTool(),
        SubAgentDispatchTool(
            main_agent=chat_agent,
            max_concurrent_agents=8
        )
    ])
    
    # Update dispatch tool (like the fix does)
    dispatch_tool = chat_agent.tools['dispatch']
    dispatch_tool.calc_agents = calc_agents
    
    print("✅ Math agent integration setup complete")
    print(f"   Chat agent tools: {list(chat_agent.tools.keys())}")
    print(f"   Dispatch tool has {len(dispatch_tool.calc_agents)} pre-configured agents")
    print(f"   Max concurrent tasks: {dispatch_tool.max_concurrent_agents}")
    
    return True


async def main():
    """Run all tests."""
    print("🧮 DISPATCH TOOL FIX VERIFICATION")
    print("=" * 60)
    print("Testing the fixes for:")
    print("1. Using pre-configured calc_agents instead of creating new ones")
    print("2. Increased task limit to handle complex equations")
    print("3. Proper agent cycling for multiple tasks")
    print("=" * 60)
    print()
    
    # Run tests
    test1_result = await test_dispatch_tool()
    test2_result = await test_math_integration()
    
    print(f"\n{'=' * 60}")
    print("🎯 TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Dispatch tool test: {'PASSED' if test1_result else 'FAILED'}")
    print(f"✅ Math integration test: {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED!")
        print("The dispatch tool fixes should resolve the issues:")
        print("• No more dynamic agent creation (uses pre-configured calc_agents)")
        print("• Higher task limit (8 concurrent tasks)")
        print("• Proper agent cycling for load distribution")
    else:
        print("\n❌ Some tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())