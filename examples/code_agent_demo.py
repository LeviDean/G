#!/usr/bin/env python3
"""
Code Agent Interactive Demo

A Code Agent-like interface for the ReAct Agent Framework.
Provides file operations, code analysis, and task execution.
"""

import os
import sys
import asyncio
import signal
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from react_agent import create_mcp_agent

class CodeAgentDemo:
    def __init__(self):
        self.agent = None
        # Get parent directory for workspace
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.workspace = os.path.abspath(os.path.join(parent_dir, "tmp", "code_workspace"))
        self.setup_workspace()
        
    def setup_workspace(self):
        """Create workspace directory"""
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        print(f"📁 Workspace: {self.workspace}")
    
    async def initialize_agent(self):
        """Initialize agent with MCP file operations"""
        api_key = os.getenv("OPENROUTE_CLAUDE_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENROUTE_CLAUDE_KEY or OPENAI_API_KEY environment variable")
            return False
        
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTE_CLAUDE_KEY") else None
        model = "anthropic/claude-sonnet-4" if os.getenv("OPENROUTE_CLAUDE_KEY") else "gpt-4"
        
        self.agent = create_mcp_agent(
            system_prompt="""You are a Code Agent assistant. You can:
- Read, write, and analyze code files
- Perform calculations and data processing
- Execute file operations (create, edit, list, search)
- Help with programming tasks and debugging
- Explain code and provide recommendations

Always be concise and practical. Show file paths when working with files.""",
            mcp_servers=[{
                "name": "filesystem",
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", self.workspace]
            }],
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
            verbose=False,
            debug=False
        )
        
        print("🤖 Code Agent initialized with file operations")
        # Suppress verbose MCP server messages
        import logging
        logging.getLogger('mcp').setLevel(logging.WARNING)
        return True
    
    def print_header(self):
        """Print welcome header"""
        print("\n" + "="*60)
        print("🤖 CODE AGENT - Interactive Demo")
        print("="*60)
        print("Features:")
        print("  📝 File operations (read, write, edit, list)")  
        print("  🧮 Calculations and data processing")
        print("  🔍 Code analysis and debugging")
        print("  💡 Programming assistance")
        print("\nNote: MCP server messages above are normal startup output.")
        print("\nCommands:")
        print("  /help    - Show this help")
        print("  /ls      - List workspace files")
        print("  /clear   - Clear screen")
        print("  /quit    - Exit")
        print("="*60)
    
    def print_help(self):
        """Print help information"""
        print("\n📖 HELP - Example Commands:")
        print("-" * 40)
        print("File Operations:")
        print("  • Create a Python file with hello world")
        print("  • Read the contents of main.py")
        print("  • List all files in the workspace")
        print("  • Edit line 5 in config.json")
        print()
        print("Code Tasks:")
        print("  • Write a function to calculate fibonacci")
        print("  • Debug this error in my Python code")
        print("  • Analyze the performance of this algorithm")
        print("  • Refactor this code to be more readable")
        print()
        print("Data Processing:")
        print("  • Calculate the average of [1,2,3,4,5]")
        print("  • Parse this JSON and extract the names")
        print("  • Convert CSV data to JSON format")
        print("-" * 40)
    
    async def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if handled."""
        command = command.strip().lower()
        
        if command == "/help":
            self.print_help()
            return True
        elif command == "/quit" or command == "/exit":
            print("👋 Goodbye!")
            return False
        elif command == "/clear":
            os.system('clear' if os.name == 'posix' else 'cls')
            self.print_header()
            return True
        elif command == "/ls":
            try:
                files = list(Path(self.workspace).iterdir())
                if files:
                    print(f"\n📁 Files in {self.workspace}:")
                    for f in sorted(files):
                        size = f.stat().st_size if f.is_file() else ""
                        print(f"  {'📄' if f.is_file() else '📁'} {f.name} {size}")
                else:
                    print(f"\n📁 {self.workspace} is empty")
            except Exception as e:
                print(f"❌ Error listing files: {e}")
            return True
        
        return True if command.startswith("/") else False
    
    async def run_interactive(self):
        """Run interactive session"""
        if not await self.initialize_agent():
            return
        
        self.print_header()
        print(f"\nType your requests or /help for examples...")
        
        try:
            while True:
                try:
                    # Get user input with green color
                    user_input = input("\n\033[32m🔧 You: \033[0m").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.startswith("/"):
                        should_continue = await self.handle_command(user_input)
                        if not should_continue:
                            break
                        continue
                    
                    # Process with agent with yellow color
                    print("\033[33m🤖 Code Agent:\n\033[0m", end=" ", flush=True)
                    
                    try:
                        # Stream response for real-time feedback
                        response_parts = []
                        async for update in self.agent.stream(user_input):
                            if update["type"] == "thinking":
                                print(f"\033[33m{update['content']}\033[0m", end="", flush=True)
                                response_parts.append(update["content"])
                            elif update["type"] == "tool_call":
                                print(f"\n   🔧 {update['content']}", flush=True)
                            elif update["type"] == "tool_result":
                                print(f"   ✅ {update['content'][:100]}{'...' if len(update['content']) > 100 else ''}")
                        
                        if not response_parts:
                            response = await self.agent.execute(user_input)
                            print(f"\033[33m{response}\033[0m")
                            
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        
                except KeyboardInterrupt:
                    print("\n\n⏸️  Use /quit to exit or continue...")
                    continue
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")

async def main():
    """Main entry point"""
    demo = CodeAgentDemo()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    await demo.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Code Agent!")
    except Exception as e:
        print(f"❌ Error: {e}")