import subprocess
import shlex
import asyncio
from ..core.tool import Tool, tool, param


@tool(name="shell", description="Execute shell commands and return the output")
class ShellTool(Tool):
    """A shell execution tool using decorator-based parameter definition."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = 30
        self.allowed_commands = [
            'ls', 'pwd', 'cat', 'echo', 'date', 'whoami', 'uname',
            'ps', 'df', 'du', 'head', 'tail', 'grep', 'find', 'wc',
            'sort', 'uniq', 'curl', 'wget', 'ping', 'which', 'whereis'
        ]
    
    @param("command", type="string", description="Shell command to execute")
    @param("allow_dangerous", type="boolean", description="Allow potentially dangerous commands", default=False, required=False)
    async def _execute(self) -> str:
        """Execute a shell command safely and return the output."""
        command = self.get_param("command")
        allow_dangerous = self.get_param("allow_dangerous", False)
        
        # Basic input validation
        if not command or not command.strip():
            return "Error: Empty command"
        
        command = command.strip()
        
        # Security check: validate command if not explicitly allowing dangerous commands
        if not allow_dangerous:
            # Extract the base command (first word)
            try:
                base_command = shlex.split(command)[0]
            except ValueError as e:
                return f"Error: Invalid command syntax: {str(e)}"
            
            # Check if base command is in allowed list
            if base_command not in self.allowed_commands:
                return f"Error: Command '{base_command}' not allowed. Set allow_dangerous=true to override."
            
            # Additional safety checks
            dangerous_patterns = ['rm -rf', 'sudo', 'su ', 'chmod +x', '>/dev/', 'dd if=', 'mkfs', 'fdisk']
            if any(pattern in command.lower() for pattern in dangerous_patterns):
                return f"Error: Potentially dangerous command detected. Set allow_dangerous=true to override."
        
        try:
            # Execute command asynchronously with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=None  # Use current working directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout
                )
                stdout = stdout.decode('utf-8') if stdout else ''
                stderr = stderr.decode('utf-8') if stderr else ''
                returncode = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {self.timeout} seconds"
            
            # Prepare output
            output_parts = []
            
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            
            if returncode != 0:
                output_parts.append(f"Return code: {returncode}")
            
            if not output_parts:
                return "Command executed successfully with no output."
            
            return "\n\n".join(output_parts)
            
        except Exception as e:
            return f"Unexpected error: {str(e)}"