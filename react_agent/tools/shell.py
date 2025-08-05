import asyncio
import subprocess
import os
from typing import Optional
from ..core.tool import Tool, tool, param


@tool(name="shell", description="Execute shell commands safely")
class ShellTool(Tool):
    """A shell execution tool using decorator-based parameter definition."""
    
    def __init__(self, **kwargs):
        # Shell commands can vary, use moderate timeout
        super().__init__(timeout=60.0, **kwargs)
    
    @param("command", type="string", description="Shell command to execute")
    @param("timeout", type="number", description="Timeout in seconds (default: 30)", required=False)
    @param("working_dir", type="string", description="Working directory for command execution", required=False)
    async def _execute(self) -> str:
        """Execute a shell command safely with timeout."""
        command = self.get_param("command")
        timeout = self.get_param("timeout", 30.0)
        working_dir = self.get_param("working_dir")
        
        # Input validation
        if not command or not command.strip():
            return "Error: Empty command"
        
        command = command.strip()
        
        # Basic security checks - prevent dangerous commands
        dangerous_patterns = [
            'rm -rf /', 'rm -rf *', ':(){ :|:& };:', 'format c:',
            'del /f /s /q', 'sudo rm', 'chmod -R 777 /',
            'dd if=/dev/zero', 'mkfs.', '> /dev/sda'
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return f"Error: Potentially dangerous command blocked: {pattern}"
        
        # Validate working directory if provided
        if working_dir:
            if not os.path.exists(working_dir):
                return f"Error: Working directory does not exist: {working_dir}"
            if not os.path.isdir(working_dir):
                return f"Error: Working directory is not a directory: {working_dir}"
        
        try:
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=float(timeout)
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                
                # Format result
                result_parts = []
                if stdout_text.strip():
                    result_parts.append(f"STDOUT:\n{stdout_text.strip()}")
                if stderr_text.strip():
                    result_parts.append(f"STDERR:\n{stderr_text.strip()}")
                
                result_parts.append(f"EXIT CODE: {process.returncode}")
                
                if not result_parts[:-1]:  # Only exit code
                    result_parts.insert(0, "Command executed successfully (no output)")
                
                return "\n\n".join(result_parts)
                
            except asyncio.TimeoutError:
                # Kill the process if timeout
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {timeout} seconds"
                
        except Exception as e:
            return f"Error executing command: {str(e)}"