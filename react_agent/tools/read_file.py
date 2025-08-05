import os
import aiofiles
from typing import Optional
from ..core.tool import Tool, tool, param


@tool(name="read_file", description="Read contents from a file")
class ReadFileTool(Tool):
    """A file reading tool using decorator-based parameter definition."""
    
    def __init__(self, **kwargs):
        # File reading is usually fast, use short timeout
        super().__init__(timeout=15.0, **kwargs)
    
    @param("file_path", type="string", description="Path to the file to read")
    @param("encoding", type="string", description="File encoding (default: utf-8)", required=False)
    @param("max_size", type="number", description="Maximum file size in bytes to read (default: 1MB)", required=False)
    async def _execute(self) -> str:
        """Read contents from a file safely."""
        file_path = self.get_param("file_path")
        encoding = self.get_param("encoding", "utf-8")
        max_size = self.get_param("max_size", 1024 * 1024)  # 1MB default
        
        # Input validation
        if not file_path or not file_path.strip():
            return "Error: Empty file path"
        
        file_path = file_path.strip()
        
        # Security check - prevent reading sensitive system files
        dangerous_paths = [
            '/etc/passwd', '/etc/shadow', '/etc/hosts', '/proc/',
            '/sys/', '/dev/', '~/.ssh/', '~/.aws/', '~/.env'
        ]
        
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        for dangerous in dangerous_paths:
            if normalized_path.startswith(os.path.normpath(os.path.abspath(os.path.expanduser(dangerous)))):
                return f"Error: Access to sensitive path blocked: {dangerous}"
        
        try:
            # Check if file exists and is a file
            if not os.path.exists(file_path):
                return f"Error: File does not exist: {file_path}"
            
            if not os.path.isfile(file_path):
                return f"Error: Path is not a file: {file_path}"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                return f"Error: File too large ({file_size} bytes). Maximum allowed: {max_size} bytes"
            
            if file_size == 0:
                return "File is empty"
            
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding=encoding) as file:
                content = await file.read()
                
            # Return content with basic info
            lines = content.count('\n') + 1 if content else 0
            chars = len(content)
            
            result = f"File: {file_path}\n"
            result += f"Size: {file_size} bytes, {lines} lines, {chars} characters\n"
            result += f"Encoding: {encoding}\n"
            result += "-" * 50 + "\n"
            result += content
            
            return result
            
        except UnicodeDecodeError as e:
            return f"Error: Unable to decode file with {encoding} encoding: {str(e)}"
        except PermissionError:
            return f"Error: Permission denied reading file: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"