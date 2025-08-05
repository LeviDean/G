import os
import aiofiles
from typing import Optional
from ..core.tool import Tool, tool, param


@tool(name="write_file", description="Write content to a file")
class WriteFileTool(Tool):
    """A file writing tool using decorator-based parameter definition."""
    
    def __init__(self, **kwargs):
        # File writing can take longer for large files
        super().__init__(timeout=30.0, **kwargs)
    
    @param("file_path", type="string", description="Path to the file to write")
    @param("content", type="string", description="Content to write to the file")
    @param("encoding", type="string", description="File encoding (default: utf-8)", required=False)
    @param("mode", type="string", description="Write mode: 'write' (overwrite) or 'append' (default: write)", required=False)
    @param("create_dirs", type="boolean", description="Create parent directories if they don't exist (default: false)", required=False)
    async def _execute(self) -> str:
        """Write content to a file safely."""
        file_path = self.get_param("file_path")
        content = self.get_param("content")
        encoding = self.get_param("encoding", "utf-8")
        mode = self.get_param("mode", "write")
        create_dirs = self.get_param("create_dirs", False)
        
        # Input validation
        if not file_path or not file_path.strip():
            return "Error: Empty file path"
        
        if content is None:
            return "Error: Content parameter is required"
        
        file_path = file_path.strip()
        
        # Validate mode
        if mode not in ["write", "append"]:
            return "Error: Mode must be 'write' or 'append'"
        
        # Security check - prevent writing to sensitive system locations
        dangerous_paths = [
            '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
            '/proc/', '/sys/', '/dev/', '/boot/', '/lib/', '/lib64/',
            '~/.ssh/', '~/.aws/'
        ]
        
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        for dangerous in dangerous_paths:
            if normalized_path.startswith(os.path.normpath(os.path.abspath(os.path.expanduser(dangerous)))):
                return f"Error: Writing to sensitive path blocked: {dangerous}"
        
        try:
            # Create parent directories if requested
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                if create_dirs:
                    os.makedirs(parent_dir, exist_ok=True)
                else:
                    return f"Error: Parent directory does not exist: {parent_dir}. Set create_dirs=true to create it."
            
            # Check if we're overwriting an existing file
            file_exists = os.path.exists(file_path)
            original_size = 0
            if file_exists:
                if os.path.isdir(file_path):
                    return f"Error: Path is a directory, not a file: {file_path}"
                original_size = os.path.getsize(file_path)
            
            # Determine file mode
            file_mode = 'a' if mode == "append" else 'w'
            
            # Write content
            async with aiofiles.open(file_path, file_mode, encoding=encoding) as file:
                await file.write(content)
            
            # Get final file info
            final_size = os.path.getsize(file_path)
            content_size = len(content.encode(encoding))
            lines_written = content.count('\n') + 1 if content else 0
            
            # Build result message
            if mode == "append":
                if file_exists:
                    result = f"Content appended to existing file: {file_path}\n"
                    result += f"Original size: {original_size} bytes\n"
                    result += f"Added: {content_size} bytes ({lines_written} lines)\n"
                    result += f"Final size: {final_size} bytes"
                else:
                    result = f"New file created: {file_path}\n"
                    result += f"Size: {final_size} bytes ({lines_written} lines)"
            else:  # write mode
                if file_exists:
                    result = f"File overwritten: {file_path}\n"
                    result += f"Previous size: {original_size} bytes\n"
                    result += f"New size: {final_size} bytes ({lines_written} lines)"
                else:
                    result = f"New file created: {file_path}\n"
                    result += f"Size: {final_size} bytes ({lines_written} lines)"
            
            result += f"\nEncoding: {encoding}"
            return result
            
        except UnicodeEncodeError as e:
            return f"Error: Unable to encode content with {encoding} encoding: {str(e)}"
        except PermissionError:
            return f"Error: Permission denied writing to file: {file_path}"
        except OSError as e:
            return f"Error: Unable to create/write file: {str(e)}"
        except Exception as e:
            return f"Error writing file: {str(e)}"