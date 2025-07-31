from typing import Optional
from pathlib import Path
try:
    import aiofiles
except ImportError:
    # Fallback if aiofiles not installed
    aiofiles = None
from ..core.tool import Tool, tool, param


@tool(name="read", description="Read a file and return its contents with line numbers")
class ReadTool(Tool):
    """A file reading tool using decorator-based parameter definition."""
    
    @param("file_path", type="string", description="Path to the file to read")
    @param("start_line", type="integer", description="Optional starting line number (1-indexed)", minimum=1, required=False)
    @param("end_line", type="integer", description="Optional ending line number (1-indexed)", minimum=1, required=False)
    async def _execute(self) -> str:
        """Read a file and return its contents with line numbers."""
        file_path = self.get_param("file_path")
        start_line = self.get_param("start_line")
        end_line = self.get_param("end_line")
        
        # Ensure agent has workspace (create default if needed)
        if hasattr(self, 'agent') and self.agent:
            self.agent._ensure_default_workspace()
            
        # Apply workspace constraint if agent has bound workspace
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'workspace') and self.agent.workspace:
            try:
                # Validate path is within workspace
                validated_path = self.agent.workspace.validate_path(file_path)
                file_path = str(validated_path)
                self._log_info(f"Using workspace-constrained path: {file_path}")
            except ValueError as e:
                return f"Error: {str(e)}"
        
        try:
            if aiofiles:
                # Use aiofiles for async file reading if available
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    lines = content.splitlines(keepends=True)
            else:
                # Fallback to synchronous file reading
                # Note: In production, you should install aiofiles for true async I/O
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            # Handle line range selection
            if start_line is not None:
                start_idx = max(0, start_line - 1)  # Convert to 0-indexed
                if end_line is not None:
                    end_idx = min(len(lines), end_line)  # Convert to 0-indexed + 1
                    selected_lines = lines[start_idx:end_idx]
                    line_offset = start_line
                else:
                    selected_lines = lines[start_idx:]
                    line_offset = start_line
            else:
                selected_lines = lines
                line_offset = 1
            
            # Format with line numbers
            formatted_lines = []
            for i, line in enumerate(selected_lines):
                line_num = line_offset + i
                # Remove trailing newline for cleaner display, then add arrow format
                clean_line = line.rstrip('\n')
                formatted_lines.append(f"{line_num:4}→{clean_line}")
            
            if not formatted_lines:
                return f"File '{file_path}' is empty or specified range is invalid"
            
            result = "\n".join(formatted_lines)
            
            # Add summary info if showing partial file
            total_lines = len(lines)
            if start_line is not None or end_line is not None:
                shown_lines = len(selected_lines)
                result = f"Showing lines {line_offset}-{line_offset + shown_lines - 1} of {total_lines} total lines:\n\n{result}"
            else:
                result = f"File: {file_path} ({total_lines} lines)\n\n{result}"
            
            return result
            
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except PermissionError:
            return f"Error: Permission denied reading '{file_path}'"
        except UnicodeDecodeError:
            return f"Error: Cannot decode '{file_path}' as text (binary file?)"
        except Exception as e:
            return f"Error reading file: {str(e)}"