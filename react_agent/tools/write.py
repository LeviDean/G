import os
from pathlib import Path
from ..core.tool import Tool, tool, param


@tool(name="write", description="Write content to a file (creates new file or overwrites existing)")
class WriteTool(Tool):
    """
    A tool for writing content to files.
    Can create new files or overwrite existing ones.
    """
    
    @param("file_path", type="string", description="Path to the file to write")
    @param("content", type="string", description="Content to write to the file")
    @param("create_dirs", type="boolean", description="Create parent directories if they don't exist", default=True, required=False)
    async def _execute(self) -> str:
        """Write content to a file."""
        file_path = self.get_param("file_path")
        content = self.get_param("content")
        create_dirs = self.get_param("create_dirs", True)
        
        if not file_path:
            return "Error: file_path parameter is required"
        
        if content is None:
            content = ""  # Allow writing empty files
        
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
            # Convert to Path object for easier handling
            path = Path(file_path)
            
            # Create parent directories if requested
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                self._log_info(f"Created directories: {path.parent}")
            
            # Check if file exists for logging
            file_existed = path.exists()
            
            # Write content to file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info
            file_size = path.stat().st_size
            line_count = len(content.splitlines()) if content else 0
            
            if file_existed:
                self._log_info(f"Overwrote existing file: {file_path}")
                return f"Successfully overwrote file '{file_path}' with {line_count} lines ({file_size} bytes)"
            else:
                self._log_info(f"Created new file: {file_path}")
                return f"Successfully created file '{file_path}' with {line_count} lines ({file_size} bytes)"
                
        except PermissionError:
            error_msg = f"Permission denied: Cannot write to '{file_path}'"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
        except FileNotFoundError:
            error_msg = f"Directory not found: '{path.parent}' (set create_dirs=true to create)"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Failed to write file '{file_path}': {str(e)}"
            self._log_error(error_msg)
            return f"Error: {error_msg}"