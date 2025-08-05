import os
import aiofiles
import difflib
from typing import Optional, List, Tuple
from ..core.tool import Tool, tool, param


@tool(name="edit_file", description="Edit a file using diff-based operations")
class EditFileTool(Tool):
    """A diff-based file editing tool using decorator-based parameter definition."""
    
    def __init__(self, **kwargs):
        # File editing with diff generation can take time
        super().__init__(timeout=45.0, **kwargs)
    
    @param("file_path", type="string", description="Path to the file to edit")
    @param("operation", type="string", description="Edit operation: 'replace', 'insert_after', 'insert_before', 'delete_lines'")
    @param("target", type="string", description="Target text to find (for replace/insert operations) or line numbers (for delete, e.g., '5-10')", required=False)
    @param("replacement", type="string", description="Replacement text (for replace/insert operations)", required=False)
    @param("encoding", type="string", description="File encoding (default: utf-8)", required=False)
    async def _execute(self) -> str:
        """Edit a file using diff-based operations."""
        file_path = self.get_param("file_path")
        operation = self.get_param("operation")
        target = self.get_param("target", "")
        replacement = self.get_param("replacement", "")
        encoding = self.get_param("encoding", "utf-8")
        
        # Input validation
        if not file_path or not file_path.strip():
            return "Error: Empty file path"
        
        if not operation:
            return "Error: Operation parameter is required"
        
        file_path = file_path.strip()
        valid_operations = ["replace", "insert_after", "insert_before", "delete_lines"]
        
        if operation not in valid_operations:
            return f"Error: Invalid operation. Must be one of: {', '.join(valid_operations)}"
        
        # Security check - prevent editing sensitive system files
        dangerous_paths = [
            '/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
            '/proc/', '/sys/', '/dev/', '/boot/', '/lib/', '/lib64/',
            '~/.ssh/', '~/.aws/'
        ]
        
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        for dangerous in dangerous_paths:
            if normalized_path.startswith(os.path.normpath(os.path.abspath(os.path.expanduser(dangerous)))):
                return f"Error: Editing sensitive path blocked: {dangerous}"
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File does not exist: {file_path}"
            
            if not os.path.isfile(file_path):
                return f"Error: Path is not a file: {file_path}"
            
            # Read original content
            async with aiofiles.open(file_path, 'r', encoding=encoding) as file:
                original_content = await file.read()
            
            original_lines = original_content.splitlines(keepends=True)
            modified_lines = original_lines.copy()
            
            # Perform the requested operation
            if operation == "replace":
                if not target:
                    return "Error: Target text is required for replace operation"
                if replacement is None:
                    replacement = ""
                
                modified_content = original_content.replace(target, replacement)
                modified_lines = modified_content.splitlines(keepends=True)
                
                if modified_content == original_content:
                    return f"Warning: Target text '{target}' not found in file"
            
            elif operation == "insert_after":
                if not target:
                    return "Error: Target text is required for insert_after operation"
                if replacement is None:
                    return "Error: Replacement text is required for insert_after operation"
                
                found = False
                for i, line in enumerate(modified_lines):
                    if target in line:
                        # Insert after this line
                        insert_text = replacement if replacement.endswith('\n') else replacement + '\n'
                        modified_lines.insert(i + 1, insert_text)
                        found = True
                        break
                
                if not found:
                    return f"Warning: Target text '{target}' not found for insertion"
            
            elif operation == "insert_before":
                if not target:
                    return "Error: Target text is required for insert_before operation"
                if replacement is None:
                    return "Error: Replacement text is required for insert_before operation"
                
                found = False
                for i, line in enumerate(modified_lines):
                    if target in line:
                        # Insert before this line
                        insert_text = replacement if replacement.endswith('\n') else replacement + '\n'
                        modified_lines.insert(i, insert_text)
                        found = True
                        break
                
                if not found:
                    return f"Warning: Target text '{target}' not found for insertion"
            
            elif operation == "delete_lines":
                if not target:
                    return "Error: Line range is required for delete_lines operation (e.g., '5' or '5-10')"
                
                try:
                    # Parse line range
                    if '-' in target:
                        start_str, end_str = target.split('-', 1)
                        start_line = int(start_str.strip()) - 1  # Convert to 0-based
                        end_line = int(end_str.strip()) - 1      # Convert to 0-based
                    else:
                        start_line = end_line = int(target.strip()) - 1  # Single line
                    
                    # Validate line numbers
                    if start_line < 0 or end_line < 0:
                        return "Error: Line numbers must be positive"
                    if start_line >= len(original_lines) or end_line >= len(original_lines):
                        return f"Error: Line number out of range (file has {len(original_lines)} lines)"
                    if start_line > end_line:
                        return "Error: Start line must be <= end line"
                    
                    # Delete lines (in reverse order to maintain indices)
                    for i in range(end_line, start_line - 1, -1):
                        del modified_lines[i]
                
                except ValueError:
                    return "Error: Invalid line range format. Use '5' or '5-10'"
            
            # Create diff
            modified_content = ''.join(modified_lines)
            
            if modified_content == original_content:
                return "No changes made to file"
            
            # Generate diff for display
            diff = list(difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"{file_path} (original)",
                tofile=f"{file_path} (modified)",
                lineterm=''
            ))
            
            # Write modified content back to file
            async with aiofiles.open(file_path, 'w', encoding=encoding) as file:
                await file.write(modified_content)
            
            # Build result
            original_line_count = len(original_lines)
            modified_line_count = len(modified_lines)
            
            result = f"File edited successfully: {file_path}\n"
            result += f"Operation: {operation}\n"
            result += f"Lines: {original_line_count} → {modified_line_count}\n"
            result += f"Encoding: {encoding}\n"
            result += "\nDiff preview:\n"
            result += "-" * 50 + "\n"
            
            # Show first 20 lines of diff
            diff_lines = diff[:20]
            if len(diff) > 20:
                diff_lines.append("... (diff truncated)")
            
            result += '\n'.join(diff_lines)
            
            return result
            
        except UnicodeDecodeError as e:
            return f"Error: Unable to decode file with {encoding} encoding: {str(e)}"
        except UnicodeEncodeError as e:
            return f"Error: Unable to encode content with {encoding} encoding: {str(e)}"
        except PermissionError:
            return f"Error: Permission denied accessing file: {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"