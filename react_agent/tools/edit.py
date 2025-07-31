import os
import difflib
from pathlib import Path
from typing import List, Tuple, Optional
from ..core.tool import Tool, tool, param


@tool(name="edit", description="Edit files using diff-based editing for precise modifications")
class EditTool(Tool):
    """
    A tool for editing existing files using diff-based editing.
    Supports applying unified diffs or line-range replacements with validation.
    """
    
    @param("file_path", type="string", description="Path to the file to edit")
    @param("operation", type="string", description="Edit operation type", 
           enum=["diff", "replace_lines", "insert_lines", "delete_lines"], default="diff")
    @param("diff", type="string", description="Unified diff to apply (for diff operation)", required=False)
    @param("start_line", type="integer", description="Starting line number (1-based)", required=False)
    @param("end_line", type="integer", description="Ending line number (1-based, inclusive)", required=False)
    @param("new_content", type="string", description="New content for the specified line range", required=False)
    @param("validate", type="boolean", description="Validate diff before applying", default=True, required=False)
    async def _execute(self) -> str:
        """Edit a file using diff-based operations."""
        file_path = self.get_param("file_path")
        operation = self.get_param("operation", "diff")
        diff_text = self.get_param("diff")
        start_line = self.get_param("start_line")
        end_line = self.get_param("end_line")
        new_content = self.get_param("new_content")
        validate = self.get_param("validate", True)
        
        if not file_path:
            return "Error: file_path parameter is required"
        
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
        
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file"
        
        try:
            # Read current file content
            with open(path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            original_lines = original_content.splitlines(keepends=True)
            
            # Perform the requested operation
            if operation == "diff":
                if not diff_text:
                    return "Error: diff parameter is required for diff operation"
                
                modified_lines, changes = self._apply_unified_diff(original_lines, diff_text, validate)
                
            elif operation == "replace_lines":
                if start_line is None or new_content is None:
                    return "Error: start_line and new_content are required for replace_lines operation"
                
                modified_lines, changes = self._replace_line_range(
                    original_lines, start_line, end_line, new_content
                )
                
            elif operation == "insert_lines":
                if start_line is None or new_content is None:
                    return "Error: start_line and new_content are required for insert_lines operation"
                
                modified_lines, changes = self._insert_lines(original_lines, start_line, new_content)
                
            elif operation == "delete_lines":
                if start_line is None:
                    return "Error: start_line is required for delete_lines operation"
                
                modified_lines, changes = self._delete_line_range(original_lines, start_line, end_line)
                
            else:
                return f"Error: Unknown operation '{operation}'"
            
            if not changes:
                return f"No changes made to '{file_path}'"
            
            # Write modified content back to file
            modified_content = ''.join(modified_lines)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            # Calculate statistics
            new_line_count = len(modified_lines)
            original_line_count = len(original_lines)
            lines_changed = new_line_count - original_line_count
            file_size = path.stat().st_size
            
            self._log_info(f"Applied {operation} to {file_path}: {changes}")
            
            return f"Successfully applied {operation} to '{file_path}'. " \
                   f"File now has {new_line_count} lines ({lines_changed:+d} lines, {file_size} bytes). " \
                   f"Changes: {changes}"
                
        except PermissionError:
            error_msg = f"Permission denied: Cannot edit '{file_path}'"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Failed to edit file '{file_path}': {str(e)}"
            self._log_error(error_msg)
            return f"Error: {error_msg}"
    
    def _apply_unified_diff(self, original_lines: List[str], diff_text: str, validate: bool) -> Tuple[List[str], str]:
        """Apply a unified diff to the original lines."""
        try:
            # Parse the unified diff
            diff_lines = diff_text.splitlines()
            
            # Find the start of the actual diff content (skip headers)
            diff_start = 0
            for i, line in enumerate(diff_lines):
                if line.startswith('@@'):
                    diff_start = i
                    break
            else:
                raise ValueError("No valid unified diff found (missing @@ markers)")
            
            # Extract line range information from @@ line
            range_line = diff_lines[diff_start]
            if not range_line.startswith('@@'):
                raise ValueError("Invalid diff format: missing @@ range line")
            
            # Parse @@ -start,count +start,count @@
            import re
            range_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', range_line)
            if not range_match:
                raise ValueError("Invalid diff range format")
            
            old_start = int(range_match.group(1))
            old_count = int(range_match.group(2) or 1)
            new_start = int(range_match.group(3))
            new_count = int(range_match.group(4) or 1)
            
            # Apply the diff
            result_lines = original_lines[:]
            
            # Convert to 0-based indexing
            old_start -= 1
            new_start -= 1
            
            # Validate that the original lines match what the diff expects
            if validate:
                validation_error = self._validate_diff_context(
                    original_lines, diff_lines[diff_start + 1:], old_start, old_count
                )
                if validation_error:
                    raise ValueError(f"Diff validation failed: {validation_error}")
            
            # Apply the changes
            new_lines = []
            for line in diff_lines[diff_start + 1:]:
                if line.startswith(' '):
                    # Context line (unchanged)
                    new_lines.append(line[1:] + '\n')
                elif line.startswith('+'):
                    # Added line
                    new_lines.append(line[1:] + '\n')
                elif line.startswith('-'):
                    # Deleted line (skip)
                    continue
                else:
                    # End of diff or invalid line
                    break
            
            # Replace the old lines with new lines
            result_lines[old_start:old_start + old_count] = new_lines
            
            changes = f"Applied unified diff at lines {old_start + 1}-{old_start + old_count}"
            return result_lines, changes
            
        except Exception as e:
            raise ValueError(f"Failed to apply diff: {str(e)}")
    
    def _validate_diff_context(self, original_lines: List[str], diff_content: List[str], 
                             old_start: int, old_count: int) -> Optional[str]:
        """Validate that the diff context matches the original file."""
        # Extract context and deletion lines from diff
        expected_lines = []
        for line in diff_content:
            if line.startswith(' ') or line.startswith('-'):
                expected_lines.append(line[1:])
            elif line.startswith('+'):
                continue
            else:
                break
        
        # Check if we have enough lines in the original file
        if old_start + len(expected_lines) > len(original_lines):
            return f"Diff extends beyond file end (file has {len(original_lines)} lines)"
        
        # Compare expected lines with actual lines
        for i, expected in enumerate(expected_lines):
            actual_line_idx = old_start + i
            if actual_line_idx >= len(original_lines):
                return f"Line {actual_line_idx + 1} not found in file"
            
            actual = original_lines[actual_line_idx].rstrip('\n\r')
            if actual != expected:
                return f"Line {actual_line_idx + 1} mismatch. Expected: '{expected}', Got: '{actual}'"
        
        return None
    
    def _replace_line_range(self, original_lines: List[str], start_line: int, end_line: Optional[int], 
                           new_content: str) -> Tuple[List[str], str]:
        """Replace a range of lines with new content."""
        # Convert to 0-based indexing
        start_idx = start_line - 1
        end_idx = (end_line - 1) if end_line is not None else start_idx
        
        # Validate range
        if start_idx < 0 or start_idx >= len(original_lines):
            raise ValueError(f"Start line {start_line} is out of range (file has {len(original_lines)} lines)")
        
        if end_idx < start_idx or end_idx >= len(original_lines):
            end_idx = start_idx
        
        # Prepare new content lines
        new_lines = []
        if new_content:
            for line in new_content.splitlines():
                new_lines.append(line + '\n')
        
        # Replace the range
        result_lines = original_lines[:]
        result_lines[start_idx:end_idx + 1] = new_lines
        
        replaced_count = end_idx - start_idx + 1
        changes = f"Replaced {replaced_count} line(s) at lines {start_line}-{end_idx + 1} with {len(new_lines)} line(s)"
        
        return result_lines, changes
    
    def _insert_lines(self, original_lines: List[str], start_line: int, new_content: str) -> Tuple[List[str], str]:
        """Insert new lines at the specified position."""
        # Convert to 0-based indexing
        insert_idx = start_line - 1
        
        # Clamp to valid range
        if insert_idx < 0:
            insert_idx = 0
        elif insert_idx > len(original_lines):
            insert_idx = len(original_lines)
        
        # Prepare new content lines
        new_lines = []
        if new_content:
            for line in new_content.splitlines():
                new_lines.append(line + '\n')
        
        # Insert the lines
        result_lines = original_lines[:]
        for i, line in enumerate(new_lines):
            result_lines.insert(insert_idx + i, line)
        
        changes = f"Inserted {len(new_lines)} line(s) at line {start_line}"
        return result_lines, changes
    
    def _delete_line_range(self, original_lines: List[str], start_line: int, 
                          end_line: Optional[int]) -> Tuple[List[str], str]:
        """Delete a range of lines."""
        # Convert to 0-based indexing
        start_idx = start_line - 1
        end_idx = (end_line - 1) if end_line is not None else start_idx
        
        # Validate range
        if start_idx < 0 or start_idx >= len(original_lines):
            raise ValueError(f"Start line {start_line} is out of range (file has {len(original_lines)} lines)")
        
        if end_idx < start_idx:
            end_idx = start_idx
        elif end_idx >= len(original_lines):
            end_idx = len(original_lines) - 1
        
        # Delete the range
        result_lines = original_lines[:]
        del result_lines[start_idx:end_idx + 1]
        
        deleted_count = end_idx - start_idx + 1
        changes = f"Deleted {deleted_count} line(s) at lines {start_line}-{end_idx + 1}"
        
        return result_lines, changes