import os
import re
from typing import Optional
from ..core.tool import Tool, tool, param


@tool(name="plan_maintain", description="Read and update plan status in todo.md file")
class PlanMaintainTool(Tool):
    """A tool for maintaining and updating todo.md plan status."""
    
    def __init__(self, **kwargs):
        # Plan maintenance operations are usually fast
        super().__init__(timeout=15.0, **kwargs)
    
    @param("operation", type="string", enum=["read", "complete", "update_status"], description="Operation to perform on the plan")
    @param("item_text", type="string", description="Text content of the item to find (for complete/update_status)", required=False)
    @param("new_status", type="string", enum=["pending", "completed"], description="New status for update_status operation", required=False)
    async def _execute(self) -> str:
        """Maintain todo.md plan file."""
        operation = self.get_param("operation")
        item_text = self.get_param("item_text", "")
        new_status = self.get_param("new_status", "")
        
        # Check if todo.md exists
        if not os.path.exists("todo.md"):
            if operation == "read":
                return "No plan found. Use plan_generate tool to create a plan first."
            else:
                return "Error: todo.md file not found. Create a plan first using plan_generate tool."
        
        try:
            # Read current todo.md content
            with open("todo.md", "r", encoding="utf-8") as f:
                content = f.read()
            
            if operation == "read":
                # Simply return the current plan
                line_count = content.count('\n') + 1
                char_count = len(content)
                
                # Count completed and pending items
                completed_items = len(re.findall(r'- \[x\]', content, re.IGNORECASE))
                pending_items = len(re.findall(r'- \[ \]', content))
                total_items = completed_items + pending_items
                
                result = f"📋 Current Plan (todo.md)\n"
                result += f"📊 Progress: {completed_items}/{total_items} items completed\n"
                result += f"📄 File: {line_count} lines, {char_count} characters\n"
                result += "-" * 50 + "\n"
                result += content
                
                return result
            
            elif operation == "complete":
                if not item_text or not item_text.strip():
                    return "Error: item_text parameter is required for complete operation"
                
                # Find and mark item as completed
                # Look for the item text in todo items
                lines = content.split('\n')
                modified = False
                
                for i, line in enumerate(lines):
                    # Check if this line contains a todo item with our text
                    if '- [ ]' in line and item_text.strip().lower() in line.lower():
                        # Mark as completed
                        lines[i] = line.replace('- [ ]', '- [x]', 1)
                        modified = True
                        break
                
                if not modified:
                    # Try partial matching for better user experience
                    for i, line in enumerate(lines):
                        if '- [ ]' in line:
                            # Extract just the task text (remove numbering and formatting)
                            task_part = re.sub(r'- \[ \]\s*\d+\.?\d*\s*', '', line).strip()
                            if item_text.strip().lower() in task_part.lower():
                                lines[i] = line.replace('- [ ]', '- [x]', 1)
                                modified = True
                                break
                
                if not modified:
                    return f"Error: Could not find pending item containing '{item_text}'. Available pending items:\n" + \
                           '\n'.join([f"- {line.strip()}" for line in lines if '- [ ]' in line])
                
                # Write back to file
                new_content = '\n'.join(lines)
                with open("todo.md", "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                return f"✅ Item marked as completed: '{item_text}'\n\nUpdated plan saved to todo.md"
            
            elif operation == "update_status":
                if not item_text or not item_text.strip():
                    return "Error: item_text parameter is required for update_status operation"
                if not new_status:
                    return "Error: new_status parameter is required for update_status operation"
                
                # Determine the target and replacement patterns
                if new_status == "completed":
                    target_pattern = '- [ ]'
                    replacement_pattern = '- [x]'
                elif new_status == "pending":
                    target_pattern = '- [x]'
                    replacement_pattern = '- [ ]'
                else:
                    return f"Error: Invalid status '{new_status}'. Must be 'pending' or 'completed'"
                
                # Find and update item status
                lines = content.split('\n')
                modified = False
                
                for i, line in enumerate(lines):
                    if target_pattern in line and item_text.strip().lower() in line.lower():
                        lines[i] = line.replace(target_pattern, replacement_pattern, 1)
                        modified = True
                        break
                
                if not modified:
                    # Try partial matching
                    for i, line in enumerate(lines):
                        if target_pattern in line:
                            task_part = re.sub(r'- \[[x ]\]\s*\d+\.?\d*\s*', '', line).strip()
                            if item_text.strip().lower() in task_part.lower():
                                lines[i] = line.replace(target_pattern, replacement_pattern, 1)
                                modified = True
                                break
                
                if not modified:
                    current_status = "completed" if new_status == "pending" else "pending"
                    available_items = [line.strip() for line in lines if target_pattern in line]
                    return f"Error: Could not find {current_status} item containing '{item_text}'. Available {current_status} items:\n" + \
                           '\n'.join([f"- {item}" for item in available_items])
                
                # Write back to file
                new_content = '\n'.join(lines)
                with open("todo.md", "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                status_symbol = "✅" if new_status == "completed" else "⏳"
                return f"{status_symbol} Item status updated to {new_status}: '{item_text}'\n\nUpdated plan saved to todo.md"
            
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error maintaining plan: {str(e)}"