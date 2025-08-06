from typing import Optional
from ..core.tool import Tool, tool, param


PLANNER_AGENT_PROMPT = """You are an expert project planner and task breakdown specialist. Your job is to create detailed, hierarchical todo lists for complex tasks.

## Your Expertise:
- Software development project structure
- Task dependencies and logical sequencing
- Breaking large tasks into manageable subtasks
- Identifying potential blockers and prerequisites
- Estimating task complexity and ordering

## Output Format:
Always output in markdown todo format with hierarchical structure:

# [Project Title]

- [ ] 1. [Major Phase]
  - [Detailed description or context]
  - [Any important notes or considerations]

- [ ] 2. [Next Major Phase] 
  - [ ] 2.1 [Specific subtask]
    - [Implementation details]
    - [Technical requirements]
  - [ ] 2.2 [Another subtask]
    - [Specific steps or considerations]

## Guidelines:
- Use numbered hierarchy (1, 2.1, 2.2, 3, etc.)
- Include descriptive bullet points under major tasks
- Order tasks by logical dependencies
- Break complex tasks into 3-7 subtasks max
- Be specific but not overly detailed
- Consider setup, implementation, testing, and deployment phases
- Include potential challenges or prerequisites

## Examples of Good Planning:
- "Set up development environment" before "Write code"
- "Design database schema" before "Implement data layer"  
- "Create basic structure" before "Add advanced features"

Focus on creating actionable, well-organized plans that guide implementation without being overly prescriptive."""


@tool(name="plan_generate", description="Generate or adjust task plans using AI planning")
class PlanGenerateTool(Tool):
    """A plan generation tool that uses a specialized planner agent."""
    
    def __init__(self, **kwargs):
        # Plan generation can take time for complex tasks
        super().__init__(timeout=90.0, **kwargs)
    
    @param("operation", type="string", enum=["create", "adjust"], description="Create new plan or adjust existing one")
    @param("task_description", type="string", description="Main task to create plan for")
    @param("requirements", type="string", description="Additional requirements or constraints", required=False)
    @param("adjustment_reason", type="string", description="Why the plan needs adjustment (for adjust operation)", required=False)
    async def _execute(self) -> str:
        """Generate or adjust a task plan using a specialized planner agent."""
        operation = self.get_param("operation")
        task_description = self.get_param("task_description")
        requirements = self.get_param("requirements", "")
        adjustment_reason = self.get_param("adjustment_reason", "")
        
        # Input validation
        if not task_description or not task_description.strip():
            return "Error: Task description is required"
        
        # Check if we have access to dispatch tool through agent
        if not self.agent or 'dispatch' not in self.agent.tools:
            return "Error: SubAgentDispatchTool not available. This tool requires access to agent dispatch functionality."
        
        try:
            # Build task for the planner agent
            if operation == "create":
                planner_task = f"""Create a detailed hierarchical todo plan for this task:

TASK: {task_description.strip()}

REQUIREMENTS: {requirements.strip() if requirements else "None specified"}

Please create a comprehensive todo.md plan following the format guidelines. Focus on logical task sequencing and proper breakdown of complex tasks."""
            
            elif operation == "adjust":
                # First read existing plan
                import os
                existing_plan = ""
                if os.path.exists("todo.md"):
                    try:
                        with open("todo.md", "r", encoding="utf-8") as f:
                            existing_plan = f.read()
                    except Exception as e:
                        existing_plan = f"[Could not read existing plan: {e}]"
                else:
                    existing_plan = "[No existing todo.md found]"
                
                planner_task = f"""Adjust the existing project plan based on new information:

ORIGINAL TASK: {task_description.strip()}
ADJUSTMENT REASON: {adjustment_reason.strip() if adjustment_reason else "Plan needs updating"}
ADDITIONAL REQUIREMENTS: {requirements.strip() if requirements else "None"}

EXISTING PLAN:
{existing_plan}

Please provide an updated todo.md plan that addresses the adjustment reason while maintaining good task structure and dependencies."""
            
            # Use dispatch tool to create planner agent
            dispatch_tool = self.agent.tools['dispatch']
            
            result = await dispatch_tool.execute(
                task=planner_task,
                agent_prompt=PLANNER_AGENT_PROMPT
            )
            
            # Extract the markdown plan from the result
            if "Error:" in result:
                return f"Planning failed: {result}"
            
            # Try to write the plan to todo.md
            try:
                # Look for markdown content in the result
                lines = result.strip().split('\n')
                plan_content = []
                capturing = False
                
                for line in lines:
                    # Start capturing when we see a markdown header
                    if line.strip().startswith('#'):
                        capturing = True
                        plan_content.append(line)
                    elif capturing:
                        plan_content.append(line)
                
                if not plan_content:
                    # If no markdown structure found, use the whole result
                    plan_content = lines
                
                final_plan = '\n'.join(plan_content).strip()
                
                # Write to todo.md
                with open("todo.md", "w", encoding="utf-8") as f:
                    f.write(final_plan)
                
                return f"✅ Plan {operation}d successfully!\n\n📋 Plan saved to todo.md:\n\n{final_plan[:500]}{'...' if len(final_plan) > 500 else ''}"
                
            except Exception as e:
                return f"Plan generated but failed to save to todo.md: {str(e)}\n\nGenerated plan:\n{result}"
            
        except Exception as e:
            return f"Error generating plan: {str(e)}"