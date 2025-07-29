import math
from ..core.tool import Tool, tool, param


@tool(name="calculator", description="Perform basic mathematical calculations")
class CalculatorTool(Tool):
    """A calculator tool using decorator-based parameter definition."""
    
    @param("expression", type="string", description="Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)')")
    async def _execute(self) -> str:
        """Execute a mathematical expression safely."""
        expression = self.get_param("expression")
        
        # Input validation
        if not expression or not expression.strip():
            return "Error: Empty expression"
        
        expression = expression.strip()
        
        # Simple safety check - only allow certain characters
        allowed_chars = set('0123456789+-*/.() ')
        
        if not all(c in allowed_chars or c.isalpha() for c in expression):
            return "Error: Invalid characters in expression"
        
        # Replace math functions
        safe_expression = expression
        safe_expression = safe_expression.replace('sqrt', 'math.sqrt')
        safe_expression = safe_expression.replace('sin', 'math.sin')
        safe_expression = safe_expression.replace('cos', 'math.cos')
        safe_expression = safe_expression.replace('tan', 'math.tan')
        safe_expression = safe_expression.replace('log', 'math.log')
        safe_expression = safe_expression.replace('exp', 'math.exp')
        safe_expression = safe_expression.replace('abs', 'abs')
        
        try:
            result = eval(safe_expression, {"__builtins__": {}, "math": math})
            return str(float(result))
        except Exception as e:
            return f"Error calculating expression: {str(e)}"