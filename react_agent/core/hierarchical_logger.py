"""
Hierarchical Logger for ReAct Agent Framework

Provides clean hierarchical logging with automatic indentation for sub-agents and tools.
This centralizes all logging formatting logic in one place.
"""

import logging
import threading
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'GRAY': '\033[90m',      # Gray for INFO
        'YELLOW': '\033[93m',    # Yellow for WARNING  
        'RED': '\033[91m',       # Red for ERROR
        'RESET': '\033[0m'       # Reset color
    }
    
    def format(self, record):
        # Apply color based on log level
        if record.levelno == logging.INFO:
            color = self.COLORS['GRAY']
        elif record.levelno == logging.WARNING:
            color = self.COLORS['YELLOW']
        elif record.levelno >= logging.ERROR:
            color = self.COLORS['RED']
        else:
            color = ''
        
        # Format the message
        formatted = super().format(record)
        
        # Add color if we have one
        if color:
            return f"{color}{formatted}{self.COLORS['RESET']}"
        return formatted


class HierarchicalLogger:
    """
    A logger that automatically handles hierarchical indentation for multi-agent systems.
    
    Features:
    - Automatic indentation based on hierarchy level
    - Thread-safe level management
    - Consistent formatting across agents and tools
    - Easy integration with existing logging infrastructure
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure consistent logging across all agents."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._loggers = {}
        self._hierarchy_levels = {}
        self._thread_levels = threading.local()
        
        # Configuration
        self.indent_string = "  "  # 2 spaces per level
        self.max_hierarchy_level = 10  # Prevent infinite nesting
        
    def get_logger(self, 
                   name: str, 
                   hierarchy_level: int = 0,
                   enable_logging: bool = True) -> Optional[logging.Logger]:
        """
        Get or create a hierarchical logger for the given name and level.
        
        Args:
            name: Logger name (e.g., "MathChatAgent", "Calculator_1.calculator")
            hierarchy_level: Indentation level (0 = root, 1 = sub-agent, etc.)
            enable_logging: Whether logging is enabled
            
        Returns:
            Logger instance or None if logging is disabled
        """
        if not enable_logging:
            return None
            
        # Clamp hierarchy level
        hierarchy_level = max(0, min(hierarchy_level, self.max_hierarchy_level))
        
        logger_key = f"{name}:{hierarchy_level}"
        
        if logger_key not in self._loggers:
            # Create new logger
            logger = logging.getLogger(f"ReActAgent.{name}")
            
            # Avoid duplicate handlers
            if not logger.handlers:
                handler = logging.StreamHandler()
                
                # Create colored formatter with proper indentation and level indicator
                indent = self.indent_string * hierarchy_level
                # level_indicator = f"[L{hierarchy_level}] " if hierarchy_level > 0 else ""
                level_indicator = ""
                formatter = ColoredFormatter(
                    f'%(asctime)s - {indent}{level_indicator}{name} - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False  # Prevent duplicate messages
            
            self._loggers[logger_key] = logger
            self._hierarchy_levels[logger_key] = hierarchy_level
        
        return self._loggers[logger_key]
    
    def create_child_logger(self, 
                           parent_logger_name: str, 
                           child_name: str, 
                           enable_logging: bool = True) -> Optional[logging.Logger]:
        """
        Create a child logger with increased hierarchy level.
        
        Args:
            parent_logger_name: Name of the parent logger
            child_name: Name for the child logger
            enable_logging: Whether logging is enabled
            
        Returns:
            Child logger with hierarchy_level + 1
        """
        # Find parent's hierarchy level
        parent_level = 0
        for key, level in self._hierarchy_levels.items():
            if key.startswith(f"{parent_logger_name}:"):
                parent_level = level
                break
        
        return self.get_logger(child_name, parent_level + 1, enable_logging)
    
    def log_agent_operation(self, 
                           logger: logging.Logger, 
                           operation: str, 
                           details: str = "",
                           level: str = "INFO"):
        """
        Log an agent operation with consistent formatting.
        
        Args:
            logger: Logger to use
            operation: Operation name (e.g., "Agent Run", "Tool Execution")
            details: Additional details
            level: Log level (INFO, WARNING, ERROR)
        """
        if not logger:
            return
            
        icons = {
            "Agent Run": "▶️",
            "Tool Execution": "🔧", 
            "Binding": "🔗",
            "Completed": "✅",
            "Error": "❌",
            "Warning": "⚠️"
        }
        
        icon = icons.get(operation, "📋")
        message = f"{icon} {operation}"
        if details:
            message += f"\n   {details}"
            
        getattr(logger, level.lower())(message)
    
    def log_hierarchy_info(self, logger: logging.Logger):
        """Debug helper to show current hierarchy structure."""
        if not logger:
            return
            
        logger.info("📊 Hierarchy structure:")
        for key, level in sorted(self._hierarchy_levels.items(), key=lambda x: x[1]):
            indent = self.indent_string * level
            name = key.split(':')[0]
            logger.info(f"   {indent}Level {level}: {name}")


# Global instance for easy access
hierarchical_logger = HierarchicalLogger()


def get_hierarchical_logger(name: str, 
                           hierarchy_level: int = 0,
                           enable_logging: bool = True) -> Optional[logging.Logger]:
    """
    Convenience function to get a hierarchical logger.
    
    Args:
        name: Logger name
        hierarchy_level: Hierarchy level (0 = root)
        enable_logging: Whether logging is enabled
        
    Returns:
        Logger instance or None if disabled
    """
    return hierarchical_logger.get_logger(name, hierarchy_level, enable_logging)


def create_child_logger(parent_name: str, 
                       child_name: str,
                       enable_logging: bool = True) -> Optional[logging.Logger]:
    """
    Convenience function to create a child logger.
    
    Args:
        parent_name: Parent logger name
        child_name: Child logger name
        enable_logging: Whether logging is enabled
        
    Returns:
        Child logger with increased hierarchy level
    """
    return hierarchical_logger.create_child_logger(parent_name, child_name, enable_logging)