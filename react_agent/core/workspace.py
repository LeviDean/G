"""
Workspace management for safe agent file operations.
Provides sandboxing and path validation to prevent agents from accessing unauthorized files.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union, List
from contextlib import contextmanager


class Workspace:
    """
    A secure workspace that constrains agent file operations to a specific directory.
    Provides path validation and prevents directory traversal attacks.
    """
    
    def __init__(self, root_path: Union[str, Path], create_if_missing: bool = True):
        """
        Initialize a workspace.
        
        Args:
            root_path: Root directory for the workspace
            create_if_missing: Create the directory if it doesn't exist
        """
        self.root_path = Path(root_path).resolve()
        self.create_if_missing = create_if_missing
        
        if create_if_missing and not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)
        
        if not self.root_path.exists():
            raise ValueError(f"Workspace directory does not exist: {self.root_path}")
        
        if not self.root_path.is_dir():
            raise ValueError(f"Workspace path is not a directory: {self.root_path}")
    
    def validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate and resolve a path within the workspace.
        
        Args:
            path: Path to validate (can be relative or absolute)
            
        Returns:
            Resolved absolute path within workspace
            
        Raises:
            ValueError: If path is outside workspace or invalid
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Handle relative paths by making them relative to workspace
        if not path.is_absolute():
            resolved_path = (self.root_path / path).resolve()
        else:
            resolved_path = path.resolve()
        
        # Ensure the path is within the workspace
        try:
            resolved_path.relative_to(self.root_path)
        except ValueError:
            raise ValueError(
                f"Path '{path}' is outside workspace '{self.root_path}'. "
                f"Resolved to: {resolved_path}"
            )
        
        return resolved_path
    
    def get_relative_path(self, path: Union[str, Path]) -> Path:
        """
        Get the relative path within the workspace.
        
        Args:
            path: Absolute or relative path
            
        Returns:
            Path relative to workspace root
        """
        validated_path = self.validate_path(path)
        return validated_path.relative_to(self.root_path)
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> List[Path]:
        """
        List files in the workspace matching a pattern.
        
        Args:
            pattern: Glob pattern to match
            recursive: Search recursively
            
        Returns:
            List of paths relative to workspace root
        """
        if recursive:
            files = list(self.root_path.rglob(pattern))
        else:
            files = list(self.root_path.glob(pattern))
        
        # Return paths relative to workspace root
        return [f.relative_to(self.root_path) for f in files if f.is_file()]
    
    def create_subdirectory(self, subdir: Union[str, Path]) -> Path:
        """
        Create a subdirectory within the workspace.
        
        Args:
            subdir: Subdirectory path (relative to workspace)
            
        Returns:
            Absolute path to created directory
        """
        subdir_path = self.validate_path(subdir)
        subdir_path.mkdir(parents=True, exist_ok=True)
        return subdir_path
    
    def get_info(self) -> dict:
        """Get workspace information."""
        try:
            file_count = len(list(self.root_path.rglob("*")))
            size_bytes = sum(f.stat().st_size for f in self.root_path.rglob("*") if f.is_file())
            
            return {
                "root_path": str(self.root_path),
                "exists": self.root_path.exists(),
                "is_directory": self.root_path.is_dir(),
                "file_count": file_count,
                "size_bytes": size_bytes,
                "readable": os.access(self.root_path, os.R_OK),
                "writable": os.access(self.root_path, os.W_OK),
            }
        except Exception as e:
            return {
                "root_path": str(self.root_path),
                "error": str(e),
                "exists": self.root_path.exists(),
            }
    
    def cleanup(self, remove_root: bool = False):
        """
        Clean up workspace contents.
        
        Args:
            remove_root: If True, remove the entire workspace directory
        """
        if remove_root and self.root_path.exists():
            import shutil
            shutil.rmtree(self.root_path)
        elif self.root_path.exists():
            # Remove all contents but keep the root directory
            for item in self.root_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
    
    def __str__(self) -> str:
        return f"Workspace({self.root_path})"
    
    def __repr__(self) -> str:
        return f"Workspace(root_path='{self.root_path}')"


class TemporaryWorkspace(Workspace):
    """
    A temporary workspace that automatically cleans up when done.
    Useful for testing or temporary agent operations.
    """
    
    def __init__(self, prefix: str = "agent_workspace_"):
        """
        Create a temporary workspace.
        
        Args:
            prefix: Prefix for the temporary directory name
        """
        self.temp_dir = tempfile.mkdtemp(prefix=prefix)
        super().__init__(self.temp_dir, create_if_missing=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(remove_root=True)


@contextmanager
def temporary_workspace(prefix: str = "agent_workspace_"):
    """
    Context manager for temporary workspaces.
    
    Args:
        prefix: Prefix for temporary directory
        
    Yields:
        Workspace instance
    """
    workspace = TemporaryWorkspace(prefix)
    try:
        yield workspace
    finally:
        workspace.cleanup(remove_root=True)


class WorkspaceManager:
    """
    Manages multiple workspaces for different agents or projects.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize workspace manager.
        
        Args:
            base_path: Base directory for all workspaces
        """
        self.base_path = Path(base_path).resolve()
        self.workspaces: dict[str, Workspace] = {}
        
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_workspace(self, name: str, subdirectory: Optional[str] = None) -> Workspace:
        """
        Create a new workspace.
        
        Args:
            name: Workspace name/identifier
            subdirectory: Optional subdirectory name (defaults to name)
            
        Returns:
            Created workspace
        """
        if name in self.workspaces:
            raise ValueError(f"Workspace '{name}' already exists")
        
        workspace_dir = self.base_path / (subdirectory or name)
        workspace = Workspace(workspace_dir, create_if_missing=True)
        self.workspaces[name] = workspace
        
        return workspace
    
    def get_workspace(self, name: str) -> Optional[Workspace]:
        """Get an existing workspace by name."""
        return self.workspaces.get(name)
    
    def list_workspaces(self) -> List[str]:
        """List all managed workspace names."""
        return list(self.workspaces.keys())
    
    def remove_workspace(self, name: str, cleanup: bool = True):
        """
        Remove a workspace.
        
        Args:
            name: Workspace name
            cleanup: Whether to delete the workspace files
        """
        if name not in self.workspaces:
            raise ValueError(f"Workspace '{name}' not found")
        
        workspace = self.workspaces[name]
        if cleanup:
            workspace.cleanup(remove_root=True)
        
        del self.workspaces[name]