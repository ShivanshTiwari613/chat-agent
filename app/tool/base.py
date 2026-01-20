# filepath: app/tool/base.py

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

class ToolResult(BaseModel):
    """
    Standard output format for all tools.
    """
    success: bool
    output_text: str
    output_data: Optional[Any] = None
    error_message: Optional[str] = None

class BaseTool(BaseModel, ABC):
    """
    Abstract base class for all tools.
    """
    name: str
    description: str
    
    # Using Union and Any to allow subclasses to override with specific Pydantic models
    args_schema: Any 

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """
        Main execution logic for the tool.
        Accepts any keyword arguments to maintain compatibility with subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")

    class Config:
        arbitrary_types_allowed = True
