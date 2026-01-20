# filepath: app/api/schema.py

from typing import Optional, Any
from pydantic import BaseModel

class AgentEvent(BaseModel):
    """
    Standard event structure for the agent to report status updates
    (e.g., "Thinking...", "Searching...", "Running Code...").
    """
    type: str # e.g., "thought", "tool", "result", "error"
    label: str # e.g., "THINKING", "SEARCHING", "EXECUTING_CODE"
    details: str # Human-readable description
    metadata: Optional[Any] = None