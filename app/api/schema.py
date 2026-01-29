# filepath: app/api/schema.py

from typing import Optional, Any, List, Dict
from pydantic import BaseModel

class SourceMetadata(BaseModel):
    """
    Structured metadata for web results or files to be displayed in the UI.
    """
    title: str
    url: Optional[str] = None
    description: Optional[str] = None

class AgentEvent(BaseModel):
    """
    Standard event structure for the agent backend to communicate with the UI.
    This format is designed to be easily mapped to UI components.
    """
    type: str  # e.g., "tool_start", "tool_end", "result", "error", "status", "source_found"
    label: Optional[str] = None # e.g., "SEARCHING WEB", "ANALYZING IMAGE"
    details: Optional[str] = None # Human readable description of what's happening
    
    # Tool Specific Fields
    tool_name: Optional[str] = None # The technical name of the tool being used
    step_id: Optional[str] = None # A unique ID for this specific step in the chain
    
    # Payload Fields
    content: Optional[str] = None # The actual text result (if type is 'result')
    sources: Optional[List[SourceMetadata]] = None # List of websites/files found
    
    metadata: Optional[Dict[str, Any]] = None # Any additional raw data

class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""
    message: str
    plan_id: str
    chat_history: Optional[List[Dict[str, str]]] = None

class UploadResponse(BaseModel):
    """Schema for file upload confirmation."""
    success: bool
    filenames: List[str]
    message: str