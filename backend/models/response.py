from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from models.question import TaqneeqQuestion
from models.session import SessionState
class ClassificationResult(BaseModel):
    """Results of the classification process"""
    
    # Primary results
    top_department: str = Field(..., description="Highest probability department")
    top_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of top department")
    
    # Secondary results
    secondary_department: Optional[str] = Field(
        default=None, 
        description="Second highest department"
    )
    secondary_probability: Optional[float] = Field(
        default=None,
        description="Probability of secondary department"
    )
    
    # Complete results
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all departments"
    )
    
    # Classification metadata
    questions_asked: int = Field(..., ge=0, description="Number of questions answered")
    confidence_level: str = Field(..., description="high/medium/low confidence")
    should_continue: bool = Field(..., description="Whether to ask more questions")
    reasoning: str = Field("Processing...", description="Explanation for stopping/continuing")
    
    # Timestamps
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When result was generated"
    )

# API Request Models

class StartSessionRequest(BaseModel):
    """Request to start new classification session"""
    
    user_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional user information"
    )
    session_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional session configuration"
    )

class AnswerQuestionRequest(BaseModel):
    """Request to submit answer to question"""
    
    session_id: str = Field(..., description="Session identifier")
    question_id: str = Field(..., description="Question being answered")
    response: int = Field(..., ge=1, le=5, description="Likert response (1-5)")
    confidence: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="User confidence in response"
    )

class GetExplanationRequest(BaseModel):
    """Request for department explanation"""
    
    session_id: str = Field(..., description="Session identifier")
    department_id: Optional[str] = Field(
        default=None,
        description="Specific department to explain (default: top match)"
    )
    include_comparison: Optional[bool] = Field(
        default=False,
        description="Include comparison with other departments"
    )

# API Response Models

class StartSessionResponse(BaseModel):
    """Response when starting new session"""
    
    session_id: str = Field(..., description="Created session identifier")
    first_question: Dict[str, Any] = Field(..., description="First question to ask")
    total_departments: int = Field(..., description="Number of departments available")
    estimated_questions: str = Field(..., description="Estimated questions needed")
    message: str = Field(..., description="Welcome message")

class AnswerQuestionResponse(BaseModel):
    """Response after submitting answer"""
    
    session_id: str = Field(..., description="Session identifier")
    next_question: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Next question (None if complete)"
    )
    classification_result: ClassificationResult = Field(
        ...,
        description="Current classification state"
    )
    current_top_traits: List[tuple] = Field(
        default_factory=list,
        description="User's current top traits"
    )
    is_complete: bool = Field(..., description="Whether classification is finished")
    message: str = Field(..., description="Status message")

class GetExplanationResponse(BaseModel):
    """Response with department explanation"""
    
    session_id: str = Field(..., description="Session identifier")
    department_id: str = Field(..., description="Explained department")
    department_name: str = Field(..., description="Department display name")
    
    # Main explanation
    explanation: Dict[str, str] = Field(
        ...,
        description="Structured explanation sections"
    )
    
    # Supporting information
    classification_confidence: float = Field(
        ...,
        description="Confidence in this classification"
    )
    user_trait_alignment: Dict[str, float] = Field(
        ...,
        description="How user traits align with department"
    )
    
    # Alternative options
    alternative_departments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Other good department matches"
    )
    
    # Metadata
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When explanation was generated"
    )

class SessionStatusResponse(BaseModel):
    """Current session status"""
    
    session_id: str = Field(..., description="Session identifier")
    state: SessionState = Field(..., description="Current session state")
    
    # Progress
    questions_answered: int = Field(..., description="Questions answered so far")
    estimated_remaining: str = Field(..., description="Estimated questions remaining")
    
    # Current results
    current_top_department: str = Field(..., description="Current top department")
    current_confidence: float = Field(..., description="Current confidence level")
    
    # Trait information
    top_traits: List[tuple] = Field(
        default_factory=list,
        description="User's strongest traits"
    )
    department_probabilities: Dict[str, float] = Field(
        ...,
        description="Current department probabilities"
    )
    
    # Metadata
    session_duration: str = Field(..., description="How long session has been active")
    is_complete: bool = Field(..., description="Whether classification is finished")

# Error Response Models

class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When error occurred"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error_type": "session_not_found",
                "message": "Session with ID 'abc123' was not found",
                "details": {"session_id": "abc123"},
                "timestamp": "2024-01-15T10:30:00"
            }
        }