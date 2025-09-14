import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

class SessionState(str, Enum):
    """Session states for tracking progress"""
    INITIALIZED = "initialized"
    SEED_QUESTIONS = "seed_questions"  
    ADAPTIVE_QUESTIONS = "adaptive_questions"
    CLASSIFICATION_COMPLETE = "complete"
    EXPIRED = "expired"

class UserResponse(BaseModel):
    """Individual question response from user"""
    question_id: str
    response: int = Field(..., ge=1, le=5, description="Likert scale response (1-5)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="User confidence in response")
    timestamp: datetime = Field(default_factory=datetime.now)

class UserSession(BaseModel):
    """Complete user session with all responses and state"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZED
    trait_scores: Dict[str, float] = Field(default_factory=dict)
    department_probabilities: Dict[str, float] = Field(default_factory=dict)
    responses: List[UserResponse] = Field(default_factory=list)
    questions_asked: List[str] = Field(default_factory=list)
    current_question_index: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    last_activity: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def get_top_traits(self, top_k: int = 5) -> List[Dict[str, any]]:
        """Get user's top traits sorted by score"""
        if not self.trait_scores:
            return []
        
        sorted_traits = sorted(
            self.trait_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = []
        for trait_name, score in sorted_traits[:top_k]:
            if score > 0.5:  # Only include traits above neutral
                result.append({
                    "trait": trait_name.replace('_', ' ').title(),
                    "score": round(score, 3),
                    "strength": "Strong" if score > 0.8 else "Moderate" if score > 0.65 else "Mild"
                })
        
        return result
    
    def get_progress_summary(self) -> Dict[str, any]:
        """Get progress summary for this session"""
        questions_answered = len(self.responses)
        
        # Estimate completion percentage
        if self.state == SessionState.SEED_QUESTIONS:
            progress = min(questions_answered / 4.0, 1.0)
        elif self.state == SessionState.ADAPTIVE_QUESTIONS:
            # Estimate 8-12 total questions typically needed
            estimated_total = 10
            progress = min(questions_answered / estimated_total, 0.95)
        else:
            progress = 1.0
        
        # Get current top department if available
        top_department = None
        top_confidence = 0.0
        if self.department_probabilities:
            top_dept, top_conf = max(
                self.department_probabilities.items(), 
                key=lambda x: x[1]
            )
            top_department = top_dept
            top_confidence = top_conf
        
        return {
            "progress_percentage": round(progress * 100, 1),
            "questions_answered": questions_answered,
            "current_state": self.state.value,
            "estimated_remaining": max(0, 8 - questions_answered) if questions_answered < 8 else "1-3",
            "current_top_department": top_department,
            "current_confidence": round(top_confidence, 3) if top_confidence else 0.0,
            "session_duration_minutes": self._calculate_duration_minutes()
        }
    
    def _calculate_duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        if not self.created_at:
            return 0.0
        
        end_time = self.completed_at or datetime.now()
        duration = end_time - self.created_at
        return round(duration.total_seconds() / 60.0, 1)
    
    def update_last_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()