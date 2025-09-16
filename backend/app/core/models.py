from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class SessionState(str, Enum):
    """Session states for tracking classification progress"""
    INITIALIZED = "initialized"
    SEED_QUESTIONS = "seed_questions"
    ADAPTIVE_QUESTIONS = "adaptive_questions"
    COMPLETE = "complete"
    EXPIRED = "expired"

class Question(BaseModel):
    """Classification question model"""
    id: str
    text: str
    type: str = "likert_5"
    options: List[str] = ["1", "2", "3", "4", "5"]
    category: str
    primary_trait: str
    secondary_traits: List[str] = []
    information_value: float = Field(..., ge=0.1, le=3.0)
    target_departments: List[str] = []
    question_stage: str = "adaptive"
    
    def get_trait_impact(self) -> Dict[str, float]:
        """Get trait impact weights for this question"""
        impact = {self.primary_trait: 1.0}
        for trait in self.secondary_traits:
            impact[trait] = 0.5
        return impact

class Department(BaseModel):
    """Department model with complete information"""
    id: str
    name: str
    description: str
    core_responsibilities: List[str]
    skills_required: List[str]
    soft_skills_required: List[str]
    skills_perks_gained: List[str]
    example_tasks: List[str]
    target_audience: List[str]
    trait_weights: Dict[str, float]
    
    def get_top_traits(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top traits for this department"""
        sorted_traits = sorted(self.trait_weights.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "trait": trait.replace('_', ' ').title(),
                "weight": round(weight, 3),
                "importance": "Critical" if weight > 0.9 else "High" if weight > 0.7 else "Moderate"
            }
            for trait, weight in sorted_traits[:top_k] if weight > 0.3
        ]
    
    def calculate_match_score(self, user_traits: Dict[str, float]) -> float:
        """Calculate user-department match score"""
        from .utils import cosine_similarity
        return cosine_similarity(user_traits, self.trait_weights)

class UserResponse(BaseModel):
    """Individual user response to a question"""
    question_id: str
    response: int = Field(..., ge=1, le=5, description="Likert scale response")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="User confidence")
    timestamp: datetime = Field(default_factory=datetime.now)

class Session(BaseModel):
    """User session with complete state"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZED
    trait_scores: Dict[str, float] = Field(default_factory=dict)
    department_probabilities: Dict[str, float] = Field(default_factory=dict)
    responses: List[UserResponse] = []
    questions_asked: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    last_activity: datetime = Field(default_factory=datetime.now)
    
    def get_top_traits(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get user's strongest traits"""
        if not self.trait_scores:
            return []
        
        sorted_traits = sorted(self.trait_scores.items(), key=lambda x: x[1], reverse=True)
        return [(trait.replace('_', ' ').title(), score) 
                for trait, score in sorted_traits[:top_k] if score > 0.5]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get session progress summary"""
        questions_answered = len(self.responses)
        
        # Estimate progress
        if self.state == SessionState.SEED_QUESTIONS:
            progress = min(questions_answered / 4.0, 1.0)
        elif self.state == SessionState.ADAPTIVE_QUESTIONS:
            progress = min(questions_answered / 10.0, 0.95)
        else:
            progress = 1.0
        
        # Get current leader
        top_department = None
        top_confidence = 0.0
        if self.department_probabilities:
            top_dept, top_conf = max(self.department_probabilities.items(), key=lambda x: x[1])
            top_department = top_dept
            top_confidence = top_conf
        
        return {
            "progress_percentage": round(progress * 100, 1),
            "questions_answered": questions_answered,
            "current_state": self.state.value,
            "current_top_department": top_department,
            "current_confidence": round(top_confidence, 3),
            "session_duration_minutes": self._get_duration_minutes()
        }
    
    def _get_duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        if not self.created_at:
            return 0.0
        end_time = self.completed_at or datetime.now()
        duration = end_time - self.created_at
        return round(duration.total_seconds() / 60.0, 1)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

# API Request/Response Models
class StartSessionRequest(BaseModel):
    """Request to start new classification session"""
    user_metadata: Optional[Dict[str, Any]] = None

class AnswerQuestionRequest(BaseModel):
    """Request to submit question answer"""
    session_id: str
    question_id: str
    response: int = Field(..., ge=1, le=5)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

class ExplanationRequest(BaseModel):
    """Request for department explanation"""
    session_id: str
    department_id: Optional[str] = None
    include_comparison: bool = False

class ClassificationResult(BaseModel):
    """Classification result model"""
    session_id: str
    top_department: str
    top_probability: float
    secondary_department: Optional[str] = None
    secondary_probability: Optional[float] = None
    all_probabilities: Dict[str, float]
    questions_asked: int
    confidence_level: str
    should_continue: bool
    is_complete: bool
    current_top_traits: List[Tuple[str, float]]
    reasoning: str