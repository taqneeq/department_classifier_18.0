import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from typing import Tuple  # CORRECT
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.data_loader import TaqneeqQuestion, TaqneeqDepartment
from core.data_loader import TRAIT_NAMES
logger = logging.getLogger(__name__)

class SessionState(str, Enum):
    """Session states for tracking progress"""
    INITIALIZED = "initialized"
    SEED_QUESTIONS = "seed_questions"  
    ADAPTIVE_QUESTIONS = "adaptive_questions"
    CLASSIFICATION_COMPLETE = "complete"
    EXPIRED = "expired"

class UserResponse(BaseModel):
    question_id: str
    response: float
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)

class UserSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZED
    trait_scores: Dict[str, float] = Field(default_factory=dict)
    department_probabilities: Dict[str, float] = Field(default_factory=dict)
    responses: List[UserResponse] = Field(default_factory=list)
    questions_asked: List[str] = Field(default_factory=list)
    current_question_index: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


    def get_top_traits(self, top_k: int = 5) -> List[Tuple[str, float]]:
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
                result.append((trait_name.replace('_', ' ').title(), score))
        
        return result


class TaqneeqSessionManager:   
    def __init__(self, departments: Dict[str, "TaqneeqDepartment"]):
        self.departments = departments
        self.sessions: Dict[str, UserSession] = {}
        self.learning_rate = 0.3

    def create_session(self, session_id: Optional[str] = None) -> UserSession:
        """Create a new session with initialized state"""
        session = UserSession()
        if session_id:
            session.session_id = session_id
            
        # Initialize uniform department probabilities
        num_depts = len(self.departments)
        session.department_probabilities = {
            dept_id: 1.0 / num_depts for dept_id in self.departments.keys()
        }
        
        # Initialize neutral trait scores (0.5 = neutral)
        session.trait_scores = {trait: 0.5 for trait in TRAIT_NAMES}
        
        # Set initial state
        session.state = SessionState.SEED_QUESTIONS
        
        self.sessions[session.session_id] = session
        logger.info(f"Created session {session.session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Retrieve session by ID"""
        return self.sessions.get(session_id)
    
    def add_response(self, session_id: str, question: "TaqneeqQuestion", 
                    response: int, confidence: float = 1.0) -> bool:
        """
        Add user response and update session state
        
        Args:
            session_id: Session identifier
            question: Question that was answered
            response: Likert response (1-5)
            confidence: User confidence in response (0.0-1.0)
        
        Returns:
            True if successful, False if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Create response record
        user_response = UserResponse(
            question_id=question.id,
            response=response,
            confidence=confidence
        )
        
        # Add to session
        session.responses.append(user_response)
        session.questions_asked.append(question.id)
        
        # Update trait scores based on response
        self._update_trait_scores(session, question, response, confidence)
        
        # Update department probabilities
        self._update_department_probabilities(session)
        
        # Update session state
        self._update_session_state(session)
        
        logger.debug(f"Session {session_id}: Added response to {question.id}, response={response}")
        
        return True
    
    def _update_trait_scores(self, session: UserSession, question: "TaqneeqQuestion", 
                           response: int, confidence: float):
        """Update user trait scores based on response"""
        # Normalize Likert response to 0.0-1.0
        normalized_response = (response - 1) / 4.0
        
        # Weight by confidence and learning rate
        update_strength = confidence * self.learning_rate
        
        # Update primary trait
        current_score = session.trait_scores[question.primary_trait]
        new_score = current_score * (1 - update_strength) + normalized_response * update_strength
        session.trait_scores[question.primary_trait] = max(0.0, min(1.0, new_score))
        
        # Update secondary traits with reduced strength
        secondary_strength = update_strength * 0.5
        for trait in question.secondary_traits:
            current_score = session.trait_scores[trait]
            new_score = current_score * (1 - secondary_strength) + normalized_response * secondary_strength
            session.trait_scores[trait] = max(0.0, min(1.0, new_score))
    
    def _update_department_probabilities(self, session: UserSession):
        """Update department probabilities based on current trait scores"""
        similarities = {}
        
        for dept_id, department in self.departments.items():
            # Calculate cosine similarity between user traits and department traits
            similarity = self._calculate_trait_similarity(
                session.trait_scores, 
                department.trait_weights
            )
            similarities[dept_id] = similarity
        
        # Convert similarities to probabilities using softmax
        session.department_probabilities = self._softmax(similarities)
    
    def _calculate_trait_similarity(self, user_traits: Dict[str, float], 
                                  dept_traits: Dict[str, float]) -> float:
        """Calculate cosine similarity between trait vectors"""
        # Only consider traits that exist in both vectors
        common_traits = set(user_traits.keys()) & set(dept_traits.keys())
        if not common_traits:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(user_traits[trait] * dept_traits[trait] for trait in common_traits)
        
        user_magnitude = sum(user_traits[trait]**2 for trait in common_traits)**0.5
        dept_magnitude = sum(dept_traits[trait]**2 for trait in common_traits)**0.5
        
        # Avoid division by zero
        if user_magnitude == 0 or dept_magnitude == 0:
            return 0.0
        
        return dot_product / (user_magnitude * dept_magnitude)
    
    def _softmax(self, scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Convert scores to probabilities using softmax"""
        import math
        
        # Apply temperature scaling
        scaled_scores = {k: v / temperature for k, v in scores.items()}
        
        # Compute softmax with numerical stability
        max_score = max(scaled_scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scaled_scores.items()}
        
        total = sum(exp_scores.values())
        if total == 0:
            # Uniform distribution as fallback
            return {k: 1.0 / len(scores) for k in scores.keys()}
        
        return {k: v / total for k, v in exp_scores.items()}
    
    def _update_session_state(self, session: UserSession):
        """Update session state based on progress"""
        questions_answered = len(session.responses)
        
        if questions_answered <= 4:
            session.state = SessionState.SEED_QUESTIONS
        else:
            session.state = SessionState.ADAPTIVE_QUESTIONS
    
    def get_top_departments(self, session_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top K departments with their probabilities"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        sorted_depts = sorted(
            session.department_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_depts[:top_k]


# Add this to core/session_manager.py or create a separate dependencies.py file

from core.session_manager import TaqneeqSessionManager
from core.data_loader import load_departments

# Global session manager instance
_session_manager_instance = None

def get_session_manager() -> TaqneeqSessionManager:
    """Dependency to get session manager instance"""
    global _session_manager_instance
    if _session_manager_instance is None:
        departments = load_departments()
        _session_manager_instance = TaqneeqSessionManager(departments)
    return _session_manager_instance
