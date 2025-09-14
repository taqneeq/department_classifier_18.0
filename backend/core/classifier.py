import math
import random
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel  # CORRECT
from datetime import datetime
from enum import Enum
import logging
from core.data_loader import TaqneeqQuestion, TaqneeqDepartment
from core.session_manager import TaqneeqSessionManager, UserSession, SessionState
from models.session import SessionState
logger = logging.getLogger(__name__)

class ClassificationResult(BaseModel):
    """Classification result with confidence metrics"""
    top_department: str
    top_probability: float
    secondary_department: Optional[str] = None
    secondary_probability: Optional[float] = None
    all_probabilities: Dict[str, float]
    questions_asked: int
    confidence_level: str  # "high", "medium", "low"
    should_continue: bool
    reasoning: str

class TaqneeqClassifier:
    """Main classification engine using information gain"""
    
    def __init__(self, departments: Dict[str, TaqneeqDepartment], 
                 questions: Dict[str, TaqneeqQuestion],
                 seed_questions: List[TaqneeqQuestion],
                 session_manager: TaqneeqSessionManager):
        
        self.departments = departments
        self.questions = questions
        self.seed_questions = seed_questions
        self.session_manager = session_manager
        
        # Classification thresholds
        self.high_confidence_threshold = 0.85
        self.medium_confidence_threshold = 0.70
        self.max_questions = 15
        self.min_questions = 6
        
        logger.info("TaqneeqClassifier initialized")
    
    def start_classification(self, session_id: Optional[str] = None) -> Tuple[str, TaqneeqQuestion]:
        """
        Start new classification session
        
        Returns:
            Tuple of (session_id, first_question)
        """
        session = self.session_manager.create_session(session_id)
        first_question = self.seed_questions[0]  # Always start with first seed question
        
        logger.info(f"Started classification for session {session.session_id}")
        return session.session_id, first_question
    
    def process_response(self, session_id: str, question_id: str, 
                        response: int, confidence: float = 1.0) -> Tuple[Optional[TaqneeqQuestion], ClassificationResult]:
        """
        Process user response and determine next question or final result
        
        Args:
            session_id: Session identifier
            question_id: Question that was answered
            response: Likert response (1-5)  
            confidence: User confidence (0.0-1.0)
        
        Returns:
            Tuple of (next_question, classification_result)
            If next_question is None, classification is complete
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        question = self.questions.get(question_id)
        if not question:
            raise ValueError(f"Question not found: {question_id}")
        
        # Add response to session
        success = self.session_manager.add_response(session_id, question, response, confidence)
        if not success:
            raise ValueError(f"Failed to add response to session {session_id}")
        
        # Always create a classification result
        questions_asked = len(session.responses)
        
        # Get current probabilities
        probs = session.department_probabilities
        sorted_depts = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_dept, top_prob = sorted_depts[0]
        second_dept, second_prob = sorted_depts[1] if len(sorted_depts) > 1 else (None, 0.0)
        
        # Determine if we should stop
        should_stop = False
        confidence_level = "low"
        reasoning = "Continuing classification..."
        
        if questions_asked >= self.max_questions:
            should_stop = True
            reasoning = f"Maximum questions ({self.max_questions}) reached"
            confidence_level = "medium"
        elif top_prob >= self.high_confidence_threshold:
            should_stop = True
            confidence_level = "high"
            reasoning = f"High confidence achieved ({top_prob:.1%})"
        elif (questions_asked >= self.min_questions and 
              top_prob >= self.medium_confidence_threshold and 
              top_prob - second_prob >= 0.25):
            should_stop = True
            confidence_level = "medium"
            reasoning = f"Clear leader identified ({top_prob:.1%} vs {second_prob:.1%})"
        else:
            if top_prob >= 0.7:
                confidence_level = "medium"
                reasoning = f"Good progress, {top_dept} leading with {top_prob:.1%}"
            elif top_prob >= 0.5:
                confidence_level = "low"
                reasoning = f"Moderate confidence, {top_dept} at {top_prob:.1%}"
            else:
                confidence_level = "low"
                reasoning = f"Still determining best match, top is {top_prob:.1%}"
        
        # Create classification result
        classification_result = ClassificationResult(
            top_department=top_dept,
            top_probability=top_prob,
            secondary_department=second_dept,
            secondary_probability=second_prob,
            all_probabilities=probs,
            questions_asked=questions_asked,
            confidence_level=confidence_level,
            should_continue=not should_stop,
            reasoning=reasoning
        )
        
        # Get next question if continuing
        next_question = None
        if not should_stop:
            next_question = self._select_next_question(session)
            if not next_question:
                # Force completion if no more questions
                should_stop = True
                classification_result.should_continue = False
                classification_result.reasoning = "No more questions available"
        
        # Update session state if complete
        if should_stop:
            session.state = SessionState.CLASSIFICATION_COMPLETE
            session.completed_at = datetime.now()
            logger.info(f"Classification complete for session {session_id}: {classification_result.top_department}")
        else:
            logger.info(f"Session {session_id}: Continuing classification, {questions_asked} questions asked")
        
        return next_question, classification_result
    def _select_next_question(self, session: UserSession) -> Optional[TaqneeqQuestion]:
        """Select the most informative next question"""
        questions_answered = len(session.responses)
        
        # Phase 1: Seed questions (1-4)
        if questions_answered < 4:
            return self.seed_questions[questions_answered]
        
        # Phase 2: Adaptive questions based on information gain
        available_questions = [
            q for q in self.questions.values()
            if q.id not in session.questions_asked and q.question_stage == "adaptive"
        ]
        
        if not available_questions:
            return None
        
        # Calculate information gain for each available question
        best_question = None
        max_information_gain = -1
        
        for question in available_questions:
            info_gain = self._calculate_information_gain(session, question)
            weighted_gain = info_gain * question.information_value
            
            if weighted_gain > max_information_gain:
                max_information_gain = weighted_gain
                best_question = question
        
        return best_question
    
    def _calculate_information_gain(self, session: UserSession, question: TaqneeqQuestion) -> float:
        """
        Calculate expected information gain from asking a question
        
        This simulates possible responses and measures entropy reduction
        """
        current_entropy = self._calculate_entropy(session.department_probabilities)
        
        # Simulate responses 1-5 with equal probability
        expected_entropy = 0.0
        
        for response in [1, 2, 3, 4, 5]:
            # Create temporary session copy for simulation
            temp_trait_scores = session.trait_scores.copy()
            
            # Simulate trait update
            normalized_response = (response - 1) / 4.0
            update_strength = self.session_manager.learning_rate
            
            # Update primary trait
            current_score = temp_trait_scores[question.primary_trait]
            temp_trait_scores[question.primary_trait] = (
                current_score * (1 - update_strength) + normalized_response * update_strength
            )
            
            # Update secondary traits
            secondary_strength = update_strength * 0.5
            for trait in question.secondary_traits:
                current_score = temp_trait_scores[trait]
                temp_trait_scores[trait] = (
                    current_score * (1 - secondary_strength) + normalized_response * secondary_strength
                )
            
            # Calculate resulting department probabilities
            temp_probs = {}
            for dept_id, department in self.departments.items():
                similarity = self.session_manager._calculate_trait_similarity(
                    temp_trait_scores, department.trait_weights
                )
                temp_probs[dept_id] = similarity
            
            # Normalize probabilities
            temp_probs = self.session_manager._softmax(temp_probs)
            
            # Add to expected entropy (uniform weighting for responses)
            expected_entropy += self._calculate_entropy(temp_probs) / 5.0
        
        # Information gain = current entropy - expected entropy
        information_gain = current_entropy - expected_entropy
        return max(0.0, information_gain)  # Ensure non-negative
    
    def _calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate Shannon entropy of probability distribution"""
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 1e-10:  # Avoid log(0)
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _should_stop_classification(self, session: UserSession, 
                                  force_completion: bool = False) -> Tuple[bool, ClassificationResult]:
        """
        Determine if classification should stop and generate result
        
        Args:
            session: Current user session
            force_completion: Force completion even if thresholds not met
        
        Returns:
            Tuple of (should_stop, classification_result)
        """
        probs = session.department_probabilities
        questions_asked = len(session.responses)
        
        # Sort departments by probability
        sorted_depts = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        top_dept, top_prob = sorted_depts[0]
        second_dept, second_prob = sorted_depts[1] if len(sorted_depts) > 1 else (None, 0.0)
        
        # Determine confidence level and stopping criteria
        should_stop = False
        reasoning = ""
        confidence_level = "low"
        
        if force_completion:
            should_stop = True
            reasoning = "Maximum questions reached or no more questions available"
        elif questions_asked >= self.max_questions:
            should_stop = True
            reasoning = f"Maximum questions ({self.max_questions}) reached"
        elif top_prob >= self.high_confidence_threshold:
            should_stop = True
            confidence_level = "high"
            reasoning = f"High confidence achieved ({top_prob:.1%})"
        elif (questions_asked >= self.min_questions and 
              top_prob >= self.medium_confidence_threshold and 
              top_prob - second_prob >= 0.25):
            should_stop = True
            confidence_level = "medium"
            reasoning = f"Clear leader identified ({top_prob:.1%} vs {second_prob:.1%})"
        
        # Determine confidence level for non-stopping cases
        if not should_stop:
            if top_prob >= 0.7:
                confidence_level = "medium"
            elif top_prob >= 0.5:
                confidence_level = "low"
            else:
                confidence_level = "low"
        
        result = ClassificationResult(
            top_department=top_dept,
            top_probability=top_prob,
            secondary_department=second_dept,
            secondary_probability=second_prob,
            all_probabilities=probs,
            questions_asked=questions_asked,
            confidence_level=confidence_level,
            should_continue=not should_stop,
            reasoning=reasoning
        )
        
        return should_stop, result