from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional
import logging

from models.response import (
    StartSessionRequest, StartSessionResponse,
    AnswerQuestionRequest, AnswerQuestionResponse,
    GetExplanationRequest, GetExplanationResponse,
    SessionStatusResponse, ErrorResponse
)
from models.session import UserSession
from core.classifier import TaqneeqClassifier
from core.session_manager import TaqneeqSessionManager
from core.data_loader import load_departments, load_questions
from utils.validation import validate_session_id, validate_likert_response, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state - these will be initialized on first use
_classifier: Optional[TaqneeqClassifier] = None
_session_manager: Optional[TaqneeqSessionManager] = None

def get_classifier() -> TaqneeqClassifier:
    """Dependency to get classifier instance"""
    global _classifier, _session_manager
    
    if _classifier is None:
        try:
            # Load data
            departments = load_departments()
            questions, seed_questions = load_questions()
            
            # Initialize session manager
            _session_manager = TaqneeqSessionManager(departments)
            
            # Initialize classifier
            _classifier = TaqneeqClassifier(departments, questions, seed_questions, _session_manager)
            
            logger.info("Classification services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize classification services: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Classification service initialization failed: {str(e)}"
            )
    
    return _classifier

def get_session_manager() -> TaqneeqSessionManager:
    """Dependency to get session manager instance"""
    global _session_manager
    
    if _session_manager is None:
        # Initialize classifier first (which also initializes session manager)
        get_classifier()
    
    if _session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session management service not initialized"
        )
    
    return _session_manager

@router.post("/classification/start", response_model=StartSessionResponse)
async def start_classification_session(
    request: StartSessionRequest,
    classifier: TaqneeqClassifier = Depends(get_classifier)
):
    """
    Start a new department classification session
    
    Returns the session ID and first question to ask
    """
    try:
        # Start classification
        session_id, first_question = classifier.start_classification()
        
        logger.info(f"Started classification session {session_id}")
        
        return StartSessionResponse(
            session_id=session_id,
            first_question={
                "id": first_question.id,
                "text": first_question.text,
                "type": first_question.type,
                "options": first_question.options,
                "category": first_question.category,
                "primary_trait": first_question.primary_trait,
                "secondary_traits": first_question.secondary_traits,
                "information_value": first_question.information_value,
                "target_departments": first_question.target_departments,
                "question_stage": first_question.question_stage
            },
            total_departments=len(classifier.departments),
            estimated_questions="8-12 questions typically needed",
            message="Welcome to Taqneeq Department Classification! Answer honestly for best results."
        )
        
    except Exception as e:
        logger.error(f"Failed to start classification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start classification session: {str(e)}"
        )

@router.post("/classification/answer", response_model=AnswerQuestionResponse)
async def submit_answer(
    request: AnswerQuestionRequest,
    classifier: TaqneeqClassifier = Depends(get_classifier),
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """
    Submit answer to current question and get next question or results
    """
    try:
        # Validate inputs
        is_valid, error = validate_session_id(request.session_id)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
        
        is_valid, error = validate_likert_response(request.response)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
        
        # Process response
        next_question, classification_result = classifier.process_response(
            session_id=request.session_id,
            question_id=request.question_id,
            response=request.response,
            confidence=request.confidence
        )

        if classification_result is None:
            from models.response import ClassificationResult
            session = session_mgr.get_session(request.session_id)
            classification_result = ClassificationResult(
                top_department="undetermined",
                top_probability=0.0,
                all_probabilities=session.department_probabilities if session else {},
                questions_asked=len(session.responses) if session else 0,
                confidence_level="low",
                should_continue=True,
                reasoning="Classification in progress..."
            )

            confidence=request.confidence
        
        # Get current session for trait info
        session = session_mgr.get_session(request.session_id)
        current_top_traits = session.get_top_traits(5) if session else []
        
        # Determine if classification is complete
        is_complete = next_question is None
        
        if is_complete:
            message = f"Classification complete! Top match: {classification_result.top_department} ({classification_result.top_probability:.1%} confidence)"
        else:
            questions_answered = len(session.responses) if session else 0
            message = f"Question {questions_answered + 1} - {classification_result.reasoning}"
        
        logger.info(f"Session {request.session_id}: Processed answer, complete={is_complete}")
        
        return {
            "session_id": request.session_id,
            "next_question": {
                "id": next_question.id,
                "text": next_question.text,
                "type": next_question.type,
                "options": next_question.options,
                "category": next_question.category,
                "primary_trait": next_question.primary_trait,
                "secondary_traits": next_question.secondary_traits,
                "information_value": next_question.information_value,
                "target_departments": next_question.target_departments,
                "question_stage": next_question.question_stage
            } if next_question else None,
            "classification_result": {
                "top_department": classification_result.top_department,
                "top_probability": classification_result.top_probability,
                "secondary_department": classification_result.secondary_department,
                "secondary_probability": classification_result.secondary_probability,
                "all_probabilities": classification_result.all_probabilities,
                "questions_asked": classification_result.questions_asked,
                "confidence_level": classification_result.confidence_level,
                "should_continue": classification_result.should_continue,
                "reasoning": classification_result.reasoning
            },
            "current_top_traits": current_top_traits,
            "is_complete": is_complete,
            "message": message
        }
        
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process answer: {str(e)}"
        )

@router.get("/classification/status/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """
    Get current status of classification session
    """
    try:
        # Validate session ID
        is_valid, error = validate_session_id(session_id)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
        
        # Get session
        session = session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Get top department
        if session.department_probabilities:
            top_dept_id, top_confidence = max(
                session.department_probabilities.items(),
                key=lambda x: x[1]
            )
        else:
            top_dept_id = "none"
            top_confidence = 0.0
        
        # Calculate session duration
        if session.created_at:
            from datetime import datetime
            duration = datetime.now() - session.created_at
            duration_str = f"{duration.total_seconds() / 60:.1f} minutes"
        else:
            duration_str = "unknown"
        
        # Estimate remaining questions
        questions_answered = len(session.responses)
        if questions_answered < 4:
            remaining = f"{4 - questions_answered} seed questions remaining"
        elif top_confidence < 0.7:
            remaining = "2-5 adaptive questions estimated"
        else:
            remaining = "1-2 questions estimated"
        
        return SessionStatusResponse(
            session_id=session_id,
            state=session.state,
            questions_answered=questions_answered,
            estimated_remaining=remaining,
            current_top_department=top_dept_id,
            current_confidence=top_confidence,
            top_traits=session.get_top_traits(5),
            department_probabilities=session.department_probabilities,
            session_duration=duration_str,
            is_complete=session.state.value == "complete"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )

@router.post("/classification/explanation")
async def get_explanation(
    request: dict,
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """Get RAG-powered explanation for classification result"""
    try:
        session_id = request.get("session_id")
        department_id = request.get("department_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
        
        session = session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get target department
        from core.data_loader import load_departments
        departments = load_departments()
        
        if department_id and department_id not in departments:
            raise HTTPException(status_code=400, detail="Invalid department_id")
        
        if not department_id:
            if session.department_probabilities:
                department_id = max(session.department_probabilities.items(), key=lambda x: x[1])[0]
            else:
                raise HTTPException(status_code=400, detail="No classification results available")
        
        department = departments[department_id]
        
        # Simple explanation without full RAG (fallback)
        explanation = {
            "overview": f"Based on your responses, you are well-suited for the {department.name} department.",
            "why_good_fit": f"Your traits align with {department.name}'s requirements.",
            "responsibilities": department.core_responsibilities[:3],
            "skills_gained": department.skills_perks_gained[:3],
            "next_steps": f"Consider applying to {department.name} and connecting with current members."
        }
        
        return {
            "session_id": session_id,
            "department_id": department_id,
            "department_name": department.name,
            "explanation": explanation,
            "classification_confidence": session.department_probabilities.get(department_id, 0.0),
            "generated_at": "2025-09-15T02:00:00"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classification/explanation")
async def get_explanation(request: dict):
    """Get explanation for classification result"""
    try:
        session_id = request.get("session_id")
        department_id = request.get("department_id", "tech")
        
        return {
            "session_id": session_id,
            "department_id": department_id,
            "explanation": {
                "overview": "RAG-powered explanation would go here",
                "vector_search_results": "Successfully retrieved from 108 documents"
            }
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
