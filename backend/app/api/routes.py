from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
import logging
from datetime import datetime

from ..core.classifier import TaqneeqClassifier
from ..core.models import (
    StartSessionRequest, AnswerQuestionRequest, ExplanationRequest,
    ClassificationResult, SessionState
)
from ..rag.engine import TaqneeqRAG
from ..config import settings

logger = logging.getLogger(__name__)

# Initialize global instances
classifier = TaqneeqClassifier()
rag_engine = TaqneeqRAG(classifier.departments) if settings.ENABLE_RAG else None

router = APIRouter()

# CLASSIFICATION ENDPOINTS

@router.post("/classification/start")
async def start_classification(request: StartSessionRequest):
    """Start new classification session"""
    try:
        session_id, first_question = classifier.start_session()
        
        response_data = {
            "session_id": session_id,
            "first_question": {
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
            "total_departments": len(classifier.departments),
            "estimated_questions": "8-12 questions typically needed",
            "message": "Welcome to Taqneeq Department Classification! Answer honestly for best results.",
            "features": {
                "adaptive_questioning": True,
                "rag_explanations": rag_engine is not None and rag_engine.initialized,
                "trait_based_matching": True
            }
        }
        
        logger.info(f"Started classification session {session_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to start classification: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start classification session: {str(e)}"
        )

@router.post("/classification/answer")
async def submit_answer(request: AnswerQuestionRequest):
    """Submit answer and get next question or results"""
    try:
        next_question, result = classifier.process_response(
            request.session_id,
            request.question_id,
            request.response,
            request.confidence
        )
        
        # Format next question data
        next_question_data = None
        if next_question:
            next_question_data = {
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
            }
        
        # Create response
        response_data = {
            "next_question": next_question_data,
            "classification_result": {
                "session_id": result.session_id,
                "top_department": result.top_department,
                "top_probability": result.top_probability,
                "secondary_department": result.secondary_department,
                "secondary_probability": result.secondary_probability,
                "all_probabilities": result.all_probabilities,
                "questions_asked": result.questions_asked,
                "confidence_level": result.confidence_level,
                "should_continue": result.should_continue,
                "is_complete": result.is_complete,
                "current_top_traits": result.current_top_traits,
                "reasoning": result.reasoning
            },
            "message": (
                f"Classification complete! Top match: {result.top_department} "
                f"({result.top_probability:.1%} confidence)"
                if result.is_complete else
                f"Question {result.questions_asked} processed - {result.reasoning}"
            )
        }
        
        logger.info(f"Processed answer for session {request.session_id}, "
                   f"complete={result.is_complete}")
        return response_data
        
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process answer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process answer: {str(e)}"
        )

@router.post("/classification/explanation")
async def get_explanation(request: ExplanationRequest):
    """Get RAG-powered or template explanation for classification result"""
    try:
        session = classifier.sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Determine target department
        dept_id = request.department_id
        if not dept_id and session.department_probabilities:
            dept_id = max(session.department_probabilities.items(), key=lambda x: x[1])[0]
        
        if not dept_id:
            raise HTTPException(status_code=400, detail="No department specified or determined")
        
        department = classifier.departments.get(dept_id)
        if not department:
            raise HTTPException(status_code=400, detail="Department not found")
        
        # Generate explanation
        if rag_engine:
            explanation = rag_engine.generate_explanation(dept_id, session)
        else:
            # Simple fallback explanation
            top_traits = session.get_top_traits(3)
            explanation = {
                "overview": f"Based on your responses, you are well-suited for the {department.name} department.",
                "why_good_fit": f"Your top traits ({', '.join([trait for trait, _ in top_traits])}) align with {department.name}'s requirements.",
                "responsibilities": "; ".join(department.core_responsibilities[:3]),
                "skills_gained": "; ".join(department.skills_perks_gained[:3]),
                "next_steps": f"Apply to {department.name} and connect with current members."
            }
        
        # Get classification confidence
        confidence = session.department_probabilities.get(dept_id, 0.0)
        
        # Build response
        response_data = {
            "session_id": request.session_id,
            "department_id": dept_id,
            "department_name": department.name,
            "explanation": explanation,
            "classification_confidence": round(confidence, 3),
            "user_top_traits": session.get_top_traits(5),
            "alternative_departments": [],
            "generated_at": datetime.now().isoformat(),
            "generation_method": "rag" if (rag_engine and rag_engine.initialized) else "template"
        }
        
        # Add alternative departments if requested
        if request.include_comparison and len(session.department_probabilities) > 1:
            sorted_depts = sorted(
                session.department_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )[1:4]  # Get top 2-4 alternatives
            
            for alt_dept_id, alt_prob in sorted_depts:
                alt_dept = classifier.departments.get(alt_dept_id)
                if alt_dept:
                    response_data["alternative_departments"].append({
                        "id": alt_dept_id,
                        "name": alt_dept.name,
                        "probability": round(alt_prob, 3),
                        "description": alt_dept.description[:150] + "..."
                    })
        
        logger.info(f"Generated explanation for session {request.session_id}, "
                   f"department {dept_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )

@router.get("/classification/status/{session_id}")
async def get_session_status(session_id: str):
    """Get current status of classification session"""
    try:
        status = classifier.get_session_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session status: {str(e)}"
        )

# DEPARTMENT ENDPOINTS

@router.get("/departments")
async def list_departments(
    include_traits: bool = Query(False, description="Include trait weights"),
    search: Optional[str] = Query(None, description="Search by name or description")
):
    """Get list of all departments with optional filtering"""
    try:
        departments = list(classifier.departments.values())
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            departments = [
                dept for dept in departments
                if (search_lower in dept.name.lower() or
                    search_lower in dept.description.lower() or
                    any(search_lower in resp.lower() for resp in dept.core_responsibilities))
            ]
        
        # Build response
        result = []
        for dept in departments:
            dept_info = {
                "id": dept.id,
                "name": dept.name,
                "description": dept.description,
                "core_responsibilities": dept.core_responsibilities[:3],
                "example_tasks": dept.example_tasks[:2],
                "target_audience": dept.target_audience
            }
            
            if include_traits:
                dept_info["top_traits"] = dept.get_top_traits(5)
                dept_info["trait_weights"] = dept.trait_weights
            
            result.append(dept_info)
        
        logger.info(f"Listed {len(result)} departments, search='{search}', "
                   f"include_traits={include_traits}")
        return {
            "departments": result,
            "total": len(result),
            "search_applied": search is not None,
            "traits_included": include_traits
        }
        
    except Exception as e:
        logger.error(f"Failed to list departments: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve departments"
        )

@router.get("/departments/{department_id}")
async def get_department_details(
    department_id: str,
    include_traits: bool = Query(True, description="Include trait analysis")
):
    """Get detailed information about a specific department"""
    try:
        department = classifier.departments.get(department_id)
        if not department:
            raise HTTPException(status_code=404, detail="Department not found")
        
        result = {
            "id": department.id,
            "name": department.name,
            "description": department.description,
            "core_responsibilities": department.core_responsibilities,
            "skills_required": department.skills_required,
            "soft_skills_required": department.soft_skills_required,
            "skills_perks_gained": department.skills_perks_gained,
            "example_tasks": department.example_tasks,
            "target_audience": department.target_audience
        }
        
        if include_traits:
            result["top_traits"] = department.get_top_traits(10)
            result["trait_weights"] = department.trait_weights
            result["trait_analysis"] = {
                "highest_weight": max(department.trait_weights.values()),
                "critical_traits": [
                    trait for trait, weight in department.trait_weights.items()
                    if weight > 0.9
                ],
                "important_traits": [
                    trait for trait, weight in department.trait_weights.items()
                    if 0.7 <= weight <= 0.9
                ]
            }
        
        logger.info(f"Retrieved details for department {department_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get department details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve department details: {str(e)}"
        )

@router.get("/departments/{department_id}/similar")
async def get_similar_departments(
    department_id: str,
    limit: int = Query(3, ge=1, le=10, description="Number of similar departments")
):
    """Get departments similar to the specified one based on trait weights"""
    try:
        target_dept = classifier.departments.get(department_id)
        if not target_dept:
            raise HTTPException(status_code=404, detail="Department not found")
        
        # Calculate similarities with other departments
        similarities = []
        for other_id, other_dept in classifier.departments.items():
            if other_id != department_id:
                from ..core.utils import cosine_similarity
                similarity = cosine_similarity(
                    target_dept.trait_weights,
                    other_dept.trait_weights
                )
                similarities.append((other_dept, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for dept, similarity in similarities[:limit]:
            result.append({
                "id": dept.id,
                "name": dept.name,
                "description": dept.description,
                "similarity_score": round(similarity, 3),
                "core_responsibilities": dept.core_responsibilities[:2],
                "relationship": (
                    "Very Similar" if similarity > 0.8 else
                    "Similar" if similarity > 0.6 else
                    "Somewhat Similar"
                )
            })
        
        logger.info(f"Found {len(result)} similar departments to {department_id}")
        return {
            "target_department": {
                "id": target_dept.id,
                "name": target_dept.name
            },
            "similar_departments": result,
            "total_found": len(result)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar departments: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to find similar departments"
        )

# MONITORING ENDPOINTS  

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Test data loading
        dept_count = len(classifier.departments)
        question_count = len(classifier.questions)
        seed_count = len(classifier.seed_questions)
        
        # Test classification engine
        if dept_count == 0 or question_count == 0:
            status = "unhealthy"
            detail = "Missing required data"
        else:
            status = "healthy"
            detail = "All systems operational"
        
        # RAG system status
        rag_status = "disabled"
        if rag_engine:
            if rag_engine.initialized:
                rag_status = "operational"
            else:
                rag_status = "failed_initialization"
        
        health_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "service": "Taqneeq Department Classifier",
            "version": "2.0.0",
            "components": {
                "departments_loaded": dept_count,
                "questions_loaded": question_count,
                "seed_questions": seed_count,
                "active_sessions": len(classifier.sessions),
                "classification_engine": "operational" if dept_count > 0 else "failed",
                "rag_system": rag_status,
                "vector_store": "ready" if (rag_engine and rag_engine.vector_store) else "unavailable"
            },
            "configuration": {
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "max_questions": settings.MAX_QUESTIONS,
                "rag_enabled": settings.ENABLE_RAG,
                "openai_configured": bool(settings.OPENAI_API_KEY)
            }
        }
        
        if status == "unhealthy":
            return HTTPException(status_code=503, detail=health_data)
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/stats")
async def get_usage_statistics():
    """Get system usage statistics and analytics"""
    try:
        sessions = list(classifier.sessions.values())
        completed_sessions = [s for s in sessions if s.state == SessionState.COMPLETE]
        active_sessions = [s for s in sessions if s.state in [SessionState.SEED_QUESTIONS, SessionState.ADAPTIVE_QUESTIONS]]
        
        # Department popularity among completed sessions
        dept_counts = {}
        for session in completed_sessions:
            if session.department_probabilities:
                top_dept = max(session.department_probabilities.items(), key=lambda x: x[1])[0]
                dept_counts[top_dept] = dept_counts.get(top_dept, 0) + 1
        
        # Response analysis
        all_responses = []
        question_counts = []
        for session in sessions:
            all_responses.extend([r.response for r in session.responses])
            question_counts.append(len(session.responses))
        
        # Calculate statistics
        avg_questions = sum(question_counts) / len(question_counts) if question_counts else 0
        completion_rate = len(completed_sessions) / len(sessions) if sessions else 0
        
        # Response distribution
        response_dist = {}
        for response in all_responses:
            response_dist[str(response)] = response_dist.get(str(response), 0) + 1
        
        stats_data = {
            "summary": {
                "total_sessions": len(sessions),
                "completed_sessions": len(completed_sessions),
                "active_sessions": len(active_sessions),
                "completion_rate": round(completion_rate, 3)
            },
            "metrics": {
                "average_questions_per_session": round(avg_questions, 1),
                "total_responses": len(all_responses),
                "response_distribution": response_dist
            },
            "popular_departments": sorted(
                dept_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "system_info": {
                "departments_available": len(classifier.departments),
                "questions_in_bank": len(classifier.questions),
                "rag_explanations_generated": "available" if rag_engine else "unavailable"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return stats_data
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve usage statistics"
        )

@router.post("/admin/cleanup")
async def cleanup_sessions(max_age_hours: int = Query(24, ge=1, le=168)):
    """Clean up old sessions (admin endpoint)"""
    try:
        initial_count = len(classifier.sessions)
        classifier.cleanup_expired_sessions(max_age_hours)
        final_count = len(classifier.sessions)
        cleaned = initial_count - final_count
        
        logger.info(f"Session cleanup: removed {cleaned} sessions older than {max_age_hours}h")
        
        return {
            "sessions_removed": cleaned,
            "sessions_remaining": final_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cleanup sessions"
        )
