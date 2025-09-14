from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
import logging
import psutil
import os
from datetime import datetime
from core.session_manager import TaqneeqSessionManager
from core.data_loader import load_departments, load_questions, validate_data_integrity
from core.classifier import TaqneeqClassifier
from rag.retriever import TaqneeqRetriever
from rag.explanation_generator import TaqneeqExplanationGenerator   
from rag.document_processor import TaqneeqDocumentProcessor 
from fastapi import Query

logger = logging.getLogger(__name__)

# Global instances
_session_manager_instance = None
_classifier_instance = None

def get_session_manager() -> TaqneeqSessionManager:
    """Dependency to get session manager instance"""
    global _session_manager_instance
    if _session_manager_instance is None:
        try:
            departments = load_departments()
            _session_manager_instance = TaqneeqSessionManager(departments)
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session manager initialization failed"
            )
    return _session_manager_instance

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    try:
        # Check if core components are loadable
        from core.data_loader import load_departments, load_questions
        
        # Basic data loading test
        departments = load_departments()
        questions, seed_questions = load_questions()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Taqneeq Department Classifier",
            "version": "1.0.0",
            "components": {
                "departments_loaded": len(departments),
                "questions_loaded": len(questions),
                "seed_questions": len(seed_questions),
                "data_integrity": "validated"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with system metrics
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Load data to test integrity
        from core.data_loader import load_departments, load_questions, validate_data_integrity
        
        departments = load_departments()
        questions, seed_questions = load_questions()
        
        # Validate data integrity
        is_valid, errors = validate_data_integrity(departments, questions)
        
        return {
            "status": "healthy" if is_valid else "degraded",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
            },
            "data": {
                "departments_count": len(departments),
                "questions_count": len(questions),
                "seed_questions_count": len(seed_questions),
                "data_valid": is_valid,
                "validation_errors": errors if not is_valid else []
            },
            "features": {
                "classification_engine": "available",
                "rag_system": "available",
                "session_management": "available"
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/stats")
async def get_classification_stats(
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """
    Get classification statistics and usage metrics
    """
    try:
        sessions = session_mgr.sessions
        
        if not sessions:
            return {
                "message": "No classification sessions yet",
                "total_sessions": 0
            }
        
        # Calculate statistics
        total_sessions = len(sessions)
        completed_sessions = [s for s in sessions.values() if s.state.value == "complete"]
        active_sessions = [s for s in sessions.values() if s.state.value in ["seed_questions", "adaptive_questions"]]
        
        # Department popularity
        dept_counts = {}
        for session in completed_sessions:
            if session.department_probabilities:
                top_dept = max(session.department_probabilities.items(), key=lambda x: x[1])[0]
                dept_counts[top_dept] = dept_counts.get(top_dept, 0) + 1
        
        # Average questions per session
        question_counts = [len(s.responses) for s in sessions.values() if s.responses]
        avg_questions = sum(question_counts) / len(question_counts) if question_counts else 0
        
        # Response distribution
        all_responses = []
        for session in sessions.values():
            all_responses.extend([r.response for r in session.responses])
        
        response_distribution = {}
        for response in all_responses:
            response_distribution[str(response)] = response_distribution.get(str(response), 0) + 1
        
        return {
            "summary": {
                "total_sessions": total_sessions,
                "completed_sessions": len(completed_sessions),
                "active_sessions": len(active_sessions),
                "completion_rate": len(completed_sessions) / total_sessions if total_sessions > 0 else 0
            },
            "metrics": {
                "average_questions_per_session": round(avg_questions, 1),
                "total_responses": len(all_responses),
                "response_distribution": response_distribution
            },
            "popular_departments": sorted(dept_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get classification stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get classification statistics"
        )

@router.get("/stats/departments")
async def get_department_stats():
    """
    Get statistics about department trait distributions
    """
    try:
        from core.data_loader import load_departments
        departments = load_departments()
        
        # Analyze trait distributions
        trait_stats = {}
        for trait_name in departments[list(departments.keys())[0]].trait_weights.keys():
            values = [dept.trait_weights[trait_name] for dept in departments.values()]
            trait_stats[trait_name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "departments_with_high_score": [
                    dept.name for dept in departments.values() 
                    if dept.trait_weights[trait_name] >= 0.8
                ]
            }
        
        return {
            "total_departments": len(departments),
            "total_traits": len(trait_stats),
            "trait_statistics": trait_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get department stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get department statistics"
        )

@router.post("/admin/clear-sessions")
async def clear_all_sessions(
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """
    Clear all classification sessions (admin only)
    """
    try:
        session_count = len(session_mgr.sessions)
        session_mgr.sessions.clear()
        
        logger.warning(f"Cleared {session_count} sessions via admin endpoint")
        
        return {
            "message": f"Cleared {session_count} sessions",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear sessions"
        )

@router.get("/admin/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=200, description="Maximum sessions to return"),
    session_mgr: TaqneeqSessionManager = Depends(get_session_manager)
):
    """
    List current sessions (admin only)
    """
    try:
        sessions = list(session_mgr.sessions.values())
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        
        result = []
        for session in sessions[:limit]:
            session_info = {
                "session_id": session.session_id,
                "state": session.state.value,
                "questions_answered": len(session.responses),
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                "top_department": None,
                "top_probability": 0.0
            }
            
            # Add top department if available
            if session.department_probabilities:
                top_dept, top_prob = max(
                    session.department_probabilities.items(),
                    key=lambda x: x[1]
                )
                session_info["top_department"] = top_dept
                session_info["top_probability"] = round(top_prob, 3)
            
            result.append(session_info)
        
        return {
            "sessions": result,
            "total_sessions": len(session_mgr.sessions),
            "showing": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions"
        )
