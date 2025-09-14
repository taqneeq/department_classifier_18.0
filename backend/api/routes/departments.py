from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
import logging

from core.data_loader import load_departments

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache departments data
_departments_cache: Optional[dict] = None

def get_departments() -> dict:
    """Get departments data with caching"""
    global _departments_cache
    if _departments_cache is None:
        _departments_cache = load_departments()
    return _departments_cache

@router.get("/departments", response_model=List[dict])
async def list_departments(
    include_traits: bool = Query(False, description="Include trait weights in response")
):
    """
    Get list of all available departments
    
    Args:
        include_traits: Whether to include trait weights (default: False for lighter response)
    """
    try:
        departments = get_departments()
        
        result = []
        for dept_id, dept in departments.items():
            dept_info = {
                "id": dept.id,
                "name": dept.name,
                "description": dept.description,
                "core_responsibilities": dept.core_responsibilities[:3],  # Limit for overview
                "example_tasks": dept.example_tasks[:2]  # Limit for overview
            }
            
            if include_traits:
                dept_info["trait_weights"] = dept.trait_weights
                dept_info["top_traits"] = dept.get_top_traits(5)
            
            result.append(dept_info)
        
        logger.info(f"Retrieved {len(result)} departments, include_traits={include_traits}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load departments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load departments"
        )

# IMPORTANT: Search route MUST come before {department_id} route to avoid conflicts
@router.get("/departments/search", response_model=List[dict])
async def search_departments(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(5, ge=1, le=20, description="Maximum results to return")
):
    """
    Search departments by name, description, or responsibilities
    """
    try:
        departments = get_departments()
        query = q.lower().strip()
        
        matches = []
        for dept_id, dept in departments.items():
            score = 0
            
            # Search in name (highest weight)
            if query in dept.name.lower():
                score += 10
            
            # Search in description
            if query in dept.description.lower():
                score += 5
            
            # Search in responsibilities
            for responsibility in dept.core_responsibilities:
                if query in responsibility.lower():
                    score += 3
                    break
            
            # Search in skills
            for skill in dept.skills_required:
                if query in skill.lower():
                    score += 2
                    break
            
            if score > 0:
                matches.append((dept, score))
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for dept, score in matches[:limit]:
            result.append({
                "id": dept.id,
                "name": dept.name,
                "description": dept.description,
                "relevance_score": score,
                "core_responsibilities": dept.core_responsibilities[:2]
            })
        
        logger.info(f"Search for '{query}' found {len(result)} results")
        return result
        
    except Exception as e:
        logger.error(f"Failed to search departments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search departments"
        )

# This route MUST come after the search route
@router.get("/departments/{department_id}", response_model=dict)
async def get_department_details(
    department_id: str,
    include_traits: bool = Query(True, description="Include trait weights")
):
    """
    Get detailed information about a specific department
    """
    try:
        departments = get_departments()
        
        if department_id not in departments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Department '{department_id}' not found"
            )
        
        dept = departments[department_id]
        
        result = {
            "id": dept.id,
            "name": dept.name,
            "description": dept.description,
            "core_responsibilities": dept.core_responsibilities,
            "skills_required": dept.skills_required,
            "soft_skills_required": dept.soft_skills_required,
            "skills_perks_gained": dept.skills_perks_gained,
            "example_tasks": dept.example_tasks,
            "target_audience": dept.target_audience
        }
        
        if include_traits:
            result["trait_weights"] = dept.trait_weights
            result["top_traits"] = dept.get_top_traits(10)
        
        logger.info(f"Retrieved details for department {department_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get department details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get department details"
        )

@router.get("/departments/{department_id}/similar", response_model=List[dict])
async def get_similar_departments(
    department_id: str,
    limit: int = Query(3, ge=1, le=10, description="Number of similar departments to return")
):
    """
    Get departments similar to the specified department based on trait weights
    """
    try:
        departments = get_departments()
        
        if department_id not in departments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Department '{department_id}' not found"
            )
        
        target_dept = departments[department_id]
        target_traits = target_dept.get_trait_vector()
        
        # Calculate similarities
        similarities = []
        for dept_id, dept in departments.items():
            if dept_id != department_id:  # Exclude self
                from utils.math_utils import cosine_similarity
                similarity = cosine_similarity(target_traits, dept.get_trait_vector())
                similarities.append((dept_id, dept, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        result = []
        for dept_id, dept, similarity in similarities[:limit]:
            result.append({
                "id": dept.id,
                "name": dept.name,
                "description": dept.description,
                "similarity_score": round(similarity, 3),
                "core_responsibilities": dept.core_responsibilities[:2]
            })
        
        logger.info(f"Found {len(result)} similar departments to {department_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar departments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar departments"
        )
