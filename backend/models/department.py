from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from utils.math_utils import cosine_similarity

class TaqneeqDepartment(BaseModel):
    """Department model with all information and trait weights"""
    id: str = Field(..., description="Unique department identifier")
    name: str = Field(..., description="Display name of department")
    description: str = Field(..., description="Department description")
    core_responsibilities: List[str] = Field(..., description="Main responsibilities")
    skills_required: List[str] = Field(..., description="Required skills")
    soft_skills_required: List[str] = Field(..., description="Required soft skills")
    skills_perks_gained: List[str] = Field(..., description="Skills and perks gained")
    example_tasks: List[str] = Field(..., description="Example tasks")
    target_audience: List[str] = Field(..., description="Target audience")
    trait_weights: Dict[str, float] = Field(..., description="Trait importance weights (0.0-1.0)")
    
    def get_trait_vector(self) -> List[float]:
        """Get trait weights as ordered vector for calculations"""
        # Standard trait order (must match validation.py)
        trait_order = [
            "coding_aptitude", "hardware_technical", "digital_design", "web_development",
            "visual_creativity", "content_creation", "hands_on_crafting", "innovation_ideation",
            "stakeholder_management", "team_collaboration", "public_interaction", "networking_ability",
            "logistics_coordination", "process_management", "event_execution", "strategic_planning",
            "business_development", "financial_management", "leadership_initiative", "analytical_thinking"
        ]
        
        return [self.trait_weights.get(trait, 0.0) for trait in trait_order]
    
    def get_top_traits(self, top_k: int = 5) -> List[Dict[str, any]]:
        """Get top traits for this department"""
        sorted_traits = sorted(
            self.trait_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = []
        for trait_name, weight in sorted_traits[:top_k]:
            if weight > 0.3:  # Only include meaningful traits
                result.append({
                    "trait": trait_name.replace('_', ' ').title(),
                    "weight": round(weight, 3),
                    "importance": "Critical" if weight > 0.9 else "High" if weight > 0.7 else "Moderate"
                })
        
        return result
    
    def calculate_match_score(self, user_traits: Dict[str, float]) -> float:
        """Calculate how well user traits match this department"""
        if not user_traits or not self.trait_weights:
            return 0.0
        
        # Use cosine similarity for matching
        try:
            return cosine_similarity(user_traits, self.trait_weights)
        except ValueError:
            # Fallback to simple weighted average if cosine similarity fails
            return self._calculate_weighted_match(user_traits)
    
    def _calculate_weighted_match(self, user_traits: Dict[str, float]) -> float:
        """Fallback matching using weighted average"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for trait, dept_weight in self.trait_weights.items():
            if trait in user_traits:
                user_score = user_traits[trait]
                # Higher department weight means this trait matters more
                contribution = user_score * dept_weight
                weighted_sum += contribution * dept_weight  # Weight the contribution again
                total_weight += dept_weight * dept_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_trait_alignment(self, user_traits: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Get detailed trait-by-trait alignment analysis"""
        alignment = {}
        
        for trait, dept_weight in self.trait_weights.items():
            if trait in user_traits:
                user_score = user_traits[trait]
                
                # Calculate alignment (1.0 = perfect match, 0.0 = opposite)
                alignment_score = 1.0 - abs(user_score - dept_weight)
                
                # Calculate importance-weighted contribution
                contribution = alignment_score * dept_weight
                
                alignment[trait] = {
                    "user_score": round(user_score, 3),
                    "dept_importance": round(dept_weight, 3),
                    "alignment": round(alignment_score, 3),
                    "weighted_contribution": round(contribution, 3),
                    "status": self._get_alignment_status(alignment_score, dept_weight)
                }
        
        return alignment
    
    def _get_alignment_status(self, alignment_score: float, dept_importance: float) -> str:
        """Get human-readable alignment status"""
        if dept_importance < 0.3:
            return "Low Importance"
        elif alignment_score > 0.8:
            return "Excellent Match"
        elif alignment_score > 0.6:
            return "Good Match"
        elif alignment_score > 0.4:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def get_summary_for_user(self, include_details: bool = False) -> Dict[str, any]:
        """Get user-friendly department summary"""
        summary = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "key_responsibilities": self.core_responsibilities[:3],  # Top 3
            "example_tasks": self.example_tasks[:2],  # Top 2
            "target_audience": self.target_audience,
            "top_traits": self.get_top_traits(3)
        }
        
        if include_details:
            summary.update({
                "all_responsibilities": self.core_responsibilities,
                "skills_required": self.skills_required,
                "soft_skills": self.soft_skills_required,
                "growth_opportunities": self.skills_perks_gained,
                "all_example_tasks": self.example_tasks,
                "trait_profile": self.get_top_traits(10)
            })
        
        return summary
    
    def compare_with(self, other_dept: 'TaqneeqDepartment') -> Dict[str, any]:
        """Compare this department with another"""
        similarity = cosine_similarity(self.trait_weights, other_dept.trait_weights)
        
        # Find key differences
        trait_differences = {}
        for trait in self.trait_weights:
            if trait in other_dept.trait_weights:
                diff = self.trait_weights[trait] - other_dept.trait_weights[trait]
                if abs(diff) > 0.2:  # Only significant differences
                    trait_differences[trait] = {
                        "self_weight": self.trait_weights[trait],
                        "other_weight": other_dept.trait_weights[trait],
                        "difference": diff,
                        "advantage": self.name if diff > 0 else other_dept.name
                    }
        
        return {
            "similarity_score": round(similarity, 3),
            "relationship": "Very Similar" if similarity > 0.8 else "Similar" if similarity > 0.6 else "Different",
            "key_differences": trait_differences,
            "complementary": similarity < 0.4  # Very different departments might be complementary
        }

class DepartmentCatalog(BaseModel):
    """Collection of all departments with search and analysis capabilities"""
    departments: Dict[str, TaqneeqDepartment] = Field(default_factory=dict)
    
    def add_department(self, department: TaqneeqDepartment):
        """Add a department to the catalog"""
        self.departments[department.id] = department
    
    def get_by_id(self, dept_id: str) -> Optional[TaqneeqDepartment]:
        """Get department by ID"""
        return self.departments.get(dept_id)
    
    def search_by_name(self, query: str) -> List[TaqneeqDepartment]:
        """Search departments by name"""
        query = query.lower().strip()
        results = []
        
        for dept in self.departments.values():
            if query in dept.name.lower():
                results.append(dept)
        
        return sorted(results, key=lambda d: d.name.lower().find(query))
    
    def search_by_description(self, query: str) -> List[Tuple[TaqneeqDepartment, float]]:
        """Search departments by description with relevance scoring"""
        query = query.lower().strip()
        results = []
        
        for dept in self.departments.values():
            score = 0
            
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
                results.append((dept, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def find_similar_departments(self, dept_id: str, top_k: int = 3) -> List[Tuple[TaqneeqDepartment, float]]:
        """Find departments similar to the given one"""
        target_dept = self.get_by_id(dept_id)
        if not target_dept:
            return []
        
        similarities = []
        for other_id, other_dept in self.departments.items():
            if other_id != dept_id:
                similarity = cosine_similarity(
                    target_dept.trait_weights, 
                    other_dept.trait_weights
                )
                similarities.append((other_dept, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_departments_by_trait(self, trait: str, min_weight: float = 0.5) -> List[Tuple[TaqneeqDepartment, float]]:
        """Get departments that value a specific trait highly"""
        results = []
        
        for dept in self.departments.values():
            weight = dept.trait_weights.get(trait, 0.0)
            if weight >= min_weight:
                results.append((dept, weight))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def analyze_trait_distribution(self) -> Dict[str, Dict[str, float]]:
        """Analyze how traits are distributed across departments"""
        trait_stats = {}
        
        # Get all unique traits
        all_traits = set()
        for dept in self.departments.values():
            all_traits.update(dept.trait_weights.keys())
        
        for trait in all_traits:
            weights = [dept.trait_weights.get(trait, 0.0) for dept in self.departments.values()]
            
            trait_stats[trait] = {
                "min": min(weights),
                "max": max(weights),
                "mean": sum(weights) / len(weights),
                "high_value_depts": len([w for w in weights if w > 0.7]),
                "total_depts": len(self.departments)
            }
        
        return trait_stats