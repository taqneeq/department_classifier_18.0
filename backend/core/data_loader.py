import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.session_manager import TaqneeqSessionManager, UserSession, SessionState
logger = logging.getLogger(__name__)    

TRAIT_NAMES = [
    "coding_aptitude",
    "hardware_technical", 
    "digital_design",
    "web_development",
    "visual_creativity",
    "content_creation",
    "hands_on_crafting",
    "innovation_ideation",
    "stakeholder_management",
    "team_collaboration",
    "public_interaction", 
    "networking_ability",
    "logistics_coordination",
    "process_management",
    "event_execution",
    "strategic_planning",
    "business_development",
    "financial_management",
    "leadership_initiative",
    "analytical_thinking"
]

class TaqneeqQuestion(BaseModel):
    id: str
    text: str
    type: str
    options: List[str]
    category: str
    primary_trait: str
    secondary_traits: List[str]
    information_value: float
    target_departments: List[str]  # This is what the model expects
    question_stage: str

class TaqneeqDepartment(BaseModel):
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

def load_departments(file_path: str = "data/departments.json") -> Dict[str, TaqneeqDepartment]:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Departments file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        departments = {}
        for dept_data in data['departments']:
            department = TaqneeqDepartment(**dept_data)
            departments[department.id] = department
        
        logger.info(f"Successfully loaded {len(departments)} departments")
        return departments
        
    except Exception as e:
        logger.error(f"Failed to load departments: {e}")
        raise

def load_questions(file_path: str = "data/question_bank.json") -> Tuple[Dict[str, TaqneeqQuestion], List[TaqneeqQuestion]]:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Questions file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load all questions with field name fix
        all_questions = {}
        for q_data in data['question_bank']:
            # Fix field name mismatch: targets_departments -> target_departments
            if 'targets_departments' in q_data:
                q_data['target_departments'] = q_data.pop('targets_departments')
            
            question = TaqneeqQuestion(**q_data)
            all_questions[question.id] = question
        
        # Load seed questions with field name fix
        seed_questions = []
        for q_data in data['seed_questions']:
            # Fix field name mismatch: targets_departments -> target_departments
            if 'targets_departments' in q_data:
                q_data['target_departments'] = q_data.pop('targets_departments')
            
            question = TaqneeqQuestion(**q_data)
            seed_questions.append(question)
            all_questions[question.id] = question  # Also add to main dict
        
        logger.info(f"Successfully loaded {len(all_questions)} total questions ({len(seed_questions)} seed)")
        return all_questions, seed_questions
        
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        raise

def validate_data_integrity(departments: Dict[str, TaqneeqDepartment], questions: Dict[str, TaqneeqQuestion]) -> Tuple[bool, List[str]]:
    errors = []
    
    # Validate trait consistency
    for dept_id, dept in departments.items():
        # Check all traits are present
        missing_traits = set(TRAIT_NAMES) - set(dept.trait_weights.keys())
        if missing_traits:
            errors.append(f"Department {dept_id} missing traits: {missing_traits}")
        
        # Check trait weights are in valid range
        for trait, weight in dept.trait_weights.items():
            if not 0.0 <= weight <= 1.0:
                errors.append(f"Department {dept_id} trait {trait} has invalid weight: {weight}")
    
    # Validate questions
    for q_id, question in questions.items():
        if question.type != "likert_5":
            errors.append(f"Question {q_id} has invalid type: {question.type}")

        if question.options != ["1", "2", "3", "4", "5"]:
            errors.append(f"Question {q_id} has invalid options: {question.options}")
        
        if question.primary_trait not in TRAIT_NAMES:
            errors.append(f"Question {q_id} has invalid primary trait: {question.primary_trait}")

        for trait in question.secondary_traits:
            if trait not in TRAIT_NAMES:
                errors.append(f"Question {q_id} has invalid secondary trait: {trait}")

        for dept_id in question.target_departments:
            if dept_id not in departments:
                errors.append(f"Question {q_id} targets unknown department: {dept_id}")
    
    if len(questions) < 20:
        errors.append(f"Insufficient questions for classification ({len(questions)} < 20)")
    
    # Check seed questions
    seed_count = sum(1 for q in questions.values() if q.question_stage == "seed")
    if seed_count != 4:
        errors.append(f"Expected exactly 4 seed questions, found {seed_count}")
    
    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Data validation passed")
    else:
        logger.warning(f"Data validation failed with {len(errors)} errors")
    
    return is_valid, errors