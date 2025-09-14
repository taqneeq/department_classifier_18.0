import re
from typing import Dict, List, Any, Union, Optional, Tuple
from models.session import SessionState, UserResponse
import logging

logger = logging.getLogger(__name__)

# Define valid trait names (must match models/department.py)
VALID_TRAIT_NAMES = [
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

def validate_trait_scores(trait_scores: Dict[str, float]) -> Tuple[bool, List[str]]:

    errors = []
    
    if not isinstance(trait_scores, dict):
        errors.append("Trait scores must be a dictionary")
        return False, errors
    
    # Check for missing traits
    missing_traits = set(VALID_TRAIT_NAMES) - set(trait_scores.keys())
    if missing_traits:
        errors.append(f"Missing traits: {sorted(missing_traits)}")
    
    # Check for invalid traits
    invalid_traits = set(trait_scores.keys()) - set(VALID_TRAIT_NAMES)
    if invalid_traits:
        errors.append(f"Invalid traits: {sorted(invalid_traits)}")
    
    # Check trait score values
    for trait, score in trait_scores.items():
        if not isinstance(score, (int, float)):
            errors.append(f"Trait '{trait}' score must be numeric, got {type(score)}")
            continue
        
        if not 0.0 <= score <= 1.0:
            errors.append(f"Trait '{trait}' score must be between 0.0 and 1.0, got {score}")
    
    return len(errors) == 0, errors

def validate_department_probabilities(probabilities: Dict[str, float]) -> Tuple[bool, List[str]]:

    errors = []
    
    if not isinstance(probabilities, dict):
        errors.append("Probabilities must be a dictionary")
        return False, errors
    
    if not probabilities:
        errors.append("Probabilities dictionary cannot be empty")
        return False, errors
    
    # Check probability values
    total = 0.0
    for dept_id, prob in probabilities.items():
        if not isinstance(prob, (int, float)):
            errors.append(f"Probability for '{dept_id}' must be numeric, got {type(prob)}")
            continue
        
        if not 0.0 <= prob <= 1.0:
            errors.append(f"Probability for '{dept_id}' must be between 0.0 and 1.0, got {prob}")
            continue
        
        total += prob
    
    # Check if probabilities sum to approximately 1.0
    if not 0.99 <= total <= 1.01:  # Allow small floating point errors
        errors.append(f"Probabilities must sum to approximately 1.0, got {total}")
    
    return len(errors) == 0, errors

def validate_likert_response(response: Any, scale_min: int = 1, scale_max: int = 5) -> Tuple[bool, Optional[str]]:

    if not isinstance(response, int):
        if isinstance(response, float) and response.is_integer():
            response = int(response)
        else:
            return False, f"Response must be an integer, got {type(response)}"
    
    if not scale_min <= response <= scale_max:
        return False, f"Response must be between {scale_min} and {scale_max}, got {response}"
    
    return True, None

def validate_session_state(state: Any) -> Tuple[bool, Optional[str]]:

    if not isinstance(state, (str, SessionState)):
        return False, f"State must be string or SessionState enum, got {type(state)}"
    
    if isinstance(state, str):
        try:
            SessionState(state)
        except ValueError:
            valid_states = [s.value for s in SessionState]
            return False, f"Invalid state '{state}'. Valid states: {valid_states}"
    
    return True, None

def validate_question_response(question_id: str, response: int, confidence: Optional[float] = None) -> Tuple[bool, List[str]]:
    """
    Validate a complete question response
    
    Args:
        question_id: Question identifier
        response: Likert response
        confidence: Optional confidence score
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate question ID
    if not isinstance(question_id, str) or not question_id.strip():
        errors.append("Question ID must be a non-empty string")
    
    # Validate response
    is_valid, error = validate_likert_response(response)
    if not is_valid:
        errors.append(f"Invalid response: {error}")
    
    # Validate confidence if provided
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence must be numeric, got {type(confidence)}")
        elif not 0.0 <= confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0, got {confidence}")
    
    return len(errors) == 0, errors

def sanitize_input(input_str: str, max_length: int = 1000, allow_html: bool = False) -> str:

    if not isinstance(input_str, str):
        raise ValueError(f"Input must be string, got {type(input_str)}")
    
    # Trim whitespace
    sanitized = input_str.strip()
    
    # Check length
    if len(sanitized) > max_length:
        raise ValueError(f"Input too long: {len(sanitized)} > {max_length}")
    
    # Remove or escape HTML if not allowed
    if not allow_html:
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Escape remaining < and > characters
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
    
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    return sanitized

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(email, str):
        return False, "Email must be a string"
    
    email = email.strip().lower()
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    if len(email) > 254:  # RFC 5321 limit
        return False, "Email address too long"
    
    return True, None

def validate_session_id(session_id: str) -> Tuple[bool, Optional[str]]:

    if not isinstance(session_id, str):
        return False, "Session ID must be a string"
    
    session_id = session_id.strip()
    
    if not session_id:
        return False, "Session ID cannot be empty"
    
    # UUID format validation (optional but recommended)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, session_id, re.IGNORECASE):
        # Allow alphanumeric session IDs as fallback
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return False, "Session ID must be UUID format or alphanumeric"
    
    if len(session_id) > 100:
        return False, "Session ID too long"
    
    return True, None

def validate_confidence_score(confidence: Union[int, float]) -> Tuple[bool, Optional[str]]:

    if not isinstance(confidence, (int, float)):
        return False, f"Confidence must be numeric, got {type(confidence)}"
    
    if not 0.0 <= confidence <= 1.0:
        return False, f"Confidence must be between 0.0 and 1.0, got {confidence}"
    
    return True, None

def validate_information_value(info_value: Union[int, float]) -> Tuple[bool, Optional[str]]:

    if not isinstance(info_value, (int, float)):
        return False, f"Information value must be numeric, got {type(info_value)}"
    
    if not 0.1 <= info_value <= 3.0:
        return False, f"Information value must be between 0.1 and 3.0, got {info_value}"
    
    return True, None

class ValidationError(Exception):
    """Custom exception for validation errors"""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []
    
    def __str__(self):
        if self.errors:
            return f"{super().__str__()}: {'; '.join(self.errors)}"
        return super().__str__()

def validate_and_raise(validation_result: Tuple[bool, Union[str, List[str]]], 
                      operation: str = "validation") -> None:

    is_valid, errors = validation_result
    
    if not is_valid:
        if isinstance(errors, str):
            errors = [errors]
        raise ValidationError(f"{operation} failed", errors)