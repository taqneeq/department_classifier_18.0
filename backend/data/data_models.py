from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Department(BaseModel):
    id: str
    name: str
    traits: Dict[str, float] = Field(..., description="Trait name → weight")

class Question(BaseModel):
    id: str
    text: str
    trait_weights: Dict[str, float] = Field(..., description="Trait name → weight")
    options: Optional[List[str]] = None

class UserSession(BaseModel):
    session_id: str
    responses: Dict[str, float] = {}
    trait_scores: Dict[str, float] = {}
    department_probs: Dict[str, float] = {}

class ClassificationResult(BaseModel):
    top_department: str
    second_department: Optional[str] = None
    probabilities: Dict[str, float]
    explanation: Optional[str] = None
