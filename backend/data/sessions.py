import uuid
from typing import Dict
from data_models import UserSession, Question, Department

def initialize_session() -> UserSession:
    return UserSession(session_id=str(uuid.uuid4()))

def add_response(session: UserSession, question: Question, likert_score: float) -> None:
    normalized = (likert_score - 1) / 4
    session.responses[question.id] = normalized
    for trait, weight in question.trait_weights.items():
        session.trait_scores[trait] = session.trait_scores.get(trait, 0) + normalized * weight

def get_current_probabilities(session: UserSession, departments: Dict[str, Department]) -> Dict[str, float]:
    scores = {}
    for dept in departments.values():
        score = sum(session.trait_scores.get(t, 0) * w for t, w in dept.traits.items())
        scores[dept.id] = score
    total = sum(scores.values()) or 1.0
    session.department_probs = {k: v / total for k, v in scores.items()}
    return session.department_probs
