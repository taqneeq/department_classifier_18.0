import json
from typing import List
from data_models import Department, Question

def load_departments(path: str) -> List[Department]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Department(**d) for d in raw]

def load_questions(path: str) -> List[Question]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Question(**q) for q in raw]

def validate_data_integrity(departments: List[Department], questions: List[Question]) -> None:
    dept_traits = {t for d in departments for t in d.traits}
    for q in questions:
        missing = set(q.trait_weights) - dept_traits
        if missing:
            raise ValueError(f"Question {q.id} references unknown traits: {missing}")
