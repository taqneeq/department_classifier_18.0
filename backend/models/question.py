from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum

class QuestionType(str, Enum):
    """Types of questions available"""
    LIKERT_5 = "likert_5"
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"

class QuestionStage(str, Enum):
    """Question stages in classification process"""
    SEED = "seed"
    ADAPTIVE = "adaptive"
    FOLLOWUP = "followup"

class TaqneeqQuestion(BaseModel):
    """Individual classification question"""
    id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., description="Question text to display to user")
    type: QuestionType = Field(QuestionType.LIKERT_5, description="Type of question")
    options: List[str] = Field(default_factory=lambda: ["1", "2", "3", "4", "5"])
    scale_labels: Optional[Dict[str, str]] = Field(
        default={
            "1": "Strongly Disagree",
            "2": "Disagree", 
            "3": "Neutral",
            "4": "Agree",
            "5": "Strongly Agree"
        },
        description="Labels for scale points"
    )
    category: str = Field(..., description="Question category for grouping")
    primary_trait: str = Field(..., description="Main trait this question measures")
    secondary_traits: List[str] = Field(default_factory=list, description="Additional traits influenced")
    information_value: float = Field(..., ge=0.1, le=3.0, description="Expected information gain")
    targets_departments: List[str] = Field(default_factory=list, description="Departments this question helps identify")
    question_stage: QuestionStage = Field(QuestionStage.ADAPTIVE, description="Stage when question is used")
    
    class Config:
        use_enum_values = True
    
    def get_trait_impact(self) -> Dict[str, float]:
        """Get the relative impact this question has on different traits"""
        impact = {self.primary_trait: 1.0}
        
        # Secondary traits have 50% impact of primary
        for trait in self.secondary_traits:
            impact[trait] = 0.5
        
        return impact
    
    def format_for_display(self) -> Dict[str, any]:
        """Format question for user display"""
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "options": [
                {
                    "value": option,
                    "label": self.scale_labels.get(option, option) if self.scale_labels else option
                }
                for option in self.options
            ],
            "category": self.category.replace('_', ' ').title(),
            "stage": self.question_stage
        }
    
    def is_seed_question(self) -> bool:
        """Check if this is a seed question"""
        return self.question_stage == QuestionStage.SEED
    
    def get_information_weight(self) -> float:
        """Get weighted information value based on question type"""
        base_weight = self.information_value
        
        # Seed questions get higher weight
        if self.question_stage == QuestionStage.SEED:
            return base_weight * 1.2
        
        return base_weight

class QuestionBank(BaseModel):
    """Collection of all classification questions"""
    questions: Dict[str, TaqneeqQuestion] = Field(default_factory=dict)
    seed_questions: List[TaqneeqQuestion] = Field(default_factory=list)
    
    def add_question(self, question: TaqneeqQuestion):
        """Add a question to the bank"""
        self.questions[question.id] = question
        
        if question.is_seed_question():
            self.seed_questions.append(question)
    
    def get_questions_by_stage(self, stage: QuestionStage) -> List[TaqneeqQuestion]:
        """Get questions filtered by stage"""
        return [q for q in self.questions.values() if q.question_stage == stage]
    
    def get_questions_by_trait(self, trait: str) -> List[TaqneeqQuestion]:
        """Get questions that measure a specific trait"""
        return [
            q for q in self.questions.values() 
            if q.primary_trait == trait or trait in q.secondary_traits
        ]
    
    def get_questions_for_departments(self, department_ids: List[str]) -> List[TaqneeqQuestion]:
        """Get questions that help identify specific departments"""
        return [
            q for q in self.questions.values()
            if any(dept in q.targets_departments for dept in department_ids)
        ]
    
    def get_high_value_questions(self, min_value: float = 1.5) -> List[TaqneeqQuestion]:
        """Get questions with high information value"""
        return [
            q for q in self.questions.values()
            if q.information_value >= min_value
        ]
    
    def validate_question_coverage(self, required_traits: List[str]) -> Dict[str, any]:
        """Validate that all required traits have question coverage"""
        covered_traits = set()
        
        for question in self.questions.values():
            covered_traits.add(question.primary_trait)
            covered_traits.update(question.secondary_traits)
        
        missing_traits = set(required_traits) - covered_traits
        
        # Count questions per trait
        trait_counts = {}
        for trait in required_traits:
            trait_counts[trait] = len(self.get_questions_by_trait(trait))
        
        return {
            "total_questions": len(self.questions),
            "seed_questions": len(self.seed_questions),
            "covered_traits": len(covered_traits),
            "missing_traits": list(missing_traits),
            "trait_question_counts": trait_counts,
            "average_info_value": sum(q.information_value for q in self.questions.values()) / len(self.questions) if self.questions else 0.0
        }