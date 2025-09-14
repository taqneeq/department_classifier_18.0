from typing import Dict, List, Optional
import logging
from datetime import datetime

# from langchain.llms import OpenAI  # Deprecated - using fallback
# from langchain.chat_models import ChatOpenAI  # Deprecated - using fallback
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.docstore.document import Document

from core.session_manager import UserSession
from core.data_loader import TaqneeqDepartment
from core.classifier import ClassificationResult

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        # Fallback for when OpenAI is not available
        ChatOpenAI = None


logger = logging.getLogger(__name__)

class DepartmentExplanation(BaseOutputParser):
    """Output parser for structured department explanations"""
    
    def parse(self, text: str) -> Dict[str, str]:
        """Parse LLM output into structured explanation"""
        try:
            # Simple parsing - in production, you might want more sophisticated parsing
            sections = {}
            current_section = None
            current_content = []
            
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('## '):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = line[3:].lower().replace(' ', '_')
                    current_content = []
                elif line and current_section:
                    current_content.append(line)
            
            # Add final section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Ensure we have required sections
            if 'overview' not in sections and 'summary' not in sections:
                sections['overview'] = text[:200] + "..."
            
            return sections
            
        except Exception as e:
            logger.error(f"Failed to parse explanation: {e}")
            return {'overview': text}

class TaqneeqExplanationGenerator:
    """Generate personalized department explanations using LLM"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 openai_api_key: Optional[str] = None):
        """
        Initialize explanation generator
        
        Args:
            model_name: OpenAI model to use
            temperature: Creativity level (0.0-1.0)
            openai_api_key: OpenAI API key (if not in environment)
        """
        
        # Initialize LLM
        if openai_api_key:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature
            )
        
        self.output_parser = DepartmentExplanation()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=self.output_parser
        )
        
        logger.info(f"TaqneeqExplanationGenerator initialized with model: {model_name}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for explanations"""
        
        system_template = """You are an expert advisor helping students find their perfect department match for Taqneeq, a technical festival. Your role is to provide personalized, encouraging explanations about why a student is a good fit for a particular department.

Guidelines:
- Be enthusiastic and encouraging while remaining honest
- Focus on the student's strengths and how they align with the department
- Use the retrieved context to provide specific, accurate information
- Structure your response with clear sections
- Keep explanations concise but comprehensive
- Highlight growth opportunities and skills they'll gain

Format your response with these sections:
## Overview
## Why You're a Great Fit  
## What You'll Do
## Skills You'll Gain
## Next Steps"""

        human_template = """Based on the following information, generate a personalized explanation for why this student is recommended for the {department_name} department:

STUDENT PROFILE:
- Questions Answered: {questions_answered}
- Top Traits: {top_traits}
- Classification Confidence: {confidence}%

DEPARTMENT CONTEXT:
{context}

DEPARTMENT INFORMATION:
Name: {department_name}
Description: {department_description}

Generate an engaging, personalized explanation that shows why this student would thrive in {department_name}."""

        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def generate_explanation(self,
                           department: TaqneeqDepartment,
                           user_session: UserSession,
                           classification_result: ClassificationResult,
                           retrieved_docs: List[Document]) -> Dict[str, str]:
        """
        Generate personalized department explanation
        
        Args:
            department: Target department
            user_session: User session with trait scores
            classification_result: Classification results
            retrieved_docs: Retrieved context documents
            
        Returns:
            Dictionary with explanation sections
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs, department.id)
            
            # Get top user traits
            top_traits = self._get_top_traits(user_session.trait_scores)
            
            # Format inputs for prompt
            prompt_inputs = {
                'department_name': department.name,
                'department_description': department.description,
                'questions_answered': len(user_session.responses),
                'top_traits': top_traits,
                'confidence': int(classification_result.top_probability * 100),
                'context': context
            }
            
            # Generate explanation
            explanation = self.chain.run(**prompt_inputs)
            
            logger.info(f"Generated explanation for {department.name}")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            # Fallback explanation
            return self._generate_fallback_explanation(department, classification_result)
    
    def generate_comparison_explanation(self,
                                      primary_dept: TaqneeqDepartment,
                                      secondary_dept: TaqneeqDepartment,
                                      user_session: UserSession,
                                      classification_result: ClassificationResult,
                                      retrieved_docs: Dict[str, List[Document]]) -> Dict[str, str]:
        """
        Generate comparison between two departments
        
        Args:
            primary_dept: Top recommended department
            secondary_dept: Second choice department
            user_session: User session
            classification_result: Classification results
            retrieved_docs: Retrieved docs for both departments
            
        Returns:
            Dictionary with comparison explanation
        """
        try:
            # Create comparison prompt (simplified for this example)
            primary_context = self._prepare_context(retrieved_docs.get('primary', []), primary_dept.id)
            secondary_context = self._prepare_context(retrieved_docs.get('secondary', []), secondary_dept.id)
            
            top_traits = self._get_top_traits(user_session.trait_scores)
            
            comparison_prompt = f"""
            Based on the user's top traits ({top_traits}), explain why {primary_dept.name} 
            is the top recommendation over {secondary_dept.name}.
            
            {primary_dept.name} Context: {primary_context[:500]}...
            {secondary_dept.name} Context: {secondary_context[:500]}...
            
            Provide a brief comparison highlighting the key differences and why 
            {primary_dept.name} is a better fit.
            """
            
            explanation = self.llm.predict(comparison_prompt)
            
            return {
                'comparison': explanation,
                'primary_department': primary_dept.name,
                'secondary_department': secondary_dept.name
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comparison: {e}")
            return {
                'comparison': f"Based on your responses, {primary_dept.name} is your top match, "
                            f"while {secondary_dept.name} is also a good fit but with lower alignment."
            }
    
    def _prepare_context(self, documents: List[Document], target_dept_id: str) -> str:
        """Prepare context string from retrieved documents"""
        if not documents:
            return "No specific context available."
        
        context_parts = []
        for doc in documents:
            # Prioritize content from target department
            if doc.metadata.get('department_id') == target_dept_id:
                context_parts.insert(0, doc.page_content)
            else:
                context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)
    
    def _get_top_traits(self, trait_scores: Dict[str, float], top_k: int = 3) -> str:
        """Get formatted string of top traits"""
        sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_traits = []
        for trait_name, score in sorted_traits[:top_k]:
            if score > 0.6:  # Only include strong traits
                formatted_trait = trait_name.replace('_', ' ').title()
                top_traits.append(f"{formatted_trait} ({score:.1%})")
        
        return ", ".join(top_traits) if top_traits else "Balanced profile across multiple areas"
    
    def _generate_fallback_explanation(self, 
                                     department: TaqneeqDepartment,
                                     classification_result: ClassificationResult) -> Dict[str, str]:
        """Generate fallback explanation when LLM fails"""
        return {
            'overview': f"You've been matched with {department.name} based on your responses to our classification questions.",
            'why_great_fit': f"Your profile shows strong alignment with {department.name}'s requirements and culture.",
            'what_you_will_do': f"In {department.name}, you'll work on: " + "; ".join(department.core_responsibilities[:3]),
            'skills_you_will_gain': f"You'll develop: " + "; ".join(department.skills_perks_gained[:3]),
            'next_steps': f"Apply for {department.name}, connect with current members, and learn more about ongoing projects."
        }