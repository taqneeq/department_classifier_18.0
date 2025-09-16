import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for RAG dependencies
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG dependencies not installed. Install with: pip install langchain langchain-community sentence-transformers faiss-cpu")
    RAG_AVAILABLE = False

# Check for OpenAI
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available. Install with: pip install langchain-openai openai")
    OPENAI_AVAILABLE = False

class TaqneeqRAG:
    """
    Simplified RAG system for generating department explanations
    """
    
    def __init__(self, departments: Dict[str, Any]):
        self.departments = departments
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.initialized = False
        
        if RAG_AVAILABLE:
            try:
                self._initialize_rag()
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {e}")
                logger.info("Falling back to simple explanations")
        else:
            logger.info("RAG system disabled - using fallback explanations")
    
    def _initialize_rag(self):
        """Initialize RAG components"""
        from ..config import settings
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM if API key available
        if settings.OPENAI_API_KEY and OPENAI_AVAILABLE:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=settings.OPENAI_API_KEY
            )
            logger.info("OpenAI LLM initialized")
        else:
            logger.info("OpenAI not configured - using template-based explanations")
        
        # Process documents
        documents = []
        
        # Create documents from department data
        dept_docs = self._create_department_documents()
        documents.extend(dept_docs)
        
        # Load PDF if available
        pdf_path = Path(settings.PDF_FILE)
        if pdf_path.exists():
            try:
                pdf_docs = self._load_pdf(str(pdf_path))
                documents.extend(pdf_docs)
                logger.info(f"Loaded PDF with {len(pdf_docs)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load PDF: {e}")
        else:
            logger.info(f"PDF file not found at {pdf_path}")
        
        # Build vector store
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"RAG initialized with {len(documents)} documents")
        else:
            logger.warning("No documents loaded for RAG")
    
    def _create_department_documents(self) -> List[Document]:
        """Create searchable documents from department data"""
        if not RAG_AVAILABLE:
            return []
        
        documents = []
        
        for dept_id, dept in self.departments.items():
            # Create comprehensive text content
            sections = [
                f"Department: {dept.name}",
                f"Description: {dept.description}",
                "",
                "Core Responsibilities:",
            ]
            
            for resp in dept.core_responsibilities:
                sections.append(f"• {resp}")
            
            sections.extend([
                "",
                "Skills Required:",
            ])
            
            for skill in dept.skills_required:
                sections.append(f"• {skill}")
            
            if dept.soft_skills_required:
                sections.extend([
                    "",
                    "Soft Skills Required:",
                ])
                for skill in dept.soft_skills_required:
                    sections.append(f"• {skill}")
            
            sections.extend([
                "",
                "Skills and Benefits You'll Gain:",
            ])
            
            for benefit in dept.skills_perks_gained:
                sections.append(f"• {benefit}")
            
            sections.extend([
                "",
                "Example Tasks:",
            ])
            
            for task in dept.example_tasks:
                sections.append(f"• {task}")
            
            sections.extend([
                "",
                f"Target Audience: {', '.join(dept.target_audience)}",
            ])
            
            # Combine into single document
            content = "\n".join(sections)
            
            doc = Document(
                page_content=content,
                metadata={
                    'department_id': dept_id,
                    'department_name': dept.name,
                    'source': 'department_data',
                    'document_type': 'structured_info'
                }
            )
            
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} department documents")
        return documents
    
    def _load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and process PDF document"""
        if not RAG_AVAILABLE:
            return []
        
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = splitter.split_documents(pages)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': pdf_path,
                    'chunk_id': i,
                    'document_type': 'pdf_content',
                    'processed_at': datetime.now().isoformat()
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            return []
    
    def generate_explanation(self, department_id: str, user_session: Any) -> Dict[str, str]:
        """
        Generate explanation for department match
        
        Args:
            department_id: Target department ID
            user_session: User session with trait scores and responses
            
        Returns:
            Dictionary with explanation sections
        """
        department = self.departments.get(department_id)
        if not department:
            return self._fallback_explanation(department_id, "Department not found")
        
        try:
            # Use RAG if fully initialized
            if self.initialized and self.vector_store and self.llm:
                return self._rag_explanation(department, user_session)
            elif self.initialized and self.vector_store:
                return self._template_explanation(department, user_session)
            else:
                return self._simple_explanation(department, user_session)
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return self._simple_explanation(department, user_session)
    
    def _rag_explanation(self, department: Any, user_session: Any) -> Dict[str, str]:
        """Generate RAG-powered explanation using LLM"""
        try:
            from .prompts import EXPLANATION_PROMPT
            
            # Retrieve relevant context
            query = f"{department.name} department responsibilities skills requirements tasks"
            docs = self.vector_store.similarity_search(
                query, 
                k=min(5, self.vector_store.index.ntotal)
            )
            
            # Build context from retrieved documents
            context_parts = []
            for doc in docs:
                # Prioritize department-specific content
                if doc.metadata.get('department_id') == department.id:
                    context_parts.insert(0, doc.page_content)
                else:
                    context_parts.append(doc.page_content)
            
            context = "\n\n---\n\n".join(context_parts[:3])  # Limit context length
            
            # Get user traits
            top_traits = user_session.get_top_traits(3)
            traits_text = ", ".join([f"{trait} ({score:.1%})" for trait, score in top_traits])
            
            # Generate explanation using LLM
            prompt = EXPLANATION_PROMPT.format(
                department_name=department.name,
                department_description=department.description,
                user_traits=traits_text or "Balanced across multiple areas",
                context=context[:1500],  # Limit context length
                questions_answered=len(user_session.responses)
            )
            
            response = self.llm.predict(prompt)
            
            # Parse response into sections
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"RAG explanation failed: {e}")
            return self._template_explanation(department, user_session)
    
    def _template_explanation(self, department: Any, user_session: Any) -> Dict[str, str]:
        """Generate explanation using templates with retrieved context"""
        try:
            # Retrieve relevant context without LLM
            query = f"{department.name} responsibilities tasks"
            docs = self.vector_store.similarity_search(query, k=2)
            
            # Extract key information from context
            context_info = []
            for doc in docs:
                if doc.metadata.get('department_id') == department.id:
                    content = doc.page_content
                    # Extract specific responsibilities or tasks
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('•') or line.startswith('-'):
                            context_info.append(line.strip('•- '))
            
            top_traits = user_session.get_top_traits(3)
            
            return {
                "overview": f"Based on your {len(user_session.responses)} responses, you're well-suited for the {department.name} department.",
                "why_good_fit": self._build_fit_explanation(top_traits, department),
                "responsibilities": self._format_responsibilities(department, context_info),
                "skills_gained": self._format_skills_gained(department),
                "next_steps": f"Apply to {department.name} and connect with current members to learn about ongoing projects and opportunities."
            }
            
        except Exception as e:
            logger.error(f"Template explanation failed: {e}")
            return self._simple_explanation(department, user_session)
    
    def _simple_explanation(self, department: Any, user_session: Any) -> Dict[str, str]:
        """Generate simple explanation without external dependencies"""
        top_traits = user_session.get_top_traits(3)
        
        return {
            "overview": f"Based on your responses, you are well-suited for the {department.name} department.",
            "why_good_fit": self._build_fit_explanation(top_traits, department),
            "responsibilities": "; ".join(department.core_responsibilities[:3]),
            "skills_gained": "; ".join(department.skills_perks_gained[:3]),
            "next_steps": f"Apply to {department.name} and connect with current members."
        }
    
    def _build_fit_explanation(self, top_traits: List[tuple], department: Any) -> str:
        """Build explanation of why user fits the department"""
        if not top_traits:
            return f"Your balanced profile aligns well with {department.name}'s diverse requirements."
        
        trait_names = [trait for trait, _ in top_traits]
        
        # Map traits to department focus areas
        trait_mappings = {
            'Coding Aptitude': 'technical development',
            'Visual Creativity': 'creative design',
            'Stakeholder Management': 'professional networking',
            'Team Collaboration': 'teamwork',
            'Leadership Initiative': 'leadership opportunities',
            'Public Interaction': 'community engagement',
            'Digital Design': 'digital media creation',
            'Content Creation': 'content development',
            'Innovation Ideation': 'creative problem-solving'
        }
        
        relevant_areas = []
        for trait in trait_names:
            if trait in trait_mappings:
                relevant_areas.append(trait_mappings[trait])
        
        if relevant_areas:
            areas_text = ", ".join(relevant_areas[:2])
            return f"Your strengths in {', '.join(trait_names[:2])} align perfectly with {department.name}'s focus on {areas_text}."
        else:
            return f"Your top traits ({', '.join(trait_names[:2])}) are valuable assets for {department.name}'s activities."
    
    def _format_responsibilities(self, department: Any, context_info: List[str]) -> str:
        """Format responsibilities section"""
        # Use context info if available, otherwise use department data
        if context_info:
            return "; ".join(context_info[:3])
        else:
            return "; ".join(department.core_responsibilities[:3])
    
    def _format_skills_gained(self, department: Any) -> str:
        """Format skills gained section"""
        skills = department.skills_perks_gained[:3]
        return "; ".join(skills)
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response into structured sections"""
        sections = {
            "overview": "",
            "why_good_fit": "",
            "responsibilities": "",
            "skills_gained": "",
            "next_steps": ""
        }
        
        current_section = "overview"
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line.startswith('##'):
                section_name = line.replace('#', '').strip().lower()
                section_name = section_name.replace(' ', '_').replace("'", "")
                
                if 'overview' in section_name:
                    current_section = "overview"
                elif 'fit' in section_name or 'good' in section_name:
                    current_section = "why_good_fit"
                elif 'do' in section_name or 'responsibilities' in section_name:
                    current_section = "responsibilities"
                elif 'gain' in section_name or 'skills' in section_name:
                    current_section = "skills_gained"
                elif 'next' in section_name or 'steps' in section_name:
                    current_section = "next_steps"
                continue
            
            # Add content to current section
            if line and current_section in sections:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        # Ensure all sections have content
        for key, value in sections.items():
            if not value:
                sections[key] = f"Information about {key.replace('_', ' ')} for this department."
        
        return sections
    
    def _fallback_explanation(self, department_id: str, reason: str) -> Dict[str, str]:
        """Fallback explanation when generation fails"""
        return {
            "overview": f"You have been matched with department {department_id}.",
            "why_good_fit": "Based on your responses to the classification questions.",
            "responsibilities": "Various department-specific tasks and activities.",
            "skills_gained": "Technical and soft skills relevant to the department.",
            "next_steps": "Contact the department for more detailed information."
        }
