from typing import Dict, List, Tuple, Optional
import logging

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

from core.session_manager import UserSession
from core.data_loader import TaqneeqDepartment

logger = logging.getLogger(__name__)

class TaqneeqRetriever:
    """Retrieve relevant context for department explanations"""
    
    def __init__(self, vector_store: FAISS, departments: Dict[str, TaqneeqDepartment]):
        """
        Initialize retriever
        
        Args:
            vector_store: FAISS vector store with department documents
            departments: Dictionary of department objects
        """
        self.vector_store = vector_store
        self.departments = departments
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,
                "k": 10
            }
        )
        
        logger.info("TaqneeqRetriever initialized")
    
    def retrieve_for_department(self, 
                               department_id: str,
                               user_session: UserSession,
                               max_chunks: int = 5) -> List[Document]:
        """
        Retrieve relevant chunks for a specific department
        
        Args:
            department_id: Target department ID
            user_session: Current user session with trait scores
            max_chunks: Maximum chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        department = self.departments.get(department_id)
        if not department:
            logger.warning(f"Department not found: {department_id}")
            return []
        
        # Create search query based on department and user traits
        query = self._create_search_query(department, user_session)
        
        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Filter for target department and limit results
        filtered_docs = []
        for doc in docs:
            # Prioritize documents specifically about the target department
            if (doc.metadata.get('department_id') == department_id or
                department.name.lower() in doc.page_content.lower()):
                filtered_docs.append(doc)
                
            if len(filtered_docs) >= max_chunks:
                break
        
        # If we don't have enough department-specific docs, add general ones
        if len(filtered_docs) < max_chunks:
            for doc in docs:
                if doc not in filtered_docs:
                    filtered_docs.append(doc)
                if len(filtered_docs) >= max_chunks:
                    break
        
        logger.debug(f"Retrieved {len(filtered_docs)} chunks for department {department_id}")
        return filtered_docs
    
    def retrieve_comparative(self,
                           primary_department_id: str,
                           secondary_department_id: str,
                           user_session: UserSession,
                           max_chunks: int = 8) -> Dict[str, List[Document]]:
        """
        Retrieve documents for comparing two departments
        
        Args:
            primary_department_id: Main department ID
            secondary_department_id: Alternative department ID
            user_session: Current user session
            max_chunks: Total chunks to retrieve (split between departments)
            
        Returns:
            Dictionary with 'primary' and 'secondary' document lists
        """
        chunks_per_dept = max_chunks // 2
        
        primary_docs = self.retrieve_for_department(
            primary_department_id, user_session, chunks_per_dept
        )
        
        secondary_docs = self.retrieve_for_department(
            secondary_department_id, user_session, chunks_per_dept
        )
        
        return {
            'primary': primary_docs,
            'secondary': secondary_docs
        }
    
    def _create_search_query(self, 
                           department: TaqneeqDepartment,
                           user_session: UserSession) -> str:
        """
        Create search query based on department and user traits
        
        Args:
            department: Target department
            user_session: User session with trait scores
            
        Returns:
            Search query string
        """
        query_parts = [
            f"{department.name} department",
            f"responsibilities {department.name}",
            f"skills needed for {department.name}"
        ]
        
        # Add user's strongest traits to query
        top_traits = sorted(
            user_session.trait_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 traits
        
        trait_keywords = {
            'coding_aptitude': 'programming coding technical',
            'visual_creativity': 'design creative visual artistic',
            'stakeholder_management': 'networking sponsors business relations',
            'hands_on_crafting': 'crafting building physical hands-on',
            'content_creation': 'content writing social media',
            'business_development': 'business sponsorship deals partnerships',
            'public_interaction': 'participants public interaction engagement'
        }
        
        for trait_name, score in top_traits:
            if score > 0.6:  # Only include strong traits
                keywords = trait_keywords.get(trait_name, trait_name.replace('_', ' '))
                query_parts.append(keywords)
        
        return " ".join(query_parts)