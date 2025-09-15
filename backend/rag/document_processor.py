import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Core imports
from core.data_loader import TaqneeqDepartment

logger = logging.getLogger(__name__)

class TaqneeqDocumentProcessor:
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
       
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use GPU if available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        self.vector_store = None
        self.documents = []
        
        logger.info(f"TaqneeqDocumentProcessor initialized with model: {embedding_model}")
    
    def process_pdf_document(self, pdf_path: str) -> List[Document]:
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            logger.info(f"Loaded PDF with {len(pages)} pages from {pdf_path}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': pdf_path,
                    'chunk_id': i,
                    'processed_at': datetime.now().isoformat(),
                    'document_type': 'department_description'
                })
            
            logger.info(f"Created {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
    
    def create_department_documents(self, departments: Dict[str, TaqneeqDepartment]) -> List[Document]:
        """
        Create documents from department data structures
        
        Args:
            departments: Dictionary of TaqneeqDepartment objects
            
        Returns:
            List of documents for each department
        """
        documents = []
        
        for dept_id, dept in departments.items():
            # Create comprehensive text for each department
            content_sections = []
            
            # Basic information
            content_sections.append(f"Department: {dept.name}")
            content_sections.append(f"Description: {dept.description}")
            
            # Core responsibilities
            if dept.core_responsibilities:
                content_sections.append("Core Responsibilities:")
                for resp in dept.core_responsibilities:
                    content_sections.append(f"- {resp}")
            
            # Skills required
            if dept.skills_required:
                content_sections.append("Skills Required:")
                for skill in dept.skills_required:
                    content_sections.append(f"- {skill}")
            
            # Soft skills
            if dept.soft_skills_required:
                content_sections.append("Soft Skills Required:")
                for skill in dept.soft_skills_required:
                    content_sections.append(f"- {skill}")
            
            # Skills and perks gained
            if dept.skills_perks_gained:
                content_sections.append("Skills and Perks Gained:")
                for perk in dept.skills_perks_gained:
                    content_sections.append(f"- {perk}")
            
            # Example tasks
            if dept.example_tasks:
                content_sections.append("Example Tasks:")
                for task in dept.example_tasks:
                    content_sections.append(f"- {task}")
            
            # Target audience
            if dept.target_audience:
                content_sections.append("Target Audience:")
                content_sections.append(", ".join(dept.target_audience))
            
            # Combine all sections
            full_content = "\n".join(content_sections)
            
            # Create document
            doc = Document(
                page_content=full_content,
                metadata={
                    'department_id': dept_id,
                    'department_name': dept.name,
                    'source': 'department_data',
                    'document_type': 'structured_department_info',
                    'processed_at': datetime.now().isoformat()
                }
            )
            
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} department documents")
        return documents
    
    def build_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS vector store from documents
        
        Args:
            documents: List of processed documents
            
        Returns:
            FAISS vector store
        """
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.documents = documents
            
            logger.info(f"Built vector store with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise
    
    def save_vector_store(self, save_path: str):
        """Save vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """Load vector store from disk"""
        self.vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vector store loaded from {load_path}")
        return self.vector_store