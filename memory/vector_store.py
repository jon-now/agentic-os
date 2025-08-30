import os
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, List, Set
import json

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

from config.settings import settings

class VectorStore:
    def __init__(self, store_type: str = "chromadb"):
        self.store_type = store_type
        self.client = None
        self.collection = None
        self.index = None
        self.metadata_store = {}

        # Initialize based on available libraries
        if store_type == "chromadb" and CHROMADB_AVAILABLE:
            self._init_chromadb()
        elif store_type == "faiss" and FAISS_AVAILABLE:
            self._init_faiss()
        else:
            logger.warning("Vector store %s not available, using fallback", store_type)
            self._init_fallback()

    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.CHROMADB_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="agentic_memory",
                metadata={"description": "Agentic system memory and context"}
            )

            logger.info("ChromaDB initialized successfully")

        except Exception as e:
            logger.error("ChromaDB initialization failed: %s", e)
            self._init_fallback()

    def _init_faiss(self):
        """Initialize FAISS"""
        try:
            # FAISS requires manual embedding management
            self.dimension = 384  # Default embedding dimension
            self.index = faiss.IndexFlatL2(self.dimension)

            # Load existing index if available
            index_path = settings.MEMORY_PATH / "faiss_index.bin"
            metadata_path = settings.MEMORY_PATH / "faiss_metadata.json"

            if index_path.exists():
                self.index = faiss.read_index(str(index_path))

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)

            logger.info("FAISS initialized successfully")

        except Exception as e:
            logger.error("FAISS initialization failed: %s", e)
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback JSON-based storage"""
        self.store_type = "fallback"
        self.memory_file = settings.MEMORY_PATH / "memory_store.json"

        # Load existing memory
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    self.metadata_store = json.load(f)
            except Exception:
                self.metadata_store = {}
        else:
            self.metadata_store = {}

        logger.info("Fallback memory store initialized")

    async def store_interaction(self, user_input: str, response: str,
                              context: Dict, intention: Dict) -> str:
        """Store user interaction in vector memory"""
        try:
            # Create unique ID
            interaction_id = self._generate_id(user_input, response)

            # Prepare document
            document = {
                "id": interaction_id,
                "user_input": user_input,
                "response": response,
                "context": context,
                "intention": intention,
                "timestamp": datetime.now().isoformat(),
                "interaction_type": intention.get("interaction_type", "unknown"),
                "primary_action": intention.get("primary_action", "unknown")
            }

            # Store based on available backend
            if self.store_type == "chromadb" and self.collection:
                await self._store_chromadb(document)
            elif self.store_type == "faiss":
                await self._store_faiss(document)
            else:
                await self._store_fallback(document)

            return interaction_id

        except Exception as e:
            logger.error("Error storing interaction: %s", e)
            return ""

    async def store_research_data(self, topic: str, research_results: Dict) -> str:
        """Store research data for future reference"""
        try:
            research_id = self._generate_id(topic, str(research_results))

            document = {
                "id": research_id,
                "type": "research",
                "topic": topic,
                "results": research_results,
                "timestamp": datetime.now().isoformat(),
                "sources_count": len(research_results.get("sources", [])),
                "summary": research_results.get("summary", "")
            }

            if self.store_type == "chromadb" and self.collection:
                await self._store_chromadb(document)
            elif self.store_type == "faiss":
                await self._store_faiss(document)
            else:
                await self._store_fallback(document)

            return research_id

        except Exception as e:
            logger.error("Error storing research data: %s", e)
            return ""

    async def store_task_result(self, task_id: str, task_data: Dict) -> str:
        """Store task execution results"""
        try:
            document = {
                "id": task_id,
                "type": "task_result",
                "task_data": task_data,
                "timestamp": datetime.now().isoformat(),
                "status": task_data.get("status", "unknown"),
                "action": task_data.get("action", "unknown")
            }

            if self.store_type == "chromadb" and self.collection:
                await self._store_chromadb(document)
            elif self.store_type == "faiss":
                await self._store_faiss(document)
            else:
                await self._store_fallback(document)

            return task_id

        except Exception as e:
            logger.error("Error storing task result: %s", e)
            return ""

    async def search_similar_interactions(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar past interactions"""
        try:
            if self.store_type == "chromadb" and self.collection:
                return await self._search_chromadb(query, limit, "interaction")
            elif self.store_type == "faiss":
                return await self._search_faiss(query, limit, "interaction")
            else:
                return await self._search_fallback(query, limit, "interaction")

        except Exception as e:
            logger.error("Error searching interactions: %s", e)
            return []

    async def search_research_history(self, topic: str, limit: int = 3) -> List[Dict]:
        """Search for past research on similar topics"""
        try:
            if self.store_type == "chromadb" and self.collection:
                return await self._search_chromadb(topic, limit, "research")
            elif self.store_type == "faiss":
                return await self._search_faiss(topic, limit, "research")
            else:
                return await self._search_fallback(topic, limit, "research")

        except Exception as e:
            logger.error("Error searching research history: %s", e)
            return []

    async def get_user_preferences(self, user_id: str = "default") -> Dict:
        """Get learned user preferences"""
        try:
            # Analyze past interactions to infer preferences
            interactions = await self.search_similar_interactions("", limit=50)

            preferences = {
                "preferred_response_style": "conversational",
                "common_topics": [],
                "preferred_actions": [],
                "response_length_preference": "medium",
                "last_updated": datetime.now().isoformat()
            }

            if interactions:
                # Analyze interaction patterns
                action_counts = {}
                topic_counts = {}

                for interaction in interactions:
                    action = interaction.get("primary_action", "")
                    if action:
                        action_counts[action] = action_counts.get(action, 0) + 1

                    # Extract topics from user input
                    user_input = interaction.get("user_input", "").lower()
                    for word in user_input.split():
                        if len(word) > 4:  # Meaningful words
                            topic_counts[word] = topic_counts.get(word, 0) + 1

                # Set preferences based on patterns
                if action_counts:
                    preferences["preferred_actions"] = sorted(
                        action_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5]

                if topic_counts:
                    preferences["common_topics"] = sorted(
                        topic_counts.items(), key=lambda x: x[1], reverse=True
                    )[:10]

            return preferences

        except Exception as e:
            logger.error("Error getting user preferences: %s", e)
            return {}

    async def get_context_for_query(self, query: str) -> Dict:
        """Get relevant context for a user query"""
        try:
            # Search for similar interactions
            similar_interactions = await self.search_similar_interactions(query, limit=3)

            # Search for relevant research
            research_history = await self.search_research_history(query, limit=2)

            # Get user preferences
            preferences = await self.get_user_preferences()

            return {
                "similar_interactions": similar_interactions,
                "research_history": research_history,
                "user_preferences": preferences,
                "context_generated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error getting context: %s", e)
            return {}

    async def _store_chromadb(self, document: Dict):
        """Store document in ChromaDB"""
        try:
            # Prepare text for embedding
            text_content = self._extract_text_content(document)

            self.collection.add(
                documents=[text_content],
                metadatas=[document],
                ids=[document["id"]]
            )

        except Exception as e:
            logger.error("ChromaDB storage error: %s", e)

    async def _store_faiss(self, document: Dict):
        """Store document in FAISS"""
        try:
            # For FAISS, we need to generate embeddings manually
            # This is a simplified implementation
            text_content = self._extract_text_content(document)

            # Simple hash-based "embedding" (not ideal, but functional)
            embedding = self._text_to_vector(text_content)

            # Add to index
            self.index.add(np.array([embedding], dtype=np.float32))

            # Store metadata
            doc_index = self.index.ntotal - 1
            self.metadata_store[str(doc_index)] = document

            # Save to disk
            await self._save_faiss()

        except Exception as e:
            logger.error("FAISS storage error: %s", e)

    async def _store_fallback(self, document: Dict):
        """Store document in fallback JSON storage"""
        try:
            self.metadata_store[document["id"]] = document

            # Save to disk
            with open(self.memory_file, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)

        except Exception as e:
            logger.error("Fallback storage error: %s", e)

    async def _search_chromadb(self, query: str, limit: int, doc_type: str = None) -> List[Dict]:
        """Search ChromaDB"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if doc_type:
                if doc_type == "interaction":
                    where_clause = {"interaction_type": {"$ne": None}}
                elif doc_type == "research":
                    where_clause = {"type": "research"}

            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None
            )

            # Format results
            formatted_results = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata in results["metadatas"][0]:
                    formatted_results.append(metadata)

            return formatted_results

        except Exception as e:
            logger.error("ChromaDB search error: %s", e)
            return []

    async def _search_faiss(self, query: str, limit: int, doc_type: str = None) -> List[Dict]:
        """Search FAISS"""
        try:
            if self.index.ntotal == 0:
                return []

            # Generate query embedding
            query_embedding = self._text_to_vector(query)

            # Search
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                min(limit, self.index.ntotal)
            )

            # Format results
            results = []
            for idx in indices[0]:
                if str(idx) in self.metadata_store:
                    doc = self.metadata_store[str(idx)]
                    if not doc_type or doc.get("type") == doc_type:
                        results.append(doc)

            return results[:limit]

        except Exception as e:
            logger.error("FAISS search error: %s", e)
            return []

    async def _search_fallback(self, query: str, limit: int, doc_type: str = None) -> List[Dict]:
        """Search fallback storage"""
        try:
            query_lower = query.lower()
            results = []

            for doc_id, document in self.metadata_store.items():
                # Filter by type if specified
                if doc_type:
                    if doc_type == "interaction" and "user_input" not in document:
                        continue
                    elif doc_type == "research" and document.get("type") != "research":
                        continue

                # Simple text matching
                text_content = self._extract_text_content(document).lower()
                if query_lower in text_content:
                    results.append(document)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error("Fallback search error: %s", e)
            return []

    async def _save_faiss(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = settings.MEMORY_PATH / "faiss_index.bin"
            metadata_path = settings.MEMORY_PATH / "faiss_metadata.json"

            faiss.write_index(self.index, str(index_path))

            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)

        except Exception as e:
            logger.error("Error saving FAISS: %s", e)

    def _extract_text_content(self, document: Dict) -> str:
        """Extract searchable text content from document"""
        text_parts = []

        if "user_input" in document:
            text_parts.append(document["user_input"])
        if "response" in document:
            text_parts.append(document["response"])
        if "topic" in document:
            text_parts.append(document["topic"])
        if "summary" in document:
            text_parts.append(document["summary"])

        return " ".join(text_parts)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector (simplified implementation)"""
        # This is a very basic implementation
        # In production, you'd use proper embeddings like sentence-transformers

        # Create a hash-based vector
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float vector
        vector = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

        # Pad or truncate to desired dimension
        if len(vector) > self.dimension:
            vector = vector[:self.dimension]
        else:
            vector = np.pad(vector, (0, self.dimension - len(vector)))

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _generate_id(self, *args) -> str:
        """Generate unique ID from arguments"""
        content = "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()

    def get_store_status(self) -> Dict:
        """Get vector store status"""
        status = {
            "store_type": self.store_type,
            "chromadb_available": CHROMADB_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
            "initialized": self.client is not None or self.index is not None or self.metadata_store is not None
        }

        if self.store_type == "chromadb" and self.collection:
            try:
                status["document_count"] = self.collection.count()
            except Exception:
                status["document_count"] = 0
        elif self.store_type == "faiss" and self.index:
            status["document_count"] = self.index.ntotal
        else:
            status["document_count"] = len(self.metadata_store)

        return status
