import os
import subprocess
import time
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import uno
    from com.sun.star.beans import PropertyValue
    from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
    LIBREOFFICE_UNO_AVAILABLE = True
except ImportError:
    LIBREOFFICE_UNO_AVAILABLE = False
    logger.warning("LibreOffice UNO bridge not available. Install with: pip install pyuno")

class DocumentController:
    def __init__(self):
        self.libreoffice_path = self._find_libreoffice_path()
        self.connection = None
        self.desktop = None
        self.documents = {}

    def _find_libreoffice_path(self) -> Optional[str]:
        """Find LibreOffice installation path"""
        possible_paths = []

        if os.name == 'nt':  # Windows
            possible_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                r"C:\Program Files\LibreOffice 7.0\program\soffice.exe",
            ]
        else:  # Linux/Mac
            possible_paths = [
                "/usr/bin/libreoffice",
                "/usr/local/bin/libreoffice",
                "/opt/libreoffice/program/soffice",
                "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info("Found LibreOffice at: %s", path)
                return path

        # Try system PATH
        try:
            result = subprocess.run(
                ["which", "libreoffice"] if os.name != 'nt' else ["where", "soffice"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                logger.info("Found LibreOffice in PATH: %s", path)
                return path
        except Exception:
            pass

        logger.warning("LibreOffice not found")
        return None

    async def create_document(self, doc_type: str = "writer", content: str = "") -> Dict:
        """Create a new document (simplified implementation without UNO)"""
        try:
            # For now, create a simple text-based document
            doc_id = f"doc_{len(self.documents)}_{int(time.time())}"

            document_data = {
                "id": doc_id,
                "type": doc_type,
                "content": content,
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat()
            }

            self.documents[doc_id] = document_data

            return {
                "document_id": doc_id,
                "type": doc_type,
                "status": "created",
                "content_length": len(content),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Failed to create document: %s", e)
            return {"error": f"Document creation failed: {str(e)}"}

    async def add_content(self, doc_id: str, content: str, content_type: str = "text") -> Dict:
        """Add content to a document"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}

        try:
            document = self.documents[doc_id]

            if content_type == "text":
                document["content"] += "\n" + content
            elif content_type == "heading":
                document["content"] += f"\n\n# {content}\n"
            elif content_type == "list":
                items = content.split('\n')
                for item in items:
                    if item.strip():
                        document["content"] += f"\n• {item.strip()}"
                document["content"] += "\n"

            document["modified"] = datetime.now().isoformat()

            return {
                "document_id": doc_id,
                "content_added": len(content),
                "content_type": content_type,
                "total_length": len(document["content"]),
                "status": "success"
            }

        except Exception as e:
            logger.error("Failed to add content: %s", e)
            return {"error": f"Content addition failed: {str(e)}"}

    async def save_document(self, doc_id: str, file_path: Optional[str] = None) -> Dict:
        """Save a document to file"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}

        try:
            document = self.documents[doc_id]

            if not file_path:
                # Generate default filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"document_{timestamp}.txt"

            file_path_obj = Path(file_path)

            # Save content to file
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                f.write(document["content"])

            return {
                "document_id": doc_id,
                "file_path": str(file_path_obj.absolute()),
                "file_size": file_path_obj.stat().st_size,
                "status": "saved",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Failed to save document: %s", e)
            return {"error": f"Document saving failed: {str(e)}"}

    async def create_report_from_data(self, data: Dict, title: str = "Report") -> Dict:
        """Create a formatted report document"""
        try:
            # Create writer document
            result = await self.create_document("writer")
            if "error" in result:
                return result

            doc_id = result["document_id"]

            # Add title
            await self.add_content(doc_id, title, "heading")

            # Add timestamp
            timestamp = datetime.now().strftime("%B %d, %Y")
            await self.add_content(doc_id, f"Generated on: {timestamp}", "text")

            # Add summary if available
            if "summary" in data:
                await self.add_content(doc_id, "Executive Summary", "heading")
                await self.add_content(doc_id, data["summary"], "text")

            # Add key points if available
            if "key_points" in data:
                await self.add_content(doc_id, "Key Points", "heading")
                key_points_text = "\n".join(data["key_points"])
                await self.add_content(doc_id, key_points_text, "list")

            # Add detailed content
            if "sources" in data:
                await self.add_content(doc_id, "Detailed Information", "heading")
                for i, source in enumerate(data["sources"], 1):
                    source_title = f"Source {i}: {source.get('name', 'Unknown')}"
                    await self.add_content(doc_id, source_title, "heading")
                    await self.add_content(doc_id, source.get('content', ''), "text")

            return {
                "document_id": doc_id,
                "type": "report",
                "status": "created",
                "sections": ["summary", "key_points", "detailed_info"],
                "title": title
            }

        except Exception as e:
            logger.error("Failed to create report: %s", e)
            return {"error": f"Report creation failed: {str(e)}"}

    async def list_documents(self) -> Dict:
        """List all open documents"""
        return {
            "open_documents": [
                {
                    "id": doc_id,
                    "type": doc["type"],
                    "created": doc["created"],
                    "modified": doc["modified"],
                    "content_length": len(doc["content"])
                }
                for doc_id, doc in self.documents.items()
            ],
            "count": len(self.documents),
            "timestamp": datetime.now().isoformat()
        }

    async def get_document_content(self, doc_id: str) -> Dict:
        """Get document content"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}

        document = self.documents[doc_id]

        return {
            "document_id": doc_id,
            "type": document["type"],
            "content": document["content"],
            "content_length": len(document["content"]),
            "word_count": len(document["content"].split()),
            "created": document["created"],
            "modified": document["modified"]
        }

    async def close_document(self, doc_id: str) -> Dict:
        """Close a document"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}

        try:
            del self.documents[doc_id]

            return {
                "document_id": doc_id,
                "status": "closed",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Failed to close document: %s", e)
            return {"error": f"Document closing failed: {str(e)}"}

    def close_all_documents(self):
        """Close all open documents"""
        self.documents.clear()

    def get_controller_status(self) -> Dict:
        """Get controller status"""
        return {
            "libreoffice_found": self.libreoffice_path is not None,
            "libreoffice_path": self.libreoffice_path,
            "uno_available": LIBREOFFICE_UNO_AVAILABLE,
            "open_documents": len(self.documents),
            "mode": "uno" if LIBREOFFICE_UNO_AVAILABLE else "text_based"
        }
