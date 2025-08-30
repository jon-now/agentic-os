import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from typing import Dict, List, Optional
import json
import base64
import tempfile
import os

# Import voice interface
try:
    from interfaces.voice_interface import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    VoiceInterface = None

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[str] = None

class ConfirmationResponse(BaseModel):
    confirmation: str  # "yes" or "no"
    original_message: str
    llm_analysis: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    status: str
    timestamp: str
    details: Optional[Dict] = None

class VoiceMessage(BaseModel):
    audio_data: str  # Base64 encoded audio
    format: str = "wav"

class ChatInterface:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.app = FastAPI(title="Agentic Orchestrator", version="1.0.0")
        self.active_connections: List[WebSocket] = []
        
        # Initialize voice interface with fallback handling
        self.voice_interface = None
        if VOICE_AVAILABLE:
            try:
                self.voice_interface = VoiceInterface(orchestrator)
                logger.info("Voice interface initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize voice interface: %s", e)
                # Try to initialize with CPU-only mode
                try:
                    logger.info("Attempting to initialize voice interface with CPU-only mode...")
                    # Create a CPU-only voice interface
                    self.voice_interface = VoiceInterface(orchestrator)
                    if self.voice_interface:
                        self.voice_interface.config["use_gpu"] = False
                        self.voice_interface.config["device"] = "cpu"
                        self.voice_interface._initialize_recognition()
                        self.voice_interface._initialize_synthesis()
                        logger.info("Voice interface initialized successfully with CPU fallback")
                except Exception as cpu_error:
                    logger.error("Failed to initialize voice interface even with CPU fallback: %s", cpu_error)
                    self.voice_interface = None
        
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_chat_interface():
            return self.get_chat_html()

        @self.app.get("/mobile", response_class=HTMLResponse)
        async def get_mobile_interface():
            return self.get_mobile_html()
        
        @self.app.get("/static/manifest.json")
        async def get_manifest():
            from pathlib import Path
            manifest_path = Path(__file__).parent.parent / "static" / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    return json.loads(f.read())
            return {"error": "Manifest not found"}
        
        @self.app.get("/static/sw.js")
        async def get_service_worker():
            from fastapi.responses import FileResponse
            from pathlib import Path
            sw_path = Path(__file__).parent.parent / "static" / "sw.js"
            if sw_path.exists():
                return FileResponse(sw_path, media_type="application/javascript")
            return {"error": "Service worker not found"}

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(message: ChatMessage):
            try:
                # Use auto_confirm=False to enable confirmation flow for web UI
                result = await self.orchestrator.process_user_intention(message.message, auto_confirm=False)

                return ChatResponse(
                    response=result.get("result", {}).get("final_output", result.get("final_output", "Task completed")),
                    status=result.get("status", "unknown"),
                    timestamp=datetime.now().isoformat(),
                    details=result
                )
            except Exception as e:
                logger.error("Chat endpoint error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/confirm", response_model=ChatResponse)
        async def confirmation_endpoint(confirmation: ConfirmationResponse):
            try:
                if confirmation.confirmation.lower() in ["yes", "y", "proceed", "continue"]:
                    # User confirmed, proceed with original message using auto_confirm=True
                    result = await self.orchestrator.process_user_intention(confirmation.original_message, auto_confirm=True)
                    
                    return ChatResponse(
                        response=result.get("result", {}).get("final_output", result.get("final_output", "Task completed")),
                        status=result.get("status", "success"),
                        timestamp=datetime.now().isoformat(),
                        details=result
                    )
                else:
                    # User declined, ask for clarification
                    return ChatResponse(
                        response="I understand you'd like me to interpret your request differently. Could you please clarify what you meant?",
                        status="clarification_needed",
                        timestamp=datetime.now().isoformat(),
                        details={"original_message": confirmation.original_message}
                    )
            except Exception as e:
                logger.error("Confirmation endpoint error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)

                    # Handle different message types
                    if message_data.get("type") == "confirmation":
                        # Handle confirmation response
                        if message_data["confirmation"].lower() in ["yes", "y", "proceed", "continue"]:
                            # User confirmed, proceed with original message
                            result = await self.orchestrator.process_user_intention(message_data["original_message"], auto_confirm=True)
                        else:
                            # User declined, ask for clarification
                            result = {
                                "status": "clarification_needed",
                                "final_output": "I understand you'd like me to interpret your request differently. Could you please clarify what you meant?",
                                "result": {
                                    "final_output": "I understand you'd like me to interpret your request differently. Could you please clarify what you meant?"
                                }
                            }
                    else:
                        # Process regular message with confirmation enabled
                        result = await self.orchestrator.process_user_intention(message_data["message"], auto_confirm=False)

                    # Send response
                    response = {
                        "type": "response",
                        "message": result.get("result", {}).get("final_output", result.get("final_output", "Task completed")),
                        "status": result.get("status", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "details": result
                    }

                    await websocket.send_text(json.dumps(response))

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

        @self.app.get("/status")
        async def get_status():
            return {
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "active_connections": len(self.active_connections),
                "llm_model": self.orchestrator.llm_client.model_name,
                "controllers": list(self.orchestrator.app_controllers.keys()),
                "automation_engine": self.orchestrator.automation_engine.get_engine_status()
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "llm": self.orchestrator.llm_client.is_model_available(),
                    "browser": self.orchestrator.app_controllers["browser"].driver is not None,
                    "email": self.orchestrator.app_controllers["email"].authenticated,
                    "automation": self.orchestrator.automation_engine.running,
                    "vector_store": self.orchestrator.vector_store.get_store_status(),
                    "intelligence": {
                        "context_analyzer": self.orchestrator.context_analyzer is not None,
                        "learning_engine": self.orchestrator.learning_engine is not None,
                        "proactive_assistant": self.orchestrator.proactive_assistant is not None
                    },
                    "communication": {
                        "slack": self.orchestrator.app_controllers["slack"].get_authentication_status(),
                        "teams": self.orchestrator.app_controllers["teams"].get_authentication_status(),
                        "message_analyzer": self.orchestrator.message_analyzer is not None,
                        "smart_responder": self.orchestrator.smart_responder is not None
                    }
                }
            }

        @self.app.get("/intelligence")
        async def get_intelligence_insights():
            """Get intelligence system insights"""
            return await self.orchestrator.get_intelligence_insights()

        @self.app.get("/communication")
        async def get_communication_insights():
            """Get communication system insights"""
            return await self.orchestrator.get_communication_insights()

        @self.app.post("/voice/process")
        async def process_voice_message(voice_message: VoiceMessage):
            """Process voice message"""
            if not self.voice_interface:
                raise HTTPException(status_code=503, detail="Voice interface not available")
            
            try:
                # Decode base64 audio data
                audio_data = base64.b64decode(voice_message.audio_data)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=f".{voice_message.format}", delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_path = tmp_file.name
                
                try:
                    # Process audio with voice interface with error handling
                    text = ""
                    
                    if hasattr(self.voice_interface, 'whisper_model'):
                        try:
                            if self.voice_interface.recognition_engine == "faster_whisper":
                                segments, info = self.voice_interface.whisper_model.transcribe(tmp_path, beam_size=5)
                                text = " ".join([segment.text for segment in segments]).strip()
                            else:
                                result = self.voice_interface.whisper_model.transcribe(tmp_path)
                                text = result["text"].strip()
                        except Exception as whisper_error:
                            logger.error(f"Whisper processing failed: {whisper_error}")
                            
                            # Try to reinitialize voice interface with CPU fallback
                            if "cublas" in str(whisper_error).lower() or "cuda" in str(whisper_error).lower():
                                logger.info("CUDA error detected, reinitializing voice interface with CPU...")
                                try:
                                    self.voice_interface.config["use_gpu"] = False
                                    self.voice_interface.config["device"] = "cpu"
                                    self.voice_interface._initialize_recognition()
                                    
                                    # Retry with CPU
                                    if hasattr(self.voice_interface, 'whisper_model'):
                                        if self.voice_interface.recognition_engine == "faster_whisper":
                                            segments, info = self.voice_interface.whisper_model.transcribe(tmp_path, beam_size=5)
                                            text = " ".join([segment.text for segment in segments]).strip()
                                        else:
                                            result = self.voice_interface.whisper_model.transcribe(tmp_path)
                                            text = result["text"].strip()
                                        logger.info("Successfully processed audio with CPU fallback")
                                    else:
                                        raise Exception("Voice interface not available after CPU fallback")
                                except Exception as fallback_error:
                                    logger.error(f"CPU fallback also failed: {fallback_error}")
                                    return {
                                        "transcription": "",
                                        "response": "Voice processing failed. Please try typing your message instead.",
                                        "status": "error",
                                        "error": "Speech recognition unavailable"
                                    }
                            else:
                                return {
                                    "transcription": "",
                                    "response": "Voice processing failed. Please try again or type your message.",
                                    "status": "error",
                                    "error": str(whisper_error)
                                }
                    else:
                        return {
                            "transcription": "",
                            "response": "Speech recognition not available. Please type your message.",
                            "status": "error",
                            "error": "No speech recognition engine available"
                        }
                    
                    if not text:
                        return {"transcription": "", "response": "No speech detected", "status": "no_speech"}
                    
                    # Process the transcribed text
                    result = await self.orchestrator.process_user_intention(text)
                    response_text = result.get("result", {}).get("final_output", "Task completed")
                    
                    # Generate speech response if TTS is available
                    audio_response = None
                    if self.voice_interface.synthesis_engine:
                        try:
                            # Generate audio response
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as response_tmp:
                                if self.voice_interface.synthesis_engine == "coqui_tts":
                                    self.voice_interface.tts_model.tts_to_file(text=response_text, file_path=response_tmp.name)
                                    
                                    # Read the generated audio file
                                    with open(response_tmp.name, 'rb') as audio_file:
                                        audio_response = base64.b64encode(audio_file.read()).decode('utf-8')
                                
                                # Clean up
                                os.unlink(response_tmp.name)
                        except Exception as e:
                            logger.warning("TTS generation failed: %s", e)
                    
                    return {
                        "transcription": text,
                        "response": response_text,
                        "audio_response": audio_response,
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                        "details": result
                    }
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
            except Exception as e:
                logger.error("Voice processing error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/voice/status")
        async def get_voice_status():
            """Get voice interface status"""
            if not self.voice_interface:
                return {"available": False, "reason": "Voice interface not initialized"}
            
            try:
                # Try to get status from voice interface
                if hasattr(self.voice_interface, 'get_status'):
                    status = self.voice_interface.get_status()
                elif hasattr(self.voice_interface, 'get_performance_stats'):
                    # Fallback to performance stats
                    stats = self.voice_interface.get_performance_stats()
                    status = {
                        "available": stats.get('recognition_engine') is not None or stats.get('synthesis_engine') is not None,
                        "recognition_available": stats.get('recognition_engine') is not None,
                        "synthesis_available": stats.get('synthesis_engine') is not None,
                        "recognition_engine": stats.get('recognition_engine'),
                        "synthesis_engine": stats.get('synthesis_engine'),
                        "device": stats.get('device', 'unknown'),
                        "gpu_available": stats.get('gpu_available', False)
                    }
                else:
                    # Basic fallback
                    status = {
                        "available": True,
                        "recognition_available": hasattr(self.voice_interface, 'recognition_engine') and self.voice_interface.recognition_engine is not None,
                        "synthesis_available": hasattr(self.voice_interface, 'synthesis_engine') and self.voice_interface.synthesis_engine is not None,
                        "recognition_engine": getattr(self.voice_interface, 'recognition_engine', None),
                        "synthesis_engine": getattr(self.voice_interface, 'synthesis_engine', None)
                    }
                
                return {
                    "available": True,
                    "status": status
                }
                
            except Exception as e:
                logger.error(f"Error getting voice status: {e}")
                return {
                    "available": False, 
                    "reason": f"Voice status check failed: {str(e)}"
                }

        @self.app.post("/voice/speak")
        async def speak_text(text_data: dict):
            """Convert text to speech"""
            if not self.voice_interface:
                raise HTTPException(status_code=503, detail="Voice interface not available")
            
            text = text_data.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="No text provided")
            
            try:
                success = await self.voice_interface.speak(text)
                return {"success": success, "message": "Speech synthesis completed" if success else "Speech synthesis failed"}
            except Exception as e:
                logger.error("Speech synthesis error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

    def get_mobile_html(self) -> str:
        """Return mobile-optimized PWA interface"""
        from pathlib import Path
        mobile_path = Path(__file__).parent / "mobile.html"
        
        if mobile_path.exists():
            with open(mobile_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return basic mobile fallback
            return """
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agentic OS Mobile</title><style>body{font-family:sans-serif;margin:0;padding:20px;background:#f5f5f5;}
.container{max-width:600px;margin:0 auto;background:white;border-radius:10px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}
h1{color:#333;text-align:center;margin-bottom:30px;}textarea{width:100%;padding:15px;border:2px solid #ddd;border-radius:8px;font-size:16px;resize:vertical;min-height:60px;}
button{background:#007bff;color:white;border:none;padding:15px 30px;border-radius:8px;font-size:16px;cursor:pointer;margin:10px 5px;}button:hover{background:#0056b3;}
.message{margin:10px 0;padding:15px;border-radius:8px;}.user{background:#007bff;color:white;text-align:right;}.assistant{background:#f8f9fa;border:1px solid #dee2e6;}</style></head>
<body><div class="container"><h1>🤖 Agentic OS Mobile</h1><div id="messages"><div class="message assistant">Hello! I'm your AI assistant. How can I help you today?</div></div>
<textarea id="messageInput" placeholder="Type your message here..."></textarea><button onclick="sendMessage()">Send Message</button><button onclick="window.location.reload()">Refresh</button></div>
<script>function sendMessage(){const input=document.getElementById('messageInput');const message=input.value.trim();if(!message)return;addMessage(message,'user');input.value='';fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:message})}).then(r=>r.json()).then(data=>addMessage(data.response,'assistant')).catch(e=>addMessage('Error: '+e.message,'assistant'));}
function addMessage(content,type){const messages=document.getElementById('messages');const div=document.createElement('div');div.className='message '+type;div.textContent=content;messages.appendChild(div);messages.scrollTop=messages.scrollHeight;}
document.getElementById('messageInput').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});</script></body></html>
"""
    
    def get_chat_html(self) -> str:
        """Return the completely redesigned modern chat interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic OS - Professional AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --secondary-gradient: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
            --dark-bg: #0f0f23;
            --card-bg: rgba(255, 255, 255, 0.95);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: rgba(255, 255, 255, 0.2);
            --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            --shadow-xl: 0 35px 60px -12px rgba(0, 0, 0, 0.3);
        }

        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--dark-bg);
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            pointer-events: none;
        }

        .app-container {
            width: 100vw;
            height: 100vh;
            display: grid;
            grid-template-columns: 320px 1fr;
            position: relative;
            z-index: 1;
        }

        .sidebar {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .sidebar-header {
            padding: 2rem;
            border-bottom: 1px solid var(--border-color);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--primary-gradient);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.25rem;
            font-weight: 600;
        }

        .logo-text {
            color: white;
            font-size: 1.25rem;
            font-weight: 700;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 8px;
            color: #22c55e;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .nav-section {
            padding: 1.5rem 2rem;
            flex: 1;
        }

        .nav-title {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s ease;
            margin-bottom: 0.25rem;
            cursor: pointer;
        }

        .nav-item:hover, .nav-item.active {
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }

        .nav-item i {
            width: 20px;
            text-align: center;
        }

        .chat-container {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: relative;
        }

        .chat-header {
            padding: 2rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-info {
            flex: 1;
        }

        .header-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .queue-badge {
            background: #3b82f6;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            min-width: 20px;
            text-align: center;
            animation: fadeIn 0.3s ease;
        }

        .queue-badge.processing {
            background: #f59e0b;
        }

        .header-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            transition: color 0.3s ease;
        }

        .header-subtitle.processing {
            color: #f59e0b;
        }

        .header-subtitle.queued {
            color: #3b82f6;
        }

        .header-actions {
            display: flex;
            gap: 0.75rem;
        }

        .action-button {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            border: none;
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            font-size: 1.1rem;
        }

        .action-button:hover {
            background: rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }

        .action-button.voice-active {
            background: var(--secondary-gradient);
            color: white;
        }

        .voice-controls {
            position: absolute;
            top: 50%;
            right: 25px;
            transform: translateY(-50%);
            display: flex;
            gap: 10px;
        }

        .voice-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .voice-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }

        .voice-btn.recording {
            background: #ff4757;
            border-color: #ff4757;
            animation: pulse 1.5s infinite;
        }

        .voice-btn.processing {
            background: #ffa502;
            border-color: #ffa502;
            animation: spin 1s linear infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .chat-messages {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            scroll-behavior: smooth;
            background: transparent;
        }

        .message-group {
            margin-bottom: 2rem;
        }

        .message {
            margin-bottom: 1rem;
            max-width: 80%;
            position: relative;
            animation: messageSlide 0.4s ease-out;
        }

        @keyframes messageSlide {
            from { 
                opacity: 0; 
                transform: translateY(20px) scale(0.95); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1); 
            }
        }

        .message-content {
            padding: 1.25rem 1.5rem;
            border-radius: 20px;
            position: relative;
            word-wrap: break-word;
            line-height: 1.6;
        }

        .user-message {
            margin-left: auto;
        }

        .user-message .message-content {
            background: var(--primary-gradient);
            color: white;
            border-bottom-right-radius: 6px;
            box-shadow: var(--shadow-lg);
        }

        .bot-message .message-content {
            background: white;
            color: var(--text-primary);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-bottom-left-radius: 6px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        .system-message .message-content {
            background: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.2);
            text-align: center;
            font-size: 0.875rem;
        }

        .message-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.75rem;
            opacity: 0.7;
        }

        .user-message .message-meta {
            color: rgba(255, 255, 255, 0.8);
        }

        .bot-message .message-meta {
            color: var(--text-secondary);
        }

        .message-time {
            flex-shrink: 0;
        }

        .message-actions {
            display: flex;
            gap: 0.25rem;
            opacity: 0;
            transition: opacity 0.2s ease;
            margin-left: auto;
            padding-left: 0.5rem;
        }

        .message:hover .message-actions {
            opacity: 1;
        }

        .message-action {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            border: none;
            background: rgba(0, 0, 0, 0.1);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            transition: all 0.2s ease;
        }

        .message-action:hover {
            background: rgba(0, 0, 0, 0.2);
            transform: scale(1.1);
        }

        .user-message .message-action {
            background: rgba(255, 255, 255, 0.2);
            color: rgba(255, 255, 255, 0.8);
        }

        .user-message .message-action:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .chat-input-area {
            padding: 2rem;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(0, 0, 0, 0.1);
        }

        .input-container {
            position: relative;
            background: white;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            border: 1px solid rgba(0, 0, 0, 0.1);
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .input-container:focus-within {
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
            border-color: #6366f1;
        }

        .input-container.processing {
            border-color: #f59e0b;
            box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2);
        }

        .input-container.queued {
            border-color: #3b82f6;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
        }

        .input-wrapper {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 1.5rem;
        }

        .chat-input {
            flex: 1;
            border: none;
            outline: none;
            font-size: 1rem;
            font-family: inherit;
            background: transparent;
            color: var(--text-primary);
            padding: 0.5rem 0;
        }

        .chat-input::placeholder {
            color: var(--text-secondary);
        }

        .input-actions {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-button {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            transition: all 0.2s ease;
            position: relative;
        }

        .voice-button {
            background: rgba(139, 92, 246, 0.1);
            color: #8b5cf6;
        }

        .voice-button:hover {
            background: rgba(139, 92, 246, 0.2);
            transform: scale(1.05);
        }

        .voice-button.recording {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            animation: pulse-recording 1.5s infinite;
        }

        @keyframes pulse-recording {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .send-button {
            background: var(--primary-gradient);
            color: white;
        }

        .send-button:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .typing-indicator {
            display: none;
            max-width: 80%;
            margin-bottom: 1rem;
        }

        .typing-content {
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 20px;
            border-bottom-left-radius: 6px;
            padding: 1rem 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        .typing-dots {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .typing-text {
            color: var(--text-secondary);
            font-size: 0.875rem;
            white-space: nowrap;
        }

        .typing-animation {
            display: flex;
            gap: 0.25rem;
            align-items: center;
        }

        .typing-animation span {
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: typing-bounce 1.4s infinite;
        }

        .typing-animation span:nth-child(1) { animation-delay: 0s; }
        .typing-animation span:nth-child(2) { animation-delay: 0.2s; }
        .typing-animation span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing-bounce {
            0%, 60%, 100% { 
                transform: translateY(0); 
                opacity: 0.4; 
            }
            30% { 
                transform: translateY(-8px); 
                opacity: 1; 
            }
        }

        .quick-suggestions {
            padding: 1rem 2rem 0;
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            justify-content: center;
        }

        .suggestion-chip {
            padding: 0.5rem 1rem;
            background: rgba(99, 102, 241, 0.1);
            color: #6366f1;
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 20px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .suggestion-chip:hover {
            background: rgba(99, 102, 241, 0.2);
            transform: translateY(-1px);
        }

        .floating-controls {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            z-index: 1000;
        }

        .floating-button {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            border: none;
            background: var(--primary-gradient);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: var(--shadow-lg);
            transition: all 0.2s ease;
        }

        .floating-button:hover {
            transform: scale(1.05);
            box-shadow: var(--shadow-xl);
        }

        /* Confirmation Dialog Styles */
        .confirmation-dialog {
            background: rgba(59, 130, 246, 0.05);
            border: 2px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
        }

        .confirmation-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #3b82f6;
            font-weight: 600;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .confirmation-header i {
            font-size: 1.25rem;
        }

        .confirmation-content {
            margin-bottom: 1.5rem;
        }

        .analysis-display {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
            color: var(--text-primary);
        }

        .confirmation-question {
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .confirmation-actions {
            display: flex;
            gap: 1rem;
            justify-content: center;
        }

        .confirmation-btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .confirm-yes {
            background: rgba(34, 197, 94, 0.1);
            color: #22c55e;
            border: 2px solid rgba(34, 197, 94, 0.3);
        }

        .confirm-yes:hover:not(:disabled) {
            background: rgba(34, 197, 94, 0.2);
            border-color: rgba(34, 197, 94, 0.5);
            transform: translateY(-1px);
        }

        .confirm-no {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 2px solid rgba(239, 68, 68, 0.3);
        }

        .confirm-no:hover:not(:disabled) {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.5);
            transform: translateY(-1px);
        }

        .confirmation-btn:disabled {
            cursor: not-allowed !important;
            transform: none !important;
        }
            transition: all 0.3s ease;
        }

        .floating-button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
        }

        .voice-visualizer {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }

        .voice-visualizer.active {
            transform: scaleX(1);
            animation: voice-pulse 1.5s infinite;
        }

        @keyframes voice-pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .app-container {
                grid-template-columns: 280px 1fr;
            }
            
            .sidebar-header {
                padding: 1.5rem;
            }
            
            .nav-section {
                padding: 1rem 1.5rem;
            }
        }

        @media (max-width: 768px) {
            .app-container {
                grid-template-columns: 1fr;
            }
            
            .sidebar {
                position: fixed;
                left: -320px;
                top: 0;
                bottom: 0;
                width: 320px;
                z-index: 1000;
                transition: left 0.3s ease;
            }
            
            .sidebar.open {
                left: 0;
            }
            
            .chat-header {
                padding: 1rem 1.5rem;
            }
            
            .header-title {
                font-size: 1.25rem;
            }
            
            .header-actions {
                gap: 0.5rem;
            }
            
            .action-button {
                width: 40px;
                height: 40px;
                font-size: 1rem;
            }
            
            .chat-messages {
                padding: 1rem 1.5rem;
            }
            
            .message {
                max-width: 90%;
            }
            
            .message-action {
                width: 24px;
                height: 24px;
                font-size: 0.7rem;
            }
            
            .chat-input-area {
                padding: 1rem 1.5rem;
            }
            
            .input-wrapper {
                padding: 0.5rem 1rem;
            }
            
            .input-button {
                width: 36px;
                height: 36px;
                font-size: 0.9rem;
            }
            
            .quick-suggestions {
                padding: 0.5rem 1.5rem 0;
            }
            
            .floating-controls {
                bottom: 1rem;
                right: 1rem;
            }
        }

        @media (max-width: 480px) {
            .message {
                max-width: 95%;
            }
            
            .message-content {
                padding: 1rem;
            }
            
            .input-wrapper {
                padding: 0.5rem 1rem;
            }
            
            .suggestion-chip {
                font-size: 0.8rem;
                padding: 0.4rem 0.8rem;
            }
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { 
                opacity: 0; 
                transform: translateY(20px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }

        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }

        .slide-up {
            animation: slideUp 0.4s ease-out;
        }

        /* Scrollbar Styling */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body>
    <div class="voice-visualizer" id="voice-visualizer"></div>
    
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <div class="logo-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="logo-text">Agentic OS</div>
                </div>
                <div class="status-indicator" id="connection-status">
                    <div class="status-dot"></div>
                    <span>Connecting...</span>
                </div>
            </div>
            
            <div class="nav-section">
                <div class="nav-title">Quick Actions</div>
                <div class="nav-item active" onclick="sendQuickMessage('What\\'s my system status?')">
                    <i class="fas fa-chart-line"></i>
                    <span>System Status</span>
                </div>
                <div class="nav-item" onclick="sendQuickMessage('Check my emails')">
                    <i class="fas fa-envelope"></i>
                    <span>Check Emails</span>
                </div>
                <div class="nav-item" onclick="sendQuickMessage('Check my calendar')">
                    <i class="fas fa-calendar"></i>
                    <span>Calendar</span>
                </div>
                <div class="nav-item" onclick="sendQuickMessage('List files in Documents')">
                    <i class="fas fa-folder"></i>
                    <span>Files</span>
                </div>
                <div class="nav-item" onclick="sendQuickMessage('Research artificial intelligence')">
                    <i class="fas fa-search"></i>
                    <span>Research AI</span>
                </div>
                
                <div class="nav-title" style="margin-top: 2rem;">Communication</div>
                <div class="nav-item" onclick="sendQuickMessage('Check Slack messages')">
                    <i class="fab fa-slack"></i>
                    <span>Slack</span>
                </div>
                <div class="nav-item" onclick="sendQuickMessage('Check Teams status')">
                    <i class="fas fa-users"></i>
                    <span>Teams</span>
                </div>
                
                <div class="nav-title" style="margin-top: 2rem;">Voice</div>
                <div class="nav-item" id="voice-mode-toggle" onclick="toggleVoiceMode()">
                    <i class="fas fa-microphone"></i>
                    <span>Voice Mode: OFF</span>
                </div>
            </div>
        </div>

        <!-- Main Chat Area -->
        <div class="chat-container">
            <div class="chat-header">
                <div class="header-content">
                    <div class="header-info">
                        <div class="header-title">
                            AI Assistant
                            <span class="queue-badge" id="queue-badge" style="display: none;">0</span>
                        </div>
                        <div class="header-subtitle" id="header-subtitle">Ready to help with your tasks</div>
                    </div>
                    <div class="header-actions">
                        <button class="action-button" onclick="toggleSidebar()" title="Menu">
                            <i class="fas fa-bars"></i>
                        </button>
                        <button class="action-button" id="voice-toggle-header" onclick="toggleVoiceMode()" title="Voice Mode">
                            <i class="fas fa-microphone"></i>
                        </button>
                        <button class="action-button" onclick="clearChat()" title="Clear Chat">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="message-group">
                    <div class="message bot-message">
                        <div class="message-content">
                            <div>👋 Welcome to Agentic OS! I'm your AI assistant with advanced capabilities:</div>
                            <ul style="margin: 1rem 0; padding-left: 1.5rem; line-height: 1.8;">
                                <li><strong>🔍 Research:</strong> Multi-source information gathering</li>
                                <li><strong>📧 Email:</strong> Smart email management and analysis</li>
                                <li><strong>💻 System:</strong> Real-time monitoring and control</li>
                                <li><strong>[NOTE] Documents:</strong> Creation and management</li>
                                <li><strong>📅 Calendar:</strong> Scheduling and reminders</li>
                                <li><strong>[CHAT] Communication:</strong> Slack, Teams integration</li>
                                <li><strong>[MIC] Voice:</strong> Natural speech interaction</li>
                                <li><strong>[AI] Automation:</strong> Intelligent workflows</li>
                            </ul>
                            <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                                <strong>[MIC] Voice Features:</strong> Use the microphone button or keyboard shortcut (Ctrl+M) to speak your commands. I can understand natural language and respond with voice!
                            </div>
                        </div>
                        <div class="message-meta">
                            <span>System ready</span>
                            <div class="message-actions">
                                <button class="message-action" onclick="speakMessage(this)" title="Speak">
                                    <i class="fas fa-volume-up"></i>
                                </button>
                                <button class="message-action" onclick="copyMessage(this)" title="Copy">
                                    <i class="fas fa-copy"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-content">
                    <div class="typing-dots">
                        <span class="typing-text">AI is thinking</span>
                        <div class="typing-animation">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="quick-suggestions">
                <div class="suggestion-chip" onclick="sendQuickMessage('What\\'s my system status?')">
                    [STATS] System Status
                </div>
                <div class="suggestion-chip" onclick="sendQuickMessage('Check my emails')">
                    📧 Check Emails
                </div>
                <div class="suggestion-chip" onclick="sendQuickMessage('Research AI trends')">
                    🔍 Research AI
                </div>
                <div class="suggestion-chip" onclick="sendQuickMessage('Create a document')">
                    [NOTE] New Document
                </div>
            </div>

            <div class="chat-input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <input type="text" 
                               class="chat-input" 
                               id="chat-input"
                               placeholder="Type your message or press Ctrl+M for voice..."
                               onkeypress="handleKeyPress(event)"
                               autocomplete="off">
                        <div class="input-actions">
                            <button class="input-button voice-button" 
                                    id="voice-record-btn" 
                                    onclick="toggleVoiceRecording()" 
                                    title="Voice Record (Ctrl+M)">
                                <i class="fas fa-microphone"></i>
                            </button>
                            <button class="input-button send-button" 
                                    id="send-button" 
                                    onclick="sendMessage()" 
                                    title="Send Message">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Controls -->
    <div class="floating-controls">
        <button class="floating-button" onclick="scrollToBottom()" title="Scroll to Bottom">
            <i class="fas fa-arrow-down"></i>
        </button>
    </div>

    <script>
        // Global state
        let ws = null;
        let isConnected = false;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let voiceEnabled = false;
        let isVoiceMode = false;
        let messageCount = 0;
        let sidebarOpen = false;
        
        // Task processing state
        let isProcessingTask = false;
        let taskQueue = [];
        let currentTaskId = null;
        let currentTaskMessage = null; // Store current task message for confirmations

        // Initialize application
        document.addEventListener('DOMContentLoaded', function() {
            initializeApp();
        });

        function initializeApp() {
            connectWebSocket();
            checkVoiceSupport();
            setupKeyboardShortcuts();
            setupAutoResize();
            document.getElementById('chat-input').focus();
            
            // Hide suggestions after first message
            setTimeout(() => {
                const suggestions = document.querySelector('.quick-suggestions');
                if (messageCount === 0) {
                    suggestions.style.display = 'flex';
                }
            }, 1000);
        }

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = function() {
                isConnected = true;
                updateConnectionStatus('Connected', true);
                updateHeaderSubtitle('Connected and ready');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Handle confirmation needed status
                if (data.status === 'needs_confirmation') {
                    displayConfirmationDialog(data.details, getCurrentTaskMessage());
                } else {
                    displayMessage(data.message, 'bot', data.timestamp);
                    
                    // Auto-speak response if voice mode is enabled
                    if (isVoiceMode && data.message) {
                        speakText(data.message);
                    }
                }
                
                // Task completed - enable input and process queue
                completeCurrentTask();
                hideTyping();
                enableInput();
                hideQuickSuggestions();
                
                // Process next task in queue if any
                processNextTask();
            };

            ws.onclose = function() {
                isConnected = false;
                updateConnectionStatus('Disconnected', false);
                updateHeaderSubtitle('Reconnecting...');
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('Error', false);
                hideTyping();
                enableInput();
            };
        }

        function updateConnectionStatus(status, connected) {
            const statusEl = document.getElementById('connection-status');
            const statusSpan = statusEl.querySelector('span');
            statusSpan.textContent = status;
            
            if (connected) {
                statusEl.style.background = 'rgba(34, 197, 94, 0.1)';
                statusEl.style.borderColor = 'rgba(34, 197, 94, 0.2)';
                statusEl.style.color = '#22c55e';
            } else {
                statusEl.style.background = 'rgba(239, 68, 68, 0.1)';
                statusEl.style.borderColor = 'rgba(239, 68, 68, 0.2)';
                statusEl.style.color = '#ef4444';
            }
        }

        function updateHeaderSubtitle(text) {
            document.getElementById('header-subtitle').textContent = text;
        }

        async function checkVoiceSupport() {
            try {
                const response = await fetch('/voice/status');
                const data = await response.json();
                
                if (data.available) {
                    voiceEnabled = true;
                    document.getElementById('voice-status').innerHTML = '<span class="voice-status active">Voice: Ready</span>';
                } else {
                    document.getElementById('voice-status').innerHTML = '<span class="voice-status error">Voice: Not Available</span>';
                }
            } catch (error) {
                console.error('Voice check failed:', error);
                document.getElementById('voice-status').innerHTML = '<span class="voice-status error">Voice: Error</span>';
            }
        }

        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();

            if (message === '' || !isConnected) return;

            // Clear input immediately
            input.value = '';
            
            // Add to task queue
            queueTask({
                type: 'text',
                message: message,
                timestamp: new Date().toISOString()
            });
        }

        function sendQuickMessage(message) {
            if (!isConnected) return;

            // Add to task queue
            queueTask({
                type: 'quick',
                message: message,
                timestamp: new Date().toISOString()
            });
        }

        function displayMessage(content, sender, timestamp = null) {
            const messagesContainer = document.getElementById('chat-messages');
            
            // Create message group if it's the first message or different sender
            let messageGroup = messagesContainer.lastElementChild;
            if (!messageGroup || !messageGroup.classList.contains('message-group')) {
                messageGroup = document.createElement('div');
                messageGroup.className = 'message-group';
                messagesContainer.appendChild(messageGroup);
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message fade-in`;

            const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
            
            const actionsHtml = sender === 'bot' ? `
                <div class="message-actions">
                    <button class="message-action" onclick="speakMessage(this)" title="Speak">
                        <i class="fas fa-volume-up"></i>
                    </button>
                    <button class="message-action" onclick="copyMessage(this)" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="message-action" onclick="regenerateResponse(this)" title="Regenerate">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>
            ` : `
                <div class="message-actions">
                    <button class="message-action" onclick="copyMessage(this)" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="message-action" onclick="editMessage(this)" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                </div>
            `;
            
            messageDiv.innerHTML = `
                <div class="message-content">${content}</div>
                <div class="message-meta">
                    <span class="message-time">${time}</span>
                    ${actionsHtml}
                </div>
            `;

            messageGroup.appendChild(messageDiv);
            messageCount++;
            
            // Smooth scroll to bottom
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 100);
        }

        function hideQuickSuggestions() {
            if (messageCount > 1) {
                const suggestions = document.querySelector('.quick-suggestions');
                if (suggestions) {
                    suggestions.style.display = 'none';
                }
            }
        }
        
        function displayConfirmationDialog(result, originalMessage) {
            const confirmationHtml = `
                <div class="message-group">
                    <div class="message bot-message">
                        <div class="message-content">
                            <div class="confirmation-dialog">
                                <div class="confirmation-header">
                                    <i class="fas fa-robot"></i>
                                    <strong>🤖 Analysis Results</strong>
                                </div>
                                <div class="confirmation-content">
                                    <div class="analysis-display">${result.analysis_display || 'Analysis completed'}</div>
                                    <p class="confirmation-question">${result.confirmation_question || 'Does this look correct?'}</p>
                                </div>
                                <div class="confirmation-actions">
                                    <button class="confirmation-btn confirm-yes" onclick="handleConfirmation('yes', '${originalMessage.replace(/'/g, "\\'")}')">
                                        ✓ Yes, proceed
                                    </button>
                                    <button class="confirmation-btn confirm-no" onclick="handleConfirmation('no', '${originalMessage.replace(/'/g, "\\'")}')">
                                        ✗ No, let me clarify
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="message-meta">
                            <span class="message-time">${new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                </div>
            `;
            
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.insertAdjacentHTML('beforeend', confirmationHtml);
            scrollToBottom();
        }
        
        async function handleConfirmation(confirmation, originalMessage) {
            try {
                // Disable confirmation buttons
                const confirmBtns = document.querySelectorAll('.confirmation-btn');
                confirmBtns.forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.cursor = 'not-allowed';
                });
                
                showTyping();
                
                // Send confirmation via WebSocket
                ws.send(JSON.stringify({
                    type: 'confirmation',
                    confirmation: confirmation,
                    original_message: originalMessage
                }));
                
            } catch (error) {
                hideTyping();
                console.error('Confirmation error:', error);
                displayMessage('[ERROR] Failed to process confirmation. Please try again.', 'system');
                
                // Re-enable buttons on error
                const confirmBtns = document.querySelectorAll('.confirmation-btn');
                confirmBtns.forEach(btn => {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    btn.style.cursor = 'pointer';
                });
            }
        }

        // Task Queue Management
        function queueTask(task) {
            // Generate unique task ID
            task.id = Date.now() + Math.random();
            
            // Add to queue
            taskQueue.push(task);
            
            // Update UI to show queued state
            updateTaskQueueUI();
            
            // Process if no task is currently running
            if (!isProcessingTask) {
                processNextTask();
            }
        }

        function processNextTask() {
            if (taskQueue.length === 0 || isProcessingTask) {
                return;
            }

            const task = taskQueue.shift();
            currentTaskId = task.id;
            currentTaskMessage = task.message; // Store for confirmations
            isProcessingTask = true;

            // Update UI
            updateTaskQueueUI();
            disableInput();
            showTyping();

            // Display user message
            displayMessage(task.message, 'user');

            // Send to server
            ws.send(JSON.stringify({
                message: task.message,
                timestamp: task.timestamp,
                taskId: task.id
            }));

            // Update header subtitle
            updateHeaderSubtitle('Processing your request...');
        }

        function completeCurrentTask() {
            isProcessingTask = false;
            currentTaskId = null;
            currentTaskMessage = null; // Clear task message
            updateTaskQueueUI();
            updateHeaderSubtitle('Ready to help');
        }
        
        function getCurrentTaskMessage() {
            return currentTaskMessage;
        }

        function updateTaskQueueUI() {
            const queueCount = taskQueue.length;
            const totalTasks = queueCount + (isProcessingTask ? 1 : 0);
            const headerSubtitle = document.getElementById('header-subtitle');
            const inputContainer = document.querySelector('.input-container');
            const input = document.getElementById('chat-input');
            const queueBadge = document.getElementById('queue-badge');
            
            // Remove all state classes
            headerSubtitle.classList.remove('processing', 'queued');
            inputContainer.classList.remove('processing', 'queued');
            queueBadge.classList.remove('processing');
            
            // Update queue badge
            if (totalTasks > 0) {
                queueBadge.textContent = totalTasks;
                queueBadge.style.display = 'inline-block';
                if (isProcessingTask) {
                    queueBadge.classList.add('processing');
                }
            } else {
                queueBadge.style.display = 'none';
            }
            
            if (isProcessingTask) {
                headerSubtitle.textContent = 'Processing your request...';
                headerSubtitle.classList.add('processing');
                inputContainer.classList.add('processing');
                input.placeholder = 'Task in progress... You can queue your next message';
            } else if (queueCount > 0) {
                headerSubtitle.textContent = `${queueCount} task${queueCount > 1 ? 's' : ''} in queue`;
                headerSubtitle.classList.add('queued');
                inputContainer.classList.add('queued');
                input.placeholder = `${queueCount} task${queueCount > 1 ? 's' : ''} queued. Add another message...`;
            } else {
                headerSubtitle.textContent = 'Ready to help';
                input.placeholder = 'Type your message or press Ctrl+M for voice...';
            }

            // Update send button state - keep enabled for queuing
            const sendButton = document.getElementById('send-button');
            const voiceButton = document.getElementById('voice-record-btn');
            
            // Always keep buttons enabled to allow queuing
            sendButton.disabled = false;
            voiceButton.disabled = false;
            
            // Visual feedback only
            if (isProcessingTask) {
                sendButton.style.opacity = '0.8';
                voiceButton.style.opacity = '0.8';
            } else {
                sendButton.style.opacity = '1';
                voiceButton.style.opacity = '1';
            }
        }

        async function toggleVoiceRecording() {
            if (!voiceEnabled) {
                alert('Voice functionality is not available');
                return;
            }

            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await processVoiceInput(audioBlob);
                    
                    // Stop all tracks to release microphone
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                isRecording = true;
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.add('recording');
                recordBtn.innerHTML = '⏹️';
                
                displayMessage('[MIC] Recording... Click again to stop', 'system');

            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Could not access microphone. Please check permissions.');
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.remove('recording');
                recordBtn.innerHTML = '[MIC]';
                recordBtn.classList.add('processing');
                
                displayMessage('🔄 Processing voice input...', 'system');
            }
        }

        async function processVoiceInput(audioBlob) {
            try {
                const arrayBuffer = await audioBlob.arrayBuffer();
                const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

                const response = await fetch('/voice/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        audio_data: base64Audio,
                        format: 'wav'
                    })
                });

                const result = await response.json();
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.remove('processing');

                if (result.status === 'success') {
                    if (result.transcription) {
                        // Add voice command to task queue instead of processing directly
                        queueTask({
                            type: 'voice',
                            message: result.transcription,
                            timestamp: new Date().toISOString(),
                            audioResponse: result.audio_response
                        });
                    } else {
                        displayMessage('[ERROR] No speech detected. Please try again.', 'system');
                    }
                } else {
                    displayMessage('[ERROR] Voice processing failed. Please try again.', 'system');
                }

            } catch (error) {
                console.error('Voice processing error:', error);
                displayMessage('[ERROR] Error processing voice input.', 'system');
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.remove('processing');
            }
        }

        function playAudioResponse(base64Audio) {
            try {
                const audioData = atob(base64Audio);
                const audioArray = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    audioArray[i] = audioData.charCodeAt(i);
                }
                
                const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                
                audio.play().catch(error => {
                    console.error('Audio playback failed:', error);
                });
                
                // Clean up URL after playback
                audio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                };
                
            } catch (error) {
                console.error('Audio playback error:', error);
            }
        }

        async function speakMessage(button) {
            const messageDiv = button.closest('.message');
            const messageText = messageDiv.querySelector('div').textContent;
            await speakText(messageText);
        }

        async function speakText(text) {
            if (!voiceEnabled) return;
            
            try {
                const response = await fetch('/voice/speak', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                if (!result.success) {
                    console.error('TTS failed:', result.message);
                }
            } catch (error) {
                console.error('TTS error:', error);
            }
        }

        function copyMessage(button) {
            const messageDiv = button.closest('.message');
            const messageText = messageDiv.querySelector('div').textContent;
            navigator.clipboard.writeText(messageText).then(() => {
                button.innerHTML = '[OK]';
                setTimeout(() => {
                    button.innerHTML = '📋';
                }, 1000);
            });
        }

        function toggleVoiceMode() {
            isVoiceMode = !isVoiceMode;
            const headerBtn = document.getElementById('voice-toggle-header');
            const sidebarBtn = document.getElementById('voice-mode-toggle');
            
            if (isVoiceMode) {
                headerBtn.classList.add('voice-active');
                sidebarBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Voice Mode: ON</span>';
                displayMessage('[MIC] Voice mode enabled - responses will be spoken automatically', 'system');
                updateHeaderSubtitle('Voice mode active');
            } else {
                headerBtn.classList.remove('voice-active');
                sidebarBtn.innerHTML = '<i class="fas fa-microphone-slash"></i><span>Voice Mode: OFF</span>';
                displayMessage('🔇 Voice mode disabled', 'system');
                updateHeaderSubtitle('Ready to help');
            }
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebarOpen = !sidebarOpen;
            
            if (sidebarOpen) {
                sidebar.classList.add('open');
            } else {
                sidebar.classList.remove('open');
            }
        }

        function clearChat() {
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.innerHTML = '';
            messageCount = 0;
            
            // Show welcome message again
            setTimeout(() => {
                displayMessage('Chat cleared. How can I help you?', 'bot');
                document.querySelector('.quick-suggestions').style.display = 'flex';
            }, 300);
        }

        function scrollToBottom() {
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function scrollToBottom() {
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function setupKeyboardShortcuts() {
            document.addEventListener('keydown', function(event) {
                // Ctrl/Cmd + M for voice recording
                if ((event.ctrlKey || event.metaKey) && event.key === 'm') {
                    event.preventDefault();
                    toggleVoiceRecording();
                }
                
                // Ctrl/Cmd + Shift + V for voice mode toggle
                if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'V') {
                    event.preventDefault();
                    toggleVoiceMode();
                }
                
                // Ctrl/Cmd + K to focus input
                if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
                    event.preventDefault();
                    document.getElementById('chat-input').focus();
                }
                
                // Escape to close sidebar
                if (event.key === 'Escape' && sidebarOpen) {
                    toggleSidebar();
                }
            });
        }

        function setupAutoResize() {
            const input = document.getElementById('chat-input');
            input.addEventListener('input', function() {
                // Auto-resize input if needed (future enhancement)
            });
        }

        function regenerateResponse(button) {
            // Future enhancement: regenerate last response
            console.log('Regenerate response requested');
        }

        function editMessage(button) {
            // Future enhancement: edit user message
            console.log('Edit message requested');
        }

        // Voice visualization
        function updateVoiceVisualizer(active) {
            const visualizer = document.getElementById('voice-visualizer');
            if (active) {
                visualizer.classList.add('active');
            } else {
                visualizer.classList.remove('active');
            }
        }

        // Enhanced voice recording with visualization
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await processVoiceInput(audioBlob);
                    
                    // Stop all tracks to release microphone
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                isRecording = true;
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.add('recording');
                
                updateVoiceVisualizer(true);
                displayMessage('[MIC] Recording... Click again to stop', 'system');

            } catch (error) {
                console.error('Error starting recording:', error);
                displayMessage('[ERROR] Could not access microphone. Please check permissions.', 'system');
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                
                const recordBtn = document.getElementById('voice-record-btn');
                recordBtn.classList.remove('recording');
                
                updateVoiceVisualizer(false);
                displayMessage('🔄 Processing voice input...', 'system');
            }
        }

        function showTyping() {
            document.getElementById('typing-indicator').style.display = 'block';
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTyping() {
            document.getElementById('typing-indicator').style.display = 'none';
        }

        function disableInput() {
            // Only disable if processing task, allow queuing new tasks
            if (isProcessingTask) {
                document.getElementById('send-button').disabled = true;
                document.getElementById('voice-record-btn').disabled = true;
            }
            // Keep input enabled for queuing but update placeholder
            updateTaskQueueUI();
        }

        function enableInput() {
            document.getElementById('chat-input').disabled = false;
            document.getElementById('send-button').disabled = false;
            document.getElementById('voice-record-btn').disabled = false;
            updateTaskQueueUI();
            
            // Only focus if no tasks are processing or queued
            if (!isProcessingTask && taskQueue.length === 0) {
                document.getElementById('chat-input').focus();
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                sendMessage();
            }
        }

        // Add system message helper
        function displaySystemMessage(content) {
            displayMessage(content, 'system');
        }

    </script>
</body>
</html>
        """
