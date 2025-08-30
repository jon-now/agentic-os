#!/usr/bin/env python3
"""
GPU-Optimized Voice Interface - Pure PyTorch Implementation
Avoids TensorFlow Lite conflicts by using only PyTorch-based components
"""

import os
import sys
import asyncio
import logging
import threading
import queue
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Force PyTorch CUDA environment before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;12.9"

# Import PyTorch first to establish CUDA context
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None

# Voice processing imports
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)

class GPUVoiceInterface:
    """
    Pure PyTorch GPU-accelerated voice interface
    Avoids TensorFlow Lite conflicts by using only PyTorch components
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.is_listening = False
        self.is_speaking = False
        self.recognition_engine = None
        self.synthesis_engine = None
        self.audio_queue = queue.Queue()
        self.response_callback: Optional[Callable] = None
        
        # GPU-optimized configuration
        self.config = {
            "use_gpu": True,
            "device": "cuda" if CUDA_AVAILABLE else "cpu",
            "whisper_model": "base",  # Good balance of speed/accuracy
            "compute_type": "float16",  # GPU optimization
            "beam_size": 1,  # Faster inference
            "vad_filter": True,
            "language": "en",
            "wake_word": "hey assistant",
            "continuous_listening": False,
            "auto_speak_responses": True,
            "voice_rate": 180,
            "voice_volume": 0.9,
            "microphone_timeout": 3,
            "phrase_timeout": 2,
            "gpu_memory_fraction": 0.7
        }
        
        # Initialize components
        self._setup_gpu_environment()
        self._initialize_recognition()
        self._initialize_synthesis()
        
        logger.info(f"GPU Voice Interface initialized - Device: {self.config['device']}")
        self._log_system_status()
    
    def _setup_gpu_environment(self):
        """Setup optimal GPU environment for voice processing"""
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available, falling back to CPU")
            self.config["device"] = "cpu"
            self.config["compute_type"] = "int8"
            return
        
        try:
            # Optimize PyTorch for voice processing
            torch.cuda.empty_cache()
            
            # Set memory fraction if specified
            if self.config["gpu_memory_fraction"] < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config["gpu_memory_fraction"])
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            logger.info("GPU environment optimized for voice processing")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
    
    def _initialize_recognition(self):
        """Initialize GPU-accelerated speech recognition"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("SpeechRecognition not available")
            return
        
        # Try Faster-Whisper first (most efficient)
        if FASTER_WHISPER_AVAILABLE and self.config["use_gpu"]:
            try:
                self.whisper_model = WhisperModel(
                    self.config["whisper_model"],
                    device=self.config["device"],
                    compute_type=self.config["compute_type"],
                    cpu_threads=0 if self.config["device"] == "cuda" else 4,
                    num_workers=1
                )
                self.recognition_engine = "faster_whisper"
                logger.info(f"[OK] Faster-Whisper GPU initialized ({self.config['whisper_model']})")
                
                # Test GPU functionality
                self._test_gpu_recognition()
                return
                
            except Exception as e:
                logger.error(f"[ERROR] Faster-Whisper GPU failed: {e}")
                if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                    logger.error("CUDA library issue detected!")
                    self._suggest_cuda_fix()
        
        # Fallback to regular Whisper
        if WHISPER_AVAILABLE:
            try:
                device = self.config["device"]
                self.whisper_model = whisper.load_model(self.config["whisper_model"], device=device)
                self.recognition_engine = "whisper"
                logger.info(f"[OK] Whisper initialized on {device}")
                return
                
            except Exception as e:
                logger.error(f"[ERROR] Whisper failed: {e}")
        
        # Final fallback to basic speech recognition
        self._fallback_recognition()
    
    def _test_gpu_recognition(self):
        """Test GPU recognition with a small audio sample"""
        try:
            import numpy as np
            
            # Create 1 second of silence for testing
            test_audio = np.zeros(16000, dtype=np.float32)
            
            if self.recognition_engine == "faster_whisper":
                segments, info = self.whisper_model.transcribe(
                    test_audio,
                    beam_size=1,
                    vad_filter=False,
                    word_timestamps=False
                )
                logger.info("[OK] GPU recognition test passed")
            else:
                result = self.whisper_model.transcribe(test_audio)
                logger.info("[OK] GPU recognition test passed")
                
        except Exception as e:
            logger.warning(f"[WARNING]  GPU recognition test failed: {e}")
    
    def _suggest_cuda_fix(self):
        """Suggest CUDA fixes"""
        logger.error("[FIX] CUDA Fix Suggestions:")
        logger.error("1. Run: python fix_gpu_voice_conflicts.py")
        logger.error("2. Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        logger.error("3. Install cuDNN 8.7+")
        logger.error("4. Reinstall PyTorch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        logger.error("5. Reboot system after CUDA installation")
    
    def _fallback_recognition(self):
        """Fallback to basic speech recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.recognition_engine = "google"
            logger.info("[OK] Google Speech Recognition initialized (fallback)")
            
        except Exception as e:
            logger.error(f"[ERROR] All recognition engines failed: {e}")
            self.recognition_engine = None
    
    def _initialize_synthesis(self):
        """Initialize text-to-speech engine"""
        # Use pyttsx3 to avoid TensorFlow conflicts
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                
                # Configure voice settings
                self.tts_engine.setProperty('rate', self.config["voice_rate"])
                self.tts_engine.setProperty('volume', self.config["voice_volume"])
                
                # Try to set a pleasant voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if any(keyword in voice.name.lower() for keyword in ['female', 'zira', 'hazel']):
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                self.synthesis_engine = "pyttsx3"
                logger.info("[OK] pyttsx3 TTS initialized")
                
            except Exception as e:
                logger.error(f"[ERROR] TTS initialization failed: {e}")
                self.synthesis_engine = None
        else:
            logger.error("[ERROR] No TTS engine available")
            self.synthesis_engine = None
    
    def _log_system_status(self):
        """Log current system status"""
        logger.info("[SYSTEM]  System Status:")
        logger.info(f"   PyTorch: {'[OK]' if TORCH_AVAILABLE else '[ERROR]'}")
        logger.info(f"   CUDA: {'[OK]' if CUDA_AVAILABLE else '[ERROR]'}")
        logger.info(f"   Recognition: {self.recognition_engine or '[ERROR]'}")
        logger.info(f"   Synthesis: {self.synthesis_engine or '[ERROR]'}")
        
        if CUDA_AVAILABLE and PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                logger.info(f"   GPU: {name}")
                logger.info(f"   VRAM: {memory_info.used // 1024**2}MB / {memory_info.total // 1024**2}MB")
            except Exception as e:
                logger.warning(f"GPU status check failed: {e}")
    
    async def listen_once(self) -> Optional[str]:
        """Listen for a single voice command with GPU acceleration"""
        if not self.recognition_engine:
            logger.error("No recognition engine available")
            return None
        
        try:
            # Record audio
            with sr.Microphone() as source:
                if not hasattr(self, 'recognizer'):
                    self.recognizer = sr.Recognizer()
                
                logger.debug("[MIC] Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config["microphone_timeout"],
                    phrase_time_limit=self.config["phrase_timeout"]
                )
            
            # Process with GPU if available
            if self.recognition_engine in ["whisper", "faster_whisper"]:
                return await self._transcribe_gpu(audio)
            else:
                return await self._transcribe_fallback(audio)
                
        except sr.WaitTimeoutError:
            logger.debug("[TIMEOUT] Listen timeout")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Listen error: {e}")
            return None
    
    async def _transcribe_gpu(self, audio) -> Optional[str]:
        """Transcribe audio using GPU acceleration"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio.get_wav_data())
                tmp_path = tmp_file.name
            
            try:
                start_time = time.time()
                
                # Monitor GPU memory before transcription
                if PYNVML_AVAILABLE and CUDA_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_percent = (memory_info.used / memory_info.total) * 100
                        if memory_percent > 90:
                            logger.warning(f"[WARNING]  High GPU memory: {memory_percent:.1f}%")
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                
                # Transcribe with GPU
                if self.recognition_engine == "faster_whisper":
                    segments, info = self.whisper_model.transcribe(
                        tmp_path,
                        beam_size=self.config["beam_size"],
                        vad_filter=self.config["vad_filter"],
                        language=self.config["language"],
                        word_timestamps=False
                    )
                    text = " ".join([segment.text for segment in segments]).strip()
                else:
                    result = self.whisper_model.transcribe(tmp_path)
                    text = result["text"].strip()
                
                duration = time.time() - start_time
                logger.debug(f"[FAST] GPU transcription: {duration:.2f}s")
                
                return text if text else None
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"[ERROR] GPU transcription failed: {e}")
            return None
    
    async def _transcribe_fallback(self, audio) -> Optional[str]:
        """Fallback transcription using online services"""
        try:
            # Try Google first
            text = self.recognizer.recognize_google(audio, language=self.config["language"])
            logger.debug("[WEB] Google transcription successful")
            return text
        except sr.UnknownValueError:
            logger.debug("[UNKNOWN] Speech not understood")
            return None
        except sr.RequestError as e:
            logger.warning(f"[WARNING]  Google API error: {e}")
            return None
    
    async def speak(self, text: str) -> bool:
        """Speak text using optimized TTS"""
        if not self.synthesis_engine or self.is_speaking:
            return False
        
        self.is_speaking = True
        
        try:
            def speak_thread():
                try:
                    # Use thread-safe TTS
                    if not hasattr(self, '_tts_lock'):
                        self._tts_lock = threading.Lock()
                    
                    with self._tts_lock:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                        
                except Exception as e:
                    logger.error(f"[ERROR] TTS error: {e}")
                finally:
                    self.is_speaking = False
            
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            
            logger.debug(f"[SPEAK] Speaking: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Speak failed: {e}")
            self.is_speaking = False
            return False
    
    async def start_listening(self, callback: Optional[Callable] = None):
        """Start continuous listening with GPU acceleration"""
        if not self.recognition_engine:
            logger.error("[ERROR] No recognition engine available")
            return False
        
        self.response_callback = callback
        self.is_listening = True
        
        # Start listening loop in thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        
        logger.info("[MIC] GPU voice listening started")
        return True
    
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        logger.info("[STOP] Voice listening stopped")
    
    def _listen_loop(self):
        """Main listening loop with GPU processing"""
        while self.is_listening:
            try:
                # Use asyncio to handle the async listen_once
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                text = loop.run_until_complete(self.listen_once())
                
                if text and text.strip():
                    # Check for wake word
                    if self.config["wake_word"]:
                        if self.config["wake_word"].lower() not in text.lower():
                            continue
                        # Remove wake word
                        text = text.lower().replace(self.config["wake_word"].lower(), "").strip()
                    
                    if text:
                        # Process command
                        loop.run_until_complete(self._process_voice_command(text))
                
                loop.close()
                
            except Exception as e:
                logger.error(f"[ERROR] Listen loop error: {e}")
                if not self.config["continuous_listening"]:
                    break
    
    async def _process_voice_command(self, text: str):
        """Process voice command with orchestrator"""
        logger.info(f"[MIC] Voice: {text}")
        
        try:
            if self.orchestrator:
                result = await self.orchestrator.process_user_intention(text)
                response = result.get("result", {}).get("final_output", "I didn't understand that.")
                
                if self.config["auto_speak_responses"]:
                    await self.speak(response)
                
                if self.response_callback:
                    await self.response_callback({
                        "input": text,
                        "output": response,
                        "result": result
                    })
            else:
                response = f"I heard: {text}"
                await self.speak(response)
                
        except Exception as e:
            logger.error(f"[ERROR] Voice command processing failed: {e}")
            await self.speak("Sorry, I encountered an error.")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "gpu_available": CUDA_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "recognition_engine": self.recognition_engine,
            "synthesis_engine": self.synthesis_engine,
            "device": self.config["device"]
        }
        
        if PYNVML_AVAILABLE and CUDA_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                stats.update({
                    "gpu_memory_used": memory_info.used,
                    "gpu_memory_total": memory_info.total,
                    "gpu_memory_percent": (memory_info.used / memory_info.total) * 100,
                    "gpu_utilization": utilization.gpu
                })
            except Exception:
                pass
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice interface status (compatibility method)"""
        return {
            "available": self.recognition_engine is not None or self.synthesis_engine is not None,
            "recognition_available": self.recognition_engine is not None,
            "synthesis_available": self.synthesis_engine is not None,
            "recognition_engine": self.recognition_engine,
            "synthesis_engine": self.synthesis_engine,
            "device": self.config["device"],
            "gpu_available": CUDA_AVAILABLE,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "performance_stats": self.get_performance_stats()
        }