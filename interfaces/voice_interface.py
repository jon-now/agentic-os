#!/usr/bin/env python3
"""
Voice Interface for Agentic OS
Provides speech-to-text and text-to-speech capabilities for hands-free interaction
GPU-optimized to avoid TensorFlow Lite conflicts
"""

# Import GPU-optimized voice interface if available
try:
    from .gpu_voice_interface import GPUVoiceInterface
    GPU_VOICE_AVAILABLE = True
except ImportError:
    GPU_VOICE_AVAILABLE = False
    GPUVoiceInterface = None

from typing import Any, Callable, Dict, Optional, Set

import asyncio
import logging
import threading
import queue
from pathlib import Path
import tempfile
import os

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
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    faster_whisper = None

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None

try:
    import TTS
    from TTS.api import TTS as TTSEngine
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTSEngine = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)

class VoiceInterface:
    """
    Advanced voice interface for the Agentic OS
    Supports multiple speech recognition and synthesis engines
    Automatically uses GPU-optimized version when available
    """

    def __new__(cls, orchestrator=None):
        """Factory method to return GPU-optimized interface when available"""
        if GPU_VOICE_AVAILABLE:
            logger.info("Using GPU-optimized voice interface")
            return GPUVoiceInterface(orchestrator)
        else:
            logger.info("Using standard voice interface")
            return super().__new__(cls)

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.is_listening = False
        self.is_speaking = False
        self.recognition_engine = None
        self.synthesis_engine = None
        self.audio_queue = queue.Queue()
        self.response_callback: Optional[Callable] = None

        # Configuration - RTX 3050 optimized settings
        self.config = {
            "recognition_engine": "faster_whisper" if FASTER_WHISPER_AVAILABLE else "whisper",
            "synthesis_engine": "coqui_tts" if TTS_AVAILABLE else "pyttsx3",
            "language": "en-US",
            "wake_word": "hey assistant",
            "continuous_listening": False,
            "auto_speak_responses": True,
            "voice_rate": 150,
            "voice_volume": 0.8,
            "microphone_timeout": 5,
            "phrase_timeout": 1,
            "use_gpu": True,
            
            # RTX 3050 6GB optimized settings
            "whisper_model": "small",  # 500MB VRAM - perfect for RTX 3050
            "device": "cuda",
            "compute_type": "float16",  # Memory efficient
            "beam_size": 1,  # Faster inference for real-time
            "vad_filter": True,
            "vad_threshold": 0.5,
            "batch_size": 8,  # Conservative for 6GB
            
            # TTS optimization for RTX 3050
            "tts_model": "tts_models/en/ljspeech/fast_pitch",  # Lightweight model
            "tts_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",  # Fast vocoder
            "tts_gpu_memory_fraction": 0.3,  # Reserve memory for other tasks
            "tts_use_streaming": True,  # Real-time synthesis
            
            # Memory management
            "max_audio_length": 30,  # Limit to prevent memory issues
            "enable_voice_activity_detection": True,
            "auto_memory_cleanup": True,
            "concurrent_voice_tasks": 1  # One at a time for stability
        }

        # Initialize GPU monitoring
        self.gpu_monitoring = False
        self._initialize_gpu_monitoring()
        
        # Initialize engines
        self._initialize_recognition()
        self._initialize_synthesis()

    def _test_gpu_recognition(self):
        """Test GPU recognition functionality"""
        try:
            if hasattr(self, 'whisper_model') and self.recognition_engine == "faster_whisper":
                # Create a small test audio (silence)
                import numpy as np
                test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                
                # Test transcription
                segments, info = self.whisper_model.transcribe(
                    test_audio,
                    beam_size=1,
                    vad_filter=False
                )
                logger.info("GPU recognition test successful")
        except Exception as e:
            logger.warning(f"GPU recognition test failed: {e}")

    def _suggest_cuda_fix(self):
        """Suggest CUDA library fixes"""
        logger.info("CUDA Fix Suggestions:")
        logger.info("1. Install CUDA Toolkit 11.8 or 12.x from NVIDIA")
        logger.info("2. Install cuDNN library")
        logger.info("3. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        logger.info("4. Or run: python install_gpu_voice.py")
        logger.info("5. Reboot after CUDA installation")

    def optimize_gpu_performance(self):
        """Optimize GPU performance for voice processing with RTX 3050 specific settings"""
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available for GPU optimization")
            return
            
        try:
            import torch
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # RTX 3050 specific memory management
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve more memory for other tasks on RTX 3050
                memory_fraction = self.config.get("tts_gpu_memory_fraction", 0.7)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set GPU memory fraction to {memory_fraction} for RTX 3050")
            
            # Enable GPU optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            logger.info("RTX 3050 GPU performance optimizations applied")
            self._log_vram_usage("GPU optimization complete")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
    
    def _log_vram_usage(self, context: str):
        """Log VRAM usage for RTX 3050 monitoring"""
        try:
            if CUDA_AVAILABLE:
                import torch
                allocated = torch.cuda.memory_allocated() // 1024**2  # MB
                cached = torch.cuda.memory_reserved() // 1024**2  # MB
                total_vram = 6144  # RTX 3050 6GB
                
                logger.info(f"📊 {context} - VRAM: {allocated}MB allocated, {cached}MB cached, {total_vram-cached}MB free")
                
                # Warning if using too much VRAM
                if cached > 4500:  # 75% of 6GB
                    logger.warning(f"⚠️ High VRAM usage: {cached}MB/{total_vram}MB. Consider using smaller models.")
                    
        except Exception as e:
            logger.debug(f"VRAM monitoring failed: {e}")
    
    def get_rtx_3050_status(self) -> Dict:
        """Get RTX 3050 specific voice interface status"""
        status = self.get_status()
        
        if CUDA_AVAILABLE:
            try:
                import torch
                allocated = torch.cuda.memory_allocated() // 1024**2
                cached = torch.cuda.memory_reserved() // 1024**2
                
                status["rtx_3050_optimization"] = {
                    "vram_allocated": f"{allocated}MB",
                    "vram_cached": f"{cached}MB",
                    "vram_free": f"{6144-cached}MB",
                    "vram_usage_percent": f"{(cached/6144)*100:.1f}%",
                    "recommended_models": {
                        "whisper": "small (500MB) or tiny (150MB)",
                        "tts": "fast_pitch (800MB) or tacotron2 (1.2GB)"
                    },
                    "memory_tips": [
                        "Use 'small' Whisper model for best balance",
                        "Enable auto_memory_cleanup in config",
                        "Close other GPU applications",
                        "Use CPU fallback if VRAM > 5GB"
                    ]
                }
            except Exception as e:
                status["rtx_3050_optimization"] = {"error": str(e)}
                
        return status

    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {
            "gpu_available": CUDA_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "recognition_engine": self.recognition_engine,
            "synthesis_engine": self.synthesis_engine,
        }
        
        if self.gpu_monitoring:
            gpu_stats = self.get_gpu_stats()
            if gpu_stats:
                stats.update(gpu_stats)
        
        return stats

    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        try:
            if PYNVML_AVAILABLE:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_monitoring = True
                logger.info("GPU monitoring initialized")
            else:
                self.gpu_monitoring = False
                logger.info("GPU monitoring not available (pynvml not installed)")
        except Exception as e:
            self.gpu_monitoring = False
            logger.warning(f"GPU monitoring initialization failed: {e}")

    def _log_gpu_status(self):
        """Log current GPU status"""
        if not self.gpu_monitoring:
            return
            
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get GPU info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            logger.info(f"GPU Status: {name}")
            logger.info(f"  Memory: {memory_info.used // 1024**2}MB / {memory_info.total // 1024**2}MB")
            logger.info(f"  Utilization: {utilization.gpu}%")
            logger.info(f"  Temperature: {temperature}°C")
            
        except Exception as e:
            logger.warning(f"Failed to get GPU status: {e}")

    def get_gpu_stats(self):
        """Get current GPU statistics"""
        if not self.gpu_monitoring:
            return None
            
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                "memory_used": memory_info.used,
                "memory_total": memory_info.total,
                "memory_percent": (memory_info.used / memory_info.total) * 100,
                "gpu_utilization": utilization.gpu,
                "memory_utilization": utilization.memory,
                "temperature": temperature
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return None

    def _initialize_recognition(self):
        """Initialize GPU-optimized speech recognition engine"""
        engine = self.config["recognition_engine"]

        # GPU-first approach with proper CUDA setup
        if engine == "faster_whisper" and FASTER_WHISPER_AVAILABLE:
            try:
                from faster_whisper import WhisperModel
                
                # Ensure CUDA is properly initialized
                if TORCH_AVAILABLE and CUDA_AVAILABLE:
                    import torch
                    torch.cuda.init()  # Initialize CUDA context
                    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                
                # GPU-optimized configuration
                self.whisper_model = WhisperModel(
                    self.config["whisper_model"], 
                    device="cuda",
                    compute_type=self.config["compute_type"],
                    cpu_threads=0,  # Use GPU threads
                    num_workers=1,  # Single worker for GPU
                    download_root=None,
                    local_files_only=False
                )
                
                self.recognition_engine = "faster_whisper"
                logger.info(f"GPU-accelerated Faster-Whisper initialized with {self.config['whisper_model']} model")
                
                # Test GPU functionality
                self._test_gpu_recognition()
                return
                
            except Exception as e:
                logger.error(f"GPU Faster-Whisper initialization failed: {e}")
                if "cublas" in str(e).lower():
                    logger.error("CUDA cuBLAS library issue detected. Please install proper CUDA libraries.")
                    self._suggest_cuda_fix()
                self._fallback_to_whisper()

        elif engine == "whisper" and WHISPER_AVAILABLE:
            try:
                device = "cuda" if self.config["use_gpu"] and CUDA_AVAILABLE else "cpu"
                self.whisper_model = whisper.load_model(self.config["whisper_model"], device=device)
                self.recognition_engine = "whisper"
                logger.info(f"Whisper speech recognition initialized on {device}")
            except Exception as e:
                logger.warning("Failed to initialize Whisper: %s", e)
                self._fallback_recognition()

        elif engine == "google" and SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()

                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)

                self.recognition_engine = "google"
                logger.info("Google speech recognition initialized")
            except Exception as e:
                logger.warning("Failed to initialize Google recognition: %s", e)
                self._fallback_recognition()

        else:
            self._fallback_recognition()

    def _fallback_to_whisper(self):
        """GPU-optimized fallback to regular whisper"""
        if WHISPER_AVAILABLE:
            try:
                # GPU-first approach for regular Whisper
                if TORCH_AVAILABLE and CUDA_AVAILABLE:
                    import torch
                    torch.cuda.empty_cache()  # Clear GPU memory
                
                self.whisper_model = whisper.load_model(
                    self.config["whisper_model"], 
                    device="cuda"
                )
                self.recognition_engine = "whisper"
                logger.info(f"GPU-accelerated Whisper fallback initialized")
                return
                    
            except Exception as e:
                logger.error(f"GPU Whisper fallback failed: {e}")
                if "cublas" in str(e).lower():
                    self._suggest_cuda_fix()
                self._fallback_recognition()
        else:
            self._fallback_recognition()

    def _fallback_recognition(self):
        """Fallback to available recognition engine"""
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                self.recognition_engine = "sphinx"
                logger.info("Sphinx speech recognition initialized (fallback)")
            except Exception as e:
                logger.error("No speech recognition available: %s", e)
                self.recognition_engine = None
        else:
            logger.error("No speech recognition libraries available")
            self.recognition_engine = None

    def _initialize_synthesis(self):
        """Initialize text-to-speech engine with RTX 3050 optimization"""
        engine = self.config["synthesis_engine"]

        # Try Coqui TTS first (GPU accelerated neural voices optimized for RTX 3050)
        if engine == "coqui_tts" and TTS_AVAILABLE:
            try:
                device = "cuda" if self.config["use_gpu"] and CUDA_AVAILABLE else "cpu"
                
                # RTX 3050 optimized TTS model selection
                if device == "cuda":
                    # Use lightweight model optimized for 6GB VRAM
                    model_name = self.config.get("tts_model", "tts_models/en/ljspeech/fast_pitch")
                    logger.info(f"Initializing Coqui TTS with RTX 3050 optimized model: {model_name}")
                    
                    # Set GPU memory fraction for TTS
                    import torch
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        memory_fraction = self.config.get("tts_gpu_memory_fraction", 0.3)
                        torch.cuda.set_per_process_memory_fraction(memory_fraction)
                        logger.info(f"Set TTS GPU memory fraction to {memory_fraction}")
                else:
                    model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Fallback model
                
                self.tts_model = TTSEngine(model_name, gpu=self.config["use_gpu"])
                self.synthesis_engine = "coqui_tts"
                logger.info(f"✅ Coqui TTS initialized on {device}")
                
                if device == "cuda":
                    self._log_vram_usage("Coqui TTS loaded")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize Coqui TTS: {e}")
                if "out of memory" in str(e).lower():
                    logger.info("TTS GPU memory insufficient, trying CPU fallback")
                    self._fallback_synthesis_cpu()
                    return
                else:
                    self._fallback_synthesis()

        elif engine == "pyttsx3" and PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()

                # Configure voice settings optimized for RTX 3050 performance
                self.tts_engine.setProperty('rate', self.config["voice_rate"])
                self.tts_engine.setProperty('volume', self.config["voice_volume"])

                # Try to set a pleasant voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Prefer female voices for assistant
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break

                self.synthesis_engine = "pyttsx3"
                logger.info("✅ pyttsx3 text-to-speech initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3: {e}")
                self.synthesis_engine = None

        else:
            self._fallback_synthesis()
    
    def _fallback_synthesis_cpu(self):
        """Fallback to CPU TTS when GPU memory is insufficient"""
        try:
            if TTS_AVAILABLE:
                # Use CPU-optimized TTS model
                self.tts_model = TTSEngine("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
                self.synthesis_engine = "coqui_tts"
                logger.info("✅ Coqui TTS CPU fallback initialized")
            else:
                self._fallback_synthesis()
        except Exception as e:
            logger.error(f"CPU TTS fallback failed: {e}")
            self._fallback_synthesis()

    def _fallback_synthesis(self):
        """Fallback to pyttsx3 if GPU TTS fails"""
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.config["voice_rate"])
                self.tts_engine.setProperty('volume', self.config["voice_volume"])
                
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                self.synthesis_engine = "pyttsx3"
                logger.info("Fallback to pyttsx3 TTS")
            except Exception as e:
                logger.error("All TTS engines failed: %s", e)
                self.synthesis_engine = None
        else:
            logger.warning("No text-to-speech engine available")
            self.synthesis_engine = None

    async def start_listening(self, callback: Optional[Callable] = None):
        """Start continuous voice listening"""
        if not self.recognition_engine:
            logger.error("No speech recognition engine available")
            return False

        self.response_callback = callback
        self.is_listening = True

        # Start listening in a separate thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()

        logger.info("Voice listening started")
        return True

    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        logger.info("Voice listening stopped")

    def _listen_loop(self):
        """Main listening loop (runs in separate thread)"""
        while self.is_listening:
            try:
                if self.recognition_engine in ["whisper", "faster_whisper"]:
                    text = self._listen_whisper()
                else:
                    text = self._listen_speech_recognition()

                if text and text.strip():
                    # Check for wake word if configured
                    if self.config["wake_word"]:
                        if self.config["wake_word"].lower() not in text.lower():
                            continue
                        # Remove wake word from text
                        text = text.lower().replace(self.config["wake_word"].lower(), "").strip()

                    if text:
                        # Process the voice command
                        asyncio.run_coroutine_threadsafe(
                            self._process_voice_command(text),
                            asyncio.get_event_loop()
                        )

            except Exception as e:
                logger.error("Error in listening loop: %s", e)
                if not self.config["continuous_listening"]:
                    break

    def _listen_whisper(self) -> Optional[str]:
        """Listen using Whisper model (regular or faster-whisper)"""
        try:
            # Record audio
            if not SPEECH_RECOGNITION_AVAILABLE:
                return None

            with sr.Microphone() as source:
                if not hasattr(self, 'recognizer'):
                    self.recognizer = sr.Recognizer()

                audio = self.recognizer.listen(
                    source,
                    timeout=self.config["microphone_timeout"],
                    phrase_time_limit=self.config["phrase_timeout"]
                )

            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio.get_wav_data())
                tmp_path = tmp_file.name

            try:
                # Monitor GPU before transcription
                if self.gpu_monitoring:
                    gpu_stats = self.get_gpu_stats()
                    if gpu_stats and gpu_stats["memory_percent"] > 90:
                        logger.warning(f"High GPU memory usage: {gpu_stats['memory_percent']:.1f}%")
                
                start_time = time.time()
                
                if self.recognition_engine == "faster_whisper":
                    # GPU-accelerated faster-whisper with optimizations
                    segments, info = self.whisper_model.transcribe(
                        tmp_path, 
                        beam_size=self.config.get("beam_size", 5),
                        vad_filter=self.config.get("vad_filter", True),
                        word_timestamps=False  # Faster processing
                    )
                    text = " ".join([segment.text for segment in segments]).strip()
                    
                    # Log transcription performance
                    duration = time.time() - start_time
                    logger.debug(f"GPU transcription completed in {duration:.2f}s")
                    
                else:
                    # Use regular whisper with GPU if available
                    result = self.whisper_model.transcribe(tmp_path)
                    text = result["text"].strip()
                    
                    duration = time.time() - start_time
                    logger.debug(f"Transcription completed in {duration:.2f}s")
                
                return text
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            logger.debug("Whisper recognition error: %s", e)
            return None

    def _listen_speech_recognition(self) -> Optional[str]:
        """Listen using speech_recognition library"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config["microphone_timeout"],
                    phrase_time_limit=self.config["phrase_timeout"]
                )

            # Try Google recognition first, fallback to offline
            try:
                text = self.recognizer.recognize_google(audio, language=self.config["language"])
                return text
            except sr.UnknownValueError:
                # Try offline recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except Exception:
                    return None
            except sr.RequestError:
                # Fallback to offline recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except Exception:
                    return None

        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logger.debug("Speech recognition error: %s", e)
            return None

    async def _process_voice_command(self, text: str):
        """Process recognized voice command"""
        logger.info("Voice command received: %s", text)

        try:
            if self.orchestrator:
                # Process through orchestrator
                result = await self.orchestrator.process_user_intention(text)
                response = result.get("result", {}).get("final_output", "I didn't understand that.")

                # Speak the response if configured
                if self.config["auto_speak_responses"]:
                    await self.speak(response)

                # Call callback if provided
                if self.response_callback:
                    await self.response_callback({
                        "input": text,
                        "output": response,
                        "result": result
                    })

            else:
                # Simple echo response
                response = f"I heard you say: {text}"
                await self.speak(response)

        except Exception as e:
            logger.error("Error processing voice command: %s", e)
            await self.speak("Sorry, I encountered an error processing your request.")

    async def speak(self, text: str) -> bool:
        """Convert text to speech with GPU acceleration"""
        if not self.synthesis_engine or self.is_speaking:
            return False

        self.is_speaking = True

        try:
            if self.synthesis_engine == "coqui_tts":
                # GPU-accelerated Coqui TTS with memory management
                def speak_thread():
                    try:
                        # Monitor GPU before synthesis
                        if self.gpu_monitoring:
                            gpu_stats = self.get_gpu_stats()
                            if gpu_stats and gpu_stats["memory_percent"] > 85:
                                logger.warning(f"High GPU memory before TTS: {gpu_stats['memory_percent']:.1f}%")
                                # Clear GPU cache if available
                                if TORCH_AVAILABLE and CUDA_AVAILABLE:
                                    import torch
                                    torch.cuda.empty_cache()
                        
                        start_time = time.time()
                        
                        # Generate audio with GPU acceleration
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        try:
                            self.tts_model.tts_to_file(text=text, file_path=tmp_path)
                            
                            synthesis_time = time.time() - start_time
                            logger.debug(f"GPU TTS synthesis completed in {synthesis_time:.2f}s")
                            
                            # Play the audio file
                            if os.name == 'nt':  # Windows
                                import winsound
                                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                            else:  # Linux/Mac
                                os.system(f"aplay {tmp_path}")
                        
                        finally:
                            # Ensure file cleanup with retry
                            import time
                            for attempt in range(3):
                                try:
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                                    break
                                except (OSError, PermissionError):
                                    if attempt < 2:
                                        time.sleep(0.1)  # Brief delay before retry
                                    else:
                                        logger.warning(f"Could not delete temporary file: {tmp_path}")
                            
                    except Exception as e:
                        logger.error(f"GPU TTS error: {e}")
                        # Fallback to basic TTS if GPU fails
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.say(text)
                            engine.runAndWait()
                        except Exception as fallback_e:
                            logger.error(f"Fallback TTS also failed: {fallback_e}")
                    finally:
                        self.is_speaking = False

                thread = threading.Thread(target=speak_thread, daemon=True)
                thread.start()
                return True

            elif self.synthesis_engine == "pyttsx3":
                # Use a lock to prevent multiple TTS instances
                if not hasattr(self, '_tts_lock'):
                    self._tts_lock = threading.Lock()

                def speak_thread():
                    with self._tts_lock:
                        try:
                            # Stop any current speech
                            self.tts_engine.stop()
                            # Clear the queue
                            self.tts_engine.say(text)
                            self.tts_engine.runAndWait()
                        except Exception as e:
                            logger.debug("TTS error: %s", e)
                        finally:
                            self.is_speaking = False

                thread = threading.Thread(target=speak_thread, daemon=True)
                thread.start()
                return True

        except Exception as e:
            logger.error("Speech synthesis error: %s", e)
            self.is_speaking = False
            return False

        return False

    async def listen_once(self) -> Optional[str]:
        """Listen for a single voice command"""
        if not self.recognition_engine:
            return None

        try:
            if self.recognition_engine in ["whisper", "faster_whisper"]:
                return self._listen_whisper()
            else:
                return self._listen_speech_recognition()
        except Exception as e:
            logger.error("Single listen error: %s", e)
            return None

    def set_config(self, config: Dict[str, Any]):
        """Update voice interface configuration"""
        self.config.update(config)

        # Reinitialize if needed
        if "recognition_engine" in config:
            self._initialize_recognition()

        if "synthesis_engine" in config:
            self._initialize_synthesis()

        # Update TTS settings
        if self.synthesis_engine == "pyttsx3" and self.tts_engine:
            if "voice_rate" in config:
                self.tts_engine.setProperty('rate', config["voice_rate"])
            if "voice_volume" in config:
                self.tts_engine.setProperty('volume', config["voice_volume"])

    def get_status(self) -> Dict[str, Any]:
        """Get voice interface status"""
        return {
            "recognition_engine": self.recognition_engine,
            "synthesis_engine": self.synthesis_engine,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
            "whisper_available": WHISPER_AVAILABLE,
            "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "coqui_tts_available": TTS_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE,
            "config": self.config
        }

    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        voices = []

        if self.synthesis_engine == "pyttsx3" and self.tts_engine:
            try:
                engine_voices = self.tts_engine.getProperty('voices')
                for voice in engine_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "languages": getattr(voice, 'languages', []),
                        "gender": getattr(voice, 'gender', 'unknown')
                    })
            except Exception as e:
                logger.error("Error getting voices: %s", e)

        return voices

    def set_voice(self, voice_id: str) -> bool:
        """Set TTS voice by ID"""
        if self.synthesis_engine == "pyttsx3" and self.tts_engine:
            try:
                self.tts_engine.setProperty('voice', voice_id)
                return True
            except Exception as e:
                logger.error("Error setting voice: %s", e)
                return False

        return False

    def close(self):
        """Clean up voice interface"""
        self.stop_listening()

        if self.synthesis_engine == "pyttsx3" and hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.stop()
            except Exception:
                pass

        logger.info("Voice interface closed")

# Example usage and testing
async def test_voice_interface():
    """Test the voice interface"""
    print("Testing Voice Interface...")

    voice = VoiceInterface()
    status = voice.get_status()

    print(f"Recognition Engine: {status['recognition_engine']}")
    print(f"Synthesis Engine: {status['synthesis_engine']}")
    print("Available Libraries:")
    print(f"  - Speech Recognition: {status['speech_recognition_available']}")
    print(f"  - Whisper: {status['whisper_available']}")
    print(f"  - pyttsx3: {status['pyttsx3_available']}")
    print(f"  - Piper: {status['piper_available']}")

    # Test TTS
    if status['synthesis_engine']:
        print("\nTesting text-to-speech...")
        await voice.speak("Hello! Voice interface is working correctly.")

        # Show available voices
        voices = voice.get_available_voices()
        if voices:
            print(f"\nAvailable voices: {len(voices)}")
            for voice_info in voices[:3]:  # Show first 3
                print(f"  - {voice_info['name']} ({voice_info['id']})")

    # Test single listen (if recognition available)
    if status['recognition_engine']:
        print("\nTesting speech recognition...")
        print("Say something (you have 5 seconds)...")

        text = await voice.listen_once()
        if text:
            print(f"Recognized: {text}")
            await voice.speak(f"I heard you say: {text}")
        else:
            print("No speech detected or recognition failed")

    voice.close()
    print("Voice interface test completed!")

if __name__ == "__main__":
    asyncio.run(test_voice_interface())
