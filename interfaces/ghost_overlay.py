#!/usr/bin/env python3
"""
GHOST (Graphical Heuristic Operating System Tool) - Voice-Activated Overlay Interface
A voice-activated compact overlay system for the Agentic OS
"""

import asyncio
import logging
import threading
import time
import queue
from datetime import datetime
from typing import Any, Dict, Optional, Callable
import json

try:
    import tkinter as tk
    from tkinter import ttk, font, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None

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
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

logger = logging.getLogger(__name__)

class GhostOverlay:
    """
    GHOST (Graphical Heuristic Operating System Tool)
    Voice-activated overlay interface for quick AI interactions
    """
    
    def __init__(self, orchestrator=None, voice_interface=None):
        self.orchestrator = orchestrator
        self.voice_interface = voice_interface
        
        # State management
        self.is_active = False
        self.is_listening = False
        self.is_speaking = False
        self.is_voice_mode = True
        
        # UI components
        self.root = None
        self.overlay_window = None
        self.chat_display = None
        self.input_entry = None
        self.voice_button = None
        self.status_label = None
        
        # Voice components
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.wake_word = "ghost"  # Our activation word
        
        # Configuration
        self.config = {
            "overlay_size": (400, 300),
            "position": "center",
            "transparency": 0.95,
            "auto_hide_delay": 10000,  # 10 seconds
            "wake_word_threshold": 0.7,
            "voice_timeout": 5.0,
            "phrase_timeout": 1.0,
            "always_on_top": True,
            "compact_mode": True
        }
        
        # Initialize components
        self._initialize_voice()
        self._initialize_ui()
        self._start_background_listening()
        
        logger.info("GHOST overlay initialized successfully")
    
    def _initialize_voice(self):
        """Initialize voice recognition and synthesis"""
        try:
            if SPEECH_RECOGNITION_AVAILABLE and PYAUDIO_AVAILABLE:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                logger.info("Voice recognition initialized")
            else:
                logger.warning("Voice recognition not available - missing dependencies")
                
            if PYTTSX3_AVAILABLE:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)  # Speech rate
                self.tts_engine.setProperty('volume', 0.8)  # Volume level
                logger.info("Text-to-speech initialized")
            else:
                logger.warning("Text-to-speech not available")
                
        except Exception as e:
            logger.error(f"Voice initialization failed: {e}")
            self.recognizer = None
            self.tts_engine = None
    
    def _initialize_ui(self):
        """Initialize the overlay UI"""
        if not TKINTER_AVAILABLE:
            logger.error("Tkinter not available - UI disabled")
            return
            
        try:
            # Create root window (hidden)
            self.root = tk.Tk()
            self.root.withdraw()
            
            # Create overlay window
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.withdraw()  # Hide initially
            
            # Configure window
            self._configure_window()
            self._create_widgets()
            
            logger.info("GHOST UI initialized")
            
        except Exception as e:
            logger.error(f"UI initialization failed: {e}")
            self.overlay_window = None
    
    def _configure_window(self):
        """Configure the overlay window"""
        if not self.overlay_window:
            return
            
        # Window properties
        self.overlay_window.title("GHOST - AI Assistant")
        width, height = self.config["overlay_size"]
        
        # Position window
        screen_width = self.overlay_window.winfo_screenwidth()
        screen_height = self.overlay_window.winfo_screenheight()
        
        if self.config["position"] == "center":
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
        else:
            x, y = 100, 100  # Default position
            
        self.overlay_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Window attributes
        self.overlay_window.attributes('-topmost', self.config["always_on_top"])
        self.overlay_window.attributes('-alpha', self.config["transparency"])
        self.overlay_window.configure(bg='#1a1a1a')
        
        # Remove window decorations for clean look
        self.overlay_window.overrideredirect(True)
        
        # Bind events
        self.overlay_window.bind('<Escape>', lambda e: self.hide_overlay())
        self.overlay_window.bind('<FocusOut>', self._on_focus_out)
    
    def _create_widgets(self):
        """Create the overlay widgets"""
        if not self.overlay_window:
            return
            
        # Main frame with rounded appearance
        self.main_frame = tk.Frame(
            self.overlay_window,
            bg='#2d2d2d',
            padx=15,
            pady=15
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with GHOST branding
        self._create_header()
        
        # Chat display area
        self._create_chat_area()
        
        # Input area
        self._create_input_area()
        
        # Control buttons
        self._create_controls()
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create the header with GHOST branding"""
        header_frame = tk.Frame(self.main_frame, bg='#2d2d2d')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # GHOST logo/title
        title_label = tk.Label(
            header_frame,
            text="👻 GHOST",
            bg='#2d2d2d',
            fg='#00d4ff',
            font=('Arial', 14, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Graphical Heuristic Operating System Tool",
            bg='#2d2d2d',
            fg='#888888',
            font=('Arial', 8)
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Close button
        close_btn = tk.Button(
            header_frame,
            text="×",
            bg='#2d2d2d',
            fg='#ffffff',
            font=('Arial', 12, 'bold'),
            bd=0,
            command=self.hide_overlay,
            width=2
        )
        close_btn.pack(side=tk.RIGHT)
    
    def _create_chat_area(self):
        """Create the chat display area"""
        # Chat frame
        chat_frame = tk.Frame(self.main_frame, bg='#2d2d2d')
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollable text area
        self.chat_display = tk.Text(
            chat_frame,
            bg='#1a1a1a',
            fg='#ffffff',
            font=('Consolas', 9),
            wrap=tk.WORD,
            height=8,
            width=50,
            bd=0,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(chat_frame, command=self.chat_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.config(yscrollcommand=scrollbar.set)
        
        # Configure text tags for different message types
        self.chat_display.tag_config("user", foreground="#00d4ff")
        self.chat_display.tag_config("ghost", foreground="#00ff88")
        self.chat_display.tag_config("system", foreground="#ffaa00")
        self.chat_display.tag_config("error", foreground="#ff4444")
        
        # Welcome message
        self._add_chat_message("👻 GHOST activated! Say 'GHOST' to wake me up, or type your message.", "system")
    
    def _create_input_area(self):
        """Create the input area"""
        input_frame = tk.Frame(self.main_frame, bg='#2d2d2d')
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text input
        self.input_entry = tk.Entry(
            input_frame,
            bg='#3d3d3d',
            fg='#ffffff',
            font=('Arial', 10),
            bd=0,
            relief=tk.FLAT,
            insertbackground='#ffffff'
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind('<Return>', self._on_text_input)
        self.input_entry.bind('<Control-Return>', self._on_text_input)
        
        # Send button
        send_btn = tk.Button(
            input_frame,
            text="Send",
            bg='#00d4ff',
            fg='#ffffff',
            font=('Arial', 9),
            bd=0,
            command=self._on_text_input,
            width=8
        )
        send_btn.pack(side=tk.RIGHT)
    
    def _create_controls(self):
        """Create control buttons"""
        controls_frame = tk.Frame(self.main_frame, bg='#2d2d2d')
        controls_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Voice button
        self.voice_button = tk.Button(
            controls_frame,
            text="🎤 Voice Mode",
            bg='#00ff88' if self.is_voice_mode else '#666666',
            fg='#ffffff',
            font=('Arial', 9),
            bd=0,
            command=self._toggle_voice_mode
        )
        self.voice_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear button
        clear_btn = tk.Button(
            controls_frame,
            text="Clear",
            bg='#666666',
            fg='#ffffff',
            font=('Arial', 9),
            bd=0,
            command=self._clear_chat
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Settings button
        settings_btn = tk.Button(
            controls_frame,
            text="⚙️",
            bg='#666666',
            fg='#ffffff',
            font=('Arial', 9),
            bd=0,
            command=self._show_settings,
            width=3
        )
        settings_btn.pack(side=tk.RIGHT)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_label = tk.Label(
            self.main_frame,
            text="Ready - Say 'GHOST' to activate voice",
            bg='#2d2d2d',
            fg='#888888',
            font=('Arial', 8),
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
    
    def _add_chat_message(self, message: str, sender: str = "ghost"):
        """Add a message to the chat display"""
        if not self.chat_display:
            return
            
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add message with appropriate tag
        if sender == "user":
            self.chat_display.insert(tk.END, f"[{timestamp}] You: {message}\n", "user")
        elif sender == "ghost":
            self.chat_display.insert(tk.END, f"[{timestamp}] GHOST: {message}\n", "ghost")
        elif sender == "system":
            self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n", "system")
        elif sender == "error":
            self.chat_display.insert(tk.END, f"[{timestamp}] Error: {message}\n", "error")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _start_background_listening(self):
        """Start background voice listening for wake word"""
        if not self.recognizer or not self.microphone:
            return
            
        def listen_for_wake_word():
            while True:
                try:
                    if not self.is_listening and self.is_voice_mode:
                        with self.microphone as source:
                            # Listen for wake word with shorter timeout
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                            
                        try:
                            # Use Google's speech recognition for wake word detection
                            text = self.recognizer.recognize_google(audio).lower()
                            
                            if self.wake_word in text:
                                logger.info(f"Wake word '{self.wake_word}' detected!")
                                self._wake_up()
                                
                        except sr.UnknownValueError:
                            pass  # No speech detected, continue listening
                        except sr.RequestError as e:
                            logger.error(f"Speech recognition error: {e}")
                            time.sleep(1)
                            
                except sr.WaitTimeoutError:
                    pass  # Timeout is expected, continue listening
                except Exception as e:
                    logger.error(f"Background listening error: {e}")
                    time.sleep(1)
                    
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        # Start background listening thread
        listen_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
        listen_thread.start()
        logger.info("Background voice listening started")
    
    def _wake_up(self):
        """Wake up GHOST and show overlay"""
        self.show_overlay()
        self._add_chat_message("I'm listening! What can I help you with?", "ghost")
        self._update_status("Listening for command...")
        
        if self.tts_engine:
            self._speak("Yes? How can I help you?")
        
        # Start listening for command
        if self.is_voice_mode:
            asyncio.create_task(self._listen_for_command())
    
    async def _listen_for_command(self):
        """Listen for voice command after wake word"""
        if not self.recognizer or not self.microphone:
            return
            
        self.is_listening = True
        self._update_status("🎤 Listening...")
        
        try:
            # Listen for actual command with longer timeout
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech
            command = self.recognizer.recognize_google(audio)
            self._add_chat_message(command, "user")
            
            # Process command
            await self._process_command(command)
            
        except sr.WaitTimeoutError:
            self._add_chat_message("No command heard. Try again!", "system")
        except sr.UnknownValueError:
            self._add_chat_message("Sorry, I didn't understand that.", "ghost")
        except sr.RequestError as e:
            self._add_chat_message(f"Speech recognition error: {e}", "error")
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            self._add_chat_message(f"Voice error: {e}", "error")
        finally:
            self.is_listening = False
            self._update_status("Ready - Say 'GHOST' to activate voice")
    
    async def _process_command(self, command: str):
        """Process a voice or text command"""
        self._update_status("🤖 Processing...")
        
        try:
            if self.orchestrator:
                # Use the orchestrator to process the command
                result = await self.orchestrator.process_user_intention(command)
                response = result.get("result", {}).get("final_output", "Command processed.")
            else:
                # Fallback response
                response = f"I heard: '{command}'. The orchestrator is not available right now."
            
            self._add_chat_message(response, "ghost")
            
            # Speak response if in voice mode
            if self.is_voice_mode and self.tts_engine:
                self._speak(response)
                
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            self._add_chat_message(error_msg, "error")
            logger.error(error_msg)
        finally:
            self._update_status("Ready - Say 'GHOST' to activate voice")
    
    def _speak(self, text: str):
        """Speak text using TTS"""
        if not self.tts_engine:
            return
            
        def speak_async():
            try:
                self.is_speaking = True
                # Limit text length for TTS
                if len(text) > 200:
                    text_to_speak = text[:200] + "..."
                else:
                    text_to_speak = text
                    
                self.tts_engine.say(text_to_speak)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self.is_speaking = False
        
        # Run TTS in separate thread
        tts_thread = threading.Thread(target=speak_async, daemon=True)
        tts_thread.start()
    
    def _update_status(self, status: str):
        """Update status label"""
        if self.status_label:
            self.status_label.config(text=status)
    
    # UI Event Handlers
    def _on_text_input(self, event=None):
        """Handle text input"""
        if not self.input_entry:
            return
            
        text = self.input_entry.get().strip()
        if not text:
            return
            
        self.input_entry.delete(0, tk.END)
        self._add_chat_message(text, "user")
        asyncio.create_task(self._process_command(text))
    
    def _toggle_voice_mode(self):
        """Toggle voice mode on/off"""
        self.is_voice_mode = not self.is_voice_mode
        
        if self.voice_button:
            if self.is_voice_mode:
                self.voice_button.config(bg='#00ff88', text="🎤 Voice Mode")
                self._update_status("Voice mode ON - Say 'GHOST' to activate")
            else:
                self.voice_button.config(bg='#666666', text="🔇 Voice OFF")
                self._update_status("Voice mode OFF - Type to chat")
        
        self._add_chat_message(f"Voice mode {'enabled' if self.is_voice_mode else 'disabled'}", "system")
    
    def _clear_chat(self):
        """Clear chat history"""
        if self.chat_display:
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self._add_chat_message("Chat cleared", "system")
    
    def _show_settings(self):
        """Show settings dialog"""
        if not self.overlay_window:
            return
            
        settings_window = tk.Toplevel(self.overlay_window)
        settings_window.title("GHOST Settings")
        settings_window.geometry("300x400")
        settings_window.configure(bg='#2d2d2d')
        settings_window.attributes('-topmost', True)
        
        # Settings content
        tk.Label(
            settings_window,
            text="⚙️ GHOST Settings",
            bg='#2d2d2d',
            fg='#00d4ff',
            font=('Arial', 12, 'bold')
        ).pack(pady=10)
        
        # Voice settings
        voice_frame = tk.LabelFrame(
            settings_window,
            text="Voice Settings",
            bg='#2d2d2d',
            fg='#ffffff'
        )
        voice_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Wake word setting
        tk.Label(voice_frame, text="Wake Word:", bg='#2d2d2d', fg='#ffffff').pack(anchor=tk.W)
        wake_word_entry = tk.Entry(voice_frame, bg='#3d3d3d', fg='#ffffff')
        wake_word_entry.insert(0, self.wake_word)
        wake_word_entry.pack(fill=tk.X, padx=5, pady=2)
        
        # Apply button
        def apply_settings():
            self.wake_word = wake_word_entry.get().lower()
            self._add_chat_message(f"Wake word changed to: {self.wake_word}", "system")
            settings_window.destroy()
        
        tk.Button(
            settings_window,
            text="Apply",
            bg='#00d4ff',
            fg='#ffffff',
            command=apply_settings
        ).pack(pady=10)
    
    def _on_focus_out(self, event=None):
        """Handle focus lost event"""
        if self.config.get("auto_hide_delay", 0) > 0:
            # Auto-hide after delay
            def auto_hide():
                time.sleep(self.config["auto_hide_delay"] / 1000)
                if not self.is_listening and not self.is_speaking:
                    self.hide_overlay()
            
            threading.Thread(target=auto_hide, daemon=True).start()
    
    # Public Methods
    def show_overlay(self):
        """Show the GHOST overlay"""
        if not self.overlay_window:
            return
            
        self.overlay_window.deiconify()
        self.overlay_window.lift()
        self.overlay_window.focus_force()
        self.is_active = True
        
        # Focus on input
        if self.input_entry:
            self.input_entry.focus_set()
        
        logger.info("GHOST overlay shown")
    
    def hide_overlay(self):
        """Hide the GHOST overlay"""
        if not self.overlay_window:
            return
            
        self.overlay_window.withdraw()
        self.is_active = False
        logger.info("GHOST overlay hidden")
    
    def toggle_overlay(self):
        """Toggle overlay visibility"""
        if self.is_active:
            self.hide_overlay()
        else:
            self.show_overlay()
    
    def get_status(self) -> Dict[str, Any]:
        """Get GHOST status"""
        return {
            "is_active": self.is_active,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "is_voice_mode": self.is_voice_mode,
            "wake_word": self.wake_word,
            "voice_available": self.recognizer is not None,
            "tts_available": self.tts_engine is not None,
            "ui_available": self.overlay_window is not None
        }
    
    def close(self):
        """Clean up GHOST overlay"""
        self.is_active = False
        self.is_listening = False
        
        if self.overlay_window:
            self.overlay_window.destroy()
        
        if self.root:
            self.root.quit()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        logger.info("GHOST overlay closed")

# Example usage and testing
async def test_ghost_overlay():
    """Test the GHOST overlay"""
    print("Testing GHOST Overlay...")
    
    ghost = GhostOverlay()
    status = ghost.get_status()
    
    print("GHOST Status:")
    for key, value in status.items():
        print(f"  - {key}: {value}")
    
    if status["ui_available"]:
        print("\nGHOST overlay ready!")
        print("Try saying 'GHOST' to activate voice mode")
        print("Or click the overlay to interact")
    
    return ghost

if __name__ == "__main__":
    asyncio.run(test_ghost_overlay())