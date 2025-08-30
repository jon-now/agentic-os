# 👻 GHOST User Guide

**GHOST (Graphical Heuristic Operating System Tool)** is a voice-activated overlay interface for the Agentic OS that provides quick AI assistance through both voice and text interactions.

## 🎯 Overview

GHOST is designed to be your always-available AI assistant that can be summoned with a simple voice command. It provides a compact, elegant overlay that appears on top of any application, allowing you to interact with your AI assistant without switching contexts.

### Key Features
- 🎤 **Voice Activation**: Say "GHOST" to wake up the assistant
- 💬 **Dual Interface**: Both voice and text input supported
- 🪟 **System Overlay**: Appears on top of any application
- 🧠 **AI Integration**: Full access to Agentic OS capabilities
- ⚡ **Quick Actions**: Fast access to common functions
- 🎨 **Modern UI**: Clean, dark-themed interface

## 🚀 Getting Started

### Installation

1. **Install Voice Dependencies** (Required for voice features):
   ```bash
   python install_ghost_voice.py
   ```

2. **Launch GHOST**:
   ```bash
   # Standalone mode
   python ghost_launcher.py
   
   # Or integrated with main system
   python main.py
   ```

### First Time Setup

1. **Microphone Test**: When GHOST starts, it will automatically test your microphone
2. **Wake Word**: The default wake word is "GHOST" (customizable in settings)
3. **Voice Mode**: Voice mode is enabled by default

## 💬 How to Use GHOST

### Voice Activation

1. **Wake Up GHOST**: Say "GHOST" clearly
2. **Wait for Response**: GHOST will acknowledge with "Yes? How can I help you?"
3. **Give Command**: Speak your command or question
4. **Listen to Response**: GHOST will respond both visually and audibly

**Example Voice Interaction:**
```
You: "GHOST"
GHOST: "Yes? How can I help you?"
You: "What's my system status?"
GHOST: "Your CPU usage is 45%, memory at 60%, and all systems are running normally."
```

### Text Interface

1. **Show Overlay**: Say "GHOST" or use the system tray
2. **Type Message**: Click in the text input area
3. **Press Enter**: Send your message
4. **View Response**: See the response in the chat area

### Overlay Controls

- **🎤 Voice Mode Button**: Toggle voice mode on/off
- **Clear Button**: Clear chat history
- **⚙️ Settings**: Open settings dialog
- **× Close**: Hide the overlay
- **Escape Key**: Quick hide overlay

## 🎤 Voice Features

### Supported Commands

GHOST understands natural language and can handle various types of requests:

#### System Commands
- "What's my system status?"
- "Check my CPU and memory usage"
- "Open calculator"
- "Lock my computer"

#### Email Commands
- "Check my emails"
- "Send email to [person] about [topic]"
- "Show my unread messages"

#### General Assistance
- "What's the weather like?"
- "Set a reminder for [time]"
- "Search for information about [topic]"
- "Help me with [task]"

#### Chat and Questions
- "Hello GHOST"
- "What can you do?"
- "Tell me a joke"
- "What time is it?"

### Voice Settings

Access voice settings through the ⚙️ Settings button:

- **Wake Word**: Change from "GHOST" to custom word
- **Voice Timeout**: How long to wait for commands
- **Speech Rate**: TTS speaking speed
- **Volume**: TTS volume level

## 🖥️ Interface Overview

### Main Components

1. **Header Bar**
   - 👻 GHOST logo and title
   - Subtitle explaining the acronym
   - × Close button

2. **Chat Display**
   - Scrollable message history
   - Color-coded messages:
     - **Blue**: Your messages
     - **Green**: GHOST responses
     - **Orange**: System messages
     - **Red**: Error messages

3. **Input Area**
   - Text input field
   - Send button
   - Enter key support

4. **Control Panel**
   - 🎤 Voice mode toggle
   - Clear chat button
   - ⚙️ Settings button

5. **Status Bar**
   - Current status indicator
   - Voice mode status

### Overlay Behavior

- **Always on Top**: Stays above other windows
- **Auto-Hide**: Automatically hides after period of inactivity
- **Focus Management**: Clicking outside hides the overlay
- **Transparency**: Configurable transparency level

## ⚙️ Configuration

### Settings Dialog

Access through the Settings button (⚙️):

#### Voice Settings
- **Wake Word**: Customize activation phrase
- **Timeout**: Voice command timeout duration
- **Recognition**: Speech recognition sensitivity

#### Display Settings
- **Position**: Overlay position on screen
- **Size**: Overlay dimensions
- **Transparency**: Window transparency level
- **Theme**: Color scheme (dark/light)

#### Behavior Settings
- **Auto-Hide**: Automatic hiding after inactivity
- **Always on Top**: Keep overlay above other windows
- **Startup**: Launch GHOST on system startup

### Configuration File

GHOST settings are stored in the main Agentic OS configuration system and can be modified programmatically:

```python
ghost_config = {
    "overlay_size": (400, 300),
    "position": "center",
    "transparency": 0.95,
    "wake_word": "ghost",
    "voice_timeout": 5.0,
    "auto_hide_delay": 10000
}
```

## 🔧 Troubleshooting

### Common Issues

#### Voice Not Working
**Problem**: GHOST doesn't respond to voice commands
**Solutions**:
1. Check microphone permissions
2. Install voice dependencies: `python install_ghost_voice.py`
3. Test microphone in system settings
4. Restart GHOST overlay

#### Wake Word Not Detected
**Problem**: Saying "GHOST" doesn't activate the overlay
**Solutions**:
1. Speak clearly and at normal volume
2. Check if voice mode is enabled (🎤 button should be green)
3. Try different wake word in settings
4. Check background noise levels

#### Overlay Not Showing
**Problem**: GHOST overlay doesn't appear
**Solutions**:
1. Check if running: look for GHOST in system tray
2. Try keyboard shortcut (if configured)
3. Restart GHOST launcher
4. Check display settings

#### TTS Not Working
**Problem**: GHOST shows responses but doesn't speak
**Solutions**:
1. Check system audio settings
2. Verify pyttsx3 installation
3. Test with different TTS engine
4. Check audio device selection

### Advanced Troubleshooting

#### Log Files
Check GHOST logs for detailed error information:
- Location: `logs/orchestrator.log`
- Look for "ghost" or "voice" related errors

#### Dependency Check
Verify all required packages are installed:
```bash
python -c "import speech_recognition, pyttsx3, pyaudio; print('All dependencies OK')"
```

#### Manual Testing
Test components individually:
```bash
# Test voice recognition
python -c "import speech_recognition as sr; r=sr.Recognizer(); print('Voice recognition OK')"

# Test TTS
python -c "import pyttsx3; e=pyttsx3.init(); e.say('Test'); e.runAndWait(); print('TTS OK')"
```

## 🎨 Customization

### Themes

GHOST supports custom themes through CSS-like configuration:

```python
custom_theme = {
    "bg_color": "#1a1a1a",      # Background color
    "fg_color": "#ffffff",       # Text color
    "accent_color": "#00d4ff",   # Accent color
    "font_family": "Arial",      # Font family
    "font_size": 10              # Font size
}
```

### Custom Wake Words

Change the wake word to anything you prefer:

1. Open Settings (⚙️)
2. Change "Wake Word" field
3. Click Apply
4. Test new wake word

**Recommended Wake Words**:
- "Ghost" (default)
- "Assistant"
- "Computer"
- "Helper"
- Custom names

### Keyboard Shortcuts

Configure keyboard shortcuts for quick access:
- `Ctrl+Alt+G`: Toggle GHOST overlay
- `Ctrl+Alt+V`: Toggle voice mode
- `Escape`: Hide overlay

## 🔮 Advanced Features

### System Integration

GHOST integrates deeply with the Agentic OS:

- **Email Management**: Full Gmail integration
- **System Control**: Computer control and monitoring
- **File Operations**: File and folder management
- **Web Automation**: Browser control and web scraping
- **Research**: Multi-source information gathering

### Voice Commands Examples

```
# System Management
"GHOST, check my system performance"
"GHOST, open the task manager"
"GHOST, what's taking up my disk space?"

# Email Operations
"GHOST, do I have any new emails?"
"GHOST, send an email to John about the meeting"
"GHOST, show me emails from last week"

# File Operations
"GHOST, find files containing 'project'"
"GHOST, backup my documents folder"
"GHOST, clean up my downloads"

# Research and Information
"GHOST, research artificial intelligence trends"
"GHOST, what's the weather forecast?"
"GHOST, translate this text to Spanish"

# Automation
"GHOST, schedule a meeting for tomorrow"
"GHOST, remind me to call mom at 3pm"
"GHOST, set up automated backups"
```

### API Integration

GHOST can be controlled programmatically:

```python
from interfaces.ghost_overlay import GhostOverlay

# Initialize GHOST
ghost = GhostOverlay(orchestrator)

# Show overlay
ghost.show_overlay()

# Add custom message
ghost._add_chat_message("Custom automation message", "system")

# Process command programmatically
await ghost._process_command("Check system status")
```

## 📊 Performance Tips

### Optimization

1. **Voice Recognition**: Use shorter phrases for better accuracy
2. **Background Listening**: Minimize background noise
3. **System Resources**: GHOST uses minimal CPU when idle
4. **Memory Management**: Chat history auto-clears after 100 messages

### Best Practices

1. **Wake Word**: Choose a unique word not commonly used
2. **Commands**: Be specific and clear in voice commands
3. **Positioning**: Place overlay where it won't interfere with work
4. **Voice Mode**: Toggle off voice mode when not needed

## 🛡️ Privacy and Security

### Local Processing
- All voice recognition happens locally (when possible)
- No voice data sent to external servers
- Chat history stored locally only

### Data Protection
- No personal data transmitted without permission
- All AI processing through local Ollama instance
- Secure credential management for integrations

## 📚 References

- **Main Documentation**: See main Agentic OS documentation
- **Voice Setup**: `install_ghost_voice.py`
- **API Reference**: Source code in `interfaces/ghost_overlay.py`
- **Configuration**: `config/settings.py`

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files for error details
3. Test individual components
4. File issues with detailed error descriptions

---

**GHOST** - Your always-available AI assistant, just a voice command away! 👻