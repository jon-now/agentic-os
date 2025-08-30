import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from contextlib import asynccontextmanager

from llm.ollama_client import OllamaClient
from llm.prompt_templates import PromptTemplates
from core.context_manager import CrossAppContextManager
from core.automation_engine import AutomationEngine, TaskPriority
from core.conversation_manager import ConversationManager
from controllers.browser_controller import BrowserController
from controllers.email_controller import EmailController
from controllers.document_controller import DocumentController
from controllers.calendar_controller import CalendarController
from controllers.file_controller import FileController
from controllers.slack_controller import SlackController
from controllers.teams_controller import TeamsController
from controllers.system_controller import SystemController
from controllers.vision_controller import VisionController
from workflows.automation_workflows import AutomationWorkflows
from core.llm_analyzer import LLMAnalyzer
from memory.vector_store import VectorStore
from intelligence.context_analyzer import ContextAnalyzer
from intelligence.learning_engine import LearningEngine
from intelligence.proactive_assistant import ProactiveAssistant
from communication.message_analyzer import MessageAnalyzer
from communication.smart_responder import SmartResponder

logger = logging.getLogger(__name__)

class SessionConfig:
    """Configuration for session management"""
    MAX_HISTORY_SIZE = 100
    MAX_SESSION_AGE_HOURS = 24
    CLEANUP_INTERVAL_MINUTES = 30

class IntentionAnalyzer:
    """Dedicated class for intention analysis with improved logic"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self._keyword_patterns = self._build_keyword_patterns()
    
    def _build_keyword_patterns(self) -> Dict[str, Dict]:
        """Build optimized keyword patterns for fast analysis"""
        return {
            "system_status": {
                "patterns": ["system status", "system info", "computer status", "how is my computer", "system health"],
                "action": "system_info",
                "confidence": 0.95,
                "apps": [],
                "steps": 0
            },
            "greeting": {
                "patterns": ["hello", "hi there", "hey there", "good morning", "good afternoon", "good evening", "\\bhi\\b", "\\bhey\\b"],
                "action": "chat",
                "confidence": 0.95,
                "apps": [],
                "steps": 0
            },
            "personal_inquiry": {
                "patterns": ["how are you", "how do you do", "what's up", "how's it going", 
                           "tell me about yourself", "who are you", "what are you"],
                "action": "chat",
                "confidence": 0.9,
                "apps": [],
                "steps": 0
            },
            "general_knowledge": {
                "patterns": ["what is", "tell me about", "explain", "how does", "why does",
                           "what are", "define", "describe", "what's the difference"],
                "action": "general_knowledge",
                "confidence": 0.85,
                "apps": [],
                "steps": 0
            },
            "opinion_discussion": {
                "patterns": ["what do you think", "your opinion", "do you believe", "thoughts on",
                           "what's your view", "how do you feel", "what would you do"],
                "action": "chat",
                "confidence": 0.8,
                "apps": [],
                "steps": 0
            },
            "casual_conversation": {
                "patterns": ["just chatting", "let's talk", "tell me something", "what's new",
                           "anything interesting", "random question", "just wondering"],
                "action": "chat",
                "confidence": 0.8,
                "apps": [],
                "steps": 0
            },
            "research": {
                "patterns": ["research", "find information", "search", "look up", "investigate"],
                "action": "research",
                "confidence": 0.85,
                "apps": ["browser"],
                "steps": 2
            },
            "send_email": {
                "patterns": ["send email", "send mail", "email someone", "compose email", "write email", 
                           "send message", "email to", "mail to", "send a mail", "send an email"],
                "action": "send_email",
                "confidence": 0.9,
                "apps": ["email"],
                "steps": 2
            },
            "system_automation": {
                "patterns": ["recycle bin", "clean temp", "system cleanup", "clear trash", 
                           "empty recycle", "temp files", "cleanup system", "maintenance", "clean system"],
                "action": "system_automation",
                "confidence": 0.85,
                "apps": ["system"],
                "steps": 2
            },
            "email": {
                "patterns": ["check email", "gmail", "inbox", "messages", "read mail", "email status"],
                "action": "email_management",
                "confidence": 0.8,
                "apps": ["email"],
                "steps": 2
            },
            "system_control": {
                "patterns": ["shutdown", "restart", "lock screen", "set volume", "kill process", 
                           "system info", "lock computer", "turn off", "reboot"],
                "action": "system_control",
                "confidence": 0.85,
                "apps": ["system"],
                "steps": 2
            },
            "open_application": {
                "patterns": ["open", "launch", "start", "run application", "open app", 
                           "launch app", "start app", "run program", "open program"],
                "action": "open_application",
                "confidence": 0.9,
                "apps": ["system"],
                "steps": 1
            },
            "file_management": {
                "patterns": ["file", "files", "folder", "directory", "copy", "move", "delete", "search files", "list files"],
                "action": "file_management",
                "confidence": 0.8,
                "apps": ["file"],
                "steps": 2
            },
            "document_creation": {
                "patterns": ["document", "doc", "write", "create document", "report", "presentation"],
                "action": "document_creation",
                "confidence": 0.8,
                "apps": ["document"],
                "steps": 2
            },
            "calendar": {
                "patterns": ["calendar", "schedule", "meeting", "appointment", "event"],
                "action": "calendar_management",
                "confidence": 0.8,
                "apps": ["calendar"],
                "steps": 2
            },
            "communication": {
                "patterns": ["slack", "teams", "message", "chat", "send message", "communication"],
                "action": "communication",
                "confidence": 0.8,
                "apps": ["slack", "teams"],
                "steps": 2
            },
            "automation": {
                "patterns": ["automate", "schedule", "task", "workflow"],
                "action": "automation",
                "confidence": 0.7,
                "apps": ["automation"],
                "steps": 3
            }
        }
    
    async def analyze_intention(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced intention analysis with context support"""
        # Sanitize input
        if not user_input or not user_input.strip():
            return self._create_fallback_intention("empty_input")
        
        user_input = user_input.strip()[:1000]  # Limit input length
        
        # Enhanced context analysis if available
        if context and context.get("context_analysis"):
            context_result = self._enhance_intention_with_context(user_input, context["context_analysis"])
            if context_result["confidence"] > 0.8:
                return context_result
        
        # Fast keyword analysis
        keyword_result = self._analyze_keywords(user_input)
        if keyword_result["confidence"] > 0.8:
            return keyword_result
        
        # Try LLM analysis for complex cases
        if context and self.llm_client:
            try:
                llm_result = await self._analyze_with_llm(user_input, context)
                if llm_result and llm_result.get("confidence", 0) > 0.6:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Return keyword result as fallback
        return keyword_result if keyword_result["confidence"] > 0.5 else self._create_fallback_intention("general_chat")
    
    def _analyze_keywords(self, user_input: str) -> Dict[str, Any]:
        """Enhanced keyword-based intention analysis"""
        import re
        user_lower = user_input.lower()
        
        for intent_type, config in self._keyword_patterns.items():
            # Check patterns with word boundaries for single words
            pattern_matches = []
            for pattern in config["patterns"]:
                if len(pattern.split()) == 1 and len(pattern) <= 3:
                    # Single short word - use word boundaries
                    if re.search(r'\b' + re.escape(pattern) + r'\b', user_lower):
                        pattern_matches.append(pattern)
                else:
                    # Multi-word or longer patterns - use substring matching
                    if pattern in user_lower:
                        pattern_matches.append(pattern)
            
            if pattern_matches:
                # Determine conversation type for chat actions
                conversation_type = "general"
                if intent_type == "greeting":
                    conversation_type = "greeting"
                elif intent_type == "personal_inquiry":
                    conversation_type = "personal_inquiry"
                elif intent_type == "opinion_discussion":
                    conversation_type = "opinion_discussion"
                elif intent_type == "casual_conversation":
                    conversation_type = "casual"
                elif intent_type == "general_knowledge":
                    conversation_type = "knowledge"
                
                return {
                    "interaction_type": "direct_response" if config["steps"] == 0 else "task_execution",
                    "primary_action": config["action"],
                    "confidence": config["confidence"],
                    "applications_needed": config["apps"],
                    "parameters": {
                        "user_input": user_input, 
                        "intent_type": intent_type,
                        "conversation_type": conversation_type,
                        "topic": user_input if config["action"] == "general_knowledge" else None,
                        "operation": user_input if config["action"] == "file_management" else None,
                        "content": user_input if config["action"] == "document_creation" else None,
                        "request": user_input if config["action"] in ["calendar_management", "communication", "automation"] else None
                    },
                    "estimated_steps": config["steps"],
                    "user_intent_summary": intent_type.replace("_", " ").title(),
                    "analysis_method": "keyword"
                }
        
        # Default to chat
        return self._create_fallback_intention("general_chat")
    
    def _enhance_intention_with_context(self, user_input: str, context_analysis: Dict) -> Dict:
        """Enhance intention analysis using context analysis results"""
        # Start with basic keyword analysis
        base_intention = self._analyze_keywords(user_input)

        # Enhance with context analysis insights
        behavioral_patterns = context_analysis.get("behavioral_patterns", {})
        preferred_actions = behavioral_patterns.get("preferred_actions", [])

        # Boost confidence if this matches user's preferred actions
        if preferred_actions:
            primary_action = base_intention.get("primary_action")
            for pref_action in preferred_actions:
                if pref_action.get("action") == primary_action:
                    base_intention["confidence"] = min(base_intention["confidence"] + 0.2, 1.0)
                    base_intention["enhancement_reason"] = "Matches user preference pattern"
                    break

        # Adjust based on complexity score
        complexity_score = context_analysis.get("complexity_score", 0)
        if complexity_score > 0.7:
            base_intention["estimated_steps"] = max(base_intention.get("estimated_steps", 1) + 1, 3)
            base_intention["complexity_adjustment"] = "High complexity detected"

        # Add proactive suggestions from context analysis
        suggested_actions = context_analysis.get("suggested_actions", [])
        if suggested_actions:
            base_intention["context_suggestions"] = suggested_actions[:3]

        return base_intention
    
    async def _analyze_with_llm(self, user_input: str, context: Dict) -> Optional[Dict]:
        """LLM-based analysis for complex cases"""
        try:
            return await self.llm_client.analyze_intention(user_input, context)
        except Exception as e:
            logger.error(f"LLM intention analysis failed: {e}")
            return None
    
    def _create_fallback_intention(self, intent_type: str) -> Dict[str, Any]:
        """Create fallback intention for edge cases"""
        return {
            "interaction_type": "direct_response",
            "primary_action": "chat",
            "confidence": 0.6,
            "applications_needed": [],
            "parameters": {"conversation_type": "general", "intent_type": intent_type},
            "estimated_steps": 0,
            "user_intent_summary": "General conversation",
            "analysis_method": "fallback"
        }

class ResponseGenerator:
    """Dedicated class for generating responses with full conversational capabilities"""
    
    def __init__(self, llm_client: OllamaClient, prompt_templates: PromptTemplates):
        self.llm_client = llm_client
        self.prompt_templates = prompt_templates
    
    async def generate_response(self, user_input: str, context: Dict, intention: Dict) -> str:
        """Generate appropriate response based on intention"""
        try:
            if intention["primary_action"] == "system_info":
                return await self._generate_system_status(context)
            elif intention["primary_action"] == "chat":
                return await self._generate_chat_response(user_input, context, intention)
            elif intention["primary_action"] == "general_knowledge":
                return await self._generate_knowledge_response(user_input, context)
            else:
                return self._get_generic_response(intention)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._get_fallback_response(user_input)
    
    async def _generate_system_status(self, context: Dict) -> str:
        """Generate system status response"""
        system_status = context.get("system_status", {})
        active_apps = context.get("active_apps", [])
        recent_files = context.get("recent_files", [])
        
        try:
            if self.prompt_templates and hasattr(self.prompt_templates, 'SYSTEM_STATUS'):
                prompt = self.prompt_templates.SYSTEM_STATUS.format(
                    cpu_percent=system_status.get("cpu_percent", "Unknown"),
                    memory_percent=system_status.get("memory_percent", "Unknown"),
                    active_apps=active_apps[:5] if active_apps else ["No applications detected"],
                    recent_files=[f["name"] for f in recent_files[:3]] if recent_files else ["No recent files"],
                    network_status="Connected" if system_status.get("network", {}).get("connected") else "Disconnected",
                    current_time=context.get("current_time", "Unknown")
                )
                
                response = await self.llm_client.generate_response(prompt)
                return response.strip() if response else self._get_basic_system_status(system_status, active_apps)
            else:
                return self._get_basic_system_status(system_status, active_apps)
                
        except Exception as e:
            logger.error(f"System status generation failed: {e}")
            return self._get_basic_system_status(system_status, active_apps)
    
    def _get_basic_system_status(self, system_status: Dict, active_apps: List) -> str:
        """Basic system status without LLM"""
        cpu = system_status.get("cpu_percent", "Unknown")
        memory = system_status.get("memory_percent", "Unknown")
        apps_count = len(active_apps)
        
        return f"""System Status:
CPU Usage: {cpu}%
Memory Usage: {memory}%
Active Applications: {apps_count} running
Status: System running normally
Current Time: {datetime.now().strftime('%I:%M %p')}"""
    
    async def _generate_chat_response(self, user_input: str, context: Dict, intention: Dict) -> str:
        """Generate natural, engaging chat response with full conversational capabilities"""
        try:
            conversation_type = intention.get("parameters", {}).get("conversation_type", "general")
            
            # Build rich context for conversation
            conversation_context = {
                "system_apps": len(context.get('active_apps', [])),
                "current_time": context.get('current_time', 'Unknown'),
                "recent_history": self._get_recent_conversation_summary(context),
                "user_tone": self._analyze_user_tone(user_input),
            }
            
            if conversation_type == "greeting":
                return await self._generate_greeting_response(user_input, conversation_context)
            elif conversation_type == "personal_inquiry":
                return await self._generate_personal_inquiry_response(user_input, conversation_context)
            elif conversation_type == "opinion_discussion":
                return await self._generate_opinion_response(user_input, conversation_context)
            elif conversation_type == "casual":
                return await self._generate_casual_response(user_input, conversation_context)
            elif conversation_type == "knowledge":
                return await self._generate_knowledge_response(user_input, context)
            else:
                return await self._generate_general_chat_response(user_input, conversation_context)
                
        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            return self._get_fallback_chat_response(user_input)
    
    async def _generate_greeting_response(self, user_input: str, context: Dict) -> str:
        """Generate personalized greeting response"""
        try:
            prompt = self._create_greeting_prompt(user_input, context)
            response = await self.llm_client.generate_response(prompt)
            return response.strip() if response else self._get_greeting_response()
        except Exception:
            return self._get_greeting_response()
    
    async def _generate_personal_inquiry_response(self, user_input: str, context: Dict) -> str:
        """Generate response to personal questions"""
        try:
            prompt = self._create_personal_inquiry_prompt(user_input, context)
            response = await self.llm_client.generate_response(prompt)
            return response.strip() if response else self._get_personal_inquiry_response()
        except Exception:
            return self._get_personal_inquiry_response()
    
    async def _generate_opinion_response(self, user_input: str, context: Dict) -> str:
        """Generate response for opinion discussions"""
        try:
            if hasattr(self.prompt_templates, 'CONVERSATIONAL_RESPONSE'):
                prompt = self.prompt_templates.CONVERSATIONAL_RESPONSE.format(
                    user_input=user_input,
                    conversation_history=context["recent_history"],
                    system_context=f"System running {context['system_apps']} apps at {context['current_time']}",
                    user_tone=context["user_tone"]
                )
                response = await self.llm_client.generate_response(prompt)
                return response.strip() if response else self._get_opinion_fallback(user_input)
            else:
                return self._get_opinion_fallback(user_input)
        except Exception:
            return self._get_opinion_fallback(user_input)
    
    async def _generate_casual_response(self, user_input: str, context: Dict) -> str:
        """Generate response for casual conversation"""
        try:
            if hasattr(self.prompt_templates, 'CASUAL_CONVERSATION'):
                prompt = self.prompt_templates.CASUAL_CONVERSATION.format(
                    user_input=user_input,
                    context=context
                )
                response = await self.llm_client.generate_response(prompt)
                return response.strip() if response else self._get_casual_fallback(user_input)
            else:
                return self._get_casual_fallback(user_input)
        except Exception:
            return self._get_casual_fallback(user_input)
    
    async def _generate_general_chat_response(self, user_input: str, context: Dict) -> str:
        """Generate general chat response"""
        try:
            if hasattr(self.prompt_templates, 'DIRECT_CHAT'):
                prompt = self.prompt_templates.DIRECT_CHAT.format(
                    user_input=user_input,
                    context=f"System has {context['system_apps']} active applications, Current time: {context['current_time']}"
                )
                response = await self.llm_client.generate_response(prompt)
                return response.strip() if response else self._get_generic_chat_response()
            else:
                return self._get_generic_chat_response()
        except Exception:
            return self._get_generic_chat_response()
    
    def _create_greeting_prompt(self, user_input: str, context: Dict) -> str:
        """Create personalized greeting prompt"""
        time_of_day = self._get_time_of_day()
        return f"""You're greeting a user who said: "{user_input}"

Context: It's {time_of_day}, system is running {context['system_apps']} applications.

Respond with a warm, personalized greeting that:
- Matches their greeting style (formal/casual)
- References the time of day appropriately
- Shows enthusiasm for helping
- Asks an engaging follow-up question about what they'd like to do

Be natural and conversational, like greeting a friend.

Response:"""

    def _create_personal_inquiry_prompt(self, user_input: str, context: Dict) -> str:
        """Create prompt for personal questions about the AI"""
        return f"""The user asked a personal question: "{user_input}"

Respond in a way that:
- Shows personality and warmth
- Explains your capabilities in an engaging way
- Demonstrates your current status (all systems running well)
- Asks about their interests or how you can help
- Is conversational, not robotic

Be friendly and show genuine interest in helping them.

Response:"""
    
    def _analyze_user_tone(self, user_input: str) -> str:
        """Analyze the emotional tone of user input"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["excited", "awesome", "great", "amazing", "love"]):
            return "enthusiastic"
        elif any(word in user_lower for word in ["tired", "stressed", "busy", "overwhelmed"]):
            return "stressed"
        elif any(word in user_lower for word in ["curious", "wondering", "interested"]):
            return "curious"
        elif any(word in user_lower for word in ["help", "need", "problem", "issue"]):
            return "seeking_help"
        else:
            return "neutral"
    
    def _get_recent_conversation_summary(self, context: Dict) -> str:
        """Get summary of recent conversation for context"""
        session_history = context.get("session_history", [])
        if not session_history:
            return "This is the start of our conversation."
        
        recent = session_history[-3:]  # Last 3 interactions
        summary_parts = []
        
        for interaction in recent:
            user_msg = interaction.get("user_input", "")[:50]
            intent = interaction.get("intention", {}).get("user_intent_summary", "")
            summary_parts.append(f"User: {user_msg}... (Intent: {intent})")
        
        return " | ".join(summary_parts)
    
    def _get_greeting_response(self) -> str:
        """Get greeting response"""
        time_of_day = self._get_time_of_day()
        return f"Good {time_of_day}! I'm your AI assistant and I'm ready to help you with research, email management, system monitoring, and more. What can I do for you today?"
    
    def _get_personal_inquiry_response(self) -> str:
        """Get personal inquiry response"""
        return """I'm your personal AI assistant! I can help you with:

- Have engaging conversations about any topic
- Research information you're curious about  
- Manage your emails intelligently
- Monitor your system and keep things running smoothly
- Help create documents and content
- Automate repetitive tasks to save you time

What would you like to work on together?"""
    
    def _get_opinion_fallback(self, user_input: str) -> str:
        """Fallback for opinion discussions"""
        return f"That's a thoughtful question! I'd love to share my perspective on '{user_input}'. While I don't have personal experiences like humans do, I can offer insights based on patterns and knowledge. What specific aspect interests you most?"
    
    def _get_casual_fallback(self, user_input: str) -> str:
        """Fallback for casual conversation"""
        return f"That's interesting! You mentioned '{user_input}' - I'd love to explore that with you. What aspect would you like to dive into first?"
    
    def _get_generic_chat_response(self) -> str:
        """Generic chat response fallback"""
        return "I'm here to help! I can research topics, manage emails, check system status, create documents, and have great conversations. What's on your mind?"
    
    async def _generate_knowledge_response(self, user_input: str, context: Dict) -> str:
        """Generate knowledge response"""
        try:
            if hasattr(self.prompt_templates, 'GENERAL_KNOWLEDGE'):
                prompt = self.prompt_templates.GENERAL_KNOWLEDGE.format(
                    user_input=user_input,
                    context=f"Current time: {context.get('current_time', 'Unknown')}, System running normally"
                )
                response = await self.llm_client.generate_response(prompt)
                return response.strip() if response else self._get_knowledge_fallback(user_input)
            else:
                return self._get_knowledge_fallback(user_input)
        except Exception as e:
            logger.error(f"Knowledge response generation failed: {e}")
            return self._get_knowledge_fallback(user_input)
    
    def _get_knowledge_fallback(self, user_input: str) -> str:
        """Fallback for knowledge questions"""
        return f"That's a great question about '{user_input}'! While I'm having some technical difficulties accessing my full knowledge base right now, I'd still love to help. Could you be more specific about what aspect you're most curious about?"
    
    def _get_fallback_chat_response(self, user_input: str) -> str:
        """Enhanced fallback responses for when LLM is unavailable"""
        user_lower = user_input.lower()

        if any(word in user_lower for word in ["hello", "hi", "hey"]):
            time_greeting = self._get_time_of_day()
            return f"Good {time_greeting}! I'm your AI assistant and I'm excited to help you today. I can research topics, manage your emails, check system status, automate tasks, and have great conversations. What's on your mind?"
        
        elif any(word in user_lower for word in ["how are you", "how do you do", "what's up"]):
            return "I'm doing well! All my systems are running smoothly and I'm energized to help you with whatever you need. I've got access to your browser, email, and various automation tools. What would you like to explore together?"
        
        elif any(word in user_lower for word in ["tell me about yourself", "who are you", "what are you"]):
            return """I'm your personal AI assistant! Think of me as your digital companion who can:

- Have engaging conversations about any topic
- Research anything you're curious about  
- Manage your emails intelligently
- Monitor your system and keep things running smoothly
- Help create documents and content
- Automate repetitive tasks to save you time

I'm powered by local AI, so our conversations stay private. I love learning about your interests and finding creative ways to help. What would you like to chat about or work on together?"""
        
        elif any(phrase in user_lower for phrase in ["what do you think", "your opinion", "thoughts on"]):
            return f"That's a thoughtful question! I'd love to share my thoughts on '{user_input}'. While I don't have personal experiences like humans do, I can offer insights based on patterns and knowledge. What specific aspect interests you most? I find these kinds of discussions really engaging!"
        
        elif any(word in user_lower for word in ["what can you do", "help", "capabilities"]):
            return """I'm excited to show you what we can do together!

**Conversation & Knowledge**
- Chat about any topic that interests you
- Answer questions and explain complex concepts
- Discuss ideas and share insights

**Productivity & Automation**
- Research topics thoroughly for you
- Manage and organize your emails
- Create documents and reports
- Monitor your system health
- Automate repetitive workflows

**Smart Assistance**
- Learn your preferences over time
- Provide proactive suggestions
- Help solve problems creatively

I'm designed to be both helpful and engaging. What sounds most interesting to you right now?"""
        
        else:
            return f"That's interesting! You mentioned '{user_input}' - I'd love to explore that with you. I'm great at having conversations, researching topics, helping with tasks, and learning about what matters to you. What aspect would you like to dive into first?"
    
    def _get_generic_response(self, intention: Dict) -> str:
        """Generic response for unhandled intentions"""
        action = intention.get("primary_action", "unknown")
        return f"I understand you want help with {action}. Let me work on that for you."
    
    def _get_fallback_response(self, user_input: str) -> str:
        """Final fallback response"""
        return "I'm experiencing some technical difficulties, but I'm still here to help! Try asking me about system status, research topics, or general questions."
    
    def _get_time_of_day(self) -> str:
        """Get time of day for greetings"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

class OutputFormatter:
    """Dedicated class for formatting different types of output"""
    
    @staticmethod
    def format_research_output(result: Dict) -> str:
        """Format research results for user"""
        if "error" in result:
            return f"Research failed: {result['error']}"

        topic = result.get("topic", "Unknown topic")
        sources_count = len(result.get("sources", []))
        summary = result.get("summary", "No summary available")

        output = f"Research completed on '{topic}'\n"
        output += f"Sources consulted: {sources_count}\n"
        output += f"Summary: {summary[:300]}..."

        if result.get("key_points"):
            output += "\n\nKey points:\n"
            for i, point in enumerate(result["key_points"][:3], 1):
                output += f"{i}. {point}\n"

        return output

    @staticmethod
    def format_email_output(result) -> str:
        """Format email results for user"""
        if isinstance(result, dict) and "error" in result:
            return f"Email check failed: {result['error']}"

        if isinstance(result, list) and result:
            # Direct email list
            email_count = len(result)
            unread_count = sum(1 for e in result if e.get("is_unread", False))
            return f"Found {email_count} recent emails, {unread_count} unread."
        elif isinstance(result, dict) and "total_emails" in result:
            # Email summary
            total = result["total_emails"]
            unread = result["unread_count"]
            high_priority = len(result.get("high_priority", []))

            output = "Email Summary:\n"
            output += f"Total emails: {total}, Unread: {unread}\n"
            if high_priority:
                output += f"High priority emails: {high_priority}\n"

            if result.get("needs_attention"):
                output += "\nEmails needing attention:\n"
                for email in result["needs_attention"][:3]:
                    output += f"- {email['subject']} (from {email['sender']})\n"

            return output
        else:
            return "Email check completed."

    @staticmethod
    def format_calendar_output(result) -> str:
        """Format calendar results for user"""
        if isinstance(result, dict) and "error" in result:
            return f"Calendar check failed: {result['error']}"

        if isinstance(result, list) and result:
            # Direct calendar events list
            event_count = len(result)
            today_events = []
            upcoming_events = []

            from datetime import datetime, date
            today = date.today()

            for event in result:
                if isinstance(event, dict) and event.get("start_time"):
                    try:
                        # Parse the start time to check if it's today
                        start_time = event["start_time"]
                        if isinstance(start_time, str):
                            # Handle different datetime formats
                            if "T" in start_time:
                                event_date = datetime.fromisoformat(start_time.replace("Z", "+00:00")).date()
                            else:
                                event_date = datetime.fromisoformat(start_time).date()
                        else:
                            event_date = start_time.date() if hasattr(start_time, 'date') else today

                        if event_date == today:
                            today_events.append(event)
                        else:
                            upcoming_events.append(event)
                    except Exception:
                        upcoming_events.append(event)

            output = "Calendar Summary\n"
            output += f"Total events: {event_count}\n"
            output += f"Today: {len(today_events)} events\n"
            output += f"Upcoming: {len(upcoming_events)} events\n"

            if today_events:
                output += "\nToday's Events:\n"
                for event in today_events[:3]:
                    title = event.get("title", "No Title")
                    start_time = event.get("start_time", "")
                    if "T" in start_time:
                        try:
                            time_part = datetime.fromisoformat(start_time.replace("Z", "+00:00")).strftime("%I:%M %p")
                            output += f"- {time_part} - {title}\n"
                        except Exception:
                            output += f"- {title}\n"
                    else:
                        output += f"- {title}\n"

            if upcoming_events:
                output += "\nUpcoming Events:\n"
                for event in upcoming_events[:3]:
                    title = event.get("title", "No Title")
                    start_time = event.get("start_time", "")
                    if "T" in start_time:
                        try:
                            event_datetime = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                            date_str = event_datetime.strftime("%m/%d")
                            time_str = event_datetime.strftime("%I:%M %p")
                            output += f"- {date_str} {time_str} - {title}\n"
                        except Exception:
                            output += f"- {title}\n"
                    else:
                        output += f"- {title}\n"

            return output

        elif isinstance(result, dict) and "total_events" in result:
            # Calendar summary from get_calendar_summary
            total = result["total_events"]
            today_count = result.get("today_events", 0)
            tomorrow_count = result.get("tomorrow_events", 0)

            output = "Calendar Summary\n"
            output += f"Total events: {total}\n"
            output += f"Today: {today_count} events\n"
            output += f"Tomorrow: {tomorrow_count} events\n"

            if result.get("next_event"):
                next_event = result["next_event"]
                output += "\nNext Event:\n"
                output += f"- {next_event.get('title', 'No Title')}\n"

            if result.get("today_schedule"):
                output += "\nToday's Schedule:\n"
                for event in result["today_schedule"][:3]:
                    title = event.get("title", "No Title")
                    start_time = event.get("start_time", "")
                    if "T" in start_time:
                        try:
                            time_part = datetime.fromisoformat(start_time.replace("Z", "+00:00")).strftime("%I:%M %p")
                            output += f"- {time_part} - {title}\n"
                        except Exception:
                            output += f"- {title}\n"
                    else:
                        output += f"- {title}\n"

            return output
        else:
            return "Calendar check completed."

class SessionManager:
    """Manages session state with proper cleanup"""
    
    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self.session_history = deque(maxlen=self.config.MAX_HISTORY_SIZE)
        self.session_start = datetime.now()
        self.last_cleanup = datetime.now()
    
    def add_interaction(self, interaction: Dict) -> None:
        """Add interaction with automatic cleanup"""
        interaction["timestamp"] = datetime.now().isoformat()
        self.session_history.append(interaction)
        
        # Periodic cleanup
        if (datetime.now() - self.last_cleanup).total_seconds() > self.config.CLEANUP_INTERVAL_MINUTES * 60:
            self._cleanup_old_interactions()
    
    def _cleanup_old_interactions(self) -> None:
        """Remove old interactions"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.MAX_SESSION_AGE_HOURS)
        
        # Convert deque to list, filter, and create new deque
        filtered_interactions = [
            interaction for interaction in self.session_history
            if datetime.fromisoformat(interaction.get("timestamp", "")) > cutoff_time
        ]
        
        self.session_history = deque(filtered_interactions, maxlen=self.config.MAX_HISTORY_SIZE)
        self.last_cleanup = datetime.now()
        
        logger.info(f"Cleaned up session history. Current size: {len(self.session_history)}")
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions"""
        return list(self.session_history)[-limit:]
    
    def clear_session(self) -> None:
        """Clear current session"""
        self.session_history.clear()
        self.session_start = datetime.now()

class AgenticOrchestrator:
    """Refactored orchestrator with full functionality restored"""
    
    def __init__(self):
        # Core components
        self.llm_client = OllamaClient()
        self.context_manager = CrossAppContextManager()
        self.automation_engine = AutomationEngine()
        self.prompt_templates = PromptTemplates()
        
        # Specialized components
        self.llm_analyzer = LLMAnalyzer()
        self.intention_analyzer = IntentionAnalyzer(self.llm_client)
        self.response_generator = ResponseGenerator(self.llm_client, self.prompt_templates)
        self.session_manager = SessionManager()
        self.output_formatter = OutputFormatter()
        
        # Application controllers
        self.app_controllers = self._initialize_controllers()
        
        # Intelligence components
        self.vector_store = None  # Disabled for now
        self.context_analyzer = ContextAnalyzer(self.vector_store)
        self.learning_engine = LearningEngine(self.vector_store)
        self.proactive_assistant = ProactiveAssistant(
            self.context_analyzer, self.learning_engine, self.vector_store
        )
        
        # Communication components
        self.message_analyzer = MessageAnalyzer(self.llm_client)
        self.smart_responder = SmartResponder(self.llm_client, self.learning_engine)
        self.conversation_manager = ConversationManager(self.llm_client)
        
        # Session state for compatibility
        self.session_history = []  # Maintained for backward compatibility
        self.current_task = None
        
        logger.info("Agentic Orchestrator initialized successfully")
    
    def _initialize_controllers(self) -> Dict[str, Any]:
        """Initialize application controllers with error handling"""
        controllers = {}
        controller_classes = {
            "browser": BrowserController,
            "email": EmailController,
            "document": DocumentController,
            "calendar": CalendarController,
            "file": FileController,
            "slack": SlackController,
            "teams": TeamsController,  # Restored Teams controller
            "vision": VisionController,  # New RTX 3050 optimized vision controller
        }
        
        # Initialize system automation components
        try:
            self.system_controller = SystemController()
            self.automation_workflows = None  # Will be initialized when needed with LLM client
            logger.info("System automation components ready")
        except Exception as e:
            logger.error(f"Failed to initialize system automation: {e}")
        
        for name, controller_class in controller_classes.items():
            try:
                controllers[name] = controller_class()
                logger.info(f"Initialized {name} controller")
            except Exception as e:
                logger.error(f"Failed to initialize {name} controller: {e}")
                # Continue with other controllers
        
        return controllers
    
    async def initialize(self) -> None:
        """Initialize async components with proper error handling"""
        try:
            await self.automation_engine.start()
            logger.info("Automation engine started successfully")
        except Exception as e:
            logger.error(f"Failed to start automation engine: {e}")
        
        logger.info("Orchestrator fully initialized")
    
    async def process_user_intention(self, user_input: str, auto_confirm: bool = True) -> Dict[str, Any]:
        """Main processing pipeline with web UI support for confirmations"""
        start_time = datetime.now()
        
        try:
            # Input validation
            if not user_input or not user_input.strip():
                return self._create_error_response("Empty input provided", user_input)
            
            # First, analyze user input with LLM
            logger.info("Analyzing user input with LLM...")
            llm_analysis_result = self.llm_analyzer.analyze_user_input(user_input)
            # Convert AnalysisResult to dictionary for backward compatibility
            llm_analysis = llm_analysis_result.to_dict() if hasattr(llm_analysis_result, 'to_dict') else llm_analysis_result
            
            # SPECIAL HANDLING: Email requests should always be processed directly for web interface
            # since they need to actually send emails, not generate confirmation dialogs
            if llm_analysis.get('action_type') == 'email' and llm_analysis.get('parameters', {}).get('to'):
                logger.info("Email request detected with recipient - processing directly with auto_send=True")
                # Force auto_confirm for email requests with valid recipients
                auto_confirm = True
                    
            # For web UI, we return confirmation request instead of blocking (except for email requests)
            if llm_analysis['confidence'] < 0.8 and not auto_confirm:
                return {
                    "status": "needs_confirmation",
                    "message": "I want to confirm my understanding before proceeding",
                    "llm_analysis": llm_analysis,
                    "analysis_display": self.llm_analyzer.format_analysis_for_user(llm_analysis_result),
                    "confirmation_question": "Does this look correct?",
                    "options": ["Yes, proceed", "No, let me clarify"],
                    "final_output": f"I analyzed your request as: {llm_analysis.get('intent', 'Unknown intent')}. Should I proceed with this understanding?"
                }
            
            # Get context with enhanced analysis
            logger.info("Getting system context...")
            context = await self._get_system_context()
            context["llm_analysis"] = llm_analysis
            
            # Enhanced context analysis
            logger.info("Performing advanced context analysis...")
            context_analysis = await self.context_analyzer.analyze_user_context(
                user_input, context, self.session_history
            )
            context["context_analysis"] = context_analysis
            
            # Use LLM analysis to guide intention analysis
            logger.info("Analyzing user intention...")
            intention = await self._analyze_intention_with_llm_guidance(user_input, context, llm_analysis)
            
            # Process based on interaction type
            if intention["interaction_type"] == "direct_response":
                result = await self._handle_direct_response(user_input, context, intention)
            elif intention["interaction_type"] == "task_execution":
                result = await self._handle_task_execution(user_input, context, intention)
            else:
                result = await self._handle_clarification(user_input, intention)
            
            # Store interaction in vector memory
            if self.vector_store:
                await self.vector_store.store_interaction(
                    user_input, result.get("final_output", ""), context, intention
                )

            # Learn from this interaction
            if self.learning_engine and hasattr(self.learning_engine, 'learn_from_interaction'):
                try:
                    await self.learning_engine.learn_from_interaction(
                        user_input, result.get("final_output", ""), context
                    )
                except Exception as e:
                    logger.warning("Learning engine error: %s", e)
            
            # Store interaction
            self._store_interaction(user_input, intention, result, start_time)
            
            # Generate proactive suggestions
            proactive_suggestions = await self.proactive_assistant.generate_proactive_suggestions(
                context, self.session_history
            )
            
            # Generate final response
            return self._create_success_response(
                intention, result, context, context_analysis, proactive_suggestions, start_time
            )
            
        except Exception as e:
            logger.error(f"Error processing user intention: {e}")
            return self._create_error_response(str(e), user_input)
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """Get system context with error handling"""
        try:
            context = await self.context_manager.get_full_context()
            # Add session history to context
            context["session_history"] = self.session_history
            return context
        except Exception as e:
            logger.error(f"Failed to get system context: {e}")
            return {
                "current_time": datetime.now().isoformat(),
                "system_status": {},
                "active_apps": [],
                "session_history": self.session_history,
                "error": str(e)
            }
    
    async def _analyze_intention_with_llm_guidance(self, user_input: str, context: Dict, llm_analysis: Dict) -> Dict[str, Any]:
        """Analyze intention using LLM analysis as guidance"""
        
        # Convert LLM analysis to intention format
        action_type = llm_analysis.get('action_type', 'automation_workflow')
        parameters = llm_analysis.get('parameters', {})
        confidence = llm_analysis.get('confidence', 0.5)
        
        # Map LLM action types to orchestrator actions
        action_mapping = {
            'email': 'send_email',
            'system_control': 'system_control',
            'file_operation': 'file_management',
            'automation_workflow': 'automation'
        }
        
        primary_action = action_mapping.get(action_type, 'chat')
        
        # Determine interaction type based on action
        interaction_type = "task_execution" if primary_action != 'chat' else "direct_response"
        
        # Build intention structure
        intention = {
            "interaction_type": interaction_type,
            "primary_action": primary_action,
            "confidence": confidence,
            "applications_needed": self._get_apps_for_action(primary_action),
            "parameters": parameters,
            "estimated_steps": 2 if interaction_type == "task_execution" else 0,
            "user_intent_summary": llm_analysis.get('intent', 'User request'),
            "analysis_method": "llm_guided",
            "llm_analysis": llm_analysis
        }
        
        return intention
    
    def _get_apps_for_action(self, action: str) -> List[str]:
        """Get required applications for an action"""
        app_mapping = {
            'send_email': ['email'],
            'system_control': ['system'],
            'file_management': ['file'],
            'automation': ['automation'],
            'chat': []
        }
        return app_mapping.get(action, [])
    
    async def _handle_direct_response(self, user_input: str, context: Dict, intention: Dict) -> Dict[str, Any]:
        """Handle direct response interactions with enhanced conversation management"""
        try:
            # Generate base response
            response = await self.response_generator.generate_response(user_input, context, intention)
            
            # Enhance with conversation context
            if self.conversation_manager:
                enhanced_response = self.conversation_manager.generate_contextual_response(
                    user_input, response
                )
                response = enhanced_response
            
            # Add to conversation history
            if self.conversation_manager:
                self.conversation_manager.add_conversation_turn(
                    user_input, response, intention.get("primary_action", "chat")
                )
            
            return {
                "final_output": response,
                "interaction_type": "direct_response",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Direct response handling failed: {e}")
            return {
                "final_output": "I'm experiencing technical difficulties. Please try again.",
                "interaction_type": "direct_response",
                "success": False,
                "error": str(e)
            }
    
    async def _handle_task_execution(self, user_input: str, context: Dict, intention: Dict) -> Dict[str, Any]:
        """Handle task execution interactions"""
        try:
            # Add user_input to context for plan creation
            enhanced_context = context.copy()
            enhanced_context["user_input"] = user_input
            
            # Use public methods for execution
            plan = await self.create_execution_plan(intention, enhanced_context)
            result = await self.execute_plan(plan)
            
            # Add to conversation history
            if self.conversation_manager:
                self.conversation_manager.add_conversation_turn(
                    user_input, result.get("final_output", "Task completed"), "task_execution"
                )
            
            result["interaction_type"] = "task_execution"
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "final_output": f"Task execution failed: {str(e)}",
                "interaction_type": "task_execution",
                "success": False,
                "error": str(e)
            }
    
    async def _handle_clarification(self, user_input: str, intention: Dict) -> Dict[str, Any]:
        """Handle clarification requests"""
        clarification = self._get_clarification_response(user_input, intention)
        
        if self.conversation_manager:
            enhanced_clarification = self.conversation_manager.generate_contextual_response(
                user_input, clarification
            )
            clarification = enhanced_clarification
        
        return {
            "final_output": clarification,
            "interaction_type": "clarification_needed",
            "success": True
        }
    
    def _get_clarification_response(self, user_input: str, intention: Dict = None) -> str:
        """Generate clarification request with full functionality"""
        return f"""I'm not quite sure what you'd like me to do with: "{user_input}"

I can help you with:
• Research topics - "Research machine learning trends"
• Email management - "Check my emails"
• System information - "What's my system status?"
• Document creation - "Create a report about [topic]"
• Calendar management - "Check my schedule today"
• File management - "List files in Documents folder"
• Communication - "Send a Slack message" or "Check Teams"
• Web automation - "Find information about [topic]"
• Task automation - "Automate my daily workflow"

Could you be more specific about what you need help with?"""
    
    # ===== RESTORED PUBLIC API METHODS =====
    
    async def create_execution_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create detailed execution plan - RESTORED PUBLIC METHOD"""
        primary_action = intention["primary_action"]

        if primary_action == "research":
            return await self._create_research_plan(intention, context)
        elif primary_action == "email_management":
            return await self._create_email_plan(intention, context)
        elif primary_action == "system_info":
            return await self._create_system_info_plan(intention, context)
        elif primary_action == "automation":
            return await self._create_automation_plan(intention, context)
        elif primary_action == "document_creation":
            return await self._create_document_plan(intention, context)
        elif primary_action == "calendar_management":
            return await self._create_calendar_plan(intention, context)
        elif primary_action == "file_management":
            return await self._create_file_plan(intention, context)
        elif primary_action == "communication":
            return await self._create_communication_plan(intention, context)
        elif primary_action == "system_automation":
            return await self._create_system_automation_plan(intention, context)
        elif primary_action == "send_email":
            return await self._create_send_email_plan(intention, context)
        elif primary_action == "system_control":
            return await self._create_system_control_plan(intention, context)
        elif primary_action == "open_application":
            return await self._create_open_application_plan(intention, context)
        else:
            # Generic plan
            return [
                {
                    "step": 1,
                    "controller": intention.get("applications_needed", ["browser"])[0] if intention.get("applications_needed") else "system",
                    "action": "generic_task",
                    "parameters": intention["parameters"],
                }
            ]
    
    async def execute_plan(self, plan: List[Dict]) -> Dict:
        """Execute the planned steps - RESTORED PUBLIC METHOD"""
        results = {
            "steps_completed": 0,
            "step_results": [],
            "final_output": "",
            "errors": [],
            "success": True
        }

        for step in plan:
            try:
                logger.info(f"Executing step {step.get('step', 'unknown')}: {step.get('action', 'unknown')}")

                controller_name = step["controller"]
                action = step["action"]
                parameters = step.get("parameters", {})

                if controller_name == "system":
                    result = await self._execute_system_action(action, parameters)
                elif controller_name == "automation":
                    result = await self._execute_automation_action(action, parameters)
                elif controller_name == "communication":
                    result = await self._execute_communication_action(action, parameters)
                elif controller_name == "system_automation":
                    result = await self._execute_system_automation_action(action, parameters)
                elif controller_name == "email_automation":
                    result = await self._execute_email_automation_action(action, parameters)
                elif controller_name == "system_control":
                    result = await self._execute_system_control_action(action, parameters)
                elif controller_name == "open_application" or controller_name == "calculator":
                    result = await self._execute_open_application_action(action, parameters)
                elif controller_name in self.app_controllers:
                    controller = self.app_controllers[controller_name]
                    result = await self._execute_controller_action(
                        controller, action, parameters, results
                    )
                else:
                    result = {"error": f"Unknown controller: {controller_name}"}

                results["step_results"].append(
                    {"step": step.get("step", 0), "action": action, "result": result}
                )

                results["steps_completed"] += 1

                if isinstance(result, dict) and "error" in result:
                    results["errors"].append(f"Step {step.get('step', 0)}: {result['error']}")
                    results["success"] = False

            except Exception as e:
                error_msg = f"Step {step.get('step', 0)} failed: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["success"] = False

        # Generate final output with enhanced formatting
        results["final_output"] = await self._generate_final_output(results, plan)

        return results
    
    async def get_intelligence_insights(self) -> Dict:
        """Get insights from the intelligence components - RESTORED METHOD"""
        try:
            insights = {
                "learning_stats": self.learning_engine.get_learning_stats() if hasattr(self.learning_engine, 'get_learning_stats') else {},
                "suggestion_effectiveness": self.proactive_assistant.get_suggestion_effectiveness() if hasattr(self.proactive_assistant, 'get_suggestion_effectiveness') else {},
                "context_patterns": await self._get_context_patterns(),
                "user_model_summary": await self._get_user_model_summary(),
                "session_stats": self.get_session_stats()
            }
            return insights
        except Exception as e:
            logger.error("Failed to get intelligence insights: %s", e)
            return {"error": str(e)}
    
    async def get_communication_insights(self) -> Dict:
        """Get insights from communication platforms - RESTORED METHOD"""
        try:
            insights = {
                "slack": {},
                "teams": {},
                "message_analysis": {},
                "communication_health": {}
            }

            # Get Slack insights
            if "slack" in self.app_controllers:
                slack_controller = self.app_controllers["slack"]
                if hasattr(slack_controller, 'authenticated') and slack_controller.authenticated:
                    if hasattr(slack_controller, 'analyze_workspace_activity'):
                        slack_analysis = await slack_controller.analyze_workspace_activity()
                        insights["slack"] = slack_analysis

            # Get Teams insights
            if "teams" in self.app_controllers:
                teams_controller = self.app_controllers["teams"]
                if hasattr(teams_controller, 'authenticated') and teams_controller.authenticated:
                    if hasattr(teams_controller, 'analyze_teams_activity'):
                        teams_analysis = await teams_controller.analyze_teams_activity()
                        insights["teams"] = teams_analysis

            return insights

        except Exception as e:
            logger.error("Failed to get communication insights: %s", e)
            return {"error": str(e)}
    
    # ===== RESTORED HELPER METHODS =====
    
    async def _get_context_patterns(self) -> Dict:
        """Get context usage patterns - RESTORED METHOD"""
        if not self.session_history:
            return {}

        patterns = {
            "most_common_actions": [],
            "peak_usage_hours": [],
            "average_session_length": 0,
            "context_relevance_trends": {}
        }

        # Analyze action patterns
        action_counts = {}
        total_duration = 0
        
        for interaction in self.session_history:
            action = interaction.get("intention", {}).get("primary_action")
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            duration = interaction.get("duration", 0)
            total_duration += duration

        patterns["most_common_actions"] = sorted(
            action_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        if self.session_history:
            patterns["average_session_length"] = total_duration / len(self.session_history)

        return patterns
    
    async def _get_user_model_summary(self) -> Dict:
        """Get summary of learned user model - RESTORED METHOD"""
        try:
            user_model = getattr(self.learning_engine, 'user_model', {})
            learning_metrics = getattr(self.learning_engine, 'learning_metrics', {})

            return {
                "preferences_learned": len(user_model.get("preferences", {})),
                "skills_assessed": len(user_model.get("skills", {})),
                "patterns_identified": len(user_model.get("patterns", {})),
                "total_interactions": learning_metrics.get("interactions_processed", len(self.session_history))
            }
        except Exception as e:
            logger.error(f"Failed to get user model summary: {e}")
            return {"error": str(e)}
    
    # ===== COMMUNICATION METHODS RESTORED =====
    
    async def _analyze_communication_request(self, parameters: Dict) -> Dict:
        """Analyze communication request to determine platform and action - RESTORED METHOD"""
        request = parameters.get("request", "").lower()

        analysis = {
            "platform": "unknown",
            "action": "unknown",
            "confidence": 0.0,
            "parameters": {}
        }

        # Determine platform
        if "slack" in request:
            analysis["platform"] = "slack"
            analysis["confidence"] = 0.8
        elif "teams" in request:
            analysis["platform"] = "teams"
            analysis["confidence"] = 0.8
        elif "message" in request or "chat" in request:
            # Default to Slack if no specific platform mentioned
            analysis["platform"] = "slack"
            analysis["confidence"] = 0.6

        # Determine action
        if "send" in request or "message" in request:
            analysis["action"] = "send_message"
        elif "check" in request or "get" in request or "recent" in request:
            analysis["action"] = "get_messages"
        elif "status" in request:
            analysis["action"] = "get_status"
        elif "channels" in request:
            analysis["action"] = "list_channels"
        else:
            analysis["action"] = "get_status"  # Default action

        return analysis

    async def _execute_communication_request(self, parameters: Dict) -> Dict:
        """Execute the actual communication request - RESTORED METHOD"""
        request = parameters.get("request", "")

        # First analyze the request
        analysis = await self._analyze_communication_request(parameters)
        platform = analysis["platform"]
        action = analysis["action"]

        try:
            if platform == "slack":
                return await self._execute_slack_action(action, request)
            elif platform == "teams":
                return await self._execute_teams_action(action, request)
            else:
                return {"error": f"Unsupported platform: {platform}"}
        except Exception as e:
            logger.error("Communication request execution failed: %s", e)
            return {"error": str(e)}

    async def _execute_slack_action(self, action: str, request: str) -> Dict:
        """Execute Slack-specific actions - RESTORED METHOD"""
        if "slack" not in self.app_controllers:
            return {"error": "Slack controller not available"}
            
        slack_controller = self.app_controllers["slack"]

        if action == "get_status":
            # Check authentication and get basic status
            auth_status = slack_controller.get_authentication_status()
            if not auth_status["authenticated"]:
                auth_result = await slack_controller.authenticate()
                if "error" in auth_result:
                    return {"error": f"Slack authentication failed: {auth_result['error']}"}

            # Get workspace info and recent activity
            channels = await slack_controller.get_channels()
            user_status = await slack_controller.get_user_status()

            return {
                "platform": "slack",
                "authenticated": True,
                "workspace": auth_status.get("workspace", ""),
                "channels_count": len(channels) if isinstance(channels, list) else 0,
                "user_status": user_status,
                "message": "Slack status retrieved successfully"
            }

        elif action == "get_messages":
            # Get recent messages from general channel or first available channel
            channels = await slack_controller.get_channels()
            if isinstance(channels, list) and channels:
                # Try to find general channel, otherwise use first channel
                target_channel = None
                for channel in channels:
                    if channel.get("name") == "general":
                        target_channel = channel["id"]
                        break

                if not target_channel:
                    target_channel = channels[0]["id"]

                messages = await slack_controller.get_recent_messages(target_channel, limit=10)
                return {
                    "platform": "slack",
                    "channel": target_channel,
                    "messages": messages,
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "message": "Retrieved recent messages from Slack"
                }
            else:
                return {"error": "No Slack channels available"}

        elif action == "list_channels":
            channels = await slack_controller.get_channels()
            return {
                "platform": "slack",
                "channels": channels,
                "channel_count": len(channels) if isinstance(channels, list) else 0,
                "message": "Slack channels retrieved successfully"
            }

        elif action == "send_message":
            return {"error": "Message sending requires specific channel and message content"}

        else:
            return {"error": f"Unknown Slack action: {action}"}

    async def _execute_teams_action(self, action: str, request: str) -> Dict:
        """Execute Teams-specific actions - RESTORED METHOD"""
        if "teams" not in self.app_controllers:
            return {"error": "Teams controller not available"}
            
        teams_controller = self.app_controllers["teams"]

        if action == "get_status":
            # Check authentication and get basic status
            auth_status = teams_controller.get_authentication_status()
            if not auth_status["authenticated"]:
                auth_result = await teams_controller.authenticate()
                if "error" in auth_result:
                    return {"error": f"Teams authentication failed: {auth_result['error']}"}

            # Get teams and presence info
            teams = await teams_controller.get_teams()
            presence = await teams_controller.get_user_presence()

            return {
                "platform": "teams",
                "authenticated": True,
                "user": auth_status.get("user", ""),
                "teams_count": len(teams) if isinstance(teams, list) else 0,
                "presence": presence,
                "message": "Teams status retrieved successfully"
            }

        elif action == "get_messages":
            # Get recent chats
            chats = await teams_controller.get_chats(limit=10)
            return {
                "platform": "teams",
                "chats": chats,
                "chat_count": len(chats) if isinstance(chats, list) else 0,
                "message": "Retrieved recent Teams chats"
            }

        elif action == "list_channels":
            teams = await teams_controller.get_teams()
            all_channels = []

            if isinstance(teams, list):
                for team in teams[:3]:  # Limit to first 3 teams
                    if "error" not in team:
                        channels = await teams_controller.get_channels(team["id"])
                        if isinstance(channels, list):
                            all_channels.extend(channels)

            return {
                "platform": "teams",
                "teams": teams,
                "channels": all_channels,
                "message": f"Retrieved {len(all_channels)} channels from Teams"
            }

        else:
            return {"error": f"Unknown Teams action: {action}"}
    
    # ===== PLAN CREATION METHODS =====
    
    async def _create_research_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for research tasks"""
        topic = intention["parameters"].get("topic", intention["parameters"].get("user_input", "unknown topic"))
        scope = intention["parameters"].get("scope", "comprehensive")

        return [
            {
                "step": 1,
                "controller": "browser",
                "action": "research_topic",
                "parameters": {"topic": topic, "depth": scope},
                "expected_output": "research_data",
            },
            {
                "step": 2,
                "controller": "system",
                "action": "synthesize_research",
                "parameters": {"topic": topic, "format": "summary"},
                "expected_output": "formatted_summary",
            },
        ]

    async def _create_email_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for email management"""
        return [
            {
                "step": 1,
                "controller": "email",
                "action": "get_recent_emails",
                "parameters": {"max_results": 15, "days_back": 3},
                "expected_output": "email_list",
            },
            {
                "step": 2,
                "controller": "email",
                "action": "analyze_emails",
                "parameters": {},
                "expected_output": "email_summary",
            },
        ]

    async def _create_system_info_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for system information requests"""
        return [
            {
                "step": 1,
                "controller": "system",
                "action": "get_system_status",
                "parameters": {},
                "expected_output": "system_info",
            }
        ]

    async def _create_automation_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for automation requests"""
        request = intention["parameters"].get("request", intention["parameters"].get("user_input", ""))

        return [
            {
                "step": 1,
                "controller": "automation",
                "action": "create_workflow",
                "parameters": {"request": request},
                "expected_output": "workflow_created",
            }
        ]

    async def _create_document_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for document creation requests"""
        content = intention["parameters"].get("content", intention["parameters"].get("user_input", ""))

        return [
            {
                "step": 1,
                "controller": "document",
                "action": "create_document",
                "parameters": {"content": content, "doc_type": "writer"},
                "expected_output": "document_created",
            }
        ]

    async def _create_calendar_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for calendar management requests"""
        request = intention["parameters"].get("request", intention["parameters"].get("user_input", ""))

        return [
            {
                "step": 1,
                "controller": "calendar",
                "action": "get_upcoming_events",
                "parameters": {"days_ahead": 7},
                "expected_output": "calendar_events",
            },
            {
                "step": 2,
                "controller": "calendar",
                "action": "get_calendar_summary",
                "parameters": {"days_ahead": 7},
                "expected_output": "calendar_summary",
            }
        ]

    async def _create_file_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for file management requests"""
        operation = intention["parameters"].get("operation", intention["parameters"].get("user_input", ""))

        return [
            {
                "step": 1,
                "controller": "file",
                "action": "list_directory",
                "parameters": {"path": ".", "recursive": False},
                "expected_output": "file_listing",
            }
        ]

    async def _create_communication_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for communication requests"""
        request = intention["parameters"].get("request", intention["parameters"].get("user_input", ""))

        return [
            {
                "step": 1,
                "controller": "communication",
                "action": "analyze_communication_request",
                "parameters": {"request": request},
                "expected_output": "communication_analysis",
            },
            {
                "step": 2,
                "controller": "communication",
                "action": "execute_communication_action",
                "parameters": {"request": request},
                "expected_output": "communication_result",
            }
        ]
    
    async def _create_system_automation_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for system automation requests (recycle bin, temp files, etc.)"""
        user_input = intention["parameters"].get("user_input", "")
        
        return [
            {
                "step": 1,
                "controller": "system_automation",
                "action": "execute_system_automation",
                "parameters": {
                    "user_input": user_input,
                    "intent_type": intention["parameters"].get("intent_type", "system_automation")
                },
                "expected_output": "automation_result",
            }
        ]
    
    async def _create_send_email_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for sending emails using LLM-generated content"""
        llm_analysis = intention.get("llm_analysis", {})
        parameters = llm_analysis.get("parameters", {})
        
        # Include the original user input for proper processing
        user_input = context.get("user_input", "")
        if not user_input and parameters.get("original_request"):
            user_input = parameters["original_request"]
        
        return [
            {
                "step": 1,
                "controller": "email_automation",
                "action": "send_email_with_llm_content",
                "parameters": {
                    "to": parameters.get("to"),
                    "subject": parameters.get("subject"),
                    "body": parameters.get("body"),
                    "original_request": parameters.get("original_request", user_input),
                    "user_input": user_input,  # Add user_input for enhanced automation
                    "llm_analysis": llm_analysis
                },
                "expected_output": "email_result",
            }
        ]
    
    async def _create_system_control_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for system control requests using LLM analysis"""
        llm_analysis = intention.get("llm_analysis", {})
        parameters = llm_analysis.get("parameters", {})
        
        return [
            {
                "step": 1,
                "controller": "system_control",
                "action": "execute_system_control_with_llm",
                "parameters": {
                    "operation": parameters.get("operation", "general"),
                    "volume": parameters.get("volume"),
                    "application": parameters.get("application"),
                    "llm_analysis": llm_analysis
                },
                "expected_output": "control_result",
            }
        ]
    
    async def _create_open_application_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Create plan for opening applications using LLM analysis"""
        llm_analysis = intention.get("llm_analysis", {})
        parameters = llm_analysis.get("parameters", {}) if llm_analysis else intention.get("parameters", {})
        
        return [
            {
                "step": 1,
                "controller": "open_application",
                "action": "execute_open_application",
                "parameters": {
                    "application": parameters.get("application"),
                    "controller_name": parameters.get("controller_name"),
                    "user_input": parameters.get("user_input", ""),
                    "llm_analysis": llm_analysis
                },
                "expected_output": "application_result",
            }
        ]
    
    # ===== EXECUTION METHODS =====
    
    async def _execute_system_action(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """Execute system actions with enhanced functionality"""
        if action == "get_system_status":
            context = await self._get_system_context()
            return {
                "system_status": context.get("system_status", {}),
                "active_apps": context.get("active_apps", []),
                "timestamp": datetime.now().isoformat()
            }
        elif action == "synthesize_research":
            # Enhanced research synthesis
            topic = parameters.get("topic", "unknown")
            return {
                "message": f"Research synthesis completed for {topic}",
                "synthesis_method": "llm_enhanced" if self.llm_client else "basic"
            }
        else:
            return {"message": f"System action {action} completed", "parameters": parameters}
    
    async def _execute_automation_action(self, action: str, parameters: Dict) -> Dict:
        """Execute automation engine actions"""
        if action == "create_workflow":
            # Check if this is a file organization workflow
            request = parameters.get("request", "")
            
            # Analyze the request to see if it's file organization
            if any(keyword in request.lower() for keyword in ['organize files', 'file organization', 'organize my files']):
                try:
                    # Initialize automation workflows if not already done
                    if not hasattr(self, 'automation_workflows') or self.automation_workflows is None:
                        from workflows.automation_workflows import AutomationWorkflows
                        self.automation_workflows = AutomationWorkflows(llm_client=self.llm_client)
                        await self.automation_workflows.initialize()
                    
                    # Execute comprehensive file organization
                    organization_result = await self.automation_workflows.comprehensive_file_organization(
                        scope="common_directories",
                        organize_duplicates=True,
                        create_categories=True,
                        analyze_before_organize=True
                    )
                    
                    # Format response for user
                    final_report = organization_result.get("final_report", "File organization completed")
                    
                    return {
                        "message": final_report,
                        "details": organization_result,
                        "success": True,
                        "workflow_type": "file_organization"
                    }
                    
                except Exception as e:
                    logger.error(f"File organization workflow failed: {str(e)}")
                    return {
                        "message": f"File organization failed: {str(e)}",
                        "success": False,
                        "error": str(e)
                    }
            
            # Default workflow creation for other requests
            task_id = await self.automation_engine.submit_task(
                "workflow_creation", parameters, TaskPriority.MEDIUM
            )
            return {
                "task_id": task_id,
                "message": "Workflow creation task submitted",
                "status": "pending"
            }
        else:
            return {"error": f"Unknown automation action: {action}"}
    
    async def _execute_system_automation_action(self, action: str, parameters: Dict) -> Dict:
        """Execute system automation actions (recycle bin, temp files, etc.)"""
        try:
            # Initialize system controller and workflows if not already done
            if not hasattr(self, 'system_controller'):
                self.system_controller = SystemController()
            if not hasattr(self, 'automation_workflows') or self.automation_workflows is None:
                self.automation_workflows = AutomationWorkflows(llm_client=self.llm_client)
                await self.automation_workflows.initialize()
            
            user_input = parameters.get("user_input", "").lower()
            
            if any(keyword in user_input for keyword in ["clear recycle bin", "empty recycle bin", "clear trash"]):
                result = await self.system_controller.clear_recycle_bin(confirm=True)
                if result.get("status") == "success":
                    return {
                        "message": f"✓ {result['message']}",
                        "details": result,
                        "success": True
                    }
                else:
                    return {
                        "message": f"Failed to clear recycle bin: {result.get('error', 'Unknown error')}",
                        "success": False,
                        "error": result.get('error')
                    }
            
            elif any(keyword in user_input for keyword in ["clean temp", "delete temp", "cleanup system", "maintenance"]):
                if "maintenance" in user_input or "cleanup system" in user_input:
                    # Full system cleanup
                    result = await self.automation_workflows.cleanup_system_files(confirm=True)
                    total_freed = result.get("total_space_freed_mb", 0)
                    return {
                        "message": f"✓ System cleanup completed. Freed {total_freed:.1f}MB of disk space",
                        "details": result,
                        "success": True
                    }
                else:
                    # Just temp files
                    result = await self.system_controller.clean_temp_files(confirm=True)
                    if result.get("status") == "success":
                        total_freed = result.get("total_size_freed_mb", 0)
                        return {
                            "message": f"✓ Temporary files cleaned. Freed {total_freed:.1f}MB of disk space",
                            "details": result,
                            "success": True
                        }
                    else:
                        return {
                            "message": f"Failed to clean temp files: {result.get('error', 'Unknown error')}",
                            "success": False,
                            "error": result.get('error')
                        }
            
            else:
                # Generic system automation
                result = await self.automation_workflows.cleanup_system_files(confirm=True)
                total_freed = result.get("total_space_freed_mb", 0)
                return {
                    "message": f"✓ System automation completed. Freed {total_freed:.1f}MB of disk space",
                    "details": result,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"System automation action failed: {e}")
            return {
                "message": f"System automation failed: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def _execute_email_automation_action(self, action: str, parameters: Dict) -> Dict:
        """Execute email automation actions with LLM support and direct Gmail sending"""
        try:
            # Initialize email controller and workflows if not already done
            if not hasattr(self, 'automation_workflows') or self.automation_workflows is None:
                from workflows.automation_workflows import AutomationWorkflows
                self.automation_workflows = AutomationWorkflows(llm_client=self.llm_client)
                await self.automation_workflows.initialize()
            
            # Handle enhanced email automation
            if action == "send_email_with_llm_content":
                # Use enhanced email automation for intelligent processing
                original_request = parameters.get("original_request", "")
                if not original_request:
                    # Build request from other parameters
                    to = parameters.get("to", "")
                    subject = parameters.get("subject", "")
                    body = parameters.get("body", "")
                    original_request = f"Send email to {to} with subject '{subject}': {body}"
                
                # CRITICAL: Always auto_send for web interface to bypass terminal confirmation
                context = {"auto_send": True, "llm_analysis": parameters.get("llm_analysis")}
                
                return await self.automation_workflows.process_intelligent_email_request(
                    original_request, 
                    context=context
                )
            
            # Legacy email handling - also use enhanced automation for better processing
            user_input = parameters.get("user_input", "")
            
            # Extract original request from LLM analysis if available
            llm_analysis = parameters.get("llm_analysis", {})
            if not user_input and llm_analysis.get("parameters", {}).get("original_request"):
                user_input = llm_analysis["parameters"]["original_request"]
            
            # Try enhanced email automation first
            if user_input and hasattr(self.automation_workflows, 'process_intelligent_email_request'):
                logger.info("Using enhanced email automation with auto_send=True (web interface)")
                
                # CRITICAL: Always auto_send=True for web interface to bypass terminal confirmation
                context = {
                    "auto_send": True, 
                    "original_parameters": parameters,
                    "web_interface": True,  # Flag to indicate this is from web interface
                    "user_name": "Assistant"  # Default user name for signature
                }
                
                return await self.automation_workflows.process_intelligent_email_request(
                    user_input,
                    context=context
                )
            
            # Fallback to basic parsing for legacy compatibility
            logger.info("Using legacy email parsing")
            email_info = self._parse_email_request(user_input)
            
            if not email_info.get("to"):
                return {
                    "message": "Please specify the recipient email address. Example: 'Send email to john@example.com'",
                    "final_output": "❌ Please specify the recipient email address. Example: 'Send email to john@example.com'",
                    "success": False,
                    "error": "Missing recipient"
                }
            
            # Send email directly through automation workflows
            result = await self.automation_workflows.send_email_on_behalf(
                to=email_info["to"],
                subject=email_info.get("subject", "Message from AI Assistant"),
                body=email_info.get("body", "This message was sent via AI Assistant."),
                cc=email_info.get("cc"),
                attachments=email_info.get("attachments")
            )
            
            if result.get("status") == "success":
                return {
                    "message": f"✅ Email sent successfully to {email_info['to']}",
                    "final_output": f"✅ Email sent successfully to {email_info['to']}\n\nSubject: {email_info.get('subject', 'Message from AI Assistant')}\nSent at: {result.get('timestamp', 'Unknown')}",
                    "details": result,
                    "success": True,
                    "email_sent": True
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                return {
                    "message": f"Failed to send email: {error_msg}",
                    "final_output": f"❌ Failed to send email to {email_info['to']}: {error_msg}",
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Email automation action failed: {e}")
            return {
                "message": f"Email automation failed: {str(e)}",
                "final_output": f"❌ Email automation failed: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def _execute_system_control_action(self, action: str, parameters: Dict) -> Dict:
        """Execute system control actions with LLM support"""
        try:
            # Initialize system controller if not already done
            if not hasattr(self, 'system_controller'):
                self.system_controller = SystemController()
            
            # Handle LLM-guided system control
            if action == "execute_system_control_with_llm":
                return await self.system_controller.execute_system_control_with_llm(
                    operation=parameters.get("operation"),
                    volume=parameters.get("volume"),
                    application=parameters.get("application"),
                    llm_analysis=parameters.get("llm_analysis")
                )
            
            # Legacy system control handling for backward compatibility
            user_input = parameters.get("user_input", "").lower()
            
            if any(keyword in user_input for keyword in ["shutdown", "turn off"]):
                result = await self.system_controller.shutdown_system(delay_minutes=0, confirm=True)
                return {
                    "message": f"✓ {result.get('message', 'System shutdown initiated')}",
                    "details": result,
                    "success": result.get("status") == "success"
                }
            
            elif any(keyword in user_input for keyword in ["restart", "reboot"]):
                result = await self.system_controller.restart_system(delay_minutes=0, confirm=True)
                return {
                    "message": f"✓ {result.get('message', 'System restart initiated')}",
                    "details": result,
                    "success": result.get("status") == "success"
                }
            
            elif "lock screen" in user_input or "lock computer" in user_input:
                result = await self.system_controller.lock_screen()
                return {
                    "message": f"✓ {result.get('message', 'Screen locked')}",
                    "details": result,
                    "success": result.get("status") == "success"
                }
            
            elif "set volume" in user_input or "volume" in user_input:
                # Extract volume level from user input
                import re
                volume_match = re.search(r'(\d+)', user_input)
                if volume_match:
                    volume_level = int(volume_match.group(1))
                    result = await self.system_controller.set_volume(volume_level)
                    return {
                        "message": f"✓ {result.get('message', f'Volume set to {volume_level}%')}",
                        "details": result,
                        "success": result.get("status") == "success"
                    }
                else:
                    return {
                        "message": "Please specify volume level (0-100). Example: 'Set volume to 75'",
                        "success": False,
                        "error": "Missing volume level"
                    }
            
            elif "system info" in user_input or "system status" in user_input:
                result = await self.system_controller.get_system_info()
                if "error" not in result:
                    cpu_usage = result['hardware']['cpu_percent']
                    memory_usage = result['hardware']['memory']['percent_used']
                    return {
                        "message": f"✓ System Info: CPU {cpu_usage}%, Memory {memory_usage}%, {result['processes']['total']} processes running",
                        "details": result,
                        "success": True
                    }
                else:
                    return {
                        "message": f"Failed to get system info: {result['error']}",
                        "success": False,
                        "error": result['error']
                    }
            
            else:
                return {
                    "message": "System control command not recognized. Available: shutdown, restart, lock screen, set volume, system info",
                    "success": False,
                    "error": "Unknown command"
                }
                
        except Exception as e:
            logger.error(f"System control action failed: {e}")
            return {
                "message": f"System control failed: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def _execute_open_application_action(self, action: str, parameters: Dict) -> Dict:
        """Execute application opening actions with LLM support"""
        try:
            # Initialize system controller if not already done
            if not hasattr(self, 'system_controller'):
                self.system_controller = SystemController()
            
            # Handle LLM-guided application opening
            if parameters.get("llm_analysis"):
                llm_analysis = parameters["llm_analysis"]
                app_name = parameters.get("application") or llm_analysis.get("parameters", {}).get("application")
            else:
                # Legacy handling
                user_input = parameters.get("user_input", "")
                app_name = self._extract_application_name(user_input)
            
            # Handle direct application names from controller mapping
            if not app_name and parameters.get("controller_name"):
                app_name = parameters["controller_name"]
            
            if not app_name:
                return {
                    "message": "Please specify which application to open. Example: 'Open notepad' or 'Launch calculator'",
                    "success": False,
                    "error": "Missing application name"
                }
            
            print(f"\n🚀 Opening application: {app_name}")
            
            result = await self.system_controller.open_application(app_name)
            
            if result.get("status") == "success":
                return {
                    "message": f"✅ {result['message']}",
                    "details": result,
                    "success": True
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                suggestions = result.get('suggestions', [])
                
                message = f"❌ Failed to open {app_name}: {error_msg}"
                if suggestions:
                    message += f"\n\n💡 Suggested applications: {', '.join(suggestions[:5])}"
                
                return {
                    "message": message,
                    "success": False,
                    "error": error_msg,
                    "suggestions": suggestions
                }
                
        except Exception as e:
            logger.error(f"Open application action failed: {e}")
            return {
                "message": f"Failed to open application: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _parse_email_request(self, user_input: str) -> Dict[str, str]:
        """Parse email request from user input"""
        import re
        
        email_info = {}
        
        # Extract email address (to)
        email_pattern = r'(?:to|send to|email to|mail to)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, user_input, re.IGNORECASE)
        if email_match:
            email_info["to"] = email_match.group(1)
        
        # Extract subject (explicit)
        subject_pattern = r'(?:subject|with subject)\s+["\']([^"\']+)["\']'
        subject_match = re.search(subject_pattern, user_input, re.IGNORECASE)
        if subject_match:
            email_info["subject"] = subject_match.group(1)
        else:
            # Try without quotes
            subject_pattern2 = r'(?:subject|with subject)\s+([^,\.]+)'
            subject_match2 = re.search(subject_pattern2, user_input, re.IGNORECASE)
            if subject_match2:
                email_info["subject"] = subject_match2.group(1).strip()
        
        # Extract body content after "telling" or similar phrases
        body_patterns = [
            r'telling (?:him|her|them) (?:that )?(.+)',
            r'saying (?:that )?(.+)',
            r'informing (?:him|her|them) (?:that )?(.+)',
            r'letting (?:him|her|them) know (?:that )?(.+)',
            r'with (?:the )?message (.+)',
            r'body (.+)',
            r'message (.+)'
        ]
        
        for pattern in body_patterns:
            body_match = re.search(pattern, user_input, re.IGNORECASE)
            if body_match:
                email_info["body"] = body_match.group(1).strip()
                break
        
        # Generate subject from body if not explicitly provided
        if "body" in email_info and "subject" not in email_info:
            body_text = email_info["body"]
            
            # Generate subject based on content
            if "meeting" in body_text.lower():
                if "can't" in body_text.lower() or "won't" in body_text.lower() or "unable" in body_text.lower():
                    email_info["subject"] = "Unable to Attend Meeting"
                else:
                    email_info["subject"] = "Regarding Meeting"
            elif "emergency" in body_text.lower():
                email_info["subject"] = "Important: Emergency Situation"
            elif "update" in body_text.lower():
                email_info["subject"] = "Update"
            else:
                # Use first few words as subject
                words = body_text.split()[:6]
                email_info["subject"] = " ".join(words)
                if len(body_text.split()) > 6:
                    email_info["subject"] += "..."
        
        # Clean up the body text
        if "body" in email_info:
            # Make it more professional
            body = email_info["body"]
            
            # Add greeting if not present
            if not any(greeting in body.lower() for greeting in ["hi", "hello", "dear"]):
                body = "Hi,\n\n" + body
            
            # Add closing if not present
            if not any(closing in body.lower() for closing in ["regards", "thanks", "sincerely", "best"]):
                body += "\n\nBest regards"
            
            email_info["body"] = body
        
        return email_info
    
    def _extract_application_name(self, user_input: str) -> str:
        """Extract application name from user input"""
        import re
        
        user_lower = user_input.lower()
        
        # Remove common trigger words
        trigger_words = ["open", "launch", "start", "run", "application", "app", "program"]
        
        # Create pattern to remove trigger words
        pattern = r'\b(?:' + '|'.join(trigger_words) + r')\b\s*'
        cleaned_input = re.sub(pattern, '', user_lower, flags=re.IGNORECASE).strip()
        
        # Handle specific patterns
        if "notepad" in cleaned_input:
            return "notepad"
        elif "calculator" in cleaned_input or "calc" in cleaned_input:
            return "calculator"
        elif "paint" in cleaned_input:
            return "paint"
        elif "chrome" in cleaned_input:
            return "chrome"
        elif "firefox" in cleaned_input:
            return "firefox"
        elif "edge" in cleaned_input:
            return "edge"
        elif "word" in cleaned_input:
            return "word"
        elif "excel" in cleaned_input:
            return "excel"
        elif "powerpoint" in cleaned_input:
            return "powerpoint"
        elif "outlook" in cleaned_input:
            return "outlook"
        elif "teams" in cleaned_input:
            return "teams"
        elif "discord" in cleaned_input:
            return "discord"
        elif "spotify" in cleaned_input:
            return "spotify"
        elif "steam" in cleaned_input:
            return "steam"
        elif "vlc" in cleaned_input:
            return "vlc"
        elif "vs code" in cleaned_input or "vscode" in cleaned_input or "visual studio code" in cleaned_input:
            return "vs code"
        elif "cmd" in cleaned_input or "command prompt" in cleaned_input:
            return "cmd"
        elif "powershell" in cleaned_input:
            return "powershell"
        elif "task manager" in cleaned_input:
            return "task manager"
        elif "control panel" in cleaned_input:
            return "control panel"
        elif "file explorer" in cleaned_input or "explorer" in cleaned_input:
            return "explorer"
        else:
            # Return the cleaned input as the app name
            return cleaned_input if cleaned_input else user_input.strip()
    
    async def _execute_communication_action(self, action: str, parameters: Dict) -> Dict:
        """Execute communication actions - RESTORED METHOD"""
        try:
            if action == "analyze_communication_request":
                return await self._analyze_communication_request(parameters)
            elif action == "execute_communication_action":
                return await self._execute_communication_request(parameters)
            else:
                return {"error": f"Unknown communication action: {action}"}
        except Exception as e:
            logger.error("Communication action failed: %s", e)
            return {"error": str(e)}
    
    async def _execute_controller_action(self, controller: Any, action: str, parameters: Dict, results: Dict) -> Dict[str, Any]:
        """Execute controller actions with enhanced error handling"""
        try:
            # Map actions to methods
            method_mapping = {
                "research_topic": "research_topic",
                "get_recent_emails": "get_recent_emails",
                "analyze_emails": "analyze_emails_for_summary",
                "get_upcoming_events": "get_upcoming_events",
                "get_calendar_summary": "get_calendar_summary",
                "list_directory": "list_directory",
                "create_document": "create_document"
            }
            
            method_name = method_mapping.get(action, action)
            
            # Special handling for analyze_emails
            if action == "analyze_emails":
                if 'emails' not in parameters:
                    # Get emails from previous step results
                    for step_result in results.get("step_results", []):
                        if step_result.get("action") == "get_recent_emails":
                            parameters['emails'] = step_result.get("result", [])
                            break
                    
                    # If still no emails, get them now
                    if 'emails' not in parameters:
                        emails = await controller.get_recent_emails()
                        parameters['emails'] = emails
            
            if hasattr(controller, method_name):
                method = getattr(controller, method_name)
                if asyncio.iscoroutinefunction(method):
                    return await method(**parameters)
                else:
                    return method(**parameters)
            elif hasattr(controller, 'execute_task'):
                # Generic task execution
                return await controller.execute_task(action, parameters)
            else:
                return {"error": f"Method {method_name} not found on {controller.__class__.__name__}"}
                
        except Exception as e:
            logger.error(f"Controller action {action} failed: {e}")
            return {"error": str(e)}
    
    # ===== OUTPUT FORMATTING METHODS RESTORED =====
    
    async def _generate_final_output(self, results: Dict, plan: List[Dict]) -> str:
        """Generate human-readable final output with enhanced formatting"""
        if results["errors"]:
            error_summary = "; ".join(results["errors"][:3])  # Limit error messages
            return f"Task completed with errors: {error_summary}"

        if not results["step_results"]:
            return "No steps were executed successfully."

        # Create summary based on the type of task
        final_result = results["step_results"][-1]["result"]

        # Use enhanced formatting methods
        if "research_data" in str(final_result) or any("research" in str(step.get("action", "")) for step in plan):
            return self.output_formatter.format_research_output(final_result)
        elif "email_list" in str(final_result) or "total_emails" in str(final_result) or any("email" in str(step.get("action", "")) for step in plan):
            return self.output_formatter.format_email_output(final_result)
        elif "total_events" in str(final_result) or "today_events" in str(final_result) or any("calendar" in str(step.get("action", "")) for step in plan):
            return self.output_formatter.format_calendar_output(final_result)
        else:
            # Check if any step returned calendar events
            for step_result in results["step_results"]:
                step_data = step_result.get("result", {})
                if isinstance(step_data, list) and step_data:
                    first_item = step_data[0] if step_data else {}
                    if isinstance(first_item, dict) and any(key in first_item for key in ["title", "start_time", "end_time"]):
                        return self.output_formatter.format_calendar_output(step_data)
            
            return f"Task completed successfully. {len(results['step_results'])} steps executed."
    
    # ===== PRIVATE HELPER METHODS =====
    
    async def _create_execution_plan(self, intention: Dict, context: Dict) -> List[Dict]:
        """Private wrapper for create_execution_plan"""
        return await self.create_execution_plan(intention, context)
    
    async def _execute_plan(self, plan: List[Dict]) -> Dict:
        """Private wrapper for execute_plan"""
        return await self.execute_plan(plan)
    
    def _store_interaction(self, user_input: str, intention: Dict, result: Dict, start_time: datetime) -> None:
        """Store interaction in both session managers"""
        try:
            interaction = {
                "timestamp": start_time.isoformat(),
                "user_input": user_input,
                "intention": intention,
                "result": result,
                "duration": (datetime.now() - start_time).total_seconds(),
            }
            
            # Store in new session manager
            self.session_manager.add_interaction(interaction)
            
            # Store in old session history for compatibility
            self.session_history.append(interaction)
            
            # Limit old session history size
            if len(self.session_history) > 100:
                self.session_history = self.session_history[-100:]
                
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    def _create_success_response(self, intention: Dict, result: Dict, context: Dict, 
                               context_analysis: Dict, proactive_suggestions: List, start_time: datetime) -> Dict[str, Any]:
        """Create comprehensive successful response structure"""
        return {
            "status": "success",
            "intention": intention,
            "result": result,
            "context": context,
            "context_analysis": context_analysis,
            "proactive_suggestions": proactive_suggestions,
            "conversation_context": self._get_conversation_context(),
            "duration": (datetime.now() - start_time).total_seconds()
        }
    
    def _create_error_response(self, error_message: str, user_input: str) -> Dict[str, Any]:
        """Create error response structure"""
        return {
            "status": "error",
            "error": error_message,
            "result": {"final_output": f"I encountered an error: {error_message}"},
            "user_input": user_input
        }
    
    def _get_conversation_context(self) -> Dict[str, Any]:
        """Get conversation context safely"""
        try:
            if self.conversation_manager:
                return self.conversation_manager.get_conversation_context()
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
        return {}
    
    @asynccontextmanager
    async def managed_session(self):
        """Context manager for proper resource management"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.close()
    
    async def close(self) -> None:
        """Cleanup resources with enhanced error handling"""
        cleanup_errors = []
        
        try:
            await self.automation_engine.stop()
        except Exception as e:
            cleanup_errors.append(f"Automation engine cleanup failed: {e}")
        
        # Enhanced controller cleanup
        for name, controller in self.app_controllers.items():
            try:
                if name == "browser" and hasattr(controller, 'close'):
                    controller.close()
                elif name == "document" and hasattr(controller, 'close_all_documents'):
                    controller.close_all_documents()
                elif hasattr(controller, 'close'):
                    if asyncio.iscoroutinefunction(controller.close):
                        await controller.close()
                    else:
                        controller.close()
            except Exception as e:
                cleanup_errors.append(f"{name} controller cleanup failed: {e}")
        
        # Clear session
        self.session_manager.clear_session()
        
        if cleanup_errors:
            logger.warning(f"Cleanup completed with errors: {'; '.join(cleanup_errors)}")
        else:
            logger.info("Orchestrator cleanup completed successfully")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_start": self.session_manager.session_start.isoformat(),
            "interactions_count": len(self.session_manager.session_history),
            "max_history_size": self.session_manager.config.MAX_HISTORY_SIZE,
            "last_cleanup": self.session_manager.last_cleanup.isoformat(),
            "active_controllers": list(self.app_controllers.keys()),
            "total_session_history": len(self.session_history)
        }