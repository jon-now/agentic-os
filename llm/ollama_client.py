import requests
import asyncio
import logging
from typing import Dict, Optional
from config.settings import settings
import json

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api"
        self.session = requests.Session()
        self.session.timeout = settings.LLM_TIMEOUT
        
        # RTX 3050 optimized model configurations
        self.rtx_3050_models = {
            "small": {
                "models": ["llama3.2:3b", "phi3:mini", "tinyllama:1.1b"],
                "vram_usage": "2-3GB",
                "use_case": "Quick responses, chat, simple tasks",
                "context_length": 4096
            },
            "medium": {
                "models": ["llama3.1:7b-q4", "codeqwen:7b-q4", "mistral:7b-q4"],
                "vram_usage": "4-5GB", 
                "use_case": "Complex reasoning, coding, analysis",
                "context_length": 8192
            },
            "large": {
                "models": ["llama3.1:8b-q4", "llava:7b-q4", "codellama:7b-q4"],
                "vram_usage": "5-6GB",
                "use_case": "Advanced tasks, vision, code generation",
                "context_length": 4096,
                "requires_optimization": True
            }
        }
        
        # Current active model info
        self.current_model = None
        self.current_model_size = None
        self.available_models = []

    def is_model_available(self) -> bool:
        """Enhanced availability check with model discovery"""
        try:
            response = self.session.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = [model["name"] for model in models]
                
                # Check for current model
                model_available = any(model["name"].startswith(self.model_name) for model in models)
                
                # If current model not available, try to find a suitable alternative
                if not model_available:
                    alternative = self._find_best_available_model()
                    if alternative:
                        logger.info(f"Switching from {self.model_name} to available model: {alternative}")
                        self.model_name = alternative
                        return True
                
                return model_available
            return False
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama service not running")
            return False
        except Exception as e:
            logger.error("Error checking model: %s", e)
            return False
    
    def _find_best_available_model(self) -> Optional[str]:
        """Find the best available model for RTX 3050"""
        if not self.available_models:
            return None
            
        # Priority order for RTX 3050
        preferred_models = [
            "llama3.2:3b", "phi3:mini", "tinyllama:1.1b",  # Small models
            "llama3.1:7b-q4", "codeqwen:7b-q4", "mistral:7b-q4",  # Medium models
            "llama3.1:8b-q4", "llama3.1:8b", "codellama:7b-q4",  # Large models
        ]
        
        for preferred in preferred_models:
            for available in self.available_models:
                if available.startswith(preferred.split(':')[0]):
                    return available
                    
        # Return first available model as last resort
        return self.available_models[0] if self.available_models else None
    
    def get_optimal_model_for_task(self, task_type: str, complexity: str = "medium") -> str:
        """Get optimal model for specific task on RTX 3050"""
        task_model_mapping = {
            "chat": {
                "simple": "small",
                "medium": "small", 
                "complex": "medium"
            },
            "coding": {
                "simple": "medium",
                "medium": "medium",
                "complex": "large"
            },
            "analysis": {
                "simple": "small",
                "medium": "medium",
                "complex": "large"
            },
            "vision": {
                "simple": "large",  # Vision models require larger models
                "medium": "large",
                "complex": "large"
            }
        }
        
        model_size = task_model_mapping.get(task_type, {}).get(complexity, "medium")
        models = self.rtx_3050_models[model_size]["models"]
        
        # Return first available model from the recommended list
        for model in models:
            for available in self.available_models:
                if available.startswith(model.split(':')[0]):
                    return available
                    
        return self.model_name  # Fallback to current model

    async def generate_response(self, prompt: str, context: Optional[Dict] = None, task_type: str = "chat") -> str:
        """Generate with robust fallbacks and automatic model optimization"""
        
        # Auto-select optimal model for task if needed
        if context and context.get("auto_optimize_model", True):
            optimal_model = self.get_optimal_model_for_task(task_type, context.get("complexity", "medium"))
            if optimal_model != self.model_name:
                logger.info(f"Auto-switching to optimal model: {optimal_model}")
                self.model_name = optimal_model

        # Check model availability first
        if not self.is_model_available():
            return self._get_fallback_response(prompt)

        # Try LLM generation with RTX 3050 optimizations
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": self._get_model_options(task_type)
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.post(f"{self.api_url}/generate", json=payload, timeout=30)
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                if generated_text and len(generated_text) > 5:
                    return generated_text

            return self._get_fallback_response(prompt)

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return self._get_fallback_response(prompt)
    
    def _get_model_options(self, task_type: str) -> Dict:
        """Get optimized model options for RTX 3050"""
        base_options = {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "num_predict": 512
        }
        
        # Task-specific optimizations
        if task_type == "coding":
            base_options.update({
                "temperature": 0.3,  # More deterministic for code
                "top_k": 20,
                "num_predict": 1024
            })
        elif task_type == "analysis":
            base_options.update({
                "temperature": 0.5,
                "num_predict": 1024
            })
        elif task_type == "creative":
            base_options.update({
                "temperature": 0.9,
                "top_p": 0.95
            })
            
        # RTX 3050 memory optimizations
        current_model_size = self._get_current_model_size()
        if current_model_size == "large":
            base_options.update({
                "num_ctx": 2048,  # Reduced context for memory efficiency
                "num_batch": 1,   # Process one at a time for large models
            })
        elif current_model_size == "medium":
            base_options.update({
                "num_ctx": 4096,
                "num_batch": 2,
            })
        else:  # small models
            base_options.update({
                "num_ctx": 4096,
                "num_batch": 4,
            })
            
        return base_options
    
    def _get_current_model_size(self) -> str:
        """Determine the size category of current model"""
        model_lower = self.model_name.lower()
        
        for size, config in self.rtx_3050_models.items():
            for model in config["models"]:
                if model.split(':')[0].lower() in model_lower:
                    return size
                    
        # Default categorization based on model name
        if any(keyword in model_lower for keyword in ["3b", "mini", "tiny"]):
            return "small"
        elif any(keyword in model_lower for keyword in ["7b", "6b", "5b"]):
            return "medium" 
        else:
            return "large"
    
    def get_model_recommendations(self) -> Dict:
        """Get RTX 3050 specific model recommendations"""
        return {
            "current_model": self.model_name,
            "current_size": self._get_current_model_size(),
            "available_models": self.available_models,
            "rtx_3050_optimized": self.rtx_3050_models,
            "recommendations": {
                "chat": "llama3.2:3b - Fast responses, low VRAM",
                "coding": "codeqwen:7b-q4 - Best for programming tasks",
                "analysis": "llama3.1:7b-q4 - Good reasoning, moderate VRAM",
                "vision": "llava:7b-q4 - Image understanding (requires 5GB+)"
            },
            "installation_commands": {
                "llama3.2:3b": "ollama pull llama3.2:3b",
                "phi3:mini": "ollama pull phi3:mini", 
                "codeqwen:7b-q4": "ollama pull codeqwen:7b-q4",
                "llama3.1:7b-q4": "ollama pull llama3.1:7b-q4",
                "llava:7b-q4": "ollama pull llava:7b-q4"
            }
        }

    async def analyze_intention(self, user_input: str, context: Dict) -> Dict:
        """Analyze user intention with LLM"""
        try:
            from llm.prompt_templates import PromptTemplates
            templates = PromptTemplates()

            prompt = templates.INTENTION_ANALYSIS.format(
                user_input=user_input,
                active_apps=context.get('active_apps', [])[:5],
                recent_files=[f.get('name', '') for f in context.get('recent_files', [])][:3],
                current_time=context.get('current_time', 'Unknown'),
                cpu_percent=context.get('system_status', {}).get('cpu_percent', 'Unknown'),
                memory_percent=context.get('system_status', {}).get('memory_percent', 'Unknown')
            )

            response = await self.generate_response(prompt)

            # Try to parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Extract JSON from response if wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

                # Fallback to keyword analysis
                return self._fallback_intention_analysis(user_input)

        except Exception as e:
            logger.error("Intention analysis failed: %s", e)
            return self._fallback_intention_analysis(user_input)

    def _fallback_intention_analysis(self, user_input: str) -> Dict:
        """Fallback intention analysis using keywords"""
        user_lower = user_input.lower().strip()

        # System status requests
        if any(phrase in user_lower for phrase in [
            "system status", "system info", "computer status", "how is my computer"
        ]):
            return {
                "interaction_type": "direct_response",
                "primary_action": "system_info",
                "confidence": 0.95,
                "direct_response": "",
                "applications_needed": [],
                "parameters": {},
                "estimated_steps": 0,
                "user_intent_summary": "System status request"
            }

        # Greetings
        if any(word in user_lower for word in ["hello", "hi", "hey", "good morning"]):
            return {
                "interaction_type": "direct_response",
                "primary_action": "chat",
                "confidence": 0.95,
                "direct_response": "Hello! I'm your AI assistant. I can help with research, system information, and various tasks. What would you like me to help you with?",
                "applications_needed": [],
                "parameters": {},
                "estimated_steps": 0,
                "user_intent_summary": "Greeting"
            }

        # Research requests
        if any(word in user_lower for word in ["research", "find information", "search", "look up"]):
            return {
                "interaction_type": "task_execution",
                "primary_action": "research",
                "confidence": 0.8,
                "applications_needed": ["browser"],
                "parameters": {"topic": user_input, "scope": "comprehensive"},
                "estimated_steps": 2,
                "user_intent_summary": "Research request"
            }

        # Email requests
        if any(phrase in user_lower for phrase in ["email", "gmail", "inbox", "messages"]):
            return {
                "interaction_type": "task_execution",
                "primary_action": "email_management",
                "confidence": 0.8,
                "applications_needed": ["email"],
                "parameters": {},
                "estimated_steps": 2,
                "user_intent_summary": "Email management request"
            }

        # Default fallback
        return {
            "interaction_type": "direct_response",
            "primary_action": "chat",
            "confidence": 0.6,
            "direct_response": f"I understand you said '{user_input}'. I can help with research, system information, or email management. Could you be more specific?",
            "applications_needed": [],
            "parameters": {"user_input": user_input},
            "estimated_steps": 0,
            "user_intent_summary": "General conversation"
        }

    def _get_fallback_response(self, prompt: str) -> str:
        """Smart fallback responses with enhanced conversational capabilities"""
        prompt_lower = prompt.lower()

        # System status requests
        if any(word in prompt_lower for word in ['system', 'status', 'computer']):
            return self._generate_system_status_fallback()

        # Conversational prompts
        if any(phrase in prompt_lower for phrase in ['conversational', 'friendly', 'engaging', 'natural']):
            return self._generate_conversational_fallback(prompt)

        # General knowledge questions
        if any(phrase in prompt_lower for phrase in ['what is', 'explain', 'tell me about', 'how does']):
            return self._generate_knowledge_fallback(prompt)

        # Greeting responses
        if any(word in prompt_lower for word in ['greeting', 'hello', 'hi', 'good morning']):
            return self._generate_greeting_fallback()

        # JSON requests (intention analysis)
        if 'json' in prompt_lower:
            if 'system status' in prompt_lower:
                return json.dumps({
                    "interaction_type": "direct_response",
                    "primary_action": "system_info",
                    "confidence": 0.9,
                    "direct_response": ""
                })
            elif any(word in prompt_lower for word in ['hello', 'hi', 'greeting']):
                return json.dumps({
                    "interaction_type": "direct_response",
                    "primary_action": "chat",
                    "confidence": 0.9,
                    "direct_response": ""
                })
            elif any(phrase in prompt_lower for phrase in ['what is', 'explain', 'tell me']):
                return json.dumps({
                    "interaction_type": "direct_response",
                    "primary_action": "general_knowledge",
                    "confidence": 0.8,
                    "direct_response": ""
                })

        return "I'm experiencing some technical difficulties with my language model, but I'm still here and ready to help you! 😊"

    def _generate_conversational_fallback(self, prompt: str) -> str:
        """Generate conversational fallback response"""
        if 'greeting' in prompt.lower():
            return "Hello there! 😊 I'm your AI assistant and I'm excited to chat with you today. What's on your mind?"
        elif 'personal' in prompt.lower():
            return "I'm doing great! I'm an AI assistant designed to be helpful, engaging, and genuinely interested in what you need. I love having conversations and helping with all sorts of tasks. What would you like to explore together?"
        elif 'opinion' in prompt.lower():
            return "That's a fascinating topic! While I don't have personal experiences like humans, I enjoy discussing ideas and sharing different perspectives. What aspect interests you most?"
        else:
            return "I'd love to have a great conversation with you! While my language model is having some technical issues, I'm still here and eager to help. What would you like to chat about?"

    def _generate_knowledge_fallback(self, prompt: str) -> str:
        """Generate fallback for knowledge questions"""
        return "That's a great question! While I'm having some technical difficulties accessing my full knowledge base right now, I'd still love to help you explore that topic. Could you tell me what specific aspect you're most curious about? 🤔"

    def _generate_greeting_fallback(self) -> str:
        """Generate friendly greeting fallback"""
        from datetime import datetime
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hello"
            
        return f"{time_greeting}! 😊 I'm your AI assistant and I'm excited to help you today. Whether you want to chat, research something, or get things done - I'm here for it all! What's on your mind?"

    def _generate_system_status_fallback(self) -> str:
        """Generate system status without LLM"""
        try:
            import psutil
            from datetime import datetime

            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent

            return """🖥️ **System Status**

⚡ CPU Usage: {cpu:.1f}%
🧠 Memory Usage: {memory:.1f}%
🕐 Current Time: {datetime.now().strftime('%I:%M %p')}
✅ Status: All systems operational

*Running in fallback mode*"""
        except Exception as e:
            return f"System status check failed: {e}"
