import asyncio
import requests
from typing import Dict, List
import re
import json

class IntentionEngine:
    def __init__(self):
        # Local LLM configuration (using Ollama)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "mistral:7b"  # or "llama2:7b", "codellama:7b"

        # Fallback patterns for when LLM is not available
        self.intention_categories = {
            'browser_control': ['search', 'navigate', 'open', 'browse', 'find', 'google'],
            'email_management': ['email', 'send', 'check', 'reply', 'compose'],
            'system_control': ['open', 'close', 'run', 'execute', 'launch'],
            'information_request': ['what', 'how', 'when', 'where', 'status'],
            'automation': ['automate', 'schedule', 'remind', 'organize']
        }

        # Check if Ollama is available
        self.local_llm_available = self.check_ollama_availability()

    def check_ollama_availability(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    async def parse_intention(self, user_input: str) -> Dict:
        """Parse user input to understand intention and extract parameters"""
        if self.local_llm_available:
            try:
                return await self.llm_intention_parsing(user_input)
            except Exception as e:
                print(f"LLM parsing failed, using fallback: {e}")
                return self.simple_intention_parsing(user_input)
        else:
            print("Local LLM not available, using simple parsing")
            return self.simple_intention_parsing(user_input)

    async def llm_intention_parsing(self, user_input: str) -> Dict:
        """Use local LLM to parse intentions"""
        prompt = f"""Analyze this user request and respond with ONLY a JSON object:

User request: "{user_input}"

Categories: browser_control, email_management, system_control, information_request, automation

JSON format:
{{
    "category": "category_name",
    "action": "specific_action",
    "parameters": {{"key": "value"}},
    "priority": "medium",
    "confidence": 0.9
}}

Examples:
- "search for AI news" → {{"category": "browser_control", "action": "search", "parameters": {{"query": "AI news"}}, "priority": "medium", "confidence": 0.9}}
- "open gmail" → {{"category": "browser_control", "action": "navigate", "parameters": {{"url": "gmail.com"}}, "priority": "medium", "confidence": 0.8}}

Response:"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 20
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                result_text = response.json()['response']

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_result = json.loads(json_str)

                    # Validate required fields
                    required_fields = ['category', 'action', 'parameters', 'priority', 'confidence']
                    if all(field in parsed_result for field in required_fields):
                        return parsed_result

                # If JSON parsing fails, fall back to simple parsing
                return self.simple_intention_parsing(user_input)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            print(f"Local LLM request failed: {e}")
            return self.simple_intention_parsing(user_input)

    def simple_intention_parsing(self, user_input: str) -> Dict:
        """Enhanced fallback intention parsing with better pattern matching"""
        user_lower = user_input.lower()

        # Enhanced pattern matching
        patterns = {
            'browser_control': {
                'search': ['search for', 'find', 'look up', 'google'],
                'navigate': ['open', 'go to', 'visit', 'navigate'],
            },
            'email_management': {
                'check': ['check email', 'read email', 'email status'],
                'compose': ['send email', 'write email', 'compose'],
            },
            'system_control': {
                'status': ['what\'s running', 'active apps', 'system status'],
                'launch': ['start', 'run', 'execute', 'launch'],
            }
        }

        # Check for specific patterns
        for category, actions in patterns.items():
            for action, keywords in actions.items():
                for keyword in keywords:
                    if keyword in user_lower:
                        # Extract parameters based on context
                        parameters = self.extract_parameters(user_input, action)
                        return {
                            "category": category,
                            "action": action,
                            "parameters": parameters,
                            "priority": "medium",
                            "confidence": 0.7
                        }

        # Simple keyword matching fallback
        for category, keywords in self.intention_categories.items():
            for keyword in keywords:
                if keyword in user_lower:
                    return {
                        "category": category,
                        "action": keyword,
                        "parameters": {"query": user_input},
                        "priority": "medium",
                        "confidence": 0.6
                    }

        # Default case
        return {
            "category": "information_request",
            "action": "general_query",
            "parameters": {"query": user_input},
            "priority": "medium",
            "confidence": 0.3
        }

    def extract_parameters(self, user_input: str, action: str) -> Dict:
        """Extract parameters based on the action type"""
        if action == 'search':
            # Extract search query after common search terms
            for term in ['search for', 'find', 'look up', 'google']:
                if term in user_input.lower():
                    query = user_input.lower().split(term, 1)[-1].strip()
                    return {"query": query}
            return {"query": user_input}

        elif action == 'navigate':
            # Extract URL or site name
            words = user_input.lower().split()
            for word in words:
                if '.' in word or word in ['gmail', 'youtube', 'github', 'stackoverflow']:
                    return {"url": word}
            return {"url": user_input}

        else:
            return {"query": user_input}