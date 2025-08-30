"""
Enhanced Conversation Manager for Natural Dialogue
Handles context-aware, engaging conversations with personality
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    timestamp: datetime
    user_input: str
    ai_response: str
    intent: str
    sentiment: str
    topics: List[str]

class ConversationManager:
    """Manages conversational context and generates engaging responses"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.conversation_history: List[ConversationTurn] = []
        self.user_preferences = {}
        self.conversation_topics = set()
        self.user_personality_profile = {}
        
    def add_conversation_turn(self, user_input: str, ai_response: str, intent: str = "unknown"):
        """Add a conversation turn to history"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=user_input,
            ai_response=ai_response,
            intent=intent,
            sentiment=self._analyze_sentiment(user_input),
            topics=self._extract_topics(user_input)
        )
        
        self.conversation_history.append(turn)
        self._update_user_profile(turn)
        
        # Keep only last 20 turns for performance
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get rich conversation context for response generation"""
        if not self.conversation_history:
            return {
                "is_first_interaction": True,
                "conversation_length": 0,
                "recent_topics": [],
                "user_mood": "neutral",
                "conversation_flow": "starting"
            }
        
        recent_turns = self.conversation_history[-5:]
        
        return {
            "is_first_interaction": False,
            "conversation_length": len(self.conversation_history),
            "recent_topics": list(set([topic for turn in recent_turns for topic in turn.topics])),
            "user_mood": self._get_current_mood(),
            "conversation_flow": self._analyze_conversation_flow(),
            "last_user_sentiment": recent_turns[-1].sentiment if recent_turns else "neutral",
            "recurring_interests": self._get_recurring_interests(),
            "conversation_style": self._get_conversation_style()
        }
    
    def generate_contextual_response(self, user_input: str, base_response: str = None) -> str:
        """Generate a contextually aware response"""
        context = self.get_conversation_context()
        
        # If we have a base response, enhance it with context
        if base_response:
            return self._enhance_response_with_context(base_response, context, user_input)
        
        # Generate response from scratch
        return self._generate_response_from_context(user_input, context)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of user input"""
        text_lower = text.lower()
        
        positive_words = ['great', 'awesome', 'love', 'excited', 'happy', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'hate', 'frustrated', 'angry', 'sad', 'disappointed', 'annoyed']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from user input"""
        text_lower = text.lower()
        topics = []
        
        # Technology topics
        tech_keywords = ['ai', 'machine learning', 'programming', 'code', 'software', 'computer', 'technology']
        if any(keyword in text_lower for keyword in tech_keywords):
            topics.append('technology')
        
        # Work/productivity topics
        work_keywords = ['work', 'job', 'project', 'meeting', 'deadline', 'task', 'productivity']
        if any(keyword in text_lower for keyword in work_keywords):
            topics.append('work')
        
        # Personal topics
        personal_keywords = ['family', 'friends', 'hobby', 'weekend', 'vacation', 'personal']
        if any(keyword in text_lower for keyword in personal_keywords):
            topics.append('personal')
        
        # Learning topics
        learning_keywords = ['learn', 'study', 'research', 'understand', 'explain', 'how to']
        if any(keyword in text_lower for keyword in learning_keywords):
            topics.append('learning')
        
        return topics if topics else ['general']
    
    def _update_user_profile(self, turn: ConversationTurn):
        """Update user personality profile based on conversation"""
        # Track interests
        for topic in turn.topics:
            self.conversation_topics.add(topic)
        
        # Track communication style
        if len(turn.user_input.split()) > 20:
            self.user_preferences['communication_style'] = 'detailed'
        elif len(turn.user_input.split()) < 5:
            self.user_preferences['communication_style'] = 'concise'
        else:
            self.user_preferences['communication_style'] = 'balanced'
        
        # Track formality
        if any(word in turn.user_input.lower() for word in ['please', 'thank you', 'could you']):
            self.user_preferences['formality'] = 'polite'
        elif any(word in turn.user_input.lower() for word in ['hey', 'yo', 'sup']):
            self.user_preferences['formality'] = 'casual'
        else:
            self.user_preferences['formality'] = 'neutral'
    
    def _get_current_mood(self) -> str:
        """Analyze current user mood from recent interactions"""
        if not self.conversation_history:
            return "neutral"
        
        recent_sentiments = [turn.sentiment for turn in self.conversation_history[-3:]]
        
        if recent_sentiments.count('positive') >= 2:
            return "positive"
        elif recent_sentiments.count('negative') >= 2:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_conversation_flow(self) -> str:
        """Analyze the flow of conversation"""
        if len(self.conversation_history) < 2:
            return "starting"
        
        recent_intents = [turn.intent for turn in self.conversation_history[-3:]]
        
        if len(set(recent_intents)) == 1:
            return "focused"  # Staying on one topic
        elif 'task_execution' in recent_intents:
            return "task_oriented"
        else:
            return "exploratory"  # Jumping between topics
    
    def _get_recurring_interests(self) -> List[str]:
        """Get topics the user frequently discusses"""
        topic_counts = {}
        for turn in self.conversation_history:
            for topic in turn.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return topics mentioned more than once
        return [topic for topic, count in topic_counts.items() if count > 1]
    
    def _get_conversation_style(self) -> str:
        """Determine the user's preferred conversation style"""
        return self.user_preferences.get('communication_style', 'balanced')
    
    def _enhance_response_with_context(self, base_response: str, context: Dict, user_input: str) -> str:
        """Enhance a base response with conversational context"""
        enhanced_response = base_response
        
        # Add personality based on user mood
        if context['user_mood'] == 'positive':
            if not any(emoji in enhanced_response for emoji in ['😊', '🎉', '✨', '🚀']):
                enhanced_response = enhanced_response.rstrip('.!') + "! 😊"
        
        # Reference previous topics if relevant
        if context['recent_topics'] and any(topic in user_input.lower() for topic in context['recent_topics']):
            enhanced_response += f"\n\nI notice we've been exploring {', '.join(context['recent_topics'])} - I love how curious you are about these topics!"
        
        # Adjust formality based on user preference
        formality = self.user_preferences.get('formality', 'neutral')
        if formality == 'casual' and 'please' in enhanced_response:
            enhanced_response = enhanced_response.replace('please', '')
        elif formality == 'polite' and not any(word in enhanced_response.lower() for word in ['please', 'thank you']):
            enhanced_response = "I'd be happy to help! " + enhanced_response
        
        return enhanced_response
    
    def _generate_response_from_context(self, user_input: str, context: Dict) -> str:
        """Generate response based purely on conversation context"""
        user_lower = user_input.lower()
        
        # First interaction
        if context['is_first_interaction']:
            return self._generate_first_interaction_response(user_input)
        
        # Continuing conversation
        if context['conversation_flow'] == 'focused':
            return f"I love how we're diving deep into this topic! {self._get_topic_specific_response(user_input, context['recent_topics'])}"
        
        elif context['user_mood'] == 'positive':
            return f"Your enthusiasm is contagious! 🌟 {self._get_enthusiastic_response(user_input)}"
        
        elif context['user_mood'] == 'negative':
            return f"I can sense you might be feeling a bit frustrated. {self._get_supportive_response(user_input)}"
        
        else:
            return self._get_balanced_response(user_input, context)
    
    def _generate_first_interaction_response(self, user_input: str) -> str:
        """Generate welcoming first interaction response"""
        return "Hello! It's great to meet you! 😊 I'm excited to chat and help you with whatever you need. What's on your mind today?"
    
    def _get_topic_specific_response(self, user_input: str, topics: List[str]) -> str:
        """Generate response specific to ongoing topics"""
        if 'technology' in topics:
            return "Technology is such a fascinating field! What aspect interests you most?"
        elif 'work' in topics:
            return "Work can be both challenging and rewarding. How can I help make things easier for you?"
        else:
            return "This is such an interesting topic to explore together!"
    
    def _get_enthusiastic_response(self, user_input: str) -> str:
        """Generate enthusiastic response matching user's positive mood"""
        return f"I can tell you're excited about this! Let's explore '{user_input}' together - I'm curious to learn more about what interests you most!"
    
    def _get_supportive_response(self, user_input: str) -> str:
        """Generate supportive response for negative mood"""
        return f"I'm here to help make things better. Let's work through '{user_input}' together - sometimes a fresh perspective can really help!"
    
    def _get_balanced_response(self, user_input: str, context: Dict) -> str:
        """Generate balanced response for neutral interactions"""
        if context['recurring_interests']:
            interests = ', '.join(context['recurring_interests'])
            return f"I notice you're interested in {interests}. How does '{user_input}' relate to what we've been discussing?"
        else:
            return f"That's interesting! Tell me more about '{user_input}' - I'd love to understand your perspective better."
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation for debugging/analysis"""
        return {
            "total_turns": len(self.conversation_history),
            "topics_discussed": list(self.conversation_topics),
            "user_preferences": self.user_preferences,
            "current_mood": self._get_current_mood(),
            "conversation_style": self._get_conversation_style(),
            "recent_context": self.get_conversation_context()
        }