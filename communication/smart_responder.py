import asyncio
import logging
from datetime import datetime, timedelta
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class SmartResponder:
    """Intelligent response generation and communication automation"""

    def __init__(self, llm_client=None, learning_engine=None):
        self.llm_client = llm_client
        self.learning_engine = learning_engine
        self.response_templates = self._load_response_templates()
        self.auto_response_rules = []

    def _load_response_templates(self) -> Dict:
        """Load response templates for different scenarios"""
        return {
            "acknowledgment": [
                "Thanks for the update!",
                "Got it, thanks!",
                "Acknowledged.",
                "Thanks for letting me know."
            ],
            "meeting_scheduling": [
                "I'm available at {time}. Does that work for everyone?",
                "Let me check my calendar and get back to you.",
                "How about {alternative_time}?",
                "I can make that work. Sending calendar invite."
            ],
            "status_update": [
                "Here's the current status: {status}",
                "Update: {progress} completed. Next steps: {next_steps}",
                "Status report: {details}"
            ],
            "question_response": [
                "Great question! {answer}",
                "Based on my understanding: {response}",
                "Let me clarify: {clarification}"
            ],
            "urgent_response": [
                "I see this is urgent. {immediate_action}",
                "Prioritizing this now. {status}",
                "Urgent item noted. {response}"
            ],
            "out_of_office": [
                "I'm currently out of office until {return_date}. For urgent matters, please contact {backup_contact}.",
                "Thanks for your message. I'm away until {return_date} and will respond when I return.",
                "Auto-reply: Out of office until {return_date}. Will respond to non-urgent messages upon return."
            ]
        }

    async def generate_smart_response(self, message: Dict, context: Dict,
                                    response_type: str = "auto") -> Dict:
        """Generate intelligent response to a message"""
        try:
            response_data = {
                "suggested_response": "",
                "confidence": 0.0,
                "response_type": response_type,
                "reasoning": "",
                "alternatives": [],
                "requires_human_review": False
            }

            # Analyze message content
            message_analysis = await self._analyze_message_for_response(message)

            # Determine response strategy
            strategy = self._determine_response_strategy(message_analysis, context)

            # Generate response based on strategy
            if strategy["type"] == "template":
                response_data = await self._generate_template_response(message, strategy, context)
            elif strategy["type"] == "llm":
                response_data = await self._generate_llm_response(message, context, message_analysis)
            elif strategy["type"] == "hybrid":
                response_data = await self._generate_hybrid_response(message, strategy, context)
            else:
                response_data["suggested_response"] = "I'll review this message and respond shortly."
                response_data["confidence"] = 0.3
                response_data["requires_human_review"] = True

            # Apply personalization if learning engine available
            if self.learning_engine:
                response_data = await self._personalize_response(response_data, context)

            return response_data

        except Exception as e:
            logger.error("Smart response generation failed: %s", e)
            return {
                "error": str(e),
                "suggested_response": "I'll review this message and respond appropriately.",
                "confidence": 0.1,
                "requires_human_review": True
            }

    async def suggest_proactive_messages(self, context: Dict,
                                       communication_history: List[Dict]) -> List[Dict]:
        """Suggest proactive messages based on context and patterns"""
        try:
            suggestions = []

            # Analyze recent communication patterns
            patterns = await self._analyze_communication_patterns(communication_history)

            # Generate different types of proactive suggestions

            # 1. Follow-up suggestions
            followup_suggestions = await self._suggest_followups(communication_history, context)
            suggestions.extend(followup_suggestions)

            # 2. Status update suggestions
            status_suggestions = await self._suggest_status_updates(context, patterns)
            suggestions.extend(status_suggestions)

            # 3. Meeting scheduling suggestions
            meeting_suggestions = await self._suggest_meeting_scheduling(context, patterns)
            suggestions.extend(meeting_suggestions)

            # 4. Check-in suggestions
            checkin_suggestions = await self._suggest_checkins(communication_history, context)
            suggestions.extend(checkin_suggestions)

            # Sort by priority and confidence
            suggestions.sort(key=lambda x: (x.get("priority", 0), x.get("confidence", 0)), reverse=True)

            return suggestions[:10]  # Return top 10 suggestions

        except Exception as e:
            logger.error("Proactive message suggestion failed: %s", e)
            return []

    async def auto_respond_to_message(self, message: Dict, context: Dict,
                                    auto_response_settings: Dict) -> Optional[Dict]:
        """Automatically respond to message if conditions are met"""
        try:
            # Check if auto-response is enabled
            if not auto_response_settings.get("enabled", False):
                return None

            # Analyze message for auto-response triggers
            triggers = await self._check_auto_response_triggers(message, auto_response_settings)

            if not triggers:
                return None

            # Generate appropriate auto-response
            response_data = await self.generate_smart_response(message, context, "auto")

            # Apply auto-response filters
            if self._should_auto_respond(response_data, auto_response_settings):
                return {
                    "auto_response": response_data["suggested_response"],
                    "trigger": triggers[0],
                    "confidence": response_data["confidence"],
                    "timestamp": datetime.now().isoformat()
                }

            return None

        except Exception as e:
            logger.error("Auto-response failed: %s", e)
            return None

    async def generate_meeting_summary_response(self, meeting_data: Dict,
                                              participants: List[str]) -> Dict:
        """Generate response with meeting summary"""
        try:
            summary_response = {
                "summary_text": "",
                "action_items": [],
                "next_steps": [],
                "follow_up_required": []
            }

            # Extract key information from meeting data
            if "transcript" in meeting_data:
                summary_response = await self._analyze_meeting_transcript(meeting_data["transcript"])
            elif "notes" in meeting_data:
                summary_response = await self._analyze_meeting_notes(meeting_data["notes"])

            # Generate formatted response
            response_text = self._format_meeting_summary_response(summary_response, participants)

            return {
                "suggested_response": response_text,
                "confidence": 0.8,
                "response_type": "meeting_summary",
                "summary_data": summary_response
            }

        except Exception as e:
            logger.error("Meeting summary response generation failed: %s", e)
            return {"error": str(e)}

    async def _analyze_message_for_response(self, message: Dict) -> Dict:
        """Analyze message to determine response requirements"""
        text = message.get("text", message.get("body", "")).lower()

        analysis = {
            "is_question": False,
            "is_urgent": False,
            "requires_action": False,
            "sentiment": "neutral",
            "topics": [],
            "mentions_user": False,
            "response_expected": False
        }

        # Check if it's a question
        analysis["is_question"] = "?" in text or any(
            text.strip().startswith(word) for word in
            ["what", "how", "when", "where", "why", "who", "can", "could", "would", "should"]
        )

        # Check urgency
        urgent_keywords = ["urgent", "asap", "emergency", "critical", "immediately", "deadline"]
        analysis["is_urgent"] = any(keyword in text for keyword in urgent_keywords)

        # Check if action is required
        action_keywords = ["please", "need", "require", "must", "should", "task", "todo", "action"]
        analysis["requires_action"] = any(keyword in text for keyword in action_keywords)

        # Simple sentiment analysis
        positive_words = ["thanks", "great", "good", "excellent", "appreciate"]
        negative_words = ["problem", "issue", "error", "wrong", "bad"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            analysis["sentiment"] = "positive"
        elif negative_count > positive_count:
            analysis["sentiment"] = "negative"

        # Check if user is mentioned
        analysis["mentions_user"] = "@" in text or "you" in text

        # Determine if response is expected
        analysis["response_expected"] = (
            analysis["is_question"] or
            analysis["requires_action"] or
            analysis["mentions_user"] or
            analysis["is_urgent"]
        )

        return analysis

    def _determine_response_strategy(self, message_analysis: Dict, context: Dict) -> Dict:
        """Determine the best strategy for generating response"""
        strategy = {"type": "template", "template_category": "acknowledgment", "confidence": 0.5}

        # High-confidence template responses
        if message_analysis["sentiment"] == "positive" and not message_analysis["is_question"]:
            strategy = {"type": "template", "template_category": "acknowledgment", "confidence": 0.8}
        elif message_analysis["is_urgent"]:
            strategy = {"type": "template", "template_category": "urgent_response", "confidence": 0.7}
        elif "meeting" in context.get("recent_topics", []):
            strategy = {"type": "template", "template_category": "meeting_scheduling", "confidence": 0.7}

        # LLM responses for complex cases
        elif message_analysis["is_question"] and message_analysis["requires_action"]:
            strategy = {"type": "llm", "confidence": 0.6}
        elif message_analysis["sentiment"] == "negative":
            strategy = {"type": "llm", "confidence": 0.6}

        # Hybrid approach for moderate complexity
        elif message_analysis["is_question"]:
            strategy = {"type": "hybrid", "template_category": "question_response", "confidence": 0.7}

        return strategy

    async def _generate_template_response(self, message: Dict, strategy: Dict, context: Dict) -> Dict:
        """Generate response using templates"""
        template_category = strategy["template_category"]
        templates = self.response_templates.get(template_category, ["Thanks for your message."])

        # Select appropriate template (for now, use first one)
        template = templates[0]

        # Fill in template variables
        response_text = self._fill_template_variables(template, message, context)

        return {
            "suggested_response": response_text,
            "confidence": strategy["confidence"],
            "response_type": "template",
            "reasoning": f"Used {template_category} template",
            "alternatives": templates[1:3] if len(templates) > 1 else []
        }

    async def _generate_llm_response(self, message: Dict, context: Dict, message_analysis: Dict) -> Dict:
        """Generate response using LLM"""
        if not self.llm_client:
            return {
                "suggested_response": "I'll review this and respond appropriately.",
                "confidence": 0.3,
                "requires_human_review": True
            }

        try:
            # Create prompt for LLM
            prompt = self._create_response_prompt(message, context, message_analysis)

            # Generate response
            llm_response = await self.llm_client.generate_response(prompt)

            return {
                "suggested_response": llm_response.strip(),
                "confidence": 0.7,
                "response_type": "llm",
                "reasoning": "Generated using LLM based on message analysis"
            }

        except Exception as e:
            logger.error("LLM response generation failed: %s", e)
            return {
                "suggested_response": "I'll review this message and respond shortly.",
                "confidence": 0.3,
                "requires_human_review": True
            }

    async def _generate_hybrid_response(self, message: Dict, strategy: Dict, context: Dict) -> Dict:
        """Generate response using hybrid template + LLM approach"""
        # Start with template
        template_response = await self._generate_template_response(message, strategy, context)

        # Enhance with LLM if available
        if self.llm_client:
            try:
                enhancement_prompt = (
                    f"Enhance this response to be more personalized and contextual: '{template_response['suggested_response']}'. Original message: '{message.get('text', message.get('body', ''))}'"
                )

                enhanced_response = await self.llm_client.generate_response(enhancement_prompt)

                return {
                    "suggested_response": enhanced_response.strip(),
                    "confidence": 0.8,
                    "response_type": "hybrid",
                    "reasoning": "Template enhanced with LLM personalization"
                }
            except Exception:
                pass

        return template_response

    def _fill_template_variables(self, template: str, message: Dict, context: Dict) -> str:
        """Fill template variables with actual values"""
        # Simple variable substitution
        filled_template = template

        # Common variables
        if "{time}" in template:
            filled_template = filled_template.replace("{time}", "2:00 PM")

        if "{status}" in template:
            filled_template = filled_template.replace("{status}", "In progress")

        if "{return_date}" in template:
            return_date = context.get("return_date", "Monday")
            filled_template = filled_template.replace("{return_date}", return_date)

        return filled_template

    def _create_response_prompt(self, message: Dict, context: Dict, message_analysis: Dict) -> str:
        """Create prompt for LLM response generation"""
        message_text = message.get("text", message.get("body", ""))
        sender = message.get("user_name", message.get("from", ""))

        prompt = """Generate a professional and helpful response to this message:

Message from {sender}: "{message_text}"

Context:
- Message is {'urgent' if message_analysis['is_urgent'] else 'normal priority'}
- {'Contains questions' if message_analysis['is_question'] else 'No questions'}
- {'Requires action' if message_analysis['requires_action'] else 'Informational'}
- Sentiment: {message_analysis['sentiment']}

Generate a concise, professional response that addresses the message appropriately. Keep it under 100 words."""

        return prompt

    async def _personalize_response(self, response_data: Dict, context: Dict) -> Dict:
        """Personalize response based on learned preferences"""
        if not self.learning_engine:
            return response_data

        try:
            # Get user communication style preferences
            user_id = context.get("user_id", "")
            if user_id:
                adaptations = await self.learning_engine.adapt_response_style("", context)

                # Adjust response based on learned preferences
                if adaptations.get("response_length") == "brie":
                    # Shorten response
                    response_text = response_data["suggested_response"]
                    if len(response_text) > 50:
                        response_data["suggested_response"] = response_text[:50] + "..."

                if adaptations.get("communication_style") == "formal":
                    # Make response more formal
                    response_text = response_data["suggested_response"]
                    response_data["suggested_response"] = response_text.replace("Thanks!", "Thank you.")

        except Exception as e:
            logger.error("Response personalization failed: %s", e)

        return response_data

    async def _analyze_communication_patterns(self, communication_history: List[Dict]) -> Dict:
        """Analyze patterns in communication history"""
        patterns = {
            "response_frequency": {},
            "common_topics": [],
            "typical_response_time": 0,
            "communication_gaps": []
        }

        # Simple pattern analysis
        if communication_history:
            # Find common topics
            all_text = " ".join([
                msg.get("text", msg.get("body", "")).lower()
                for msg in communication_history
            ])

            # Basic word frequency
            words = all_text.split()
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Only meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get top topics
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            patterns["common_topics"] = [word for word, count in sorted_words[:5]]

        return patterns

    async def _suggest_followups(self, communication_history: List[Dict], context: Dict) -> List[Dict]:
        """Suggest follow-up messages"""
        suggestions = []

        # Look for messages that might need follow-up
        for message in communication_history[-10:]:  # Last 10 messages
            text = message.get("text", message.get("body", "")).lower()

            # Check for action items that might need follow-up
            if any(word in text for word in ["will", "going to", "plan to", "next week"]):
                suggestions.append({
                    "type": "followup",
                    "priority": 0.7,
                    "confidence": 0.6,
                    "suggested_message": "Hi! Just following up on our previous discussion. Any updates?",
                    "reasoning": "Previous message mentioned future action",
                    "original_message_id": message.get("id", message.get("ts", ""))
                })

        return suggestions[:3]  # Return top 3

    async def _suggest_status_updates(self, context: Dict, patterns: Dict) -> List[Dict]:
        """Suggest status update messages"""
        suggestions = []

        # Check if it's been a while since last update
        last_update = context.get("last_status_update")
        if last_update:
            # If more than 3 days since last update
            suggestions.append({
                "type": "status_update",
                "priority": 0.6,
                "confidence": 0.7,
                "suggested_message": "Quick status update: [Current progress and next steps]",
                "reasoning": "Regular status update due"
            })

        return suggestions

    async def _suggest_meeting_scheduling(self, context: Dict, patterns: Dict) -> List[Dict]:
        """Suggest meeting scheduling messages"""
        suggestions = []

        # Check if meeting-related topics are common
        if "meeting" in patterns.get("common_topics", []):
            suggestions.append({
                "type": "meeting_scheduling",
                "priority": 0.8,
                "confidence": 0.6,
                "suggested_message": "Should we schedule a meeting to discuss this further? I'm available [time slots].",
                "reasoning": "Meeting discussions detected in recent communication"
            })

        return suggestions

    async def _suggest_checkins(self, communication_history: List[Dict], context: Dict) -> List[Dict]:
        """Suggest check-in messages"""
        suggestions = []

        # Check for communication gaps
        if communication_history:
            last_message_time = communication_history[-1].get("timestamp", "")
            if last_message_time:
                try:
                    last_time = datetime.fromisoformat(last_message_time.replace('Z', '+00:00'))
                    if datetime.now() - last_time > timedelta(days=7):
                        suggestions.append({
                            "type": "checkin",
                            "priority": 0.5,
                            "confidence": 0.6,
                            "suggested_message": "Hi! Hope you're doing well. Just checking in to see how things are going.",
                            "reasoning": "Long gap since last communication"
                        })
                except Exception:
                    pass

        return suggestions

    async def _check_auto_response_triggers(self, message: Dict, settings: Dict) -> List[str]:
        """Check if message triggers auto-response"""
        triggers = []

        text = message.get("text", message.get("body", "")).lower()

        # Check for out-of-office triggers
        if settings.get("out_of_office", {}).get("enabled", False):
            triggers.append("out_of_office")

        # Check for urgent message triggers
        if settings.get("urgent_response", {}).get("enabled", False):
            urgent_keywords = ["urgent", "asap", "emergency", "critical"]
            if any(keyword in text for keyword in urgent_keywords):
                triggers.append("urgent_response")

        # Check for acknowledgment triggers
        if settings.get("acknowledgment", {}).get("enabled", False):
            ack_keywords = ["thanks", "thank you", "received", "got it"]
            if any(keyword in text for keyword in ack_keywords):
                triggers.append("acknowledgment")

        return triggers

    def _should_auto_respond(self, response_data: Dict, settings: Dict) -> bool:
        """Determine if auto-response should be sent"""
        # Check confidence threshold
        min_confidence = settings.get("min_confidence", 0.7)
        if response_data.get("confidence", 0) < min_confidence:
            return False

        # Check if human review is required
        if response_data.get("requires_human_review", False):
            return False

        # Check rate limiting
        # (This would require tracking recent auto-responses)

        return True

    async def _analyze_meeting_transcript(self, transcript: str) -> Dict:
        """Analyze meeting transcript to extract key information"""
        # Simple keyword-based analysis
        summary = {
            "summary_text": "Meeting covered key topics and action items were identified.",
            "action_items": [],
            "next_steps": [],
            "follow_up_required": []
        }

        # Look for action items
        sentences = transcript.split(".")
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["will", "action", "todo", "task", "assign"]):
                summary["action_items"].append(sentence.strip())

        return summary

    async def _analyze_meeting_notes(self, notes: str) -> Dict:
        """Analyze meeting notes to extract key information"""
        # Similar to transcript analysis but for structured notes
        return await self._analyze_meeting_transcript(notes)

    def _format_meeting_summary_response(self, summary_data: Dict, participants: List[str]) -> str:
        """Format meeting summary into response message"""
        response_parts = []

        response_parts.append("Meeting Summary:")
        response_parts.append(summary_data.get("summary_text", ""))

        if summary_data.get("action_items"):
            response_parts.append("\nAction Items:")
            for item in summary_data["action_items"][:5]:
                response_parts.append(f"• {item}")

        if summary_data.get("next_steps"):
            response_parts.append("\nNext Steps:")
            for step in summary_data["next_steps"][:3]:
                response_parts.append(f"• {step}")

        return "\n".join(response_parts)
