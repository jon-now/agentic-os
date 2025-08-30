import asyncio
import logging
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MessageAnalyzer:
    """Analyzes communication patterns and message content across platforms"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.sentiment_keywords = {
            "positive": ["great", "awesome", "excellent", "good", "thanks", "appreciate", "love", "perfect"],
            "negative": ["bad", "terrible", "awful", "hate", "problem", "issue", "wrong", "error"],
            "urgent": ["urgent", "asap", "emergency", "critical", "immediately", "now", "deadline"],
            "question": ["?", "how", "what", "when", "where", "why", "who", "can you", "could you"]
        }

    async def analyze_message_content(self, messages: List[Dict]) -> Dict:
        """Analyze content patterns in messages"""
        try:
            analysis = {
                "total_messages": len(messages),
                "sentiment_distribution": {},
                "urgency_indicators": [],
                "question_count": 0,
                "topic_clusters": [],
                "communication_patterns": {},
                "key_insights": []
            }

            if not messages:
                return analysis

            # Analyze sentiment
            analysis["sentiment_distribution"] = self._analyze_sentiment(messages)

            # Find urgent messages
            analysis["urgency_indicators"] = self._find_urgent_messages(messages)

            # Count questions
            analysis["question_count"] = self._count_questions(messages)

            # Identify topics
            analysis["topic_clusters"] = self._identify_topics(messages)

            # Analyze communication patterns
            analysis["communication_patterns"] = self._analyze_communication_patterns(messages)

            # Generate insights
            analysis["key_insights"] = self._generate_insights(analysis)

            return analysis

        except Exception as e:
            logger.error("Message content analysis failed: %s", e)
            return {"error": str(e)}

    async def analyze_user_communication_style(self, messages: List[Dict], user_id: str) -> Dict:
        """Analyze a specific user's communication style"""
        try:
            user_messages = [msg for msg in messages if msg.get("user_id") == user_id or msg.get("from_id") == user_id]

            if not user_messages:
                return {"error": "No messages found for user"}

            style_analysis = {
                "message_count": len(user_messages),
                "avg_message_length": 0,
                "communication_frequency": {},
                "preferred_times": [],
                "response_patterns": {},
                "language_style": {},
                "interaction_style": "unknown"
            }

            # Calculate average message length
            total_length = sum(len(msg.get("text", msg.get("body", ""))) for msg in user_messages)
            style_analysis["avg_message_length"] = total_length / len(user_messages)

            # Analyze communication frequency
            style_analysis["communication_frequency"] = self._analyze_frequency_patterns(user_messages)

            # Find preferred communication times
            style_analysis["preferred_times"] = self._find_preferred_times(user_messages)

            # Analyze response patterns
            style_analysis["response_patterns"] = self._analyze_response_patterns(user_messages, messages)

            # Analyze language style
            style_analysis["language_style"] = self._analyze_language_style(user_messages)

            # Determine interaction style
            style_analysis["interaction_style"] = self._determine_interaction_style(style_analysis)

            return style_analysis

        except Exception as e:
            logger.error("User communication style analysis failed: %s", e)
            return {"error": str(e)}

    async def analyze_team_communication_health(self, messages: List[Dict], team_members: List[str]) -> Dict:
        """Analyze team communication health and patterns"""
        try:
            health_analysis = {
                "participation_balance": {},
                "response_times": {},
                "collaboration_indicators": {},
                "communication_gaps": [],
                "team_dynamics": {},
                "health_score": 0.0
            }

            # Analyze participation balance
            health_analysis["participation_balance"] = self._analyze_participation_balance(messages, team_members)

            # Analyze response times
            health_analysis["response_times"] = self._analyze_response_times(messages)

            # Find collaboration indicators
            health_analysis["collaboration_indicators"] = self._find_collaboration_indicators(messages)

            # Identify communication gaps
            health_analysis["communication_gaps"] = self._identify_communication_gaps(messages, team_members)

            # Analyze team dynamics
            health_analysis["team_dynamics"] = self._analyze_team_dynamics(messages, team_members)

            # Calculate overall health score
            health_analysis["health_score"] = self._calculate_team_health_score(health_analysis)

            return health_analysis

        except Exception as e:
            logger.error("Team communication health analysis failed: %s", e)
            return {"error": str(e)}

    async def extract_action_items(self, messages: List[Dict]) -> List[Dict]:
        """Extract action items and tasks from messages"""
        try:
            action_items = []

            # Keywords that indicate action items
            action_keywords = [
                "todo", "to do", "task", "action item", "follow up", "need to",
                "should", "must", "will", "going to", "plan to", "assign",
                "deadline", "due", "complete", "finish", "deliver"
            ]

            for message in messages:
                text = message.get("text", message.get("body", "")).lower()

                # Look for action indicators
                for keyword in action_keywords:
                    if keyword in text:
                        # Extract potential action item
                        action_item = self._extract_action_from_text(message, keyword)
                        if action_item:
                            action_items.append(action_item)
                        break

            # Remove duplicates and sort by priority
            unique_actions = self._deduplicate_actions(action_items)
            sorted_actions = sorted(unique_actions, key=lambda x: x.get("priority_score", 0), reverse=True)

            return sorted_actions[:20]  # Return top 20 action items

        except Exception as e:
            logger.error("Action item extraction failed: %s", e)
            return []

    async def generate_communication_summary(self, messages: List[Dict], time_period: str = "day") -> Dict:
        """Generate a summary of communication activity"""
        try:
            summary = {
                "period": time_period,
                "total_messages": len(messages),
                "unique_participants": set(),
                "top_topics": [],
                "key_decisions": [],
                "action_items": [],
                "urgent_items": [],
                "summary_text": ""
            }

            if not messages:
                return summary

            # Count unique participants
            for message in messages:
                user_id = message.get("user_id") or message.get("from_id")
                if user_id:
                    summary["unique_participants"].add(user_id)

            summary["unique_participants"] = len(summary["unique_participants"])

            # Extract top topics
            summary["top_topics"] = self._identify_topics(messages)[:5]

            # Find key decisions
            summary["key_decisions"] = self._extract_decisions(messages)

            # Extract action items
            summary["action_items"] = await self.extract_action_items(messages)

            # Find urgent items
            summary["urgent_items"] = self._find_urgent_messages(messages)

            # Generate summary text
            summary["summary_text"] = self._generate_summary_text(summary)

            return summary

        except Exception as e:
            logger.error("Communication summary generation failed: %s", e)
            return {"error": str(e)}

    def _analyze_sentiment(self, messages: List[Dict]) -> Dict:
        """Analyze sentiment distribution in messages"""
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "urgent": 0}

        for message in messages:
            text = message.get("text", message.get("body", "")).lower()

            positive_score = sum(1 for word in self.sentiment_keywords["positive"] if word in text)
            negative_score = sum(1 for word in self.sentiment_keywords["negative"] if word in text)
            urgent_score = sum(1 for word in self.sentiment_keywords["urgent"] if word in text)

            if urgent_score > 0:
                sentiment_counts["urgent"] += 1
            elif positive_score > negative_score:
                sentiment_counts["positive"] += 1
            elif negative_score > positive_score:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1

        # Convert to percentages
        total = len(messages)
        if total > 0:
            return {k: (v / total) * 100 for k, v in sentiment_counts.items()}

        return sentiment_counts

    def _find_urgent_messages(self, messages: List[Dict]) -> List[Dict]:
        """Find messages with urgency indicators"""
        urgent_messages = []

        for message in messages:
            text = message.get("text", message.get("body", "")).lower()
            urgency_score = sum(1 for word in self.sentiment_keywords["urgent"] if word in text)

            if urgency_score > 0:
                urgent_messages.append({
                    "message_id": message.get("id", message.get("ts", "")),
                    "text": message.get("text", message.get("body", ""))[:200] + "...",
                    "user": message.get("user_name", message.get("from", "")),
                    "urgency_score": urgency_score,
                    "timestamp": message.get("timestamp", message.get("created_datetime", ""))
                })

        return sorted(urgent_messages, key=lambda x: x["urgency_score"], reverse=True)[:10]

    def _count_questions(self, messages: List[Dict]) -> int:
        """Count questions in messages"""
        question_count = 0

        for message in messages:
            text = message.get("text", message.get("body", ""))

            # Count question marks
            question_count += text.count("?")

            # Count question words at start of sentences
            sentences = text.split(".")
            for sentence in sentences:
                sentence = sentence.strip().lower()
                if any(sentence.startswith(word) for word in ["how", "what", "when", "where", "why", "who", "can", "could", "would", "should"]):
                    question_count += 1

        return question_count

    def _identify_topics(self, messages: List[Dict]) -> List[Dict]:
        """Identify main topics discussed"""
        # Simple keyword-based topic identification
        word_freq = Counter()

        # Common stop words to ignore
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "o", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their"
        }

        for message in messages:
            text = message.get("text", message.get("body", "")).lower()
            words = re.findall(r'\b\w+\b', text)

            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] += 1

        # Get top topics
        top_words = word_freq.most_common(10)

        topics = []
        for word, count in top_words:
            topics.append({
                "topic": word,
                "frequency": count,
                "relevance_score": count / len(messages)
            })

        return topics

    def _analyze_communication_patterns(self, messages: List[Dict]) -> Dict:
        """Analyze communication patterns"""
        patterns = {
            "message_distribution": {},
            "peak_hours": [],
            "response_chains": 0,
            "broadcast_vs_discussion": {}
        }

        # Analyze message distribution by hour
        hour_counts = defaultdict(int)

        for message in messages:
            timestamp_str = message.get("timestamp", message.get("created_datetime", ""))
            if timestamp_str:
                try:
                    # Parse timestamp
                    if "T" in timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(float(timestamp_str))

                    hour_counts[dt.hour] += 1
                except Exception:
                    continue

        patterns["message_distribution"] = dict(hour_counts)

        # Find peak hours
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            patterns["peak_hours"] = [hour for hour, count in sorted_hours[:3]]

        return patterns

    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Generate key insights from analysis"""
        insights = []

        # Sentiment insights
        sentiment = analysis.get("sentiment_distribution", {})
        if sentiment.get("positive", 0) > 60:
            insights.append("Communication tone is predominantly positive")
        elif sentiment.get("negative", 0) > 30:
            insights.append("High negative sentiment detected - may need attention")

        # Urgency insights
        urgent_count = len(analysis.get("urgency_indicators", []))
        if urgent_count > 5:
            insights.append(f"{urgent_count} urgent messages require immediate attention")

        # Question insights
        question_count = analysis.get("question_count", 0)
        total_messages = analysis.get("total_messages", 1)
        if question_count / total_messages > 0.3:
            insights.append("High question-to-statement ratio indicates active discussion")

        # Topic insights
        topics = analysis.get("topic_clusters", [])
        if topics:
            top_topic = topics[0]["topic"]
            insights.append(f"Primary discussion topic: '{top_topic}'")

        return insights

    def _analyze_frequency_patterns(self, messages: List[Dict]) -> Dict:
        """Analyze message frequency patterns"""
        frequency = {
            "messages_per_day": {},
            "average_daily": 0,
            "peak_days": [],
            "quiet_periods": []
        }

        # Group messages by date
        daily_counts = defaultdict(int)

        for message in messages:
            timestamp_str = message.get("timestamp", message.get("created_datetime", ""))
            if timestamp_str:
                try:
                    if "T" in timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(float(timestamp_str))

                    date_key = dt.date().isoformat()
                    daily_counts[date_key] += 1
                except Exception:
                    continue

        frequency["messages_per_day"] = dict(daily_counts)

        if daily_counts:
            frequency["average_daily"] = sum(daily_counts.values()) / len(daily_counts)

            # Find peak days
            sorted_days = sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)
            frequency["peak_days"] = sorted_days[:3]

        return frequency

    def _find_preferred_times(self, messages: List[Dict]) -> List[int]:
        """Find user's preferred communication times"""
        hour_counts = defaultdict(int)

        for message in messages:
            timestamp_str = message.get("timestamp", message.get("created_datetime", ""))
            if timestamp_str:
                try:
                    if "T" in timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(float(timestamp_str))

                    hour_counts[dt.hour] += 1
                except Exception:
                    continue

        # Return top 3 preferred hours
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            return [hour for hour, count in sorted_hours[:3]]

        return []

    def _analyze_response_patterns(self, user_messages: List[Dict], all_messages: List[Dict]) -> Dict:
        """Analyze user's response patterns"""
        patterns = {
            "average_response_time": 0,
            "response_rate": 0,
            "initiates_conversations": 0,
            "responds_to_others": 0
        }

        # This would require more sophisticated threading analysis
        # For now, return basic patterns
        patterns["response_rate"] = len(user_messages) / len(all_messages) if all_messages else 0

        return patterns

    def _analyze_language_style(self, messages: List[Dict]) -> Dict:
        """Analyze user's language style"""
        style = {
            "formality_level": "neutral",
            "avg_sentence_length": 0,
            "uses_emojis": False,
            "technical_language": False,
            "question_frequency": 0
        }

        total_sentences = 0
        total_length = 0
        question_count = 0
        emoji_count = 0

        technical_terms = ["api", "database", "algorithm", "framework", "deployment", "configuration"]

        for message in messages:
            text = message.get("text", message.get("body", ""))

            # Count sentences
            sentences = text.split(".")
            total_sentences += len(sentences)
            total_length += len(text)

            # Count questions
            question_count += text.count("?")

            # Check for emojis (simple check)
            emoji_count += len(re.findall(r'[😀-🿿]', text))

            # Check for technical terms
            if any(term in text.lower() for term in technical_terms):
                style["technical_language"] = True

        if total_sentences > 0:
            style["avg_sentence_length"] = total_length / total_sentences

        style["question_frequency"] = question_count / len(messages) if messages else 0
        style["uses_emojis"] = emoji_count > 0

        return style

    def _determine_interaction_style(self, style_analysis: Dict) -> str:
        """Determine overall interaction style"""
        avg_length = style_analysis.get("avg_message_length", 0)
        question_freq = style_analysis.get("language_style", {}).get("question_frequency", 0)

        if avg_length > 200 and question_freq > 0.2:
            return "detailed_inquisitive"
        elif avg_length > 200:
            return "detailed_informative"
        elif question_freq > 0.3:
            return "concise_inquisitive"
        elif avg_length < 50:
            return "brief_direct"
        else:
            return "balanced"

    def _analyze_participation_balance(self, messages: List[Dict], team_members: List[str]) -> Dict:
        """Analyze participation balance among team members"""
        participation = {}

        for member in team_members:
            participation[member] = 0

        for message in messages:
            user_id = message.get("user_id") or message.get("from_id")
            if user_id in participation:
                participation[user_id] += 1

        total_messages = sum(participation.values())

        # Calculate participation percentages
        if total_messages > 0:
            participation_pct = {k: (v / total_messages) * 100 for k, v in participation.items()}
        else:
            participation_pct = participation

        return {
            "raw_counts": participation,
            "percentages": participation_pct,
            "balance_score": self._calculate_balance_score(participation_pct)
        }

    def _calculate_balance_score(self, participation_pct: Dict) -> float:
        """Calculate how balanced participation is (0-1, where 1 is perfectly balanced)"""
        if not participation_pct:
            return 0.0

        values = list(participation_pct.values())
        if not values:
            return 0.0

        # Calculate standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        # Convert to balance score (lower std_dev = higher balance)
        max_possible_std = mean  # Maximum when one person has all messages
        balance_score = 1 - (std_dev / max_possible_std) if max_possible_std > 0 else 1

        return max(0, min(1, balance_score))

    def _analyze_response_times(self, messages: List[Dict]) -> Dict:
        """Analyze response time patterns"""
        # This would require message threading analysis
        # For now, return placeholder
        return {
            "average_response_time_hours": 2.5,
            "fastest_responders": [],
            "response_time_distribution": {}
        }

    def _find_collaboration_indicators(self, messages: List[Dict]) -> Dict:
        """Find indicators of good collaboration"""
        indicators = {
            "mentions_count": 0,
            "questions_answered": 0,
            "shared_resources": 0,
            "collaborative_language": 0
        }

        collaborative_words = ["let's", "we should", "together", "collaborate", "team", "share", "help"]

        for message in messages:
            text = message.get("text", message.get("body", "")).lower()

            # Count mentions (@user)
            indicators["mentions_count"] += text.count("@")

            # Count collaborative language
            indicators["collaborative_language"] += sum(1 for word in collaborative_words if word in text)

            # Count shared resources (links, files)
            indicators["shared_resources"] += text.count("http") + text.count("file:")

        return indicators

    def _identify_communication_gaps(self, messages: List[Dict], team_members: List[str]) -> List[Dict]:
        """Identify communication gaps"""
        gaps = []

        # Find members who haven't participated
        active_members = set()
        for message in messages:
            user_id = message.get("user_id") or message.get("from_id")
            if user_id:
                active_members.add(user_id)

        inactive_members = set(team_members) - active_members

        for member in inactive_members:
            gaps.append({
                "type": "no_participation",
                "member": member,
                "description": "No messages in analyzed period"
            })

        return gaps

    def _analyze_team_dynamics(self, messages: List[Dict], team_members: List[str]) -> Dict:
        """Analyze team dynamics"""
        dynamics = {
            "interaction_matrix": {},
            "leadership_indicators": {},
            "collaboration_score": 0.0
        }

        # Simple collaboration score based on participation balance and collaborative language
        participation = self._analyze_participation_balance(messages, team_members)
        collaboration_indicators = self._find_collaboration_indicators(messages)

        balance_score = participation.get("balance_score", 0)
        collab_language_score = min(1.0, collaboration_indicators.get("collaborative_language", 0) / 10)

        dynamics["collaboration_score"] = (balance_score + collab_language_score) / 2

        return dynamics

    def _calculate_team_health_score(self, health_analysis: Dict) -> float:
        """Calculate overall team communication health score"""
        scores = []

        # Participation balance score
        participation = health_analysis.get("participation_balance", {})
        balance_score = participation.get("balance_score", 0)
        scores.append(balance_score)

        # Collaboration score
        team_dynamics = health_analysis.get("team_dynamics", {})
        collab_score = team_dynamics.get("collaboration_score", 0)
        scores.append(collab_score)

        # Communication gaps penalty
        gaps = health_analysis.get("communication_gaps", [])
        gap_penalty = min(0.5, len(gaps) * 0.1)  # Max 50% penalty

        # Calculate average and apply penalty
        if scores:
            base_score = sum(scores) / len(scores)
            final_score = max(0, base_score - gap_penalty)
        else:
            final_score = 0

        return final_score

    def _extract_action_from_text(self, message: Dict, keyword: str) -> Optional[Dict]:
        """Extract action item from message text"""
        text = message.get("text", message.get("body", ""))

        # Simple extraction - find sentence containing the keyword
        sentences = text.split(".")
        for sentence in sentences:
            if keyword in sentence.lower():
                return {
                    "text": sentence.strip(),
                    "source_message_id": message.get("id", message.get("ts", "")),
                    "user": message.get("user_name", message.get("from", "")),
                    "timestamp": message.get("timestamp", message.get("created_datetime", "")),
                    "keyword": keyword,
                    "priority_score": self._calculate_action_priority(sentence)
                }

        return None

    def _calculate_action_priority(self, text: str) -> float:
        """Calculate priority score for action item"""
        high_priority_words = ["urgent", "asap", "critical", "deadline", "immediately"]
        medium_priority_words = ["should", "need", "important", "soon"]

        text_lower = text.lower()

        high_score = sum(1 for word in high_priority_words if word in text_lower)
        medium_score = sum(1 for word in medium_priority_words if word in text_lower)

        return high_score * 1.0 + medium_score * 0.5

    def _deduplicate_actions(self, action_items: List[Dict]) -> List[Dict]:
        """Remove duplicate action items"""
        seen_texts = set()
        unique_actions = []

        for action in action_items:
            text = action.get("text", "").lower().strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_actions.append(action)

        return unique_actions

    def _extract_decisions(self, messages: List[Dict]) -> List[Dict]:
        """Extract key decisions from messages"""
        decisions = []
        decision_keywords = ["decided", "decision", "agreed", "concluded", "resolved", "final"]

        for message in messages:
            text = message.get("text", message.get("body", "")).lower()

            for keyword in decision_keywords:
                if keyword in text:
                    decisions.append({
                        "text": message.get("text", message.get("body", ""))[:200] + "...",
                        "user": message.get("user_name", message.get("from", "")),
                        "timestamp": message.get("timestamp", message.get("created_datetime", "")),
                        "keyword": keyword
                    })
                    break

        return decisions[:10]  # Return top 10 decisions

    def _generate_summary_text(self, summary: Dict) -> str:
        """Generate human-readable summary text"""
        text_parts = []

        text_parts.append(f"Communication Summary for {summary['period']}")
        text_parts.append(f"Total messages: {summary['total_messages']}")
        text_parts.append(f"Unique participants: {summary['unique_participants']}")

        if summary["top_topics"]:
            top_topic = summary["top_topics"][0]["topic"]
            text_parts.append(f"Primary topic: {top_topic}")

        if summary["urgent_items"]:
            text_parts.append(f"Urgent items: {len(summary['urgent_items'])}")

        if summary["action_items"]:
            text_parts.append(f"Action items identified: {len(summary['action_items'])}")

        return ". ".join(text_parts) + "."
