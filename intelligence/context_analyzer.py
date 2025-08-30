import asyncio
import logging
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
from typing import Dict, List

logger = logging.getLogger(__name__)

class ContextAnalyzer:
    """Advanced context analysis and pattern recognition"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.pattern_cache = {}
        self.user_behavior_patterns = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        self.task_sequences = []

    async def analyze_user_context(self, user_input: str, current_context: Dict,
                                 session_history: List[Dict]) -> Dict:
        """Comprehensive context analysis for better understanding"""
        try:
            analysis = {
                "user_intent_confidence": 0.0,
                "context_relevance": {},
                "behavioral_patterns": {},
                "temporal_context": {},
                "suggested_actions": [],
                "proactive_suggestions": [],
                "learning_insights": {},
                "complexity_score": 0.0
            }

            # Analyze user intent with historical context
            analysis["user_intent_confidence"] = await self._analyze_intent_confidence(
                user_input, session_history
            )

            # Analyze context relevance
            analysis["context_relevance"] = await self._analyze_context_relevance(
                user_input, current_context
            )

            # Detect behavioral patterns
            analysis["behavioral_patterns"] = await self._detect_behavioral_patterns(
                user_input, session_history
            )

            # Analyze temporal context
            analysis["temporal_context"] = await self._analyze_temporal_context(
                user_input, current_context, session_history
            )

            # Generate proactive suggestions
            analysis["suggested_actions"] = await self._generate_action_suggestions(
                user_input, analysis
            )

            # Generate proactive suggestions based on patterns
            analysis["proactive_suggestions"] = await self._generate_proactive_suggestions(
                analysis, current_context
            )

            # Extract learning insights
            analysis["learning_insights"] = await self._extract_learning_insights(
                user_input, session_history, analysis
            )

            # Calculate task complexity
            analysis["complexity_score"] = self._calculate_complexity_score(
                user_input, analysis
            )

            return analysis

        except Exception as e:
            logger.error("Context analysis failed: %s", e)
            return {"error": str(e)}

    async def _analyze_intent_confidence(self, user_input: str,
                                       session_history: List[Dict]) -> float:
        """Analyze confidence in understanding user intent"""
        try:
            confidence_factors = []

            # Factor 1: Clarity of language
            clarity_score = self._assess_language_clarity(user_input)
            confidence_factors.append(("clarity", clarity_score, 0.3))

            # Factor 2: Specificity of request
            specificity_score = self._assess_request_specificity(user_input)
            confidence_factors.append(("specificity", specificity_score, 0.25))

            # Factor 3: Historical context match
            if session_history:
                history_match = await self._assess_historical_match(user_input, session_history)
                confidence_factors.append(("history_match", history_match, 0.2))

            # Factor 4: Domain knowledge coverage
            domain_coverage = self._assess_domain_coverage(user_input)
            confidence_factors.append(("domain_coverage", domain_coverage, 0.25))

            # Calculate weighted confidence
            total_confidence = sum(score * weight for _, score, weight in confidence_factors)

            return min(max(total_confidence, 0.0), 1.0)

        except Exception as e:
            logger.error("Intent confidence analysis failed: %s", e)
            return 0.5

    async def _analyze_context_relevance(self, user_input: str,
                                       current_context: Dict) -> Dict:
        """Analyze relevance of different context elements"""
        relevance = {
            "system_status": 0.0,
            "active_apps": 0.0,
            "recent_files": 0.0,
            "time_context": 0.0,
            "location_context": 0.0
        }

        user_lower = user_input.lower()

        # System status relevance
        if any(word in user_lower for word in ["system", "performance", "cpu", "memory", "status"]):
            relevance["system_status"] = 0.9

        # Active apps relevance
        active_apps = current_context.get("active_apps", [])
        for app in active_apps:
            if app.lower() in user_lower:
                relevance["active_apps"] = 0.8
                break

        # Recent files relevance
        if any(word in user_lower for word in ["file", "document", "recent", "open"]):
            relevance["recent_files"] = 0.7

        # Time context relevance
        if any(word in user_lower for word in ["today", "tomorrow", "schedule", "calendar", "time"]):
            relevance["time_context"] = 0.8

        return relevance

    async def _detect_behavioral_patterns(self, user_input: str,
                                        session_history: List[Dict]) -> Dict:
        """Detect user behavioral patterns"""
        patterns = {
            "preferred_actions": [],
            "common_workflows": [],
            "time_patterns": {},
            "interaction_style": "unknown",
            "expertise_level": "intermediate"
        }

        if not session_history:
            return patterns

        # Analyze preferred actions
        action_counts = Counter()
        for interaction in session_history[-20:]:  # Last 20 interactions
            intention = interaction.get("intention", {})
            primary_action = intention.get("primary_action")
            if primary_action:
                action_counts[primary_action] += 1

        patterns["preferred_actions"] = [
            {"action": action, "frequency": count}
            for action, count in action_counts.most_common(5)
        ]

        # Detect common workflows
        patterns["common_workflows"] = self._detect_workflow_patterns(session_history)

        # Analyze time patterns
        patterns["time_patterns"] = self._analyze_time_patterns(session_history)

        # Assess interaction style
        patterns["interaction_style"] = self._assess_interaction_style(session_history)

        # Assess expertise level
        patterns["expertise_level"] = self._assess_expertise_level(session_history)

        return patterns

    async def _analyze_temporal_context(self, user_input: str, current_context: Dict,
                                      session_history: List[Dict]) -> Dict:
        """Analyze temporal context and patterns"""
        temporal = {
            "current_time_relevance": 0.0,
            "session_duration": 0.0,
            "interaction_frequency": 0.0,
            "time_based_suggestions": [],
            "urgency_indicators": []
        }

        current_time = datetime.now()

        # Time relevance
        time_keywords = ["now", "today", "tonight", "tomorrow", "urgent", "asap", "quickly"]
        if any(keyword in user_input.lower() for keyword in time_keywords):
            temporal["current_time_relevance"] = 0.8

        # Session duration
        if session_history:
            first_interaction = session_history[0].get("timestamp")
            if first_interaction:
                try:
                    start_time = datetime.fromisoformat(first_interaction.replace('Z', '+00:00'))
                    temporal["session_duration"] = (current_time - start_time).total_seconds() / 60
                except Exception:
                    pass

        # Interaction frequency
        if len(session_history) > 1:
            temporal["interaction_frequency"] = len(session_history) / max(temporal["session_duration"], 1)

        # Urgency indicators
        urgency_words = ["urgent", "asap", "emergency", "critical", "immediately", "now"]
        temporal["urgency_indicators"] = [
            word for word in urgency_words if word in user_input.lower()
        ]

        # Time-based suggestions
        hour = current_time.hour
        if 9 <= hour <= 17:  # Work hours
            temporal["time_based_suggestions"].append("work_productivity")
        elif 17 <= hour <= 22:  # Evening
            temporal["time_based_suggestions"].append("personal_tasks")

        return temporal

    async def _generate_action_suggestions(self, user_input: str,
                                         analysis: Dict) -> List[Dict]:
        """Generate contextual action suggestions"""
        suggestions = []

        # Based on behavioral patterns
        preferred_actions = analysis.get("behavioral_patterns", {}).get("preferred_actions", [])
        for action_info in preferred_actions[:3]:
            action = action_info["action"]
            suggestions.append({
                "type": "behavioral",
                "action": action,
                "reason": f"You frequently use {action}",
                "confidence": 0.7
            })

        # Based on context relevance
        context_relevance = analysis.get("context_relevance", {})
        for context_type, relevance in context_relevance.items():
            if relevance > 0.7:
                suggestions.append({
                    "type": "contextual",
                    "action": f"analyze_{context_type}",
                    "reason": f"High relevance to {context_type}",
                    "confidence": relevance
                })

        # Based on temporal context
        temporal = analysis.get("temporal_context", {})
        if temporal.get("urgency_indicators"):
            suggestions.append({
                "type": "temporal",
                "action": "prioritize_urgent",
                "reason": "Urgency detected in request",
                "confidence": 0.9
            })

        return suggestions[:5]  # Limit to top 5

    async def _generate_proactive_suggestions(self, analysis: Dict,
                                            current_context: Dict) -> List[Dict]:
        """Generate proactive suggestions based on patterns and context"""
        suggestions = []

        # System health suggestions
        system_status = current_context.get("system_status", {})
        cpu_percent = system_status.get("cpu_percent", 0)
        memory_percent = system_status.get("memory_percent", 0)

        if cpu_percent > 80:
            suggestions.append({
                "type": "system_health",
                "suggestion": "High CPU usage detected. Would you like me to identify resource-heavy processes?",
                "priority": "high",
                "action": "analyze_system_performance"
            })

        if memory_percent > 85:
            suggestions.append({
                "type": "system_health",
                "suggestion": "Memory usage is high. Consider closing unused applications.",
                "priority": "medium",
                "action": "optimize_memory_usage"
            })

        # Workflow suggestions based on patterns
        behavioral_patterns = analysis.get("behavioral_patterns", {})
        common_workflows = behavioral_patterns.get("common_workflows", [])

        for workflow in common_workflows[:2]:
            suggestions.append({
                "type": "workflow",
                "suggestion": f"Based on your patterns, you might want to {workflow['description']}",
                "priority": "low",
                "action": workflow["next_action"]
            })

        # Time-based suggestions
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11:  # Morning
            suggestions.append({
                "type": "temporal",
                "suggestion": "Good morning! Would you like me to check your calendar and emails for today?",
                "priority": "medium",
                "action": "morning_briefing"
            })

        return suggestions[:3]  # Limit to top 3

    async def _extract_learning_insights(self, user_input: str, session_history: List[Dict],
                                       analysis: Dict) -> Dict:
        """Extract insights for continuous learning"""
        insights = {
            "new_patterns": [],
            "preference_updates": {},
            "skill_level_indicators": [],
            "domain_interests": [],
            "interaction_improvements": []
        }

        # Detect new patterns
        if len(session_history) >= 5:
            recent_actions = [
                h.get("intention", {}).get("primary_action")
                for h in session_history[-5:]
            ]
            if len(set(recent_actions)) == 1 and recent_actions[0]:  # Same action 5 times
                insights["new_patterns"].append({
                    "pattern": "repeated_action",
                    "action": recent_actions[0],
                    "frequency": 5
                })

        # Extract domain interests
        domain_keywords = {
            "technology": ["ai", "machine learning", "programming", "software", "tech"],
            "business": ["market", "finance", "strategy", "management", "business"],
            "science": ["research", "study", "analysis", "data", "experiment"],
            "creative": ["design", "art", "creative", "writing", "content"]
        }

        user_lower = user_input.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                insights["domain_interests"].append(domain)

        # Skill level indicators
        technical_terms = ["api", "database", "algorithm", "framework", "architecture"]
        if any(term in user_lower for term in technical_terms):
            insights["skill_level_indicators"].append("technical_proficiency")

        return insights

    def _assess_language_clarity(self, user_input: str) -> float:
        """Assess clarity of user language"""
        # Simple heuristics for language clarity
        score = 0.5  # Base score

        # Length factor
        word_count = len(user_input.split())
        if 3 <= word_count <= 20:
            score += 0.2
        elif word_count > 20:
            score -= 0.1

        # Question marks and clear intent
        if '?' in user_input:
            score += 0.1

        # Action words
        action_words = ["create", "make", "find", "search", "analyze", "check", "show"]
        if any(word in user_input.lower() for word in action_words):
            score += 0.2

        return min(score, 1.0)

    def _assess_request_specificity(self, user_input: str) -> float:
        """Assess how specific the user request is"""
        score = 0.3  # Base score

        # Specific nouns and entities
        specific_indicators = ["about", "for", "in", "with", "from", "to"]
        score += sum(0.1 for indicator in specific_indicators if indicator in user_input.lower())

        # Numbers and dates
        if re.search(r'\d+', user_input):
            score += 0.2

        # Proper nouns (capitalized words)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', user_input)
        if capitalized_words:
            score += 0.2

        return min(score, 1.0)

    async def _assess_historical_match(self, user_input: str,
                                     session_history: List[Dict]) -> float:
        """Assess how well the request matches historical patterns"""
        if not session_history:
            return 0.5

        # Simple similarity check with recent interactions
        user_words = set(user_input.lower().split())

        similarity_scores = []
        for interaction in session_history[-10:]:  # Last 10 interactions
            past_input = interaction.get("user_input", "")
            past_words = set(past_input.lower().split())

            if past_words:
                similarity = len(user_words & past_words) / len(user_words | past_words)
                similarity_scores.append(similarity)

        return max(similarity_scores) if similarity_scores else 0.5

    def _assess_domain_coverage(self, user_input: str) -> float:
        """Assess how well we can handle the domain of the request"""
        # Define domains we handle well
        strong_domains = {
            "research": ["research", "find", "search", "investigate", "study"],
            "system": ["system", "computer", "performance", "status", "health"],
            "email": ["email", "mail", "inbox", "message"],
            "files": ["file", "folder", "directory", "document"],
            "calendar": ["calendar", "schedule", "meeting", "appointment"],
            "automation": ["automate", "workflow", "task", "process"]
        }

        user_lower = user_input.lower()
        domain_matches = []

        for domain, keywords in strong_domains.items():
            match_score = sum(1 for keyword in keywords if keyword in user_lower)
            if match_score > 0:
                domain_matches.append(match_score / len(keywords))

        return max(domain_matches) if domain_matches else 0.4

    def _detect_workflow_patterns(self, session_history: List[Dict]) -> List[Dict]:
        """Detect common workflow patterns"""
        workflows = []

        if len(session_history) < 3:
            return workflows

        # Look for sequences of actions
        action_sequences = []
        for i in range(len(session_history) - 2):
            sequence = [
                session_history[i].get("intention", {}).get("primary_action"),
                session_history[i+1].get("intention", {}).get("primary_action"),
                session_history[i+2].get("intention", {}).get("primary_action")
            ]
            if all(action for action in sequence):
                action_sequences.append(tuple(sequence))

        # Find common sequences
        sequence_counts = Counter(action_sequences)
        for sequence, count in sequence_counts.most_common(3):
            if count >= 2:  # Appeared at least twice
                workflows.append({
                    "sequence": list(sequence),
                    "frequency": count,
                    "description": f"Often do {' → '.join(sequence)}",
                    "next_action": sequence[-1]  # Suggest continuing the pattern
                })

        return workflows

    def _analyze_time_patterns(self, session_history: List[Dict]) -> Dict:
        """Analyze temporal usage patterns"""
        patterns = {
            "peak_hours": [],
            "session_lengths": [],
            "interaction_gaps": []
        }

        hour_counts = defaultdict(int)
        session_durations = []

        for interaction in session_history:
            timestamp_str = interaction.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    hour_counts[timestamp.hour] += 1
                except Exception:
                    continue

        # Find peak hours
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            patterns["peak_hours"] = [hour for hour, count in sorted_hours[:3]]

        return patterns

    def _assess_interaction_style(self, session_history: List[Dict]) -> str:
        """Assess user's interaction style"""
        if not session_history:
            return "unknown"

        # Analyze recent interactions
        recent_inputs = [
            h.get("user_input", "") for h in session_history[-10:]
        ]

        avg_length = sum(len(inp.split()) for inp in recent_inputs) / len(recent_inputs)
        question_ratio = sum(1 for inp in recent_inputs if '?' in inp) / len(recent_inputs)

        if avg_length < 5 and question_ratio < 0.3:
            return "concise"
        elif avg_length > 15 or question_ratio > 0.7:
            return "detailed"
        else:
            return "balanced"

    def _assess_expertise_level(self, session_history: List[Dict]) -> str:
        """Assess user's technical expertise level"""
        if not session_history:
            return "intermediate"

        technical_indicators = 0
        total_interactions = len(session_history)

        technical_terms = [
            "api", "database", "algorithm", "framework", "architecture",
            "configuration", "optimization", "integration", "deployment"
        ]

        for interaction in session_history:
            user_input = interaction.get("user_input", "").lower()
            if any(term in user_input for term in technical_terms):
                technical_indicators += 1

        technical_ratio = technical_indicators / total_interactions

        if technical_ratio > 0.3:
            return "advanced"
        elif technical_ratio > 0.1:
            return "intermediate"
        else:
            return "beginner"

    def _calculate_complexity_score(self, user_input: str, analysis: Dict) -> float:
        """Calculate task complexity score"""
        complexity = 0.0

        # Base complexity from word count
        word_count = len(user_input.split())
        complexity += min(word_count / 20, 0.3)

        # Multiple actions or steps
        action_words = ["and", "then", "after", "also", "plus"]
        if any(word in user_input.lower() for word in action_words):
            complexity += 0.3

        # Technical terms
        technical_terms = ["integrate", "configure", "optimize", "analyze", "process"]
        if any(term in user_input.lower() for term in technical_terms):
            complexity += 0.2

        # Conditional logic
        conditional_words = ["i", "when", "unless", "depending"]
        if any(word in user_input.lower() for word in conditional_words):
            complexity += 0.2

        return min(complexity, 1.0)
