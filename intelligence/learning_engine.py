import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LearningEngine:
    """Advanced learning system that adapts to user behavior and preferences"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.user_model = {
            "preferences": {},
            "skills": {},
            "patterns": {},
            "goals": [],
            "context_history": []
        }
        self.adaptation_rules = []
        self.learning_metrics = {
            "interactions_processed": 0,
            "patterns_learned": 0,
            "adaptations_made": 0,
            "accuracy_improvements": []
        }

    async def learn_from_interaction(self, user_input: str, response: str,
                                   context: Dict, feedback: Optional[Dict] = None) -> Dict:
        """Learn from a single user interaction"""
        try:
            learning_results = {
                "patterns_updated": [],
                "preferences_learned": [],
                "skills_assessed": [],
                "adaptations_made": [],
                "confidence_changes": {}
            }

            # Update interaction count
            self.learning_metrics["interactions_processed"] += 1

            # Learn patterns from the interaction
            patterns = await self._extract_interaction_patterns(user_input, response, context)
            learning_results["patterns_updated"] = patterns

            # Learn user preferences
            preferences = await self._learn_user_preferences(user_input, response, context)
            learning_results["preferences_learned"] = preferences

            # Assess user skills and knowledge
            skills = await self._assess_user_skills(user_input, context)
            learning_results["skills_assessed"] = skills

            # Apply adaptations based on learning
            adaptations = await self._apply_learning_adaptations(learning_results)
            learning_results["adaptations_made"] = adaptations

            # Update confidence scores
            confidence_changes = await self._update_confidence_scores(
                user_input, response, feedback
            )
            learning_results["confidence_changes"] = confidence_changes

            # Store learning results
            await self._store_learning_results(learning_results)

            return learning_results

        except Exception as e:
            logger.error("Learning from interaction failed: %s", e)
            return {"error": str(e)}

    async def predict_user_needs(self, current_context: Dict,
                               session_history: List[Dict]) -> Dict:
        """Predict what the user might need based on learned patterns"""
        try:
            predictions = {
                "likely_next_actions": [],
                "proactive_suggestions": [],
                "context_predictions": {},
                "timing_predictions": {},
                "confidence_scores": {}
            }

            # Predict likely next actions
            predictions["likely_next_actions"] = await self._predict_next_actions(
                current_context, session_history
            )

            # Generate proactive suggestions
            predictions["proactive_suggestions"] = await self._generate_proactive_suggestions(
                current_context, session_history
            )

            # Predict context needs
            predictions["context_predictions"] = await self._predict_context_needs(
                current_context, session_history
            )

            # Predict optimal timing
            predictions["timing_predictions"] = await self._predict_optimal_timing(
                session_history
            )

            # Calculate confidence scores
            predictions["confidence_scores"] = self._calculate_prediction_confidence(
                predictions
            )

            return predictions

        except Exception as e:
            logger.error("User needs prediction failed: %s", e)
            return {"error": str(e)}

    async def adapt_response_style(self, user_input: str, context: Dict) -> Dict:
        """Adapt response style based on learned user preferences"""
        try:
            adaptations = {
                "communication_style": "balanced",
                "detail_level": "medium",
                "technical_level": "intermediate",
                "response_length": "medium",
                "formatting_preferences": {},
                "tone_adjustments": {}
            }

            # Analyze user's communication style preference
            adaptations["communication_style"] = self._determine_communication_style()

            # Determine preferred detail level
            adaptations["detail_level"] = self._determine_detail_preference()

            # Assess technical level preference
            adaptations["technical_level"] = self._determine_technical_level()

            # Determine response length preference
            adaptations["response_length"] = self._determine_length_preference()

            # Learn formatting preferences
            adaptations["formatting_preferences"] = self._learn_formatting_preferences()

            # Adjust tone based on context
            adaptations["tone_adjustments"] = self._determine_tone_adjustments(context)

            return adaptations

        except Exception as e:
            logger.error("Response style adaptation failed: %s", e)
            return {"communication_style": "balanced"}

    async def optimize_workflow_suggestions(self, current_task: str,
                                          context: Dict) -> List[Dict]:
        """Optimize workflow suggestions based on learned patterns"""
        try:
            suggestions = []

            # Get learned workflow patterns
            workflow_patterns = self.user_model.get("patterns", {}).get("workflows", [])

            # Find relevant patterns for current task
            relevant_patterns = [
                pattern for pattern in workflow_patterns
                if self._is_pattern_relevant(pattern, current_task, context)
            ]

            # Generate optimized suggestions
            for pattern in relevant_patterns[:5]:
                suggestion = {
                    "workflow": pattern["sequence"],
                    "confidence": pattern["confidence"],
                    "estimated_time": pattern.get("avg_duration", 0),
                    "success_rate": pattern.get("success_rate", 0.8),
                    "optimization_tips": pattern.get("optimizations", [])
                }
                suggestions.append(suggestion)

            # Add novel suggestions based on similar contexts
            novel_suggestions = await self._generate_novel_workflow_suggestions(
                current_task, context
            )
            suggestions.extend(novel_suggestions)

            # Sort by confidence and relevance
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)

            return suggestions[:3]  # Return top 3

        except Exception as e:
            logger.error("Workflow optimization failed: %s", e)
            return []

    async def _extract_interaction_patterns(self, user_input: str, response: str,
                                          context: Dict) -> List[Dict]:
        """Extract patterns from user interactions"""
        patterns = []

        # Time-based patterns
        current_time = datetime.now()
        time_pattern = {
            "type": "temporal",
            "hour": current_time.hour,
            "day_of_week": current_time.weekday(),
            "action": self._extract_primary_action(user_input),
            "context_type": self._classify_context_type(context)
        }
        patterns.append(time_pattern)

        # Language patterns
        language_pattern = {
            "type": "language",
            "input_length": len(user_input.split()),
            "question_type": self._classify_question_type(user_input),
            "formality_level": self._assess_formality_level(user_input),
            "technical_terms": self._extract_technical_terms(user_input)
        }
        patterns.append(language_pattern)

        # Task complexity patterns
        complexity_pattern = {
            "type": "complexity",
            "task_complexity": self._assess_task_complexity(user_input),
            "multi_step": self._is_multi_step_task(user_input),
            "domain": self._identify_domain(user_input),
            "success_indicators": self._extract_success_indicators(response)
        }
        patterns.append(complexity_pattern)

        return patterns

    async def _learn_user_preferences(self, user_input: str, response: str,
                                    context: Dict) -> List[Dict]:
        """Learn user preferences from interactions"""
        preferences = []

        # Response format preferences
        if "list" in user_input.lower() or "bullet" in user_input.lower():
            preferences.append({
                "type": "format",
                "preference": "lists",
                "confidence": 0.7
            })

        # Detail level preferences
        if any(word in user_input.lower() for word in ["detailed", "comprehensive", "thorough"]):
            preferences.append({
                "type": "detail_level",
                "preference": "high",
                "confidence": 0.8
            })
        elif any(word in user_input.lower() for word in ["brie", "quick", "summary"]):
            preferences.append({
                "type": "detail_level",
                "preference": "low",
                "confidence": 0.8
            })

        # Domain preferences
        domain = self._identify_domain(user_input)
        if domain != "general":
            preferences.append({
                "type": "domain_interest",
                "preference": domain,
                "confidence": 0.6
            })

        # Update user model with learned preferences
        for pref in preferences:
            self._update_user_preference(pref)

        return preferences

    async def _assess_user_skills(self, user_input: str, context: Dict) -> List[Dict]:
        """Assess user skills and knowledge level"""
        skills = []

        # Technical skill assessment
        technical_terms = self._extract_technical_terms(user_input)
        if technical_terms:
            skill_level = "advanced" if len(technical_terms) > 3 else "intermediate"
            skills.append({
                "domain": "technical",
                "level": skill_level,
                "evidence": technical_terms,
                "confidence": 0.7
            })

        # Domain expertise assessment
        domain = self._identify_domain(user_input)
        if domain != "general":
            # Assess based on question sophistication
            sophistication = self._assess_question_sophistication(user_input)
            skills.append({
                "domain": domain,
                "level": sophistication,
                "evidence": [user_input[:50] + "..."],
                "confidence": 0.6
            })

        # Update user model with skill assessments
        for skill in skills:
            self._update_user_skill(skill)

        return skills

    async def _apply_learning_adaptations(self, learning_results: Dict) -> List[Dict]:
        """Apply adaptations based on learning results"""
        adaptations = []

        # Adapt based on preferences
        for pref in learning_results.get("preferences_learned", []):
            if pref["type"] == "detail_level":
                adaptation = {
                    "type": "response_adaptation",
                    "parameter": "detail_level",
                    "value": pref["preference"],
                    "reason": "Learned user preference"
                }
                adaptations.append(adaptation)
                self.learning_metrics["adaptations_made"] += 1

        # Adapt based on skill assessments
        for skill in learning_results.get("skills_assessed", []):
            if skill["level"] == "advanced":
                adaptation = {
                    "type": "complexity_adaptation",
                    "parameter": "technical_level",
                    "value": "advanced",
                    "reason": f"User shows advanced {skill['domain']} skills"
                }
                adaptations.append(adaptation)
                self.learning_metrics["adaptations_made"] += 1

        return adaptations

    async def _update_confidence_scores(self, user_input: str, response: str,
                                      feedback: Optional[Dict]) -> Dict:
        """Update confidence scores based on interaction success"""
        confidence_changes = {}

        # If explicit feedback is provided
        if feedback:
            if feedback.get("helpful", False):
                confidence_changes["overall"] = 0.1  # Increase confidence
            else:
                confidence_changes["overall"] = -0.1  # Decrease confidence

        # Implicit feedback from user behavior
        # (This would be enhanced with actual user behavior tracking)

        return confidence_changes

    async def _predict_next_actions(self, current_context: Dict,
                                  session_history: List[Dict]) -> List[Dict]:
        """Predict likely next actions based on patterns"""
        predictions = []

        if not session_history:
            return predictions

        # Analyze recent action sequences
        recent_actions = [
            h.get("intention", {}).get("primary_action")
            for h in session_history[-5:]
            if h.get("intention", {}).get("primary_action")
        ]

        # Look for common follow-up actions
        action_transitions = self.user_model.get("patterns", {}).get("action_transitions", {})

        if recent_actions:
            last_action = recent_actions[-1]
            if last_action in action_transitions:
                for next_action, probability in action_transitions[last_action].items():
                    predictions.append({
                        "action": next_action,
                        "probability": probability,
                        "reason": f"Often follows {last_action}"
                    })

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)[:3]

    async def _generate_proactive_suggestions(self, current_context: Dict,
                                            session_history: List[Dict]) -> List[Dict]:
        """Generate proactive suggestions based on learned patterns"""
        suggestions = []

        # Time-based suggestions
        current_hour = datetime.now().hour
        time_patterns = self.user_model.get("patterns", {}).get("time_based", {})

        if str(current_hour) in time_patterns:
            common_actions = time_patterns[str(current_hour)]
            for action, frequency in common_actions.items():
                if frequency > 0.3:  # Threshold for suggestion
                    suggestions.append({
                        "type": "time_based",
                        "suggestion": f"You usually {action} around this time",
                        "action": action,
                        "confidence": frequency
                    })

        # Context-based suggestions
        system_status = current_context.get("system_status", {})
        if system_status.get("cpu_percent", 0) > 80:
            suggestions.append({
                "type": "system_health",
                "suggestion": "High CPU usage detected. Would you like me to investigate?",
                "action": "analyze_system_performance",
                "confidence": 0.8
            })

        return suggestions[:3]

    async def _predict_context_needs(self, current_context: Dict,
                                   session_history: List[Dict]) -> Dict:
        """Predict what context information will be most relevant"""
        context_predictions = {
            "system_status": 0.3,
            "active_apps": 0.2,
            "recent_files": 0.2,
            "calendar_events": 0.1,
            "email_summary": 0.2
        }

        # Adjust based on recent interaction patterns
        if session_history:
            recent_actions = [
                h.get("intention", {}).get("primary_action")
                for h in session_history[-3:]
            ]

            if "system_info" in recent_actions:
                context_predictions["system_status"] = 0.8
            if "email_management" in recent_actions:
                context_predictions["email_summary"] = 0.7
            if "calendar_management" in recent_actions:
                context_predictions["calendar_events"] = 0.7

        return context_predictions

    async def _predict_optimal_timing(self, session_history: List[Dict]) -> Dict:
        """Predict optimal timing for different types of interactions"""
        timing_predictions = {
            "best_hours": [],
            "peak_productivity": [],
            "preferred_session_length": 0,
            "interaction_frequency": 0
        }

        if not session_history:
            return timing_predictions

        # Analyze interaction times
        interaction_hours = []
        for interaction in session_history:
            timestamp_str = interaction.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    interaction_hours.append(timestamp.hour)
                except Exception:
                    continue

        if interaction_hours:
            # Find most common hours
            hour_counts = Counter(interaction_hours)
            timing_predictions["best_hours"] = [
                hour for hour, count in hour_counts.most_common(3)
            ]

        return timing_predictions

    def _calculate_prediction_confidence(self, predictions: Dict) -> Dict:
        """Calculate confidence scores for predictions"""
        confidence_scores = {}

        # Base confidence on amount of historical data
        data_points = self.learning_metrics["interactions_processed"]
        base_confidence = min(data_points / 100, 0.8)  # Max 0.8 base confidence

        confidence_scores["overall"] = base_confidence
        confidence_scores["next_actions"] = base_confidence * 0.9
        confidence_scores["proactive_suggestions"] = base_confidence * 0.7
        confidence_scores["context_predictions"] = base_confidence * 0.8

        return confidence_scores

    def _determine_communication_style(self) -> str:
        """Determine user's preferred communication style"""
        preferences = self.user_model.get("preferences", {})

        if "communication_style" in preferences:
            return preferences["communication_style"]

        # Default based on interaction patterns
        return "balanced"

    def _determine_detail_preference(self) -> str:
        """Determine user's preferred level of detail"""
        preferences = self.user_model.get("preferences", {})

        detail_prefs = [p for p in preferences.get("detail_level", []) if p["confidence"] > 0.6]
        if detail_prefs:
            return max(detail_prefs, key=lambda x: x["confidence"])["preference"]

        return "medium"

    def _determine_technical_level(self) -> str:
        """Determine user's technical level preference"""
        skills = self.user_model.get("skills", {})

        if "technical" in skills:
            return skills["technical"]["level"]

        return "intermediate"

    def _determine_length_preference(self) -> str:
        """Determine user's preferred response length"""
        preferences = self.user_model.get("preferences", {})

        if "response_length" in preferences:
            return preferences["response_length"]

        return "medium"

    def _learn_formatting_preferences(self) -> Dict:
        """Learn user's formatting preferences"""
        preferences = self.user_model.get("preferences", {})

        formatting_prefs = {}
        if "format" in preferences:
            for pref in preferences["format"]:
                if pref["confidence"] > 0.6:
                    formatting_prefs[pref["preference"]] = pref["confidence"]

        return formatting_prefs

    def _determine_tone_adjustments(self, context: Dict) -> Dict:
        """Determine tone adjustments based on context"""
        adjustments = {}

        # Adjust based on time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 11:
            adjustments["greeting"] = "morning"
        elif 12 <= current_hour <= 17:
            adjustments["greeting"] = "afternoon"
        elif 18 <= current_hour <= 22:
            adjustments["greeting"] = "evening"

        # Adjust based on system status
        system_status = context.get("system_status", {})
        if system_status.get("cpu_percent", 0) > 80:
            adjustments["urgency"] = "high"

        return adjustments

    def _extract_primary_action(self, user_input: str) -> str:
        """Extract the primary action from user input"""
        action_keywords = {
            "research": ["research", "find", "search", "investigate"],
            "create": ["create", "make", "generate", "build"],
            "analyze": ["analyze", "examine", "check", "review"],
            "manage": ["manage", "organize", "handle", "control"]
        }

        user_lower = user_input.lower()
        for action, keywords in action_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return action

        return "general"

    def _classify_context_type(self, context: Dict) -> str:
        """Classify the type of context"""
        if context.get("system_status"):
            return "system_focused"
        elif context.get("active_apps"):
            return "application_focused"
        elif context.get("recent_files"):
            return "file_focused"
        else:
            return "general"

    def _classify_question_type(self, user_input: str) -> str:
        """Classify the type of question"""
        if "?" not in user_input:
            return "statement"

        question_starters = {
            "what": "factual",
            "how": "procedural",
            "why": "explanatory",
            "when": "temporal",
            "where": "locational",
            "who": "personal"
        }

        user_lower = user_input.lower()
        for starter, q_type in question_starters.items():
            if user_lower.startswith(starter):
                return q_type

        return "general"

    def _assess_formality_level(self, user_input: str) -> str:
        """Assess the formality level of user input"""
        formal_indicators = ["please", "could you", "would you", "thank you"]
        informal_indicators = ["hey", "yo", "gonna", "wanna"]

        user_lower = user_input.lower()

        formal_count = sum(1 for indicator in formal_indicators if indicator in user_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in user_lower)

        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"

    def _extract_technical_terms(self, user_input: str) -> List[str]:
        """Extract technical terms from user input"""
        technical_terms = [
            "api", "database", "algorithm", "framework", "architecture",
            "configuration", "optimization", "integration", "deployment",
            "authentication", "encryption", "protocol", "interface"
        ]

        user_lower = user_input.lower()
        found_terms = [term for term in technical_terms if term in user_lower]

        return found_terms

    def _assess_task_complexity(self, user_input: str) -> str:
        """Assess the complexity of the requested task"""
        complexity_indicators = {
            "high": ["integrate", "configure", "optimize", "analyze", "process", "multiple"],
            "medium": ["create", "find", "check", "update", "manage"],
            "low": ["show", "list", "get", "tell", "what"]
        }

        user_lower = user_input.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in user_lower for indicator in indicators):
                return level

        return "medium"

    def _is_multi_step_task(self, user_input: str) -> bool:
        """Check if the task involves multiple steps"""
        multi_step_indicators = ["and", "then", "after", "also", "plus", "first", "next"]
        user_lower = user_input.lower()

        return any(indicator in user_lower for indicator in multi_step_indicators)

    def _identify_domain(self, user_input: str) -> str:
        """Identify the domain of the user request"""
        domains = {
            "technology": ["tech", "software", "programming", "code", "development"],
            "business": ["business", "market", "finance", "strategy", "management"],
            "science": ["research", "study", "analysis", "data", "experiment"],
            "creative": ["design", "art", "creative", "writing", "content"],
            "system": ["system", "computer", "performance", "hardware", "software"]
        }

        user_lower = user_input.lower()

        for domain, keywords in domains.items():
            if any(keyword in user_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_success_indicators(self, response: str) -> List[str]:
        """Extract indicators of successful task completion"""
        success_indicators = []

        if "completed" in response.lower():
            success_indicators.append("task_completed")
        if "found" in response.lower():
            success_indicators.append("information_found")
        if "created" in response.lower():
            success_indicators.append("content_created")

        return success_indicators

    def _assess_question_sophistication(self, user_input: str) -> str:
        """Assess the sophistication level of the question"""
        sophisticated_indicators = [
            "implications", "considerations", "trade-offs", "alternatives",
            "best practices", "optimization", "scalability", "architecture"
        ]

        user_lower = user_input.lower()

        if any(indicator in user_lower for indicator in sophisticated_indicators):
            return "advanced"
        elif len(user_input.split()) > 10:
            return "intermediate"
        else:
            return "beginner"

    def _update_user_preference(self, preference: Dict):
        """Update user preference in the model"""
        pref_type = preference["type"]
        if "preferences" not in self.user_model:
            self.user_model["preferences"] = {}

        if pref_type not in self.user_model["preferences"]:
            self.user_model["preferences"][pref_type] = []

        self.user_model["preferences"][pref_type].append(preference)

    def _update_user_skill(self, skill: Dict):
        """Update user skill assessment in the model"""
        domain = skill["domain"]
        if "skills" not in self.user_model:
            self.user_model["skills"] = {}

        self.user_model["skills"][domain] = skill

    def _is_pattern_relevant(self, pattern: Dict, current_task: str, context: Dict) -> bool:
        """Check if a workflow pattern is relevant to the current task"""
        # Simple relevance check based on task similarity
        pattern_actions = pattern.get("sequence", [])
        current_action = self._extract_primary_action(current_task)

        return current_action in pattern_actions

    async def _generate_novel_workflow_suggestions(self, current_task: str,
                                                 context: Dict) -> List[Dict]:
        """Generate novel workflow suggestions"""
        suggestions = []

        # This would use more sophisticated ML techniques in practice
        # For now, provide basic suggestions based on task type
        task_type = self._extract_primary_action(current_task)

        if task_type == "research":
            suggestions.append({
                "workflow": ["research", "document_creation", "email_management"],
                "confidence": 0.6,
                "estimated_time": 30,
                "success_rate": 0.8,
                "optimization_tips": ["Use multiple sources", "Create summary document"]
            })

        return suggestions

    async def _store_learning_results(self, learning_results: Dict):
        """Store learning results for future reference"""
        if self.vector_store:
            try:
                await self.vector_store.store_task_result(
                    f"learning_{datetime.now().isoformat()}",
                    learning_results
                )
            except Exception as e:
                logger.error("Failed to store learning results: %s", e)

    def get_learning_stats(self) -> Dict:
        """Get current learning statistics"""
        return {
            "metrics": self.learning_metrics,
            "user_model_size": {
                "preferences": len(self.user_model.get("preferences", {})),
                "skills": len(self.user_model.get("skills", {})),
                "patterns": len(self.user_model.get("patterns", {}))
            },
            "adaptation_rules": len(self.adaptation_rules)
        }
