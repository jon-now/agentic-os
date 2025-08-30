import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import statistics
from collections import defaultdict, Counter
import pickle
import hashlib
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class LearningType(Enum):
    USER_PREFERENCES = "user_preferences"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    SKILL_ASSESSMENT = "skill_assessment"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    PREDICTIVE_MODELING = "predictive_modeling"
    ADAPTIVE_RESPONSES = "adaptive_responses"
    CONTEXT_LEARNING = "context_learning"
    PERSONALIZATION = "personalization"

class PersonalizationLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    ADAPTIVE = "adaptive"

class LearningEngine:
    """Advanced learning and personalization system"""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.user_profiles = {}
        self.learning_models = {}
        self.behavioral_patterns = defaultdict(list)
        self.skill_assessments = {}
        self.preference_models = {}
        self.adaptation_history = []
        self.learning_cache = {}
        self.personalization_rules = self._initialize_personalization_rules()

    def _initialize_personalization_rules(self) -> Dict:
        """Initialize personalization rules and patterns"""
        return {
            "communication_style": {
                "formal": {
                    "indicators": ["please", "thank you", "would you", "could you"],
                    "response_style": "professional",
                    "vocabulary": "formal",
                    "structure": "detailed"
                },
                "casual": {
                    "indicators": ["hey", "thanks", "can you", "what's up"],
                    "response_style": "friendly",
                    "vocabulary": "conversational",
                    "structure": "concise"
                },
                "technical": {
                    "indicators": ["implement", "configure", "debug", "optimize"],
                    "response_style": "technical",
                    "vocabulary": "specialized",
                    "structure": "step_by_step"
                }
            },
            "expertise_levels": {
                "beginner": {
                    "indicators": ["how do I", "what is", "explain", "help me understand"],
                    "response_depth": "detailed_explanations",
                    "examples": "basic",
                    "guidance": "step_by_step"
                },
                "intermediate": {
                    "indicators": ["optimize", "improve", "best practices", "alternatives"],
                    "response_depth": "practical_focus",
                    "examples": "real_world",
                    "guidance": "options_based"
                },
                "advanced": {
                    "indicators": ["architecture", "scalability", "performance", "integration"],
                    "response_depth": "comprehensive",
                    "examples": "complex_scenarios",
                    "guidance": "strategic"
                }
            },
            "task_preferences": {
                "automation_focused": {
                    "indicators": ["automate", "schedule", "batch", "workflow"],
                    "suggestions": "automation_opportunities",
                    "tools": "scripting_focused",
                    "approach": "efficiency_first"
                },
                "analysis_focused": {
                    "indicators": ["analyze", "report", "insights", "trends"],
                    "suggestions": "analytical_tools",
                    "tools": "data_focused",
                    "approach": "insight_driven"
                },
                "creative_focused": {
                    "indicators": ["create", "design", "generate", "brainstorm"],
                    "suggestions": "creative_tools",
                    "tools": "content_focused",
                    "approach": "innovation_driven"
                }
            }
        }

    async def learn_from_interaction(self, user_id: str, interaction_data: Dict) -> Dict:
        """Learn from user interaction and update models"""
        try:
            learning_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            learning_result = {
                "learning_id": learning_id,
                "user_id": user_id,
                "interaction_timestamp": datetime.now().isoformat(),
                "learning_types": [],
                "adaptations_made": [],
                "confidence_scores": {},
                "learning_metadata": {}
            }

            # Initialize user profile if not exists
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = await self._initialize_user_profile(user_id)

            user_profile = self.user_profiles[user_id]

            # Learn user preferences
            preference_learning = await self._learn_user_preferences(user_id, interaction_data)
            learning_result["learning_types"].append(LearningType.USER_PREFERENCES.value)
            learning_result["confidence_scores"]["preferences"] = preference_learning.get("confidence", 0.0)

            # Learn behavioral patterns
            pattern_learning = await self._learn_behavioral_patterns(user_id, interaction_data)
            learning_result["learning_types"].append(LearningType.BEHAVIORAL_PATTERNS.value)
            learning_result["confidence_scores"]["patterns"] = pattern_learning.get("confidence", 0.0)

            # Assess skill level
            skill_assessment = await self._assess_user_skills(user_id, interaction_data)
            learning_result["learning_types"].append(LearningType.SKILL_ASSESSMENT.value)
            learning_result["confidence_scores"]["skills"] = skill_assessment.get("confidence", 0.0)

            # Learn workflow preferences
            workflow_learning = await self._learn_workflow_preferences(user_id, interaction_data)
            learning_result["learning_types"].append(LearningType.WORKFLOW_OPTIMIZATION.value)
            learning_result["confidence_scores"]["workflows"] = workflow_learning.get("confidence", 0.0)

            # Update personalization model
            personalization_update = await self._update_personalization_model(user_id, learning_result)
            learning_result["adaptations_made"] = personalization_update.get("adaptations", [])

            # Store learning results
            self.learning_cache[learning_id] = learning_result

            # Update adaptation history
            self.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "learning_id": learning_id,
                "adaptations": learning_result["adaptations_made"]
            })

            return learning_result

        except Exception as e:
            logger.error("Learning from interaction failed: %s", e)
            return {"error": str(e)}

    async def _initialize_user_profile(self, user_id: str) -> Dict:
        """Initialize a new user profile"""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "interaction_count": 0,
            "preferences": {
                "communication_style": "neutral",
                "response_length": "medium",
                "technical_level": "intermediate",
                "explanation_depth": "balanced"
            },
            "skills": {
                "technical_skills": {},
                "domain_expertise": {},
                "tool_proficiency": {},
                "learning_speed": "medium"
            },
            "behavioral_patterns": {
                "active_hours": [],
                "task_types": [],
                "interaction_frequency": {},
                "session_patterns": []
            },
            "workflow_preferences": {
                "automation_preference": 0.5,
                "detail_preference": 0.5,
                "guidance_preference": 0.5,
                "efficiency_focus": 0.5
            },
            "personalization_level": PersonalizationLevel.BASIC.value,
            "last_updated": datetime.now().isoformat()
        }

    async def _learn_user_preferences(self, user_id: str, interaction_data: Dict) -> Dict:
        """Learn and update user preferences"""
        user_profile = self.user_profiles[user_id]
        user_input = interaction_data.get("user_input", "")
        response_feedback = interaction_data.get("feedback", {})

        preference_updates = {}
        confidence_score = 0.0

        # Analyze communication style
        comm_style = await self._analyze_communication_style(user_input)
        if comm_style:
            current_style = user_profile["preferences"]["communication_style"]
            if current_style == "neutral" or comm_style != current_style:
                preference_updates["communication_style"] = comm_style
                confidence_score += 0.3

        # Analyze technical level from questions and responses
        tech_level = await self._analyze_technical_level(user_input, response_feedback)
        if tech_level:
            preference_updates["technical_level"] = tech_level
            confidence_score += 0.3

        # Analyze response length preference from feedback
        if response_feedback.get("length_feedback"):
            length_pref = response_feedback["length_feedback"]
            preference_updates["response_length"] = length_pref
            confidence_score += 0.2

        # Analyze explanation depth preference
        depth_pref = await self._analyze_explanation_preference(user_input, response_feedback)
        if depth_pref:
            preference_updates["explanation_depth"] = depth_pref
            confidence_score += 0.2

        # Update user profile
        user_profile["preferences"].update(preference_updates)
        user_profile["last_updated"] = datetime.now().isoformat()

        return {
            "updates": preference_updates,
            "confidence": min(confidence_score, 1.0),
            "learning_type": "preferences"
        }

    async def _learn_behavioral_patterns(self, user_id: str, interaction_data: Dict) -> Dict:
        """Learn user behavioral patterns"""
        user_profile = self.user_profiles[user_id]
        timestamp = datetime.now()

        pattern_updates = {}
        confidence_score = 0.0

        # Track active hours
        hour = timestamp.hour
        user_profile["behavioral_patterns"]["active_hours"].append(hour)

        # Keep only last 100 interactions for active hours
        if len(user_profile["behavioral_patterns"]["active_hours"]) > 100:
            user_profile["behavioral_patterns"]["active_hours"] = \
                user_profile["behavioral_patterns"]["active_hours"][-100:]

        # Analyze most active hours
        if len(user_profile["behavioral_patterns"]["active_hours"]) >= 10:
            hour_counts = Counter(user_profile["behavioral_patterns"]["active_hours"])
            most_active_hours = [hour for hour, count in hour_counts.most_common(3)]
            pattern_updates["peak_hours"] = most_active_hours
            confidence_score += 0.3

        # Track task types
        task_type = interaction_data.get("task_type", "general")
        user_profile["behavioral_patterns"]["task_types"].append(task_type)

        if len(user_profile["behavioral_patterns"]["task_types"]) > 50:
            user_profile["behavioral_patterns"]["task_types"] = \
                user_profile["behavioral_patterns"]["task_types"][-50:]

        # Analyze preferred task types
        if len(user_profile["behavioral_patterns"]["task_types"]) >= 10:
            task_counts = Counter(user_profile["behavioral_patterns"]["task_types"])
            preferred_tasks = [task for task, count in task_counts.most_common(3)]
            pattern_updates["preferred_task_types"] = preferred_tasks
            confidence_score += 0.3

        # Track interaction frequency
        date_key = timestamp.strftime("%Y-%m-%d")
        if date_key not in user_profile["behavioral_patterns"]["interaction_frequency"]:
            user_profile["behavioral_patterns"]["interaction_frequency"][date_key] = 0
        user_profile["behavioral_patterns"]["interaction_frequency"][date_key] += 1

        # Analyze interaction frequency patterns
        recent_days = list(user_profile["behavioral_patterns"]["interaction_frequency"].keys())[-7:]
        if len(recent_days) >= 3:
            daily_counts = [user_profile["behavioral_patterns"]["interaction_frequency"][day]
                          for day in recent_days]
            avg_daily = statistics.mean(daily_counts)
            pattern_updates["average_daily_interactions"] = avg_daily
            confidence_score += 0.2

        # Track session patterns
        session_data = {
            "timestamp": timestamp.isoformat(),
            "duration": interaction_data.get("session_duration", 0),
            "interaction_count": interaction_data.get("interaction_count", 1)
        }
        user_profile["behavioral_patterns"]["session_patterns"].append(session_data)

        if len(user_profile["behavioral_patterns"]["session_patterns"]) > 20:
            user_profile["behavioral_patterns"]["session_patterns"] = \
                user_profile["behavioral_patterns"]["session_patterns"][-20:]

        # Analyze session patterns
        if len(user_profile["behavioral_patterns"]["session_patterns"]) >= 5:
            durations = [s["duration"] for s in user_profile["behavioral_patterns"]["session_patterns"]
                        if s["duration"] > 0]
            if durations:
                avg_session_duration = statistics.mean(durations)
                pattern_updates["average_session_duration"] = avg_session_duration
                confidence_score += 0.2

        return {
            "updates": pattern_updates,
            "confidence": min(confidence_score, 1.0),
            "learning_type": "behavioral_patterns"
        }

    async def _assess_user_skills(self, user_id: str, interaction_data: Dict) -> Dict:
        """Assess and update user skill levels"""
        user_profile = self.user_profiles[user_id]
        user_input = interaction_data.get("user_input", "")
        task_success = interaction_data.get("task_success", None)

        skill_updates = {}
        confidence_score = 0.0

        # Assess technical skills from user input
        technical_indicators = await self._extract_technical_indicators(user_input)
        for skill, level in technical_indicators.items():
            current_level = user_profile["skills"]["technical_skills"].get(skill, 0)
            # Update skill level with weighted average
            new_level = (current_level * 0.7) + (level * 0.3)
            user_profile["skills"]["technical_skills"][skill] = new_level
            skill_updates[f"technical_{skill}"] = new_level
            confidence_score += 0.1

        # Assess domain expertise
        domain_indicators = await self._extract_domain_indicators(user_input)
        for domain, expertise in domain_indicators.items():
            current_expertise = user_profile["skills"]["domain_expertise"].get(domain, 0)
            new_expertise = (current_expertise * 0.8) + (expertise * 0.2)
            user_profile["skills"]["domain_expertise"][domain] = new_expertise
            skill_updates[f"domain_{domain}"] = new_expertise
            confidence_score += 0.1

        # Assess tool proficiency
        tool_usage = interaction_data.get("tools_used", [])
        for tool in tool_usage:
            current_proficiency = user_profile["skills"]["tool_proficiency"].get(tool, 0)
            success_rate = 1.0 if task_success else 0.5 if task_success is None else 0.0
            new_proficiency = (current_proficiency * 0.9) + (success_rate * 0.1)
            user_profile["skills"]["tool_proficiency"][tool] = new_proficiency
            skill_updates[f"tool_{tool}"] = new_proficiency
            confidence_score += 0.1

        # Assess learning speed
        if task_success is not None:
            interaction_count = user_profile["interaction_count"]
            if interaction_count > 0:
                success_rate = interaction_data.get("cumulative_success_rate", 0.5)
                if success_rate > 0.8:
                    learning_speed = "fast"
                elif success_rate > 0.6:
                    learning_speed = "medium"
                else:
                    learning_speed = "slow"

                if user_profile["skills"]["learning_speed"] != learning_speed:
                    user_profile["skills"]["learning_speed"] = learning_speed
                    skill_updates["learning_speed"] = learning_speed
                    confidence_score += 0.2

        return {
            "updates": skill_updates,
            "confidence": min(confidence_score, 1.0),
            "learning_type": "skill_assessment"
        }

    async def _learn_workflow_preferences(self, user_id: str, interaction_data: Dict) -> Dict:
        """Learn user workflow preferences"""
        user_profile = self.user_profiles[user_id]
        user_input = interaction_data.get("user_input", "")
        task_type = interaction_data.get("task_type", "")

        workflow_updates = {}
        confidence_score = 0.0

        # Analyze automation preference
        automation_indicators = ["automate", "schedule", "batch", "script", "workflow"]
        automation_score = sum(1 for indicator in automation_indicators if indicator in user_input.lower())
        if automation_score > 0:
            current_pref = user_profile["workflow_preferences"]["automation_preference"]
            new_pref = min(1.0, current_pref + (automation_score * 0.1))
            user_profile["workflow_preferences"]["automation_preference"] = new_pref
            workflow_updates["automation_preference"] = new_pref
            confidence_score += 0.3

        # Analyze detail preference
        detail_indicators = ["explain", "details", "step by step", "comprehensive", "thorough"]
        detail_score = sum(1 for indicator in detail_indicators if indicator in user_input.lower())
        if detail_score > 0:
            current_pref = user_profile["workflow_preferences"]["detail_preference"]
            new_pref = min(1.0, current_pref + (detail_score * 0.1))
            user_profile["workflow_preferences"]["detail_preference"] = new_pref
            workflow_updates["detail_preference"] = new_pref
            confidence_score += 0.3

        # Analyze guidance preference
        guidance_indicators = ["help", "guide", "assist", "support", "show me"]
        guidance_score = sum(1 for indicator in guidance_indicators if indicator in user_input.lower())
        if guidance_score > 0:
            current_pref = user_profile["workflow_preferences"]["guidance_preference"]
            new_pref = min(1.0, current_pref + (guidance_score * 0.1))
            user_profile["workflow_preferences"]["guidance_preference"] = new_pref
            workflow_updates["guidance_preference"] = new_pref
            confidence_score += 0.2

        # Analyze efficiency focus
        efficiency_indicators = ["fast", "quick", "efficient", "optimize", "streamline"]
        efficiency_score = sum(1 for indicator in efficiency_indicators if indicator in user_input.lower())
        if efficiency_score > 0:
            current_pref = user_profile["workflow_preferences"]["efficiency_focus"]
            new_pref = min(1.0, current_pref + (efficiency_score * 0.1))
            user_profile["workflow_preferences"]["efficiency_focus"] = new_pref
            workflow_updates["efficiency_focus"] = new_pref
            confidence_score += 0.2

        return {
            "updates": workflow_updates,
            "confidence": min(confidence_score, 1.0),
            "learning_type": "workflow_preferences"
        }

    async def _update_personalization_model(self, user_id: str, learning_result: Dict) -> Dict:
        """Update personalization model based on learning results"""
        user_profile = self.user_profiles[user_id]
        adaptations = []

        # Update interaction count
        user_profile["interaction_count"] += 1

        # Determine personalization level
        interaction_count = user_profile["interaction_count"]
        confidence_scores = learning_result.get("confidence_scores", {})
        avg_confidence = statistics.mean(confidence_scores.values()) if confidence_scores else 0.0

        if interaction_count >= 50 and avg_confidence > 0.8:
            new_level = PersonalizationLevel.EXPERT.value
        elif interaction_count >= 20 and avg_confidence > 0.6:
            new_level = PersonalizationLevel.ADVANCED.value
        elif interaction_count >= 10 and avg_confidence > 0.4:
            new_level = PersonalizationLevel.INTERMEDIATE.value
        else:
            new_level = PersonalizationLevel.BASIC.value

        if user_profile["personalization_level"] != new_level:
            user_profile["personalization_level"] = new_level
            adaptations.append(f"Personalization level updated to {new_level}")

        # Update last interaction timestamp
        user_profile["last_updated"] = datetime.now().isoformat()

        return {
            "adaptations": adaptations,
            "personalization_level": new_level,
            "confidence": avg_confidence
        }

    async def generate_personalized_response(self, user_id: str, query: str,
                                           base_response: str, context: Dict = None) -> Dict:
        """Generate personalized response based on user profile"""
        try:
            if user_id not in self.user_profiles:
                return {
                    "personalized_response": base_response,
                    "personalization_applied": [],
                    "confidence": 0.0
                }

            user_profile = self.user_profiles[user_id]
            personalization_applied = []

            # Apply communication style personalization
            style_adaptation = await self._adapt_communication_style(
                base_response, user_profile["preferences"]["communication_style"]
            )
            personalized_response = style_adaptation["adapted_response"]
            if style_adaptation["changes_made"]:
                personalization_applied.extend(style_adaptation["changes_made"])

            # Apply technical level adaptation
            tech_adaptation = await self._adapt_technical_level(
                personalized_response, user_profile["preferences"]["technical_level"]
            )
            personalized_response = tech_adaptation["adapted_response"]
            if tech_adaptation["changes_made"]:
                personalization_applied.extend(tech_adaptation["changes_made"])

            # Apply response length preference
            length_adaptation = await self._adapt_response_length(
                personalized_response, user_profile["preferences"]["response_length"]
            )
            personalized_response = length_adaptation["adapted_response"]
            if length_adaptation["changes_made"]:
                personalization_applied.extend(length_adaptation["changes_made"])

            # Apply workflow-specific adaptations
            workflow_adaptation = await self._adapt_workflow_suggestions(
                personalized_response, user_profile["workflow_preferences"], query
            )
            personalized_response = workflow_adaptation["adapted_response"]
            if workflow_adaptation["changes_made"]:
                personalization_applied.extend(workflow_adaptation["changes_made"])

            # Calculate personalization confidence
            personalization_level = user_profile["personalization_level"]
            interaction_count = user_profile["interaction_count"]

            confidence = min(1.0, (interaction_count / 50) * 0.7 +
                           (len(personalization_applied) / 10) * 0.3)

            return {
                "personalized_response": personalized_response,
                "personalization_applied": personalization_applied,
                "confidence": confidence,
                "personalization_level": personalization_level,
                "user_profile_summary": await self._generate_profile_summary(user_profile)
            }

        except Exception as e:
            logger.error("Response personalization failed: %s", e)
            return {
                "personalized_response": base_response,
                "personalization_applied": [],
                "confidence": 0.0,
                "error": str(e)
            }

    async def _analyze_communication_style(self, user_input: str) -> Optional[str]:
        """Analyze communication style from user input"""
        user_input_lower = user_input.lower()

        style_scores = {}
        for style, rules in self.personalization_rules["communication_style"].items():
            score = sum(1 for indicator in rules["indicators"] if indicator in user_input_lower)
            if score > 0:
                style_scores[style] = score

        if style_scores:
            return max(style_scores.keys(), key=lambda x: style_scores[x])
        return None

    async def _analyze_technical_level(self, user_input: str, feedback: Dict) -> Optional[str]:
        """Analyze technical level from user input and feedback"""
        user_input_lower = user_input.lower()

        level_scores = {}
        for level, rules in self.personalization_rules["expertise_levels"].items():
            score = sum(1 for indicator in rules["indicators"] if indicator in user_input_lower)
            if score > 0:
                level_scores[level] = score

        # Consider feedback about response complexity
        if feedback.get("complexity_feedback") == "too_simple":
            return "advanced"
        elif feedback.get("complexity_feedback") == "too_complex":
            return "beginner"

        if level_scores:
            return max(level_scores.keys(), key=lambda x: level_scores[x])
        return None

    async def _analyze_explanation_preference(self, user_input: str, feedback: Dict) -> Optional[str]:
        """Analyze explanation depth preference"""
        if feedback.get("explanation_feedback") == "more_detail":
            return "detailed"
        elif feedback.get("explanation_feedback") == "less_detail":
            return "concise"
        elif "explain" in user_input.lower() or "how" in user_input.lower():
            return "detailed"
        elif "quick" in user_input.lower() or "brie" in user_input.lower():
            return "concise"
        return None

    async def _extract_technical_indicators(self, user_input: str) -> Dict[str, float]:
        """Extract technical skill indicators from user input"""
        technical_skills = {
            "programming": ["code", "script", "function", "variable", "loop", "api"],
            "system_admin": ["server", "network", "database", "backup", "security"],
            "data_analysis": ["data", "analysis", "chart", "report", "statistics"],
            "automation": ["automate", "workflow", "batch", "schedule", "trigger"],
            "web_development": ["html", "css", "javascript", "website", "frontend", "backend"]
        }

        user_input_lower = user_input.lower()
        skill_scores = {}

        for skill, keywords in technical_skills.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                skill_scores[skill] = min(1.0, score / len(keywords))

        return skill_scores

    async def _extract_domain_indicators(self, user_input: str) -> Dict[str, float]:
        """Extract domain expertise indicators from user input"""
        domains = {
            "business": ["revenue", "profit", "customer", "market", "sales", "strategy"],
            "healthcare": ["patient", "medical", "diagnosis", "treatment", "clinical"],
            "education": ["student", "learning", "curriculum", "assessment", "teaching"],
            "finance": ["investment", "portfolio", "risk", "trading", "banking"],
            "marketing": ["campaign", "brand", "advertising", "social media", "seo"]
        }

        user_input_lower = user_input.lower()
        domain_scores = {}

        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                domain_scores[domain] = min(1.0, score / len(keywords))

        return domain_scores

    async def _adapt_communication_style(self, response: str, style: str) -> Dict:
        """Adapt response to match communication style"""
        changes_made = []
        adapted_response = response

        if style == "formal":
            # Make response more formal
            if "can't" in adapted_response:
                adapted_response = adapted_response.replace("can't", "cannot")
                changes_made.append("Formalized contractions")

            if not adapted_response.startswith(("Please", "I would", "I recommend")):
                adapted_response = f"I recommend that you {adapted_response.lower()}"
                changes_made.append("Added formal introduction")

        elif style == "casual":
            # Make response more casual
            if "I recommend" in adapted_response:
                adapted_response = adapted_response.replace("I recommend", "I'd suggest")
                changes_made.append("Made language more casual")

            if "cannot" in adapted_response:
                adapted_response = adapted_response.replace("cannot", "can't")
                changes_made.append("Used contractions")

        elif style == "technical":
            # Make response more technical
            if not any(word in adapted_response.lower() for word in ["configure", "implement", "execute", "process"]):
                adapted_response = f"To implement this solution: {adapted_response}"
                changes_made.append("Added technical framing")

        return {
            "adapted_response": adapted_response,
            "changes_made": changes_made
        }

    async def _adapt_technical_level(self, response: str, level: str) -> Dict:
        """Adapt response to match technical level"""
        changes_made = []
        adapted_response = response

        if level == "beginner":
            # Add more explanations
            if "API" in adapted_response and "Application Programming Interface" not in adapted_response:
                adapted_response = adapted_response.replace("API", "API (Application Programming Interface)")
                changes_made.append("Added technical term explanations")

        elif level == "advanced":
            # Make more concise and technical
            if "step by step" in adapted_response.lower():
                adapted_response = adapted_response.replace("step by step", "systematically")
                changes_made.append("Used advanced terminology")

        return {
            "adapted_response": adapted_response,
            "changes_made": changes_made
        }

    async def _adapt_response_length(self, response: str, length_pref: str) -> Dict:
        """Adapt response length to user preference"""
        changes_made = []
        adapted_response = response

        if length_pref == "short" and len(response.split()) > 50:
            # Summarize response
            sentences = response.split('. ')
            if len(sentences) > 2:
                adapted_response = '. '.join(sentences[:2]) + '.'
                changes_made.append("Shortened response")

        elif length_pref == "long" and len(response.split()) < 30:
            # Expand response
            adapted_response = (
                f"{response}\n\nFor additional context and detailed steps, please let me know if you need more specific guidance on any aspect of this solution."
            )
            changes_made.append("Expanded response with additional context")

        return {
            "adapted_response": adapted_response,
            "changes_made": changes_made
        }

    async def _adapt_workflow_suggestions(self, response: str, workflow_prefs: Dict, query: str) -> Dict:
        """Adapt workflow suggestions based on user preferences"""
        changes_made = []
        adapted_response = response

        # Add automation suggestions if user prefers automation
        if workflow_prefs.get("automation_preference", 0) > 0.7:
            if "automat" not in adapted_response.lower():
                adapted_response += "\n\nConsider automating this process for future efficiency."
                changes_made.append("Added automation suggestion")

        # Add efficiency tips if user focuses on efficiency
        if workflow_prefs.get("efficiency_focus", 0) > 0.7:
            if "efficien" not in adapted_response.lower():
                adapted_response += "\n\nFor optimal efficiency, consider batching similar tasks."
                changes_made.append("Added efficiency tip")

        return {
            "adapted_response": adapted_response,
            "changes_made": changes_made
        }

    async def _generate_profile_summary(self, user_profile: Dict) -> Dict:
        """Generate a summary of user profile"""
        return {
            "personalization_level": user_profile["personalization_level"],
            "interaction_count": user_profile["interaction_count"],
            "communication_style": user_profile["preferences"]["communication_style"],
            "technical_level": user_profile["preferences"]["technical_level"],
            "top_skills": list(user_profile["skills"]["technical_skills"].keys())[:3],
            "workflow_focus": max(user_profile["workflow_preferences"].keys(),
                                key=lambda x: user_profile["workflow_preferences"][x])
        }

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile"""
        return self.user_profiles.get(user_id)

    def get_learning_insights(self, user_id: Optional[str] = None) -> Dict:
        """Get learning insights for user or system"""
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return {
                "user_id": user_id,
                "personalization_level": profile["personalization_level"],
                "interaction_count": profile["interaction_count"],
                "learning_progress": {
                    "preferences_learned": len([k for k, v in profile["preferences"].items() if v != "neutral"]),
                    "skills_identified": len(profile["skills"]["technical_skills"]),
                    "patterns_recognized": len(profile["behavioral_patterns"]["task_types"])
                },
                "adaptation_history": [a for a in self.adaptation_history if a["user_id"] == user_id][-5:]
            }
        else:
            return {
                "total_users": len(self.user_profiles),
                "total_adaptations": len(self.adaptation_history),
                "learning_cache_size": len(self.learning_cache),
                "personalization_levels": Counter(p["personalization_level"] for p in self.user_profiles.values()),
                "recent_adaptations": self.adaptation_history[-10:]
            }

    async def predict_user_needs(self, user_id: str, current_context: Dict) -> Dict:
        """Predict user needs based on learned patterns"""
        try:
            if user_id not in self.user_profiles:
                return {"predictions": [], "confidence": 0.0}

            user_profile = self.user_profiles[user_id]
            predictions = []
            confidence_scores = []

            # Predict based on behavioral patterns
            current_hour = datetime.now().hour
            active_hours = user_profile["behavioral_patterns"]["active_hours"]

            if active_hours and current_hour in Counter(active_hours).most_common(3):
                predictions.append({
                    "type": "high_activity_period",
                    "description": "User is typically active during this time",
                    "suggestion": "Proactive assistance may be welcomed"
                })
                confidence_scores.append(0.8)

            # Predict based on task patterns
            recent_tasks = user_profile["behavioral_patterns"]["task_types"][-10:]
            if recent_tasks:
                most_common_task = Counter(recent_tasks).most_common(1)[0][0]
                predictions.append({
                    "type": "task_preference",
                    "description": f"User frequently works on {most_common_task} tasks",
                    "suggestion": f"Prepare {most_common_task}-related tools and suggestions"
                })
                confidence_scores.append(0.7)

            # Predict based on workflow preferences
            if user_profile["workflow_preferences"]["automation_preference"] > 0.7:
                predictions.append({
                    "type": "automation_opportunity",
                    "description": "User prefers automated solutions",
                    "suggestion": "Look for automation opportunities in current task"
                })
                confidence_scores.append(0.6)

            overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0

            return {
                "predictions": predictions,
                "confidence": overall_confidence,
                "prediction_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("User need prediction failed: %s", e)
            return {"predictions": [], "confidence": 0.0, "error": str(e)}
