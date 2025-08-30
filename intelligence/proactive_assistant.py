import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

class ProactiveAssistant:
    """Proactive assistant that anticipates user needs and provides suggestions"""

    def __init__(self, context_analyzer=None, learning_engine=None, vector_store=None):
        self.context_analyzer = context_analyzer
        self.learning_engine = learning_engine
        self.vector_store = vector_store
        self.proactive_rules = []
        self.suggestion_history = []
        self.user_feedback = defaultdict(list)

    async def generate_proactive_suggestions(self, current_context: Dict,
                                           session_history: List[Dict]) -> List[Dict]:
        """Generate proactive suggestions based on context and patterns"""
        try:
            suggestions = []

            # System health suggestions
            system_suggestions = await self._generate_system_health_suggestions(current_context)
            suggestions.extend(system_suggestions)

            # Workflow optimization suggestions
            workflow_suggestions = await self._generate_workflow_suggestions(
                current_context, session_history
            )
            suggestions.extend(workflow_suggestions)

            # Time-based suggestions
            time_suggestions = await self._generate_time_based_suggestions(
                current_context, session_history
            )
            suggestions.extend(time_suggestions)

            # Context-aware suggestions
            context_suggestions = await self._generate_context_aware_suggestions(
                current_context, session_history
            )
            suggestions.extend(context_suggestions)

            # Learning-based suggestions
            if self.learning_engine:
                learning_suggestions = await self._generate_learning_based_suggestions(
                    current_context, session_history
                )
                suggestions.extend(learning_suggestions)

            # Filter and rank suggestions
            filtered_suggestions = await self._filter_and_rank_suggestions(
                suggestions, current_context
            )

            # Store suggestions for feedback learning
            self._store_suggestions(filtered_suggestions)

            return filtered_suggestions[:5]  # Return top 5

        except Exception as e:
            logger.error("Proactive suggestion generation failed: %s", e)
            return []

    async def anticipate_user_needs(self, current_context: Dict,
                                  session_history: List[Dict]) -> Dict:
        """Anticipate what the user might need next"""
        try:
            anticipation = {
                "likely_next_tasks": [],
                "resource_needs": {},
                "potential_blockers": [],
                "optimization_opportunities": [],
                "preparation_suggestions": []
            }

            # Predict likely next tasks
            anticipation["likely_next_tasks"] = await self._predict_next_tasks(
                current_context, session_history
            )

            # Identify resource needs
            anticipation["resource_needs"] = await self._identify_resource_needs(
                current_context, session_history
            )

            # Detect potential blockers
            anticipation["potential_blockers"] = await self._detect_potential_blockers(
                current_context
            )

            # Find optimization opportunities
            anticipation["optimization_opportunities"] = await self._find_optimization_opportunities(
                current_context, session_history
            )

            # Generate preparation suggestions
            anticipation["preparation_suggestions"] = await self._generate_preparation_suggestions(
                anticipation
            )

            return anticipation

        except Exception as e:
            logger.error("User needs anticipation failed: %s", e)
            return {}

    async def provide_contextual_help(self, user_input: str, current_context: Dict,
                                    session_history: List[Dict]) -> Dict:
        """Provide contextual help and guidance"""
        try:
            help_response = {
                "suggestions": [],
                "tips": [],
                "warnings": [],
                "alternatives": [],
                "learning_resources": []
            }

            # Analyze user input for help opportunities
            help_opportunities = await self._analyze_help_opportunities(
                user_input, current_context
            )

            # Generate contextual suggestions
            for opportunity in help_opportunities:
                if opportunity["type"] == "clarification":
                    help_response["suggestions"].append({
                        "type": "clarification",
                        "message": opportunity["message"],
                        "confidence": opportunity["confidence"]
                    })
                elif opportunity["type"] == "optimization":
                    help_response["tips"].append({
                        "type": "optimization",
                        "message": opportunity["message"],
                        "benefit": opportunity.get("benefit", "")
                    })
                elif opportunity["type"] == "warning":
                    help_response["warnings"].append({
                        "type": "warning",
                        "message": opportunity["message"],
                        "severity": opportunity.get("severity", "medium")
                    })

            # Generate alternatives
            help_response["alternatives"] = await self._generate_alternatives(
                user_input, current_context
            )

            # Suggest learning resources
            help_response["learning_resources"] = await self._suggest_learning_resources(
                user_input, session_history
            )

            return help_response

        except Exception as e:
            logger.error("Contextual help generation failed: %s", e)
            return {}

    async def monitor_user_progress(self, current_task: Dict,
                                  session_history: List[Dict]) -> Dict:
        """Monitor user progress and provide assistance"""
        try:
            progress_analysis = {
                "completion_estimate": 0.0,
                "potential_issues": [],
                "assistance_suggestions": [],
                "efficiency_tips": [],
                "milestone_tracking": {}
            }

            # Estimate task completion
            progress_analysis["completion_estimate"] = await self._estimate_task_completion(
                current_task, session_history
            )

            # Identify potential issues
            progress_analysis["potential_issues"] = await self._identify_task_issues(
                current_task, session_history
            )

            # Generate assistance suggestions
            progress_analysis["assistance_suggestions"] = await self._generate_assistance_suggestions(
                current_task, progress_analysis["potential_issues"]
            )

            # Provide efficiency tips
            progress_analysis["efficiency_tips"] = await self._generate_efficiency_tips(
                current_task, session_history
            )

            # Track milestones
            progress_analysis["milestone_tracking"] = await self._track_milestones(
                current_task, session_history
            )

            return progress_analysis

        except Exception as e:
            logger.error("Progress monitoring failed: %s", e)
            return {}

    async def _generate_system_health_suggestions(self, current_context: Dict) -> List[Dict]:
        """Generate system health-related suggestions"""
        suggestions = []

        system_status = current_context.get("system_status", {})

        # CPU usage suggestions
        cpu_percent = system_status.get("cpu_percent", 0)
        if cpu_percent > 85:
            suggestions.append({
                "type": "system_health",
                "priority": "high",
                "title": "High CPU Usage Detected",
                "message": f"CPU usage is at {cpu_percent}%. Consider closing unnecessary applications.",
                "action": "analyze_cpu_usage",
                "confidence": 0.9
            })
        elif cpu_percent > 70:
            suggestions.append({
                "type": "system_health",
                "priority": "medium",
                "title": "Elevated CPU Usage",
                "message": f"CPU usage is at {cpu_percent}%. Monitor for performance impact.",
                "action": "monitor_cpu_usage",
                "confidence": 0.7
            })

        # Memory usage suggestions
        memory_percent = system_status.get("memory_percent", 0)
        if memory_percent > 90:
            suggestions.append({
                "type": "system_health",
                "priority": "high",
                "title": "Critical Memory Usage",
                "message": f"Memory usage is at {memory_percent}%. System may become unstable.",
                "action": "free_memory",
                "confidence": 0.95
            })
        elif memory_percent > 80:
            suggestions.append({
                "type": "system_health",
                "priority": "medium",
                "title": "High Memory Usage",
                "message": f"Memory usage is at {memory_percent}%. Consider closing unused applications.",
                "action": "optimize_memory",
                "confidence": 0.8
            })

        # Disk usage suggestions
        disk_usage = system_status.get("disk_usage", 0)
        if disk_usage > 90:
            suggestions.append({
                "type": "system_health",
                "priority": "high",
                "title": "Low Disk Space",
                "message": f"Disk usage is at {disk_usage}%. Clean up files to prevent issues.",
                "action": "cleanup_disk",
                "confidence": 0.9
            })

        return suggestions

    async def _generate_workflow_suggestions(self, current_context: Dict,
                                           session_history: List[Dict]) -> List[Dict]:
        """Generate workflow optimization suggestions"""
        suggestions = []

        if not session_history:
            return suggestions

        # Analyze recent workflow patterns
        recent_actions = [
            h.get("intention", {}).get("primary_action")
            for h in session_history[-5:]
            if h.get("intention", {}).get("primary_action")
        ]

        # Suggest workflow optimizations
        if len(set(recent_actions)) == 1 and len(recent_actions) >= 3:
            # User is repeating the same action
            action = recent_actions[0]
            suggestions.append({
                "type": "workflow",
                "priority": "medium",
                "title": "Workflow Automation Opportunity",
                "message": f"You've been doing {action} repeatedly. Would you like me to automate this?",
                "action": f"automate_{action}",
                "confidence": 0.8
            })

        # Suggest complementary actions
        if "research" in recent_actions:
            suggestions.append({
                "type": "workflow",
                "priority": "low",
                "title": "Document Your Research",
                "message": "Consider creating a document to save your research findings.",
                "action": "create_research_document",
                "confidence": 0.6
            })

        if "email_management" in recent_actions:
            suggestions.append({
                "type": "workflow",
                "priority": "low",
                "title": "Calendar Integration",
                "message": "Check your calendar for any meetings mentioned in emails.",
                "action": "check_calendar_from_emails",
                "confidence": 0.5
            })

        return suggestions

    async def _generate_time_based_suggestions(self, current_context: Dict,
                                             session_history: List[Dict]) -> List[Dict]:
        """Generate time-based suggestions"""
        suggestions = []

        current_time = datetime.now()
        current_hour = current_time.hour

        # Morning suggestions (9-11 AM)
        if 9 <= current_hour <= 11:
            suggestions.append({
                "type": "time_based",
                "priority": "low",
                "title": "Morning Productivity",
                "message": "Good morning! Would you like me to check your emails and calendar for today?",
                "action": "morning_briefing",
                "confidence": 0.7
            })

        # Lunch time suggestions (12-1 PM)
        elif 12 <= current_hour <= 13:
            suggestions.append({
                "type": "time_based",
                "priority": "low",
                "title": "Lunch Break Reminder",
                "message": "It's lunch time! Consider taking a break to recharge.",
                "action": "schedule_break",
                "confidence": 0.5
            })

        # End of day suggestions (5-6 PM)
        elif 17 <= current_hour <= 18:
            suggestions.append({
                "type": "time_based",
                "priority": "medium",
                "title": "End of Day Summary",
                "message": "Would you like me to summarize today's activities and prepare for tomorrow?",
                "action": "daily_summary",
                "confidence": 0.6
            })

        # Weekend suggestions
        if current_time.weekday() >= 5:  # Saturday or Sunday
            suggestions.append({
                "type": "time_based",
                "priority": "low",
                "title": "Weekend Planning",
                "message": "It's the weekend! Would you like help organizing personal tasks?",
                "action": "weekend_planning",
                "confidence": 0.4
            })

        return suggestions

    async def _generate_context_aware_suggestions(self, current_context: Dict,
                                                session_history: List[Dict]) -> List[Dict]:
        """Generate context-aware suggestions"""
        suggestions = []

        # Active applications context
        active_apps = current_context.get("active_apps", [])

        if any("code" in app.lower() or "editor" in app.lower() for app in active_apps):
            suggestions.append({
                "type": "context_aware",
                "priority": "low",
                "title": "Development Assistant",
                "message": "I see you're coding. Need help with documentation or research?",
                "action": "development_assistance",
                "confidence": 0.6
            })

        if any("browser" in app.lower() or "chrome" in app.lower() for app in active_apps):
            suggestions.append({
                "type": "context_aware",
                "priority": "low",
                "title": "Research Assistant",
                "message": "I can help organize your research or create summaries from web content.",
                "action": "research_assistance",
                "confidence": 0.5
            })

        # Recent files context
        recent_files = current_context.get("recent_files", [])
        if recent_files:
            doc_files = [f for f in recent_files if f.get("type", "").lower() in [".doc", ".docx", ".pdf"]]
            if doc_files:
                suggestions.append({
                    "type": "context_aware",
                    "priority": "low",
                    "title": "Document Management",
                    "message": "I can help organize or analyze your recent documents.",
                    "action": "document_management",
                    "confidence": 0.5
                })

        return suggestions

    async def _generate_learning_based_suggestions(self, current_context: Dict,
                                                 session_history: List[Dict]) -> List[Dict]:
        """Generate suggestions based on learning patterns"""
        suggestions = []

        if not self.learning_engine:
            return suggestions

        try:
            # Get user predictions from learning engine
            predictions = await self.learning_engine.predict_user_needs(
                current_context, session_history
            )

            # Convert predictions to suggestions
            for action in predictions.get("likely_next_actions", []):
                if action["probability"] > 0.6:
                    suggestions.append({
                        "type": "learning_based",
                        "priority": "medium",
                        "title": f"Predicted Next Action: {action['action']}",
                        "message": f"Based on your patterns, you might want to {action['action']}. {action['reason']}",
                        "action": action["action"],
                        "confidence": action["probability"]
                    })

            # Add proactive suggestions from learning engine
            for suggestion in predictions.get("proactive_suggestions", []):
                suggestions.append({
                    "type": "learning_based",
                    "priority": suggestion.get("priority", "low"),
                    "title": "Personalized Suggestion",
                    "message": suggestion["suggestion"],
                    "action": suggestion["action"],
                    "confidence": suggestion["confidence"]
                })

        except Exception as e:
            logger.error("Learning-based suggestions failed: %s", e)

        return suggestions

    async def _filter_and_rank_suggestions(self, suggestions: List[Dict],
                                         current_context: Dict) -> List[Dict]:
        """Filter and rank suggestions by relevance and priority"""
        if not suggestions:
            return []

        # Filter out low-confidence suggestions
        filtered = [s for s in suggestions if s.get("confidence", 0) > 0.3]

        # Remove duplicates based on action
        seen_actions = set()
        unique_suggestions = []
        for suggestion in filtered:
            action = suggestion.get("action", "")
            if action not in seen_actions:
                seen_actions.add(action)
                unique_suggestions.append(suggestion)

        # Rank by priority and confidence
        priority_weights = {"high": 3, "medium": 2, "low": 1}

        def rank_score(suggestion):
            priority = suggestion.get("priority", "low")
            confidence = suggestion.get("confidence", 0)
            return priority_weights.get(priority, 1) * confidence

        ranked_suggestions = sorted(unique_suggestions, key=rank_score, reverse=True)

        return ranked_suggestions

    async def _predict_next_tasks(self, current_context: Dict,
                                session_history: List[Dict]) -> List[Dict]:
        """Predict likely next tasks"""
        predictions = []

        if not session_history:
            return predictions

        # Simple pattern-based prediction
        recent_actions = [
            h.get("intention", {}).get("primary_action")
            for h in session_history[-3:]
            if h.get("intention", {}).get("primary_action")
        ]

        # Common task sequences
        task_sequences = {
            ("research", "research"): "document_creation",
            ("email_management", "calendar_management"): "meeting_preparation",
            ("file_management", "file_management"): "file_organization"
        }

        if len(recent_actions) >= 2:
            sequence_key = tuple(recent_actions[-2:])
            if sequence_key in task_sequences:
                predictions.append({
                    "task": task_sequences[sequence_key],
                    "probability": 0.7,
                    "reason": f"Common follow-up to {' → '.join(sequence_key)}"
                })

        return predictions

    async def _identify_resource_needs(self, current_context: Dict,
                                     session_history: List[Dict]) -> Dict:
        """Identify potential resource needs"""
        resource_needs = {
            "system_resources": {},
            "external_services": [],
            "data_sources": [],
            "tools": []
        }

        system_status = current_context.get("system_status", {})

        # System resource needs
        if system_status.get("memory_percent", 0) > 80:
            resource_needs["system_resources"]["memory"] = "high"

        if system_status.get("cpu_percent", 0) > 70:
            resource_needs["system_resources"]["cpu"] = "high"

        # Analyze recent actions for service needs
        if session_history:
            recent_actions = [
                h.get("intention", {}).get("primary_action")
                for h in session_history[-5:]
            ]

            if "research" in recent_actions:
                resource_needs["external_services"].append("web_search")
                resource_needs["data_sources"].append("online_databases")

            if "email_management" in recent_actions:
                resource_needs["external_services"].append("email_api")

            if "document_creation" in recent_actions:
                resource_needs["tools"].append("document_editor")

        return resource_needs

    async def _detect_potential_blockers(self, current_context: Dict) -> List[Dict]:
        """Detect potential blockers or issues"""
        blockers = []

        system_status = current_context.get("system_status", {})

        # System-related blockers
        if system_status.get("memory_percent", 0) > 95:
            blockers.append({
                "type": "system",
                "severity": "high",
                "description": "Critical memory usage may cause system instability",
                "suggested_action": "free_memory_immediately"
            })

        if not system_status.get("network", {}).get("connected", True):
            blockers.append({
                "type": "network",
                "severity": "high",
                "description": "No network connection detected",
                "suggested_action": "check_network_connection"
            })

        # Application-related blockers
        active_apps = current_context.get("active_apps", [])
        if len(active_apps) > 20:
            blockers.append({
                "type": "performance",
                "severity": "medium",
                "description": "Many applications running may slow down system",
                "suggested_action": "close_unused_applications"
            })

        return blockers

    async def _find_optimization_opportunities(self, current_context: Dict,
                                             session_history: List[Dict]) -> List[Dict]:
        """Find optimization opportunities"""
        opportunities = []

        if not session_history:
            return opportunities

        # Analyze repetitive tasks
        action_counts = {}
        for interaction in session_history[-10:]:
            action = interaction.get("intention", {}).get("primary_action")
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1

        for action, count in action_counts.items():
            if count >= 3:
                opportunities.append({
                    "type": "automation",
                    "description": f"Automate repetitive {action} tasks",
                    "potential_time_saved": count * 5,  # Estimate 5 minutes per task
                    "implementation_effort": "medium"
                })

        # System optimization opportunities
        system_status = current_context.get("system_status", {})
        if system_status.get("memory_percent", 0) > 70:
            opportunities.append({
                "type": "system_optimization",
                "description": "Optimize memory usage by closing unused applications",
                "potential_benefit": "improved_performance",
                "implementation_effort": "low"
            })

        return opportunities

    async def _generate_preparation_suggestions(self, anticipation: Dict) -> List[Dict]:
        """Generate preparation suggestions based on anticipation"""
        suggestions = []

        # Prepare for likely next tasks
        for task in anticipation.get("likely_next_tasks", []):
            if task["probability"] > 0.6:
                suggestions.append({
                    "type": "preparation",
                    "message": f"Prepare for {task['task']} by gathering necessary resources",
                    "action": f"prepare_{task['task']}",
                    "priority": "medium"
                })

        # Address potential blockers
        for blocker in anticipation.get("potential_blockers", []):
            if blocker["severity"] == "high":
                suggestions.append({
                    "type": "prevention",
                    "message": f"Address {blocker['description']} before it becomes critical",
                    "action": blocker["suggested_action"],
                    "priority": "high"
                })

        return suggestions

    def _store_suggestions(self, suggestions: List[Dict]):
        """Store suggestions for feedback learning"""
        timestamp = datetime.now().isoformat()

        for suggestion in suggestions:
            suggestion["timestamp"] = timestamp
            suggestion["shown"] = True
            suggestion["feedback"] = None

        self.suggestion_history.extend(suggestions)

        # Keep only recent suggestions (last 100)
        if len(self.suggestion_history) > 100:
            self.suggestion_history = self.suggestion_history[-100:]

    async def record_suggestion_feedback(self, suggestion_id: str, feedback: Dict):
        """Record user feedback on suggestions"""
        for suggestion in self.suggestion_history:
            if suggestion.get("id") == suggestion_id:
                suggestion["feedback"] = feedback

                # Learn from feedback
                action = suggestion.get("action", "")
                helpful = feedback.get("helpful", False)

                self.user_feedback[action].append({
                    "helpful": helpful,
                    "timestamp": datetime.now().isoformat(),
                    "context": feedback.get("context", {})
                })

                break

    def get_suggestion_effectiveness(self) -> Dict:
        """Get effectiveness metrics for suggestions"""
        total_suggestions = len(self.suggestion_history)
        if total_suggestions == 0:
            return {"total": 0, "feedback_rate": 0, "helpfulness_rate": 0}

        suggestions_with_feedback = [
            s for s in self.suggestion_history if s.get("feedback") is not None
        ]

        helpful_suggestions = [
            s for s in suggestions_with_feedback
            if s.get("feedback", {}).get("helpful", False)
        ]

        return {
            "total_suggestions": total_suggestions,
            "feedback_rate": len(suggestions_with_feedback) / total_suggestions,
            "helpfulness_rate": len(helpful_suggestions) / max(len(suggestions_with_feedback), 1),
            "most_helpful_types": self._get_most_helpful_types(),
            "least_helpful_types": self._get_least_helpful_types()
        }

    def _get_most_helpful_types(self) -> List[str]:
        """Get most helpful suggestion types"""
        type_scores = defaultdict(list)

        for suggestion in self.suggestion_history:
            if suggestion.get("feedback"):
                suggestion_type = suggestion.get("type", "unknown")
                helpful = suggestion.get("feedback", {}).get("helpful", False)
                type_scores[suggestion_type].append(helpful)

        # Calculate average helpfulness by type
        type_averages = {
            stype: sum(scores) / len(scores)
            for stype, scores in type_scores.items()
            if len(scores) >= 3  # Minimum 3 feedback instances
        }

        return sorted(type_averages.keys(), key=lambda x: type_averages[x], reverse=True)[:3]

    def _get_least_helpful_types(self) -> List[str]:
        """Get least helpful suggestion types"""
        most_helpful = self._get_most_helpful_types()

        type_scores = defaultdict(list)
        for suggestion in self.suggestion_history:
            if suggestion.get("feedback"):
                suggestion_type = suggestion.get("type", "unknown")
                helpful = suggestion.get("feedback", {}).get("helpful", False)
                type_scores[suggestion_type].append(helpful)

        type_averages = {
            stype: sum(scores) / len(scores)
            for stype, scores in type_scores.items()
            if len(scores) >= 3
        }

        return sorted(type_averages.keys(), key=lambda x: type_averages[x])[:3]
