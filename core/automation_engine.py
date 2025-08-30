import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class AutomationTask:
    def __init__(self, task_id: str, action: str, parameters: Dict, priority: TaskPriority = TaskPriority.MEDIUM):
        self.task_id = task_id
        self.action = action
        self.parameters = parameters
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.progress = 0.0
    
    def __lt__(self, other):
        """Less than comparison for priority queue sorting"""
        if not isinstance(other, AutomationTask):
            return NotImplemented
        # Lower priority value = higher priority (opposite of normal comparison)
        return self.priority.value < other.priority.value
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, AutomationTask):
            return NotImplemented
        return self.task_id == other.task_id
    
    def __hash__(self):
        """Hash function for task"""
        return hash(self.task_id)

class AutomationEngine:
    def __init__(self):
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.running = False
        self.worker_task = None
        self.max_concurrent_tasks = 3

    async def start(self):
        """Start the automation engine"""
        if not self.running:
            self.running = True
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Automation engine started")

    async def stop(self):
        """Stop the automation engine"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Automation engine stopped")

    async def submit_task(self, action: str, parameters: Dict, priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Submit a new automation task"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_tasks)}"
        task = AutomationTask(task_id, action, parameters, priority)

        # Add to priority queue - tasks are now directly comparable
        await self.task_queue.put(task)

        logger.info(f"Task submitted: {task_id} - {action}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self._task_to_dict(task)

        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return self._task_to_dict(task)

        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            logger.info("Task cancelled: %s", task_id)
            return True
        return False

    async def _worker(self):
        """Main worker loop for processing tasks"""
        while self.running:
            try:
                # Check if we can process more tasks
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue

                # Get next task from queue
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Start processing task
                self.active_tasks[task.task_id] = task
                asyncio.create_task(self._process_task(task))

            except Exception as e:
                logger.error("Worker error: %s", e)
                await asyncio.sleep(1)

    async def _process_task(self, task: AutomationTask):
        """Process a single automation task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

            logger.info("Processing task: {task.task_id} - %s", task.action)

            # Route task to appropriate handler
            result = await self._execute_task_action(task)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0

        except Exception as e:
            logger.error("Task {task.task_id} failed: %s", e)
            task.error = str(e)
            task.status = TaskStatus.FAILED

        finally:
            task.completed_at = datetime.now()

            # Move to completed tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task

            # Clean up old completed tasks (keep last 100)
            if len(self.completed_tasks) > 100:
                oldest_tasks = sorted(self.completed_tasks.items(),
                                    key=lambda x: x[1].completed_at)
                for task_id, _ in oldest_tasks[:-100]:
                    del self.completed_tasks[task_id]

    async def _execute_task_action(self, task: AutomationTask) -> Dict:
        """Execute the specific action for a task"""
        action = task.action
        parameters = task.parameters

        if action == "research_topic":
            return await self._handle_research_task(parameters)
        elif action == "check_emails":
            return await self._handle_email_task(parameters)
        elif action == "system_analysis":
            return await self._handle_system_task(parameters)
        elif action == "file_operation":
            return await self._handle_file_task(parameters)
        elif action == "web_automation":
            return await self._handle_web_task(parameters)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _handle_research_task(self, parameters: Dict) -> Dict:
        """Handle research automation tasks"""
        topic = parameters.get("topic", "")
        depth = parameters.get("depth", "comprehensive")

        # This would integrate with browser controller
        return {
            "action": "research_topic",
            "topic": topic,
            "status": "completed",
            "summary": f"Research completed for: {topic}",
            "sources_found": 3,
            "key_insights": ["Insight 1", "Insight 2", "Insight 3"]
        }

    async def _handle_email_task(self, parameters: Dict) -> Dict:
        """Handle email automation tasks"""
        action_type = parameters.get("type", "check")

        # This would integrate with email controller
        return {
            "action": "email_management",
            "type": action_type,
            "status": "completed",
            "emails_processed": 10,
            "urgent_emails": 2,
            "summary": "Email check completed successfully"
        }

    async def _handle_system_task(self, parameters: Dict) -> Dict:
        """Handle system analysis tasks"""
        analysis_type = parameters.get("type", "status")

        # This would integrate with context manager
        return {
            "action": "system_analysis",
            "type": analysis_type,
            "status": "completed",
            "cpu_usage": "Normal",
            "memory_usage": "Normal",
            "recommendations": ["System running optimally"]
        }

    async def _handle_file_task(self, parameters: Dict) -> Dict:
        """Handle file operation tasks"""
        operation = parameters.get("operation", "list")
        path = parameters.get("path", "")

        return {
            "action": "file_operation",
            "operation": operation,
            "path": path,
            "status": "completed",
            "files_processed": 5
        }

    async def _handle_web_task(self, parameters: Dict) -> Dict:
        """Handle web automation tasks"""
        url = parameters.get("url", "")
        action_type = parameters.get("type", "navigate")

        return {
            "action": "web_automation",
            "url": url,
            "type": action_type,
            "status": "completed",
            "data_extracted": True
        }

    def _task_to_dict(self, task: AutomationTask) -> Dict:
        """Convert task object to dictionary"""
        return {
            "task_id": task.task_id,
            "action": task.action,
            "parameters": task.parameters,
            "priority": task.priority.name,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "progress": task.progress,
            "result": task.result,
            "error": task.error
        }

    def get_engine_status(self) -> Dict:
        """Get overall engine status"""
        return {
            "running": self.running,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queue_size": self.task_queue.qsize(),
            "max_concurrent": self.max_concurrent_tasks
        }
