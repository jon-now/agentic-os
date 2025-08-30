import psutil
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CrossAppContextManager:
    def __init__(self):
        self.context_cache = {}
        self.cache_duration = timedelta(minutes=5)

    async def get_full_context(self) -> Dict:
        """Get comprehensive system context"""
        current_time = datetime.now()

        # Check cache freshness
        if (self.context_cache.get('timestamp') and
            current_time - self.context_cache['timestamp'] < self.cache_duration):
            return self.context_cache['data']

        context = {
            "current_time": current_time.isoformat(),
            "active_apps": self._get_active_applications(),
            "recent_files": self._get_recent_files(),
            "system_status": self._get_system_status(),
            "browser_tabs": [],  # Will be populated by browser controller
            "email_summary": {},  # Will be populated by email controller
            "calendar_events": [],  # Will be populated by calendar controller
            "memory_context": await self._get_memory_context()
        }

        # Cache the context
        self.context_cache = {
            'timestamp': current_time,
            'data': context
        }

        return context

    def _get_active_applications(self) -> List[str]:
        """Get list of currently running applications"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    # Filter for common applications
                    if any(app in proc_name for app in [
                        'chrome', 'firefox', 'safari', 'edge',
                        'code', 'sublime', 'atom', 'notepad',
                        'word', 'excel', 'powerpoint', 'outlook',
                        'slack', 'teams', 'zoom', 'discord',
                        'spotify', 'vlc', 'photoshop', 'illustrator'
                    ]):
                        processes.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return list(set(processes))[:10]  # Limit to 10 most relevant
        except Exception as e:
            logger.error("Error getting active applications: %s", e)
            return []

    def _get_recent_files(self) -> List[Dict]:
        """Get recently modified files in common directories"""
        try:
            recent_files = []
            home_dir = Path.home()

            # Check common directories
            search_dirs = [
                home_dir / "Documents",
                home_dir / "Desktop",
                home_dir / "Downloads"
            ]

            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue

                try:
                    for file_path in search_dir.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            # Get file stats
                            stat = file_path.stat()
                            modified_time = datetime.fromtimestamp(stat.st_mtime)

                            # Only include files modified in last 7 days
                            if datetime.now() - modified_time < timedelta(days=7):
                                recent_files.append({
                                    "name": file_path.name,
                                    "path": str(file_path),
                                    "modified": modified_time.isoformat(),
                                    "size": stat.st_size,
                                    "type": file_path.suffix
                                })
                except PermissionError:
                    continue

            # Sort by modification time and return top 20
            recent_files.sort(key=lambda x: x["modified"], reverse=True)
            return recent_files[:20]

        except Exception as e:
            logger.error("Error getting recent files: %s", e)
            return []

    def _get_system_status(self) -> Dict:
        """Get basic system status information"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                "battery": self._get_battery_status(),
                "network": self._get_network_status(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total
            }
        except Exception as e:
            logger.error("Error getting system status: %s", e)
            return {}

    def _get_battery_status(self) -> Optional[Dict]:
        """Get battery status if available"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "percent": battery.percent,
                    "plugged": battery.power_plugged,
                    "time_left": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
        except Exception:
            pass
        return None

    def _get_network_status(self) -> Dict:
        """Get basic network connectivity status"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('8.8.8.8', 53))
            sock.close()

            # Get network interfaces
            interfaces = []
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        interfaces.append({
                            "interface": interface,
                            "ip": addr.address
                        })

            return {
                "connected": result == 0,
                "interfaces": interfaces[:3]  # Limit to 3 interfaces
            }
        except Exception:
            return {"connected": False, "interfaces": []}

    async def _get_memory_context(self) -> Dict:
        """Get context from vector memory (placeholder for now)"""
        # This will be implemented when vector database is added
        return {
            "recent_conversations": [],
            "learned_preferences": {},
            "task_history": []
        }

    async def store_interaction(self, user_input: str, response: str, context: Dict):
        """Store interaction in memory for future context"""
        # Placeholder for vector database storage
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "context_snapshot": {
                "active_apps": context.get("active_apps", []),
                "system_status": context.get("system_status", {})
            }
        }

        # For now, just log it
        logger.info("Storing interaction: %s...", user_input[:50])

    def get_context_summary(self, context: Dict) -> str:
        """Generate a human-readable context summary"""
        try:
            active_apps = len(context.get("active_apps", []))
            recent_files = len(context.get("recent_files", []))
            cpu = context.get("system_status", {}).get("cpu_percent", "Unknown")
            memory = context.get("system_status", {}).get("memory_percent", "Unknown")

            return f"System: {active_apps} apps running, CPU {cpu}%, Memory {memory}%, {recent_files} recent files"
        except Exception:
            return "Context summary unavailable"
