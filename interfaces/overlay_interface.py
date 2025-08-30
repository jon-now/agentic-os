#!/usr/bin/env python3
from typing import Any, Dict, Set
"""
Overlay Interface for Agentic OS
Provides system-wide overlay capabilities for notifications, quick actions, and status display
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, font
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None

try:
    import pystray
    from PIL import Image, ImageDraw
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False
    pystray = None

try:
    import plyer
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    plyer = None

logger = logging.getLogger(__name__)

class OverlayInterface:
    """
    Advanced overlay interface for system-wide Agentic OS integration
    Provides notifications, quick actions, status display, and system tray integration
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.is_active = False
        self.overlay_window = None
        self.tray_icon = None
        self.notifications = []
        self.quick_actions = []
        self.status_widgets = {}

        # Configuration
        self.config = {
            "overlay_position": "top-right",  # top-left, top-right, bottom-left, bottom-right, center
            "overlay_size": (300, 400),
            "auto_hide_delay": 5000,  # milliseconds
            "transparency": 0.9,
            "always_on_top": True,
            "show_system_tray": True,
            "notification_duration": 5000,
            "quick_action_hotkeys": True,
            "theme": "dark",  # dark, light, auto
            "font_size": 10,
            "show_status_bar": True,
            "compact_mode": False
        }

        # Initialize components
        self._initialize_overlay()
        self._initialize_tray()
        self._setup_quick_actions()

        logger.info("Overlay interface initialized")

    def _initialize_overlay(self):
        """Initialize the overlay window"""
        if not TKINTER_AVAILABLE:
            logger.warning("Tkinter not available - overlay disabled")
            return

        try:
            # Create root window (hidden)
            self.root = tk.Tk()
            self.root.withdraw()  # Hide initially

            # Create overlay window
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.withdraw()  # Hide initially

            # Configure overlay window
            self._configure_overlay_window()
            self._create_overlay_widgets()

            logger.info("Overlay window initialized")

        except Exception as e:
            logger.error("Failed to initialize overlay: %s", e)
            self.overlay_window = None

    def _configure_overlay_window(self):
        """Configure overlay window properties"""
        if not self.overlay_window:
            return

        # Window properties
        self.overlay_window.title("Agentic OS Overlay")
        self.overlay_window.geometry(f"{self.config['overlay_size'][0]}x{self.config['overlay_size'][1]}")

        # Make window stay on top
        if self.config["always_on_top"]:
            self.overlay_window.attributes('-topmost', True)

        # Set transparency
        self.overlay_window.attributes('-alpha', self.config["transparency"])

        # Remove window decorations for clean overlay look
        self.overlay_window.overrideredirect(True)

        # Position window
        self._position_overlay()

        # Configure theme
        self._apply_theme()

        # Bind events
        self.overlay_window.bind('<Button-1>', self._on_overlay_click)
        self.overlay_window.bind('<Double-Button-1>', self._on_overlay_double_click)
        self.overlay_window.bind('<Button-3>', self._show_context_menu)

    def _position_overlay(self):
        """Position overlay window based on configuration"""
        if not self.overlay_window:
            return

        # Get screen dimensions
        screen_width = self.overlay_window.winfo_screenwidth()
        screen_height = self.overlay_window.winfo_screenheight()

        # Get window dimensions
        window_width, window_height = self.config["overlay_size"]

        # Calculate position
        position = self.config["overlay_position"]

        if position == "top-left":
            x, y = 10, 10
        elif position == "top-right":
            x, y = screen_width - window_width - 10, 10
        elif position == "bottom-left":
            x, y = 10, screen_height - window_height - 50
        elif position == "bottom-right":
            x, y = screen_width - window_width - 10, screen_height - window_height - 50
        elif position == "center":
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
        else:
            x, y = 10, 10  # Default to top-left

        self.overlay_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def _apply_theme(self):
        """Apply theme to overlay window"""
        if not self.overlay_window:
            return

        theme = self.config["theme"]

        if theme == "dark":
            bg_color = "#2b2b2b"
            fg_color = "#fffff"
            accent_color = "#0078d4"
        elif theme == "light":
            bg_color = "#fffff"
            fg_color = "#000000"
            accent_color = "#0078d4"
        else:  # auto - detect system theme
            bg_color = "#2b2b2b"  # Default to dark
            fg_color = "#fffff"
            accent_color = "#0078d4"

        self.overlay_window.configure(bg=bg_color)

        # Store theme colors for widgets
        self.theme_colors = {
            "bg": bg_color,
            "fg": fg_color,
            "accent": accent_color
        }

    def _create_overlay_widgets(self):
        """Create overlay window widgets"""
        if not self.overlay_window:
            return

        # Main frame
        self.main_frame = tk.Frame(
            self.overlay_window,
            bg=self.theme_colors["bg"],
            padx=10,
            pady=10
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title bar
        self.title_frame = tk.Frame(self.main_frame, bg=self.theme_colors["bg"])
        self.title_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = tk.Label(
            self.title_frame,
            text="🤖 Agentic OS",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["accent"],
            font=("Arial", 12, "bold")
        )
        self.title_label.pack(side=tk.LEFT)

        # Close button
        self.close_button = tk.Button(
            self.title_frame,
            text="×",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 12, "bold"),
            bd=0,
            command=self.hide_overlay,
            width=2
        )
        self.close_button.pack(side=tk.RIGHT)

        # Status section
        if self.config["show_status_bar"]:
            self._create_status_section()

        # Quick actions section
        self._create_quick_actions_section()

        # Notifications section
        self._create_notifications_section()

        # Input section
        self._create_input_section()

    def _create_status_section(self):
        """Create system status section"""
        self.status_frame = tk.LabelFrame(
            self.main_frame,
            text="System Status",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 9)
        )
        self.status_frame.pack(fill=tk.X, pady=(0, 10))

        # CPU and Memory status
        self.cpu_label = tk.Label(
            self.status_frame,
            text="CPU: ---%",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 8)
        )
        self.cpu_label.pack(anchor=tk.W, padx=5, pady=2)

        self.memory_label = tk.Label(
            self.status_frame,
            text="Memory: ---%",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 8)
        )
        self.memory_label.pack(anchor=tk.W, padx=5, pady=2)

        # AI Status
        self.ai_status_label = tk.Label(
            self.status_frame,
            text="AI: Ready",
            bg=self.theme_colors["bg"],
            fg="#00ff00",
            font=("Arial", 8)
        )
        self.ai_status_label.pack(anchor=tk.W, padx=5, pady=2)

    def _create_quick_actions_section(self):
        """Create quick actions section"""
        self.actions_frame = tk.LabelFrame(
            self.main_frame,
            text="Quick Actions",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 9)
        )
        self.actions_frame.pack(fill=tk.X, pady=(0, 10))

        # Action buttons frame
        self.buttons_frame = tk.Frame(self.actions_frame, bg=self.theme_colors["bg"])
        self.buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create action buttons
        self.action_buttons = {}
        self._update_quick_actions()

    def _create_notifications_section(self):
        """Create notifications section"""
        self.notifications_frame = tk.LabelFrame(
            self.main_frame,
            text="Notifications",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 9)
        )
        self.notifications_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable notifications list
        self.notifications_canvas = tk.Canvas(
            self.notifications_frame,
            bg=self.theme_colors["bg"],
            highlightthickness=0,
            height=100
        )
        self.notifications_scrollbar = ttk.Scrollbar(
            self.notifications_frame,
            orient="vertical",
            command=self.notifications_canvas.yview
        )
        self.notifications_scrollable_frame = tk.Frame(
            self.notifications_canvas,
            bg=self.theme_colors["bg"]
        )

        self.notifications_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.notifications_canvas.configure(
                scrollregion=self.notifications_canvas.bbox("all")
            )
        )

        self.notifications_canvas.create_window(
            (0, 0),
            window=self.notifications_scrollable_frame,
            anchor="nw"
        )
        self.notifications_canvas.configure(yscrollcommand=self.notifications_scrollbar.set)

        self.notifications_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.notifications_scrollbar.pack(side="right", fill="y")

    def _create_input_section(self):
        """Create input section for quick commands"""
        self.input_frame = tk.Frame(self.main_frame, bg=self.theme_colors["bg"])
        self.input_frame.pack(fill=tk.X)

        self.input_entry = tk.Entry(
            self.input_frame,
            bg="#333333" if self.config["theme"] == "dark" else "#fffff",
            fg=self.theme_colors["fg"],
            font=("Arial", 9),
            bd=1,
            relief=tk.SOLID
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind('<Return>', self._on_input_submit)

        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            bg=self.theme_colors["accent"],
            fg="#fffff",
            font=("Arial", 8),
            bd=0,
            command=self._on_input_submit
        )
        self.send_button.pack(side=tk.RIGHT)

    def _initialize_tray(self):
        """Initialize system tray icon"""
        if not self.config["show_system_tray"] or not PYSTRAY_AVAILABLE:
            return

        try:
            # Create tray icon image
            image = self._create_tray_icon()

            # Create tray menu
            menu = pystray.Menu(
                pystray.MenuItem("Show Overlay", self.show_overlay),
                pystray.MenuItem("Hide Overlay", self.hide_overlay),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Voice Control", self._toggle_voice),
                pystray.MenuItem("System Status", self._show_system_status),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Settings", self._show_settings),
                pystray.MenuItem("Exit", self._exit_application)
            )

            # Create tray icon
            self.tray_icon = pystray.Icon(
                "agentic_os",
                image,
                "Agentic OS",
                menu
            )

            # Start tray icon in separate thread
            tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            tray_thread.start()

            logger.info("System tray icon initialized")

        except Exception as e:
            logger.error("Failed to initialize system tray: %s", e)
            self.tray_icon = None

    def _create_tray_icon(self):
        """Create system tray icon image"""
        # Create a simple icon
        width = height = 64
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw a simple robot icon
        # Head
        draw.ellipse([16, 8, 48, 40], fill=(0, 120, 212, 255))
        # Eyes
        draw.ellipse([22, 18, 26, 22], fill=(255, 255, 255, 255))
        draw.ellipse([38, 18, 42, 22], fill=(255, 255, 255, 255))
        # Body
        draw.rectangle([20, 35, 44, 55], fill=(0, 120, 212, 255))

        return image

    def _setup_quick_actions(self):
        """Setup default quick actions"""
        self.quick_actions = [
            {
                "name": "System Status",
                "icon": "📊",
                "action": self._action_system_status,
                "hotkey": "Ctrl+Alt+S"
            },
            {
                "name": "Check Email",
                "icon": "📧",
                "action": self._action_check_email,
                "hotkey": "Ctrl+Alt+E"
            },
            {
                "name": "Research",
                "icon": "🔍",
                "action": self._action_research,
                "hotkey": "Ctrl+Alt+R"
            },
            {
                "name": "Voice Mode",
                "icon": "🎤",
                "action": self._action_voice_mode,
                "hotkey": "Ctrl+Alt+V"
            }
        ]

    def _update_quick_actions(self):
        """Update quick action buttons"""
        if not hasattr(self, 'buttons_frame'):
            return

        # Clear existing buttons
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

        # Create new buttons
        for i, action in enumerate(self.quick_actions):
            btn = tk.Button(
                self.buttons_frame,
                text=f"{action['icon']} {action['name']}",
                bg=self.theme_colors["accent"],
                fg="#fffff",
                font=("Arial", 8),
                bd=0,
                command=action["action"],
                width=15
            )

            # Arrange buttons in grid
            row = i // 2
            col = i % 2
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")

        # Configure grid weights
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)

    async def show_overlay(self):
        """Show the overlay window"""
        if not self.overlay_window:
            return

        self.overlay_window.deiconify()
        self.overlay_window.lift()
        self.is_active = True

        # Update status
        await self._update_status()

        # Auto-hide timer
        if self.config["auto_hide_delay"] > 0:
            self.overlay_window.after(
                self.config["auto_hide_delay"],
                self.hide_overlay
            )

        logger.info("Overlay shown")

    def hide_overlay(self):
        """Hide the overlay window"""
        if not self.overlay_window:
            return

        self.overlay_window.withdraw()
        self.is_active = False
        logger.info("Overlay hidden")

    def toggle_overlay(self):
        """Toggle overlay visibility"""
        if self.is_active:
            self.hide_overlay()
        else:
            asyncio.create_task(self.show_overlay())

    async def show_notification(self, title: str, message: str,
                              notification_type: str = "info",
                              duration: int = None):
        """Show a notification"""
        if duration is None:
            duration = self.config["notification_duration"]

        notification = {
            "id": len(self.notifications),
            "title": title,
            "message": message,
            "type": notification_type,
            "timestamp": datetime.now(),
            "duration": duration
        }

        self.notifications.append(notification)

        # Show system notification if available
        if PLYER_AVAILABLE:
            try:
                plyer.notification.notify(
                    title=title,
                    message=message,
                    timeout=duration // 1000
                )
            except Exception as e:
                logger.debug("System notification failed: %s", e)

        # Update overlay notifications
        self._update_notifications_display()

        # Auto-remove notification
        if duration > 0:
            def remove_notification():
                self.notifications = [n for n in self.notifications if n["id"] != notification["id"]]
                # Schedule UI update on main thread
                if self.overlay_window:
                    self.overlay_window.after(0, self._update_notifications_display)

            threading.Timer(duration / 1000, remove_notification).start()

        logger.info("Notification shown: %s", title)

    def _update_notifications_display(self):
        """Update notifications display in overlay"""
        if not hasattr(self, 'notifications_scrollable_frame'):
            return

        try:
            # Clear existing notifications
            for widget in self.notifications_scrollable_frame.winfo_children():
                widget.destroy()

            # Show recent notifications (last 5)
            recent_notifications = self.notifications[-5:]

            for notification in recent_notifications:
                self._create_notification_widget(notification)
        except Exception as e:
            logger.debug("Error updating notifications display: %s", e)

    def _create_notification_widget(self, notification):
        """Create a notification widget"""
        # Notification frame
        notif_frame = tk.Frame(
            self.notifications_scrollable_frame,
            bg="#333333" if self.config["theme"] == "dark" else "#f0f0f0",
            relief=tk.RAISED,
            bd=1
        )
        notif_frame.pack(fill=tk.X, padx=2, pady=2)

        # Type icon
        type_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        icon = type_icons.get(notification["type"], "ℹ️")

        # Title
        title_label = tk.Label(
            notif_frame,
            text=f"{icon} {notification['title']}",
            bg=notif_frame["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 8, "bold")
        )
        title_label.pack(anchor=tk.W, padx=5, pady=(2, 0))

        # Message
        message_label = tk.Label(
            notif_frame,
            text=notification["message"][:100] + ("..." if len(notification["message"]) > 100 else ""),
            bg=notif_frame["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 7),
            wraplength=250,
            justify=tk.LEFT
        )
        message_label.pack(anchor=tk.W, padx=5, pady=(0, 2))

        # Timestamp
        time_str = notification["timestamp"].strftime("%H:%M")
        time_label = tk.Label(
            notif_frame,
            text=time_str,
            bg=notif_frame["bg"],
            fg="#888888",
            font=("Arial", 6)
        )
        time_label.pack(anchor=tk.E, padx=5)

    async def _update_status(self):
        """Update system status display"""
        if not hasattr(self, 'cpu_label'):
            return

        try:
            import psutil

            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")

            # Update memory usage
            memory = psutil.virtual_memory()
            self.memory_label.config(text=f"Memory: {memory.percent:.1f}%")

            # Update AI status
            if self.orchestrator:
                ai_status = "Ready"
                ai_color = "#00ff00"
            else:
                ai_status = "Disconnected"
                ai_color = "#ff0000"

            self.ai_status_label.config(text=f"AI: {ai_status}", fg=ai_color)

        except Exception as e:
            logger.debug("Status update error: %s", e)

    # Event handlers
    def _on_overlay_click(self, event):
        """Handle overlay click"""
        pass

    def _on_overlay_double_click(self, event):
        """Handle overlay double-click"""
        self.hide_overlay()

    def _show_context_menu(self, event):
        """Show context menu"""
        context_menu = tk.Menu(self.overlay_window, tearoff=0)
        context_menu.add_command(label="Settings", command=self._show_settings)
        context_menu.add_command(label="Hide", command=self.hide_overlay)
        context_menu.add_separator()
        context_menu.add_command(label="Exit", command=self._exit_application)

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def _on_input_submit(self, event=None):
        """Handle input submission"""
        text = self.input_entry.get().strip()
        if not text:
            return

        self.input_entry.delete(0, tk.END)

        # Process command
        asyncio.create_task(self._process_overlay_command(text))

    async def _process_overlay_command(self, text: str):
        """Process overlay command"""
        try:
            if self.orchestrator:
                result = await self.orchestrator.process_user_intention(text)
                response = result.get("result", {}).get("final_output", "No response")

                await self.show_notification(
                    "Command Processed",
                    response[:100] + ("..." if len(response) > 100 else ""),
                    "success"
                )
            else:
                await self.show_notification(
                    "Command Received",
                    f"Processed: {text}",
                    "info"
                )

        except Exception as e:
            await self.show_notification(
                "Command Error",
                f"Error processing command: {str(e)}",
                "error"
            )

    # Quick action handlers
    def _action_system_status(self):
        """System status quick action"""
        asyncio.create_task(self._show_system_status())

    async def _show_system_status(self):
        """Show system status"""
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            status_message = f"CPU: {cpu:.1f}%\nMemory: {memory.percent:.1f}%\nDisk: {disk.percent:.1f}%"

            await self.show_notification(
                "System Status",
                status_message,
                "info"
            )
        except Exception as e:
            await self.show_notification(
                "System Status Error",
                f"Could not get system status: {e}",
                "error"
            )

    def _action_check_email(self):
        """Check email quick action"""
        asyncio.create_task(self._check_email())

    async def _check_email(self):
        """Check email"""
        if self.orchestrator:
            try:
                result = await self.orchestrator.process_user_intention("check my emails")
                response = result.get("result", {}).get("final_output", "No email response")

                await self.show_notification(
                    "Email Check",
                    response[:100] + ("..." if len(response) > 100 else ""),
                    "info"
                )
            except Exception as e:
                await self.show_notification(
                    "Email Error",
                    f"Could not check emails: {e}",
                    "error"
                )
        else:
            await self.show_notification(
                "Email Check",
                "Email checking not available - orchestrator not connected",
                "warning"
            )

    def _action_research(self):
        """Research quick action"""
        # Show input dialog for research topic
        if TKINTER_AVAILABLE:
            topic = tk.simpledialog.askstring("Research", "Enter research topic:")
            if topic:
                asyncio.create_task(self._do_research(topic))

    async def _do_research(self, topic: str):
        """Perform research"""
        if self.orchestrator:
            try:
                result = await self.orchestrator.process_user_intention(f"research {topic}")
                response = result.get("result", {}).get("final_output", "No research response")

                await self.show_notification(
                    f"Research: {topic}",
                    "Research completed successfully",
                    "success"
                )
            except Exception as e:
                await self.show_notification(
                    "Research Error",
                    f"Research failed: {e}",
                    "error"
                )
        else:
            await self.show_notification(
                "Research",
                "Research not available - orchestrator not connected",
                "warning"
            )

    def _action_voice_mode(self):
        """Voice mode quick action"""
        asyncio.create_task(self._toggle_voice())

    async def _toggle_voice(self):
        """Toggle voice mode"""
        await self.show_notification(
            "Voice Mode",
            "Voice mode toggle requested",
            "info"
        )

    def _show_settings(self):
        """Show settings dialog"""
        if not TKINTER_AVAILABLE:
            return

        settings_window = tk.Toplevel(self.root)
        settings_window.title("Overlay Settings")
        settings_window.geometry("400x500")
        settings_window.configure(bg=self.theme_colors["bg"])

        # Settings content would go here
        tk.Label(
            settings_window,
            text="Overlay Settings",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
            font=("Arial", 14, "bold")
        ).pack(pady=20)

        tk.Label(
            settings_window,
            text="Settings panel coming soon...",
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"]
        ).pack(pady=10)

    def _exit_application(self):
        """Exit the application"""
        self.close()

    def set_config(self, config: Dict[str, Any]):
        """Update overlay configuration"""
        self.config.update(config)

        # Apply configuration changes
        if self.overlay_window:
            self._configure_overlay_window()
            self._apply_theme()

    def get_status(self) -> Dict[str, Any]:
        """Get overlay interface status"""
        return {
            "is_active": self.is_active,
            "tkinter_available": TKINTER_AVAILABLE,
            "pystray_available": PYSTRAY_AVAILABLE,
            "plyer_available": PLYER_AVAILABLE,
            "overlay_initialized": self.overlay_window is not None,
            "tray_initialized": self.tray_icon is not None,
            "notifications_count": len(self.notifications),
            "quick_actions_count": len(self.quick_actions),
            "config": self.config
        }

    def close(self):
        """Clean up overlay interface"""
        self.is_active = False

        if self.overlay_window:
            self.overlay_window.destroy()

        if self.root:
            self.root.quit()

        if self.tray_icon:
            self.tray_icon.stop()

        logger.info("Overlay interface closed")

# Example usage and testing
async def test_overlay_interface():
    """Test the overlay interface"""
    print("Testing Overlay Interface...")

    overlay = OverlayInterface()
    status = overlay.get_status()

    print("Overlay Status:")
    print(f"  - Tkinter Available: {status['tkinter_available']}")
    print(f"  - Pystray Available: {status['pystray_available']}")
    print(f"  - Plyer Available: {status['plyer_available']}")
    print(f"  - Overlay Initialized: {status['overlay_initialized']}")
    print(f"  - Tray Initialized: {status['tray_initialized']}")

    if status['overlay_initialized']:
        print("\nShowing overlay...")
        await overlay.show_overlay()

        # Test notifications
        await overlay.show_notification(
            "Test Notification",
            "This is a test notification from the overlay interface",
            "info"
        )

        await asyncio.sleep(2)

        await overlay.show_notification(
            "Success Test",
            "This is a success notification",
            "success"
        )

        print("Overlay test running... Close the overlay window to continue.")

        # Keep running until overlay is closed
        while overlay.is_active:
            await asyncio.sleep(1)

    overlay.close()
    print("Overlay interface test completed!")

if __name__ == "__main__":
    asyncio.run(test_overlay_interface())
