import os
import pickle
from datetime import datetime, timedelta
import logging
from pathlib import Path
from config.settings import settings
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

try:
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import dateutil.parser
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    logger.warning("Google Calendar API dependencies not available")

try:
    import caldav
    from caldav import DAVClient
    CALDAV_AVAILABLE = True
except ImportError:
    CALDAV_AVAILABLE = False
    logger.warning("CalDAV dependencies not available. Install with: pip install caldav")

class CalendarController:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.google_service = None
        self.caldav_client = None
        self.authenticated = False

        # Try to initialize available calendar services
        if GOOGLE_CALENDAR_AVAILABLE:
            self._authenticate_google()

        if CALDAV_AVAILABLE:
            self._setup_caldav()

    def _authenticate_google(self):
        """Authenticate with Google Calendar API"""
        try:
            creds = None
            token_path = settings.CREDENTIALS_PATH / "calendar_token.pickle"
            creds_path = settings.CREDENTIALS_PATH / "credentials.json"

            # Load existing token
            if token_path.exists():
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh or get new token
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not creds_path.exists():
                        logger.info("Google Calendar credentials not found")
                        return

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_path), self.SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save token
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)

            self.google_service = build('calendar', 'v3', credentials=creds)
            self.authenticated = True
            logger.info("Google Calendar authenticated successfully")

        except Exception as e:
            logger.error("Google Calendar authentication failed: %s", e)

    def _setup_caldav(self):
        """Setup CalDAV client for local/open calendar servers"""
        # This would be configured for specific CalDAV servers
        # For now, it's a placeholder for future implementation
        pass

    async def get_upcoming_events(self, days_ahead: int = 7, max_results: int = 20) -> List[Dict]:
        """Get upcoming calendar events"""
        if not self.authenticated:
            return [{"error": "Calendar not authenticated"}]

        try:
            # Calculate time range
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'

            # Get events from Google Calendar
            events_result = self.google_service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            # Format events
            formatted_events = []
            for event in events:
                formatted_event = self._format_event(event)
                if formatted_event:
                    formatted_events.append(formatted_event)

            return formatted_events

        except HttpError as e:
            logger.error("Google Calendar API error: %s", e)
            return [{"error": f"Calendar API error: {str(e)}"}]
        except Exception as e:
            logger.error("Error fetching events: %s", e)
            return [{"error": f"Failed to fetch events: {str(e)}"}]

    async def create_event(self, title: str, start_time: str, end_time: str,
                          description: str = "", location: str = "") -> Dict:
        """Create a new calendar event"""
        if not self.authenticated:
            return {"error": "Calendar not authenticated"}

        try:
            # Parse datetime strings
            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)

            if not start_dt or not end_dt:
                return {"error": "Invalid datetime format"}

            # Create event object
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': 'UTC',
                },
            }

            if location:
                event['location'] = location

            # Create event
            created_event = self.google_service.events().insert(
                calendarId='primary', body=event
            ).execute()

            return {
                "event_id": created_event['id'],
                "title": title,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "status": "created",
                "link": created_event.get('htmlLink', '')
            }

        except Exception as e:
            logger.error("Failed to create event: %s", e)
            return {"error": f"Event creation failed: {str(e)}"}

    async def update_event(self, event_id: str, updates: Dict) -> Dict:
        """Update an existing calendar event"""
        if not self.authenticated:
            return {"error": "Calendar not authenticated"}

        try:
            # Get existing event
            event = self.google_service.events().get(
                calendarId='primary', eventId=event_id
            ).execute()

            # Apply updates
            if 'title' in updates:
                event['summary'] = updates['title']
            if 'description' in updates:
                event['description'] = updates['description']
            if 'location' in updates:
                event['location'] = updates['location']
            if 'start_time' in updates:
                start_dt = self._parse_datetime(updates['start_time'])
                if start_dt:
                    event['start']['dateTime'] = start_dt.isoformat()
            if 'end_time' in updates:
                end_dt = self._parse_datetime(updates['end_time'])
                if end_dt:
                    event['end']['dateTime'] = end_dt.isoformat()

            # Update event
            updated_event = self.google_service.events().update(
                calendarId='primary', eventId=event_id, body=event
            ).execute()

            return {
                "event_id": event_id,
                "status": "updated",
                "title": updated_event.get('summary', ''),
                "link": updated_event.get('htmlLink', '')
            }

        except Exception as e:
            logger.error("Failed to update event: %s", e)
            return {"error": f"Event update failed: {str(e)}"}

    async def delete_event(self, event_id: str) -> Dict:
        """Delete a calendar event"""
        if not self.authenticated:
            return {"error": "Calendar not authenticated"}

        try:
            self.google_service.events().delete(
                calendarId='primary', eventId=event_id
            ).execute()

            return {
                "event_id": event_id,
                "status": "deleted"
            }

        except Exception as e:
            logger.error("Failed to delete event: %s", e)
            return {"error": f"Event deletion failed: {str(e)}"}

    async def find_free_time(self, duration_minutes: int, days_ahead: int = 7) -> List[Dict]:
        """Find available time slots"""
        if not self.authenticated:
            return [{"error": "Calendar not authenticated"}]

        try:
            # Get busy times
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'

            freebusy_query = {
                'timeMin': time_min,
                'timeMax': time_max,
                'items': [{'id': 'primary'}]
            }

            freebusy_result = self.google_service.freebusy().query(
                body=freebusy_query
            ).execute()

            busy_times = freebusy_result['calendars']['primary']['busy']

            # Find free slots
            free_slots = self._find_free_slots(
                now, now + timedelta(days=days_ahead),
                busy_times, duration_minutes
            )

            return free_slots[:10]  # Return top 10 slots

        except Exception as e:
            logger.error("Failed to find free time: %s", e)
            return [{"error": f"Free time search failed: {str(e)}"}]

    async def get_calendar_summary(self, days_ahead: int = 7) -> Dict:
        """Get a summary of upcoming calendar events"""
        events = await self.get_upcoming_events(days_ahead)

        if not events or (len(events) == 1 and 'error' in events[0]):
            return {"error": "No events found or calendar not available"}

        # Analyze events
        today_events = []
        tomorrow_events = []
        upcoming_events = []

        now = datetime.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)

        for event in events:
            if 'error' in event:
                continue

            event_date = self._parse_datetime(event.get('start_time', '')).date()

            if event_date == today:
                today_events.append(event)
            elif event_date == tomorrow:
                tomorrow_events.append(event)
            else:
                upcoming_events.append(event)

        return {
            "total_events": len(events),
            "today_events": len(today_events),
            "tomorrow_events": len(tomorrow_events),
            "upcoming_events": len(upcoming_events),
            "today_schedule": today_events[:5],
            "tomorrow_schedule": tomorrow_events[:5],
            "next_event": events[0] if events else None,
            "summary_generated": datetime.now().isoformat()
        }

    async def schedule_meeting(self, title: str, participants: List[str],
                             duration_minutes: int = 60, preferred_time: str = None) -> Dict:
        """Schedule a meeting with participants"""
        try:
            # Find available time
            if preferred_time:
                start_time = self._parse_datetime(preferred_time)
                end_time = start_time + timedelta(minutes=duration_minutes)
            else:
                # Find next available slot
                free_slots = await self.find_free_time(duration_minutes)
                if not free_slots or 'error' in free_slots[0]:
                    return {"error": "No available time slots found"}

                slot = free_slots[0]
                start_time = self._parse_datetime(slot['start'])
                end_time = self._parse_datetime(slot['end'])

            # Create meeting event
            description = f"Meeting with: {', '.join(participants)}"

            result = await self.create_event(
                title=title,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                description=description
            )

            if 'error' not in result:
                result['participants'] = participants
                result['duration_minutes'] = duration_minutes

            return result

        except Exception as e:
            logger.error("Failed to schedule meeting: %s", e)
            return {"error": f"Meeting scheduling failed: {str(e)}"}

    def _format_event(self, event: Dict) -> Optional[Dict]:
        """Format a Google Calendar event"""
        try:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))

            return {
                "id": event['id'],
                "title": event.get('summary', 'No Title'),
                "description": event.get('description', ''),
                "location": event.get('location', ''),
                "start_time": start,
                "end_time": end,
                "all_day": 'date' in event['start'],
                "status": event.get('status', 'confirmed'),
                "link": event.get('htmlLink', ''),
                "attendees": [att.get('email', '') for att in event.get('attendees', [])]
            }
        except Exception as e:
            logger.error("Error formatting event: %s", e)
            return None

    def _parse_datetime(self, dt_string: str) -> Optional[datetime]:
        """Parse datetime string"""
        try:
            if isinstance(dt_string, datetime):
                return dt_string
            return dateutil.parser.parse(dt_string)
        except Exception:
            return None

    def _find_free_slots(self, start_time: datetime, end_time: datetime,
                        busy_times: List[Dict], duration_minutes: int) -> List[Dict]:
        """Find free time slots between busy periods"""
        free_slots = []

        # Convert busy times to datetime objects
        busy_periods = []
        for busy in busy_times:
            busy_start = self._parse_datetime(busy['start'])
            busy_end = self._parse_datetime(busy['end'])
            if busy_start and busy_end:
                busy_periods.append((busy_start, busy_end))

        # Sort busy periods
        busy_periods.sort(key=lambda x: x[0])

        # Find gaps between busy periods
        current_time = start_time
        duration_delta = timedelta(minutes=duration_minutes)

        for busy_start, busy_end in busy_periods:
            # Check if there's a gap before this busy period
            if current_time + duration_delta <= busy_start:
                free_slots.append({
                    "start": current_time.isoformat(),
                    "end": (current_time + duration_delta).isoformat(),
                    "duration_minutes": duration_minutes
                })

            current_time = max(current_time, busy_end)

        # Check for time after the last busy period
        if current_time + duration_delta <= end_time:
            free_slots.append({
                "start": current_time.isoformat(),
                "end": (current_time + duration_delta).isoformat(),
                "duration_minutes": duration_minutes
            })

        return free_slots

    def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            "authenticated": self.authenticated,
            "google_calendar_available": GOOGLE_CALENDAR_AVAILABLE,
            "caldav_available": CALDAV_AVAILABLE,
            "credentials_path": str(settings.CREDENTIALS_PATH / "calendar_credentials.json")
        }
