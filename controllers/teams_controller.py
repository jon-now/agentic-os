import asyncio
import logging
from datetime import datetime, timedelta
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

try:
    import msal
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False
    logger.warning("MSAL not available. Install with: pip install msal")

from config.settings import settings

class TeamsController:
    def __init__(self):
        self.app = None
        self.access_token = None
        self.authenticated = False
        self.user_info = {}
        self.teams_cache = {}
        self.channels_cache = {}

        if MSAL_AVAILABLE:
            self._initialize_app()

    def _initialize_app(self):
        """Initialize Microsoft Graph application"""
        try:
            # Load app configuration
            config_file = settings.CREDENTIALS_PATH / "teams_config.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

                self.app = msal.PublicClientApplication(
                    client_id=config["client_id"],
                    authority=f"https://login.microsoftonline.com/{config.get('tenant_id', 'common')}"
                )

                logger.info("Microsoft Teams app initialized")
            else:
                logger.warning("No Teams config found. Create teams_config.json in credentials/")

        except Exception as e:
            logger.error("Failed to initialize Teams app: %s", e)

    async def authenticate(self) -> Dict:
        """Authenticate with Microsoft Graph"""
        if not self.app:
            return {"error": "Teams app not initialized"}

        try:
            # Define scopes for Microsoft Graph
            scopes = [
                "https://graph.microsoft.com/User.Read",
                "https://graph.microsoft.com/Team.ReadBasic.All",
                "https://graph.microsoft.com/Channel.ReadBasic.All",
                "https://graph.microsoft.com/ChannelMessage.Read.All",
                "https://graph.microsoft.com/ChannelMessage.Send",
                "https://graph.microsoft.com/Chat.Read",
                "https://graph.microsoft.com/Chat.ReadWrite"
            ]

            # Try to get token silently first
            accounts = self.app.get_accounts()
            if accounts:
                result = self.app.acquire_token_silent(scopes, account=accounts[0])
                if result and "access_token" in result:
                    self.access_token = result["access_token"]
                    self.authenticated = True

                    # Get user info
                    await self._get_user_info()

                    return {
                        "authenticated": True,
                        "user": self.user_info.get("displayName", ""),
                        "email": self.user_info.get("mail", "")
                    }

            # If silent auth fails, need interactive auth
            result = self.app.acquire_token_interactive(scopes)

            if result and "access_token" in result:
                self.access_token = result["access_token"]
                self.authenticated = True

                # Get user info
                await self._get_user_info()

                # Cache teams and channels
                await self._cache_teams_data()

                return {
                    "authenticated": True,
                    "user": self.user_info.get("displayName", ""),
                    "email": self.user_info.get("mail", "")
                }
            else:
                error_msg = result.get("error_description", "Authentication failed")
                return {"error": error_msg}

        except Exception as e:
            logger.error("Teams authentication error: %s", e)
            return {"error": str(e)}

    async def _get_user_info(self):
        """Get current user information"""
        if not self.access_token:
            return

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        self.user_info = await response.json()
                        logger.info("Got user info for: %s", self.user_info.get('displayName'))
                    else:
                        logger.error("Failed to get user info: %s", response.status)

        except Exception as e:
            logger.error("Error getting user info: %s", e)

    async def _cache_teams_data(self):
        """Cache teams and channels for faster access"""
        if not self.access_token:
            return

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                # Get teams
                async with session.get(
                    "https://graph.microsoft.com/v1.0/me/joinedTeams",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        teams_data = await response.json()
                        for team in teams_data.get("value", []):
                            self.teams_cache[team["id"]] = team
                            if team.get("displayName"):
                                self.teams_cache[team["displayName"]] = team

                # Get channels for each team
                for team_id, team in self.teams_cache.items():
                    if isinstance(team, dict) and "id" in team:
                        async with session.get(
                            f"https://graph.microsoft.com/v1.0/teams/{team['id']}/channels",
                            headers=headers
                        ) as response:
                            if response.status == 200:
                                channels_data = await response.json()
                                for channel in channels_data.get("value", []):
                                    channel_key = f"{team['id']}:{channel['id']}"
                                    self.channels_cache[channel_key] = {
                                        **channel,
                                        "team_id": team["id"],
                                        "team_name": team["displayName"]
                                    }

            logger.info("Cached {len(self.teams_cache)//2} teams and %s channels", len(self.channels_cache))

        except Exception as e:
            logger.error("Failed to cache teams data: %s", e)

    async def get_teams(self) -> List[Dict]:
        """Get list of teams user is member o"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Teams"}]

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://graph.microsoft.com/v1.0/me/joinedTeams",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        teams_data = await response.json()
                        teams = []

                        for team in teams_data.get("value", []):
                            teams.append({
                                "id": team["id"],
                                "name": team["displayName"],
                                "description": team.get("description", ""),
                                "visibility": team.get("visibility", ""),
                                "web_url": team.get("webUrl", "")
                            })

                        return teams
                    else:
                        error_text = await response.text()
                        return [{"error": f"Failed to get teams: {response.status} - {error_text}"}]

        except Exception as e:
            logger.error("Error getting teams: %s", e)
            return [{"error": str(e)}]

    async def get_channels(self, team_id: str) -> List[Dict]:
        """Get channels for a specific team"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Teams"}]

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        channels_data = await response.json()
                        channels = []

                        for channel in channels_data.get("value", []):
                            channels.append({
                                "id": channel["id"],
                                "name": channel["displayName"],
                                "description": channel.get("description", ""),
                                "membership_type": channel.get("membershipType", ""),
                                "web_url": channel.get("webUrl", ""),
                                "team_id": team_id
                            })

                        return channels
                    else:
                        error_text = await response.text()
                        return [{"error": f"Failed to get channels: {response.status} - {error_text}"}]

        except Exception as e:
            logger.error("Error getting channels: %s", e)
            return [{"error": str(e)}]

    async def send_message(self, team_id: str, channel_id: str, message: str) -> Dict:
        """Send a message to a Teams channel"""
        if not self.authenticated:
            return {"error": "Not authenticated to Teams"}

        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            message_data = {
                "body": {
                    "content": message,
                    "contentType": "text"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages",
                    headers=headers,
                    json=message_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result["id"],
                            "created_datetime": result["createdDateTime"],
                            "team_id": team_id,
                            "channel_id": channel_id
                        }
                    else:
                        error_text = await response.text()
                        return {"error": f"Failed to send message: {response.status} - {error_text}"}

        except Exception as e:
            logger.error("Error sending message: %s", e)
            return {"error": str(e)}

    async def get_recent_messages(self, team_id: str, channel_id: str, limit: int = 20) -> List[Dict]:
        """Get recent messages from a Teams channel"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Teams"}]

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$top={limit}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        messages_data = await response.json()
                        messages = []

                        for message in messages_data.get("value", []):
                            formatted_message = self._format_message(message)
                            messages.append(formatted_message)

                        return messages
                    else:
                        error_text = await response.text()
                        return [{"error": f"Failed to get messages: {response.status} - {error_text}"}]

        except Exception as e:
            logger.error("Error getting messages: %s", e)
            return [{"error": str(e)}]

    async def get_chats(self, limit: int = 20) -> List[Dict]:
        """Get recent chats"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Teams"}]

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://graph.microsoft.com/v1.0/me/chats?$top={limit}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        chats_data = await response.json()
                        chats = []

                        for chat in chats_data.get("value", []):
                            chats.append({
                                "id": chat["id"],
                                "topic": chat.get("topic", ""),
                                "chat_type": chat.get("chatType", ""),
                                "created_datetime": chat.get("createdDateTime", ""),
                                "last_updated": chat.get("lastUpdatedDateTime", ""),
                                "web_url": chat.get("webUrl", "")
                            })

                        return chats
                    else:
                        error_text = await response.text()
                        return [{"error": f"Failed to get chats: {response.status} - {error_text}"}]

        except Exception as e:
            logger.error("Error getting chats: %s", e)
            return [{"error": str(e)}]

    async def send_chat_message(self, chat_id: str, message: str) -> Dict:
        """Send a message to a chat"""
        if not self.authenticated:
            return {"error": "Not authenticated to Teams"}

        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            message_data = {
                "body": {
                    "content": message,
                    "contentType": "text"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://graph.microsoft.com/v1.0/chats/{chat_id}/messages",
                    headers=headers,
                    json=message_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result["id"],
                            "created_datetime": result["createdDateTime"],
                            "chat_id": chat_id
                        }
                    else:
                        error_text = await response.text()
                        return {"error": f"Failed to send chat message: {response.status} - {error_text}"}

        except Exception as e:
            logger.error("Error sending chat message: %s", e)
            return {"error": str(e)}

    async def get_user_presence(self, user_id: Optional[str] = None) -> Dict:
        """Get user presence status"""
        if not self.authenticated:
            return {"error": "Not authenticated to Teams"}

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}

            # Use current user if no user_id provided
            endpoint = "me/presence" if not user_id else f"users/{user_id}/presence"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://graph.microsoft.com/v1.0/{endpoint}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        presence_data = await response.json()
                        return {
                            "availability": presence_data.get("availability", ""),
                            "activity": presence_data.get("activity", ""),
                            "status_message": presence_data.get("statusMessage", {}).get("message", {}).get("content", "")
                        }
                    else:
                        error_text = await response.text()
                        return {"error": f"Failed to get presence: {response.status} - {error_text}"}

        except Exception as e:
            logger.error("Error getting presence: %s", e)
            return {"error": str(e)}

    async def set_presence(self, availability: str, activity: str, expiration_duration: str = "PT1H") -> Dict:
        """Set user presence status"""
        if not self.authenticated:
            return {"error": "Not authenticated to Teams"}

        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            presence_data = {
                "sessionId": "22skdnf-asdklfj-asdklfj-asdklfj",
                "availability": availability,
                "activity": activity,
                "expirationDuration": expiration_duration
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://graph.microsoft.com/v1.0/me/presence/setUserPreferredPresence",
                    headers=headers,
                    json=presence_data
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "availability": availability,
                            "activity": activity
                        }
                    else:
                        error_text = await response.text()
                        return {"error": f"Failed to set presence: {response.status} - {error_text}"}

        except Exception as e:
            logger.error("Error setting presence: %s", e)
            return {"error": str(e)}

    async def analyze_teams_activity(self, days_back: int = 7) -> Dict:
        """Analyze Teams activity patterns"""
        if not self.authenticated:
            return {"error": "Not authenticated to Teams"}

        try:
            analysis = {
                "active_teams": [],
                "message_patterns": {},
                "team_activity": {},
                "chat_activity": {}
            }

            # Get teams
            teams = await self.get_teams()

            for team in teams[:5]:  # Analyze top 5 teams
                if "error" in team:
                    continue

                team_id = team["id"]
                team_name = team["name"]

                # Get channels for this team
                channels = await self.get_channels(team_id)

                total_messages = 0
                for channel in channels[:3]:  # Top 3 channels per team
                    if "error" in channel:
                        continue

                    messages = await self.get_recent_messages(
                        team_id, channel["id"], limit=20
                    )

                    if messages and "error" not in messages[0]:
                        total_messages += len(messages)

                analysis["team_activity"][team_name] = {
                    "message_count": total_messages,
                    "channel_count": len(channels)
                }

            # Get chat activity
            chats = await self.get_chats(limit=10)
            if chats and "error" not in chats[0]:
                analysis["chat_activity"]["total_chats"] = len(chats)
                analysis["chat_activity"]["recent_chats"] = len([
                    chat for chat in chats
                    if self._is_recent(chat.get("last_updated", ""), days_back)
                ])

            # Sort teams by activity
            analysis["active_teams"] = sorted(
                analysis["team_activity"].items(),
                key=lambda x: x[1]["message_count"],
                reverse=True
            )

            return analysis

        except Exception as e:
            logger.error("Error analyzing Teams activity: %s", e)
            return {"error": str(e)}

    def _format_message(self, message: Dict) -> Dict:
        """Format a Teams message for consistent output"""
        try:
            return {
                "id": message.get("id", ""),
                "created_datetime": message.get("createdDateTime", ""),
                "last_modified": message.get("lastModifiedDateTime", ""),
                "message_type": message.get("messageType", ""),
                "importance": message.get("importance", ""),
                "subject": message.get("subject", ""),
                "body": message.get("body", {}).get("content", ""),
                "body_type": message.get("body", {}).get("contentType", ""),
                "from": message.get("from", {}).get("user", {}).get("displayName", ""),
                "from_id": message.get("from", {}).get("user", {}).get("id", ""),
                "web_url": message.get("webUrl", ""),
                "attachments": message.get("attachments", []),
                "mentions": message.get("mentions", [])
            }
        except Exception as e:
            logger.error("Error formatting message: %s", e)
            return message

    def _is_recent(self, datetime_str: str, days_back: int) -> bool:
        """Check if datetime is within the specified days back"""
        try:
            if not datetime_str:
                return False

            # Parse ISO datetime
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            cutoff = datetime.now() - timedelta(days=days_back)

            return dt > cutoff
        except Exception:
            return False

    def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            "authenticated": self.authenticated,
            "msal_available": MSAL_AVAILABLE,
            "user": self.user_info.get("displayName", ""),
            "email": self.user_info.get("mail", ""),
            "teams_cached": len(self.teams_cache) // 2,
            "channels_cached": len(self.channels_cache)
        }
