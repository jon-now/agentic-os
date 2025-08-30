import asyncio
import logging
from datetime import datetime, timedelta
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False
    logger.warning("Slack SDK not available. Install with: pip install slack-sdk")

from config.settings import settings

class SlackController:
    def __init__(self):
        self.client = None
        self.bot_token = None
        self.user_token = None
        self.authenticated = False
        self.workspace_info = {}
        self.channels_cache = {}
        self.users_cache = {}

        if SLACK_SDK_AVAILABLE:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize Slack client with tokens"""
        try:
            # Load tokens from environment or config
            token_file = settings.CREDENTIALS_PATH / "slack_tokens.json"

            if token_file.exists():
                with open(token_file, 'r') as f:
                    tokens = json.load(f)
                    self.bot_token = tokens.get("bot_token")
                    self.user_token = tokens.get("user_token")

            if self.bot_token:
                self.client = AsyncWebClient(token=self.bot_token)
                logger.info("Slack client initialized with bot token")
            else:
                logger.warning("No Slack bot token found. Create slack_tokens.json in credentials/")

        except Exception as e:
            logger.error("Failed to initialize Slack client: %s", e)

    async def authenticate(self) -> Dict:
        """Test authentication and get workspace info"""
        if not self.client:
            return {"error": "Slack client not initialized"}

        try:
            # Test authentication
            auth_response = await self.client.auth_test()

            if auth_response["ok"]:
                self.authenticated = True
                self.workspace_info = {
                    "team": auth_response["team"],
                    "team_id": auth_response["team_id"],
                    "user": auth_response["user"],
                    "user_id": auth_response["user_id"],
                    "bot_id": auth_response.get("bot_id")
                }

                logger.info("Authenticated to Slack workspace: %s", self.workspace_info['team'])

                # Cache channels and users
                await self._cache_workspace_data()

                return {
                    "authenticated": True,
                    "workspace": self.workspace_info["team"],
                    "user": self.workspace_info["user"]
                }
            else:
                return {"error": "Authentication failed", "details": auth_response}

        except SlackApiError as e:
            logger.error("Slack authentication error: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Authentication error: %s", e)
            return {"error": str(e)}

    async def _cache_workspace_data(self):
        """Cache channels and users for faster access"""
        try:
            # Cache channels
            channels_response = await self.client.conversations_list(
                types="public_channel,private_channel,mpim,im"
            )

            if channels_response["ok"]:
                for channel in channels_response["channels"]:
                    self.channels_cache[channel["id"]] = channel
                    if channel.get("name"):
                        self.channels_cache[channel["name"]] = channel

            # Cache users
            users_response = await self.client.users_list()

            if users_response["ok"]:
                for user in users_response["members"]:
                    self.users_cache[user["id"]] = user
                    if user.get("name"):
                        self.users_cache[user["name"]] = user

            logger.info("Cached {len(self.channels_cache)//2} channels and %s users", len(self.users_cache)//2)

        except Exception as e:
            logger.error("Failed to cache workspace data: %s", e)

    async def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None) -> Dict:
        """Send a message to a Slack channel"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            # Resolve channel name to ID if needed
            channel_id = await self._resolve_channel(channel)
            if not channel_id:
                return {"error": f"Channel '{channel}' not found"}

            response = await self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts
            )

            if response["ok"]:
                return {
                    "success": True,
                    "message_ts": response["ts"],
                    "channel": channel_id,
                    "text": text
                }
            else:
                return {"error": f"Failed to send message: {response.get('error', 'Unknown error')}"}

        except SlackApiError as e:
            logger.error("Slack API error sending message: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Error sending message: %s", e)
            return {"error": str(e)}

    async def get_recent_messages(self, channel: str, limit: int = 20, hours_back: int = 24) -> List[Dict]:
        """Get recent messages from a channel"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Slack"}]

        try:
            # Resolve channel name to ID if needed
            channel_id = await self._resolve_channel(channel)
            if not channel_id:
                return [{"error": f"Channel '{channel}' not found"}]

            # Calculate oldest timestamp
            oldest_time = datetime.now() - timedelta(hours=hours_back)
            oldest_ts = str(oldest_time.timestamp())

            response = await self.client.conversations_history(
                channel=channel_id,
                limit=limit,
                oldest=oldest_ts
            )

            if response["ok"]:
                messages = []
                for message in response["messages"]:
                    formatted_message = await self._format_message(message)
                    messages.append(formatted_message)

                return messages
            else:
                return [{"error": f"Failed to get messages: {response.get('error', 'Unknown error')}"}]

        except SlackApiError as e:
            logger.error("Slack API error getting messages: %s", e)
            return [{"error": f"Slack API error: {e.response['error']}"}]
        except Exception as e:
            logger.error("Error getting messages: %s", e)
            return [{"error": str(e)}]

    async def get_channels(self, include_private: bool = False) -> List[Dict]:
        """Get list of channels"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Slack"}]

        try:
            channel_types = "public_channel"
            if include_private:
                channel_types += ",private_channel"

            response = await self.client.conversations_list(types=channel_types)

            if response["ok"]:
                channels = []
                for channel in response["channels"]:
                    channels.append({
                        "id": channel["id"],
                        "name": channel.get("name", ""),
                        "is_private": channel.get("is_private", False),
                        "is_member": channel.get("is_member", False),
                        "member_count": channel.get("num_members", 0),
                        "topic": channel.get("topic", {}).get("value", ""),
                        "purpose": channel.get("purpose", {}).get("value", "")
                    })

                return channels
            else:
                return [{"error": f"Failed to get channels: {response.get('error', 'Unknown error')}"}]

        except SlackApiError as e:
            logger.error("Slack API error getting channels: %s", e)
            return [{"error": f"Slack API error: {e.response['error']}"}]
        except Exception as e:
            logger.error("Error getting channels: %s", e)
            return [{"error": str(e)}]

    async def search_messages(self, query: str, count: int = 20) -> List[Dict]:
        """Search for messages across the workspace"""
        if not self.authenticated:
            return [{"error": "Not authenticated to Slack"}]

        try:
            response = await self.client.search_messages(
                query=query,
                count=count
            )

            if response["ok"]:
                messages = []
                for match in response["messages"]["matches"]:
                    formatted_message = await self._format_message(match)
                    formatted_message["relevance_score"] = match.get("score", 0)
                    messages.append(formatted_message)

                return messages
            else:
                return [{"error": f"Search failed: {response.get('error', 'Unknown error')}"}]

        except SlackApiError as e:
            logger.error("Slack API error searching: %s", e)
            return [{"error": f"Slack API error: {e.response['error']}"}]
        except Exception as e:
            logger.error("Error searching messages: %s", e)
            return [{"error": str(e)}]

    async def get_user_status(self, user_id: Optional[str] = None) -> Dict:
        """Get user status and presence"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            if not user_id:
                user_id = self.workspace_info.get("user_id")

            # Get user info
            user_response = await self.client.users_info(user=user_id)

            if not user_response["ok"]:
                return {"error": f"Failed to get user info: {user_response.get('error')}"}

            user = user_response["user"]

            # Get presence
            presence_response = await self.client.users_getPresence(user=user_id)

            presence = "unknown"
            if presence_response["ok"]:
                presence = presence_response.get("presence", "unknown")

            return {
                "user_id": user["id"],
                "name": user.get("name", ""),
                "real_name": user.get("real_name", ""),
                "display_name": user.get("profile", {}).get("display_name", ""),
                "status_text": user.get("profile", {}).get("status_text", ""),
                "status_emoji": user.get("profile", {}).get("status_emoji", ""),
                "presence": presence,
                "is_online": presence == "active",
                "timezone": user.get("tz", ""),
                "is_bot": user.get("is_bot", False)
            }

        except SlackApiError as e:
            logger.error("Slack API error getting user status: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Error getting user status: %s", e)
            return {"error": str(e)}

    async def set_status(self, status_text: str, status_emoji: str = "", expiration: Optional[int] = None) -> Dict:
        """Set user status"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            profile = {
                "status_text": status_text,
                "status_emoji": status_emoji
            }

            if expiration:
                profile["status_expiration"] = expiration

            response = await self.client.users_profile_set(profile=profile)

            if response["ok"]:
                return {
                    "success": True,
                    "status_text": status_text,
                    "status_emoji": status_emoji
                }
            else:
                return {"error": f"Failed to set status: {response.get('error', 'Unknown error')}"}

        except SlackApiError as e:
            logger.error("Slack API error setting status: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Error setting status: %s", e)
            return {"error": str(e)}

    async def create_channel(self, name: str, is_private: bool = False) -> Dict:
        """Create a new channel"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            response = await self.client.conversations_create(
                name=name,
                is_private=is_private
            )

            if response["ok"]:
                channel = response["channel"]
                return {
                    "success": True,
                    "channel_id": channel["id"],
                    "channel_name": channel["name"],
                    "is_private": channel.get("is_private", False)
                }
            else:
                return {"error": f"Failed to create channel: {response.get('error', 'Unknown error')}"}

        except SlackApiError as e:
            logger.error("Slack API error creating channel: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Error creating channel: %s", e)
            return {"error": str(e)}

    async def invite_to_channel(self, channel: str, users: List[str]) -> Dict:
        """Invite users to a channel"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            # Resolve channel name to ID if needed
            channel_id = await self._resolve_channel(channel)
            if not channel_id:
                return {"error": f"Channel '{channel}' not found"}

            # Resolve user names to IDs
            user_ids = []
            for user in users:
                user_id = await self._resolve_user(user)
                if user_id:
                    user_ids.append(user_id)

            if not user_ids:
                return {"error": "No valid users found"}

            response = await self.client.conversations_invite(
                channel=channel_id,
                users=",".join(user_ids)
            )

            if response["ok"]:
                return {
                    "success": True,
                    "channel": channel_id,
                    "invited_users": user_ids
                }
            else:
                return {"error": f"Failed to invite users: {response.get('error', 'Unknown error')}"}

        except SlackApiError as e:
            logger.error("Slack API error inviting users: %s", e)
            return {"error": f"Slack API error: {e.response['error']}"}
        except Exception as e:
            logger.error("Error inviting users: %s", e)
            return {"error": str(e)}

    async def analyze_workspace_activity(self, days_back: int = 7) -> Dict:
        """Analyze workspace activity patterns"""
        if not self.authenticated:
            return {"error": "Not authenticated to Slack"}

        try:
            analysis = {
                "active_channels": [],
                "top_users": [],
                "message_patterns": {},
                "channel_activity": {},
                "time_analysis": {}
            }

            # Get active channels
            channels = await self.get_channels(include_private=False)

            for channel in channels[:10]:  # Analyze top 10 channels
                if "error" in channel:
                    continue

                messages = await self.get_recent_messages(
                    channel["id"], limit=50, hours_back=days_back * 24
                )

                if messages and "error" not in messages[0]:
                    activity_score = len(messages)
                    analysis["channel_activity"][channel["name"]] = {
                        "message_count": activity_score,
                        "unique_users": len(set(msg.get("user_id", "") for msg in messages)),
                        "avg_message_length": sum(len(msg.get("text", "")) for msg in messages) / len(messages)
                    }

            # Sort channels by activity
            analysis["active_channels"] = sorted(
                analysis["channel_activity"].items(),
                key=lambda x: x[1]["message_count"],
                reverse=True
            )[:5]

            return analysis

        except Exception as e:
            logger.error("Error analyzing workspace activity: %s", e)
            return {"error": str(e)}

    async def _resolve_channel(self, channel: str) -> Optional[str]:
        """Resolve channel name to ID"""
        if channel.startswith("C") and len(channel) == 11:  # Already an ID
            return channel

        # Look in cache
        if channel in self.channels_cache:
            return self.channels_cache[channel]["id"]

        # Search by name
        for cached_channel in self.channels_cache.values():
            if isinstance(cached_channel, dict) and cached_channel.get("name") == channel:
                return cached_channel["id"]

        return None

    async def _resolve_user(self, user: str) -> Optional[str]:
        """Resolve user name to ID"""
        if user.startswith("U") and len(user) == 11:  # Already an ID
            return user

        # Look in cache
        if user in self.users_cache:
            return self.users_cache[user]["id"]

        # Search by name
        for cached_user in self.users_cache.values():
            if isinstance(cached_user, dict) and cached_user.get("name") == user:
                return cached_user["id"]

        return None

    async def _format_message(self, message: Dict) -> Dict:
        """Format a Slack message for consistent output"""
        try:
            # Get user info
            user_id = message.get("user", "")
            user_name = "Unknown"

            if user_id in self.users_cache:
                user_info = self.users_cache[user_id]
                user_name = user_info.get("real_name") or user_info.get("name", "Unknown")

            # Format timestamp
            ts = message.get("ts", "")
            timestamp = datetime.fromtimestamp(float(ts)).isoformat() if ts else ""

            return {
                "ts": ts,
                "timestamp": timestamp,
                "user_id": user_id,
                "user_name": user_name,
                "text": message.get("text", ""),
                "channel": message.get("channel", ""),
                "thread_ts": message.get("thread_ts"),
                "reply_count": message.get("reply_count", 0),
                "reactions": message.get("reactions", []),
                "attachments": message.get("attachments", []),
                "files": message.get("files", [])
            }

        except Exception as e:
            logger.error("Error formatting message: %s", e)
            return message

    def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            "authenticated": self.authenticated,
            "slack_sdk_available": SLACK_SDK_AVAILABLE,
            "workspace": self.workspace_info.get("team", ""),
            "user": self.workspace_info.get("user", ""),
            "bot_id": self.workspace_info.get("bot_id", ""),
            "channels_cached": len(self.channels_cache) // 2,
            "users_cached": len(self.users_cache) // 2
        }
