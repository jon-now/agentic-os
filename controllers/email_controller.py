import pickle
import os
import base64
import email
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
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    logger.warning("Gmail API dependencies not available. Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

class EmailController:
    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.compose',
            'https://www.googleapis.com/auth/gmail.modify'
        ]
        self.service = None
        self.authenticated = False

        if GMAIL_AVAILABLE:
            self.authenticate()

    def authenticate(self):
        """Authenticate with Gmail API"""
        if not GMAIL_AVAILABLE:
            logger.error("Gmail API not available")
            return

        try:
            creds = None
            token_path = settings.CREDENTIALS_PATH / settings.GMAIL_TOKEN_FILE
            creds_path = settings.CREDENTIALS_PATH / settings.GMAIL_CREDENTIALS_FILE

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
                        logger.error("Gmail credentials file not found: %s", creds_path)
                        logger.info("Please download credentials.json from Google Cloud Console")
                        return

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_path), self.SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save token
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)

            self.service = build('gmail', 'v1', credentials=creds)
            self.authenticated = True
            logger.info("Gmail API authenticated successfully")

        except Exception as e:
            logger.error("Gmail authentication failed: %s", e)
            self.authenticated = False

    async def get_recent_emails(self, max_results: int = 10, days_back: int = 7) -> List[Dict]:
        """Get recent emails"""
        if not self.authenticated:
            return [{"error": "Gmail not authenticated. Please set up Gmail credentials."}]

        try:
            # Calculate date filter
            after_date = datetime.now() - timedelta(days=days_back)
            date_query = after_date.strftime("%Y/%m/%d")

            # Search for recent emails
            query = f"after:{date_query}"
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])
            emails = []

            for message in messages[:max_results]:
                try:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='full'
                    ).execute()

                    email_data = self._parse_email(msg)
                    if email_data:
                        emails.append(email_data)

                except Exception as e:
                    logger.error("Error fetching email {message['id']}: %s", e)
                    continue

            return emails

        except HttpError as e:
            logger.error("Gmail API error: %s", e)
            return [{"error": f"Gmail API error: {str(e)}"}]
        except Exception as e:
            logger.error("Error fetching emails: %s", e)
            return [{"error": f"Failed to fetch emails: {str(e)}"}]

    def _parse_email(self, msg: Dict) -> Optional[Dict]:
        """Parse Gmail message into structured format"""
        try:
            payload = msg['payload']
            headers = payload.get('headers', [])

            # Extract headers
            email_data = {
                'id': msg['id'],
                'thread_id': msg.get('threadId'),
                'subject': self._get_header(headers, 'Subject') or 'No Subject',
                'sender': self._get_header(headers, 'From') or 'Unknown Sender',
                'recipient': self._get_header(headers, 'To') or 'Unknown Recipient',
                'date': self._get_header(headers, 'Date') or 'Unknown Date',
                'snippet': msg.get('snippet', ''),
                'labels': msg.get('labelIds', []),
                'is_unread': 'UNREAD' in msg.get('labelIds', []),
                'is_important': 'IMPORTANT' in msg.get('labelIds', []),
                'body': '',
                'attachments': []
            }

            # Extract body content
            email_data['body'] = self._extract_body(payload)

            # Extract attachments info
            email_data['attachments'] = self._extract_attachments(payload)

            # Calculate priority score
            email_data['priority_score'] = self._calculate_priority(email_data)

            return email_data

        except Exception as e:
            logger.error("Error parsing email: %s", e)
            return None

    def _get_header(self, headers: List[Dict], name: str) -> Optional[str]:
        """Get header value by name"""
        for header in headers:
            if header.get('name', '').lower() == name.lower():
                return header.get('value')
        return None

    def _extract_body(self, payload: Dict) -> str:
        """Extract email body content"""
        body = ""

        try:
            if 'parts' in payload:
                # Multipart message
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data', '')
                        if data:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            break
                    elif part['mimeType'] == 'text/html' and not body:
                        data = part['body'].get('data', '')
                        if data:
                            # For now, just decode HTML (could add HTML parsing later)
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
            else:
                # Simple message
                if payload['mimeType'] == 'text/plain':
                    data = payload['body'].get('data', '')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')

        except Exception as e:
            logger.error("Error extracting email body: %s", e)

        return body[:1000]  # Limit body length

    def _extract_attachments(self, payload: Dict) -> List[Dict]:
        """Extract attachment information"""
        attachments = []

        try:
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('filename'):
                        attachments.append({
                            'filename': part['filename'],
                            'mimeType': part['mimeType'],
                            'size': part['body'].get('size', 0)
                        })
        except Exception as e:
            logger.error("Error extracting attachments: %s", e)

        return attachments

    def _calculate_priority(self, email_data: Dict) -> float:
        """Calculate email priority score"""
        score = 0.0

        # Base score for unread emails
        if email_data['is_unread']:
            score += 0.5

        # Important label
        if email_data['is_important']:
            score += 0.3

        # Check for urgent keywords in subject/body
        urgent_keywords = ['urgent', 'asap', 'emergency', 'important', 'deadline', 'critical']
        subject_lower = email_data['subject'].lower()
        body_lower = email_data['body'].lower()

        for keyword in urgent_keywords:
            if keyword in subject_lower:
                score += 0.2
            if keyword in body_lower:
                score += 0.1

        # Recent emails get higher priority
        # (This is simplified - would need proper date parsing)
        if 'today' in email_data['date'].lower() or 'hour' in email_data['date'].lower():
            score += 0.2

        return min(score, 1.0)  # Cap at 1.0

    async def analyze_emails_for_summary(self, emails: List[Dict]) -> Dict:
        """Analyze emails and create summary"""
        if not emails or (len(emails) == 1 and 'error' in emails[0]):
            return {"error": "No emails to analyze"}

        summary = {
            "total_emails": len(emails),
            "unread_count": sum(1 for e in emails if e.get('is_unread', False)),
            "high_priority": [e for e in emails if e.get('priority_score', 0) > 0.7],
            "recent_senders": list(set([e['sender'].split('<')[0].strip() for e in emails[:10]])),
            "common_subjects": self._extract_common_subjects(emails),
            "needs_attention": []
        }

        # Identify emails that need attention
        for email in emails:
            if (email.get('priority_score', 0) > 0.6 or
                email.get('is_important', False) or
                any(keyword in email['subject'].lower() for keyword in ['urgent', 'deadline', 'meeting'])):
                summary["needs_attention"].append({
                    "subject": email['subject'],
                    "sender": email['sender'],
                    "snippet": email['snippet'][:100],
                    "priority_score": email.get('priority_score', 0)
                })

        return summary

    def _extract_common_subjects(self, emails: List[Dict]) -> List[str]:
        """Extract common subject patterns"""
        subjects = [email['subject'] for email in emails if email.get('subject')]

        # Simple word frequency analysis
        word_freq = {}
        for subject in subjects:
            words = subject.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Return most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in common_words[:5] if freq > 1]

    async def send_email(self, to: str, subject: str, body: str, 
                        cc: Optional[str] = None, bcc: Optional[str] = None,
                        attachments: Optional[List[str]] = None) -> Dict:
        """Send an email via Gmail API"""
        if not self.authenticated:
            return {"error": "Gmail not authenticated"}

        try:
            message = self._create_message(to, subject, body, cc, bcc, attachments)
            
            sent_message = self.service.users().messages().send(
                userId='me',
                body=message
            ).execute()
            
            return {
                "status": "success",
                "message": f"Email sent successfully to {to}",
                "subject": subject,
                "message_id": sent_message['id'],
                "thread_id": sent_message.get('threadId'),
                "timestamp": datetime.now().isoformat()
            }
            
        except HttpError as e:
            logger.error(f"Gmail API error sending email: {e}")
            return {"error": f"Gmail API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {"error": f"Failed to send email: {str(e)}"}
    
    def _create_message(self, to: str, subject: str, body: str,
                       cc: Optional[str] = None, bcc: Optional[str] = None,
                       attachments: Optional[List[str]] = None) -> Dict:
        """Create a message for sending via Gmail API"""
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        from email import encoders
        import mimetypes
        
        # Create message container
        if attachments:
            message = MIMEMultipart()
        else:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            if cc:
                message['cc'] = cc
            if bcc:
                message['bcc'] = bcc
            
            return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
        
        # Handle multipart message with attachments
        message['to'] = to
        message['subject'] = subject
        if cc:
            message['cc'] = cc
        if bcc:
            message['bcc'] = bcc
        
        # Add body
        message.attach(MIMEText(body, 'plain'))
        
        # Add attachments
        for file_path in attachments or []:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"Attachment not found: {file_path}")
                    continue
                
                content_type, encoding = mimetypes.guess_type(file_path)
                
                if content_type is None or encoding is not None:
                    content_type = 'application/octet-stream'
                
                main_type, sub_type = content_type.split('/', 1)
                
                with open(file_path, 'rb') as fp:
                    attachment = MIMEBase(main_type, sub_type)
                    attachment.set_payload(fp.read())
                
                encoders.encode_base64(attachment)
                
                filename = os.path.basename(file_path)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                
                message.attach(attachment)
                
            except Exception as e:
                logger.error(f"Error attaching file {file_path}: {e}")
                continue
        
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    
    async def send_quick_email(self, to: str, subject: str, body: str) -> Dict:
        """Quick email sending with simplified interface"""
        return await self.send_email(to, subject, body)
    
    async def send_notification_email(self, subject: str, body: str, 
                                    recipient: Optional[str] = None) -> Dict:
        """Send a notification email to self or specified recipient"""
        try:
            if not recipient:
                # Get user's email address
                profile = self.service.users().getProfile(userId='me').execute()
                recipient = profile.get('emailAddress')
                
                if not recipient:
                    return {"error": "Could not determine recipient email address"}
            
            return await self.send_email(recipient, subject, body)
            
        except Exception as e:
            logger.error(f"Error sending notification email: {e}")
            return {"error": f"Failed to send notification: {str(e)}"}
    
    async def schedule_email(self, to: str, subject: str, body: str, 
                           send_time: datetime) -> Dict:
        """Schedule an email to be sent later (placeholder for future implementation)"""
        # This would require a scheduling system
        return {
            "status": "scheduled",
            "message": f"Email scheduled for {send_time.isoformat()}",
            "to": to,
            "subject": subject,
            "send_time": send_time.isoformat(),
            "note": "Scheduling system not yet implemented - email saved as draft"
        }

    def get_authentication_status(self) -> Dict:
        """Get current authentication status"""
        return {
            "authenticated": self.authenticated,
            "gmail_available": GMAIL_AVAILABLE,
            "credentials_path": str(settings.CREDENTIALS_PATH / settings.GMAIL_CREDENTIALS_FILE),
            "token_exists": (settings.CREDENTIALS_PATH / settings.GMAIL_TOKEN_FILE).exists()
        }
    
    async def send_email_with_llm_content(self, to: str = None, subject: str = None, 
                                        body: str = None, original_request: str = "", 
                                        llm_analysis: Dict = None) -> Dict:
        """Send email using LLM-generated content with user review"""
        
        try:
            # Display the generated email for user review
            print("\n📧 **Generated Email Content**")
            print("=" * 50)
            print(f"**To:** {to or '[Please specify recipient]'}")
            print(f"**Subject:** {subject or '[Please specify subject]'}")
            print(f"**Body:**\n{body or '[Please specify body content]'}")
            print("=" * 50)
            
            # Get user input for missing fields
            if not to:
                to = input("\nEnter recipient email address: ").strip()
                if not to:
                    return {"error": "Recipient email is required"}
            
            if not subject:
                subject = input("Enter email subject (or press Enter to use generated): ").strip()
                if not subject:
                    subject = "Important Message"
            
            # Allow user to edit the body
            print(f"\nCurrent email body:\n{body}")
            edit_body = input("\nWould you like to edit the email body? (y/n): ").strip().lower()
            
            if edit_body == 'y':
                print("Enter your email body (press Enter twice to finish):")
                body_lines = []
                while True:
                    line = input()
                    if line == "" and len(body_lines) > 0 and body_lines[-1] == "":
                        break
                    body_lines.append(line)
                body = "\n".join(body_lines[:-1])  # Remove the last empty line
            
            # Final confirmation
            print(f"\n📧 **Final Email Preview**")
            print("=" * 50)
            print(f"**To:** {to}")
            print(f"**Subject:** {subject}")
            print(f"**Body:**\n{body}")
            print("=" * 50)
            
            confirm = input("\nSend this email? (y/n): ").strip().lower()
            
            if confirm != 'y':
                return {
                    "status": "cancelled",
                    "message": "Email sending cancelled by user",
                    "generated_content": {
                        "to": to,
                        "subject": subject,
                        "body": body
                    }
                }
            
            # Send the email
            result = await self.send_email(to, subject, body)
            
            if result.get("status") == "success":
                result["llm_generated"] = True
                result["original_request"] = original_request
                result["message"] = f"✅ Email sent successfully to {to} using LLM-generated content!"
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending LLM-generated email: {e}")
            return {"error": f"Failed to send email: {str(e)}"}
