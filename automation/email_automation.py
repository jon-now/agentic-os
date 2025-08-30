#!/usr/bin/env python3
"""
Enhanced Email Automation System
Automatically parses user input, identifies recipients, and generates professional emails
Optimized for RTX 3050 6GB with local LLM integration
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import email.utils

from controllers.email_controller import EmailController
from llm.prompt_templates import PromptTemplates
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedEmailAutomation:
    """Enhanced email automation with intelligent parsing and generation"""
    
    def __init__(self, llm_client=None, user_name: str = None):
        self.email_controller = EmailController()
        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()
        
        # User configuration
        self.user_name = user_name or "Assistant"
        
        # RTX 3050 optimized settings
        self.config = {
            "max_retries": 3,
            "fallback_enabled": True,
            "auto_send": False,  # Always require confirmation by default
            "confidence_threshold": 0.7,
            "rtx_3050_mode": True
        }
        
        # Contact database (can be expanded)
        self.contact_database = {}
        
        logger.info("Enhanced Email Automation initialized for RTX 3050")
    
    def set_user_name(self, user_name: str):
        """Set the user name for email signatures"""
        self.user_name = user_name
    
    async def process_email_request(self, user_input: str, context: Optional[Dict] = None, auto_send: bool = False) -> Dict[str, Any]:
        """
        Main entry point for processing email requests
        Automatically parses input, identifies recipients, generates content, and handles sending
        """
        try:
            logger.info(f"Processing email request: {user_input}")
            
            # Step 1: Analyze the email request using LLM
            email_analysis = await self._analyze_email_request(user_input, context or {})
            
            if not email_analysis or email_analysis.get("error"):
                return {"error": "Failed to analyze email request", "details": email_analysis}
            
            # Step 2: Identify and resolve recipients
            recipient_info = await self._resolve_recipients(email_analysis, context or {})
            
            # Step 3: Generate complete email content
            email_content = await self._generate_email_content(
                email_analysis, recipient_info, user_input, context or {}
            )
            
            # Step 4: Auto-send if requested (for web interface), otherwise present for confirmation
            if auto_send and email_content.get('to'):
                return await self._send_email(email_content)
            else:
                confirmation_result = await self._present_for_confirmation(
                    email_content, email_analysis, user_input
                )
                return confirmation_result
            
        except Exception as e:
            logger.error(f"Email processing failed: {e}")
            return {"error": f"Email processing failed: {str(e)}"}
    
    async def _analyze_email_request(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Use LLM to analyze the email request and extract components"""
        try:
            if self.llm_client:
                # Use LLM for intelligent analysis
                prompt = self.prompt_templates.EMAIL_AUTOMATION.format(user_input=user_input)
                response = await self.llm_client.generate_response(prompt, task_type="analysis")
                
                # Parse LLM response
                try:
                    analysis = json.loads(response)
                    analysis["analysis_method"] = "llm"
                    return analysis
                except json.JSONDecodeError:
                    logger.warning("LLM response not valid JSON, using fallback")
                    return self._fallback_email_analysis(user_input)
            else:
                return self._fallback_email_analysis(user_input)
                
        except Exception as e:
            logger.error(f"LLM email analysis failed: {e}")
            return self._fallback_email_analysis(user_input)
    
    def _fallback_email_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback rule-based email analysis for when LLM is unavailable"""
        logger.info("Using fallback email analysis")
        
        user_lower = user_input.lower()
        
        # Extract email addresses using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, user_input)
        
        # Extract names (simple pattern - words that could be names)
        name_pattern = r'\b(?:to|email|send)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        name_matches = re.findall(name_pattern, user_input, re.IGNORECASE)
        names = [match for match in name_matches if not '@' in match and not any(word in match.lower() for word in ['email', 'send', 'about', 'regarding'])]
        
        # Also try to extract recipient from email context
        recipient_patterns = [
            r'email\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'send.*?to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in recipient_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                if match and not any(word in match.lower() for word in ['email', 'send', 'about', 'regarding', 'asking', 'help']):
                    names.append(match)
                    break
        
        # Determine purpose based on keywords
        purpose = "other"
        if any(word in user_lower for word in ["meeting", "schedule", "appointment"]):
            purpose = "meeting_request"
        elif any(word in user_lower for word in ["follow", "update", "status"]):
            purpose = "follow_up"
        elif any(word in user_lower for word in ["thank", "appreciate"]):
            purpose = "thank_you"
        elif any(word in user_lower for word in ["question", "ask", "inquiry"]):
            purpose = "inquiry"
        
        # Generate basic subject
        subject = self._generate_fallback_subject(user_input, purpose)
        
        # Generate basic body
        body = self._generate_fallback_body(user_input, purpose)
        
        return {
            "recipient": {
                "email": emails[0] if emails else None,
                "name": names[0] if names else None
            },
            "subject": subject,
            "body": body,
            "tone": "professional",
            "purpose": purpose,
            "confidence": 0.6,
            "extracted_info": {
                "has_recipient": bool(emails or names),
                "has_specific_content": len(user_input.split()) > 5,
                "needs_clarification": not bool(emails),
                "extracted_emails": emails,
                "extracted_names": names
            },
            "analysis_method": "fallback"
        }
    
    def _generate_fallback_subject(self, user_input: str, purpose: str) -> str:
        """Generate subject line using fallback method"""
        subject_map = {
            "meeting_request": "Meeting Request",
            "follow_up": "Follow-up",
            "thank_you": "Thank You",
            "inquiry": "Inquiry",
            "update": "Update",
            "other": "Message"
        }
        return subject_map.get(purpose, "Important Message")
    
    def _generate_fallback_body(self, user_input: str, purpose: str) -> str:
        """Generate email body using fallback templates with proper date handling"""
        # Extract and format dates from user input
        date_info = self._extract_and_format_dates(user_input)
        
        # Extract key information from user input without including the raw text
        user_lower = user_input.lower()
        
        if purpose == "meeting_request":
            return f"""Dear [Recipient],

I hope this email finds you well. I would like to schedule a meeting to discuss an important matter.

Please let me know your availability for the coming week, and I'll send a calendar invitation accordingly.

Best regards,
[Your Name]"""
        
        elif purpose == "follow_up":
            return f"""Dear [Recipient],

I wanted to follow up on our previous discussion.

Please let me know if you need any additional information or if there are any updates on your end.

Best regards,
[Your Name]"""
        
        elif purpose == "thank_you":
            return f"""Dear [Recipient],

Thank you for your assistance and support.

I appreciate your time and help with this matter.

Best regards,
[Your Name]"""
        
        elif "leave" in user_lower or "time off" in user_lower or "vacation" in user_lower:
            # Handle leave requests with proper date formatting
            days_match = re.search(r'(\d+)\s*days?', user_input, re.IGNORECASE)
            days = days_match.group(1) if days_match else "a few"
            
            reason = "personal reasons"
            if "grandmother" in user_lower or "grandma" in user_lower:
                reason = "visit my grandmother"
            elif "sick" in user_lower or "medical" in user_lower:
                reason = "medical reasons"
            elif "family" in user_lower:
                reason = "family matters"
            elif "emergency" in user_lower:
                reason = "an emergency situation"
            
            # Use actual dates instead of placeholders
            start_date = date_info['start_date']
            if days != "a few":
                try:
                    num_days = int(days)
                    end_date = datetime.strptime(start_date, '%B %d, %Y') + timedelta(days=num_days-1)
                    end_date_str = end_date.strftime('%B %d, %Y')
                    date_range = f"from {start_date} to {end_date_str}"
                except:
                    date_range = f"starting from {start_date}"
            else:
                date_range = f"starting from {start_date}"
            
            return f"""Dear [Recipient],

I hope this email finds you well.

I am writing to request {days} days of leave {date_range} to {reason}. This is an important matter that requires my attention.

I am happy to discuss coverage arrangements for my responsibilities during my absence and will ensure a smooth transition of any urgent tasks.

Please let me know if these dates would be acceptable.

Thank you for your consideration.

Best regards,
[Your Name]"""
        
        else:
            return f"""Dear [Recipient],

I hope this message finds you well.

I wanted to reach out regarding an important matter that requires your attention.

Please let me know if you have any questions or need additional information.

Best regards,
[Your Name]"""
    
    async def _resolve_recipients(self, email_analysis: Dict, context: Dict) -> Dict[str, Any]:
        """Resolve recipient information from analysis"""
        recipient_data = email_analysis.get("recipient", {})
        
        # If we have a direct email, use it
        if recipient_data.get("email"):
            return {
                "resolved": True,
                "email": recipient_data["email"],
                "name": recipient_data.get("name"),
                "method": "direct_email"
            }
        
        # If we have a name, try to resolve to email
        if recipient_data.get("name"):
            resolved_email = await self._lookup_contact_by_name(recipient_data["name"])
            if resolved_email:
                return {
                    "resolved": True,
                    "email": resolved_email,
                    "name": recipient_data["name"],
                    "method": "name_lookup"
                }
            else:
                return {
                    "resolved": False,
                    "name": recipient_data["name"],
                    "method": "needs_email_input",
                    "message": f"Please provide email address for {recipient_data['name']}"
                }
        
        # No recipient identified
        return {
            "resolved": False,
            "method": "needs_recipient_input",
            "message": "Please specify the recipient email address"
        }
    
    async def _lookup_contact_by_name(self, name: str) -> Optional[str]:
        """Lookup contact email by name (can be expanded with actual contact database)"""
        # Simple lookup in our contact database
        name_lower = name.lower()
        for contact_name, email in self.contact_database.items():
            if name_lower in contact_name.lower() or contact_name.lower() in name_lower:
                return email
        return None
    
    async def _generate_email_content(self, email_analysis: Dict, recipient_info: Dict, 
                                    original_request: str, context: Dict) -> Dict[str, Any]:
        """Generate complete email content with all components and proper placeholder replacement"""
        
        # Use LLM-generated content if available, otherwise use fallback
        content = {
            "to": recipient_info.get("email"),
            "to_name": recipient_info.get("name"),
            "subject": email_analysis.get("subject"),
            "body": email_analysis.get("body"),
            "tone": email_analysis.get("tone", "professional"),
            "purpose": email_analysis.get("purpose", "other"),
            "confidence": email_analysis.get("confidence", 0.6)
        }
        
        # CRITICAL: Replace all placeholders with actual information
        if content["body"]:
            # Get date information for proper replacement
            date_info = self._extract_and_format_dates(original_request)
            
            # Replace recipient placeholder
            if content["to_name"]:
                content["body"] = content["body"].replace("[Recipient]", content["to_name"])
            elif content["to"]:
                # Use email local part as fallback name
                name_from_email = content["to"].split("@")[0].replace(".", " ").title()
                content["body"] = content["body"].replace("[Recipient]", name_from_email)
            else:
                content["body"] = content["body"].replace("[Recipient]", "Sir/Madam")
            
            # Replace sender name placeholder - get from context or use configured name
            sender_name = context.get("user_name") or self.user_name
            content["body"] = content["body"].replace("[Your Name]", sender_name)
            
            # Replace date placeholders with actual dates
            content["body"] = content["body"].replace("[current date]", date_info['start_date'])
            content["body"] = content["body"].replace("[start date]", date_info['start_date'])
            content["body"] = content["body"].replace("[date]", date_info['start_date'])
        
        # Validate that no placeholders remain
        if content["body"] and ("[Recipient]" in content["body"] or "[Your Name]" in content["body"] or "[current date]" in content["body"] or "[start date]" in content["body"] or "[date]" in content["body"]):
            logger.warning("Placeholders still present in email body - replacing with defaults")
            content["body"] = content["body"].replace("[Recipient]", "Sir/Madam")
            content["body"] = content["body"].replace("[Your Name]", self.user_name)
            # Replace any remaining date placeholders
            current_date = datetime.now().strftime('%B %d, %Y')
            content["body"] = content["body"].replace("[current date]", current_date)
            content["body"] = content["body"].replace("[start date]", current_date)
            content["body"] = content["body"].replace("[date]", current_date)
        
        # CRITICAL: Ensure no original request text appears in the body
        if content["body"] and original_request.lower() in content["body"].lower():
            logger.error("Original request text found in email body - this should never happen!")
            # Emergency fallback - generate clean content
            content["body"] = self._generate_clean_emergency_body(content["purpose"], content.get("to_name", "Sir/Madam"), original_request)
        
        # Additional validation: Remove any debugging text or unwanted content
        if content["body"]:
            # Remove common debugging patterns
            debug_patterns = [
                r"Processing email request:.*?\n",
                r"Original Request:.*?\n",
                r"Analysis Method:.*?\n",
                r"Confidence:.*?\n",
                r"Purpose:.*?\n",
                r"Tone:.*?\n"
            ]
            for pattern in debug_patterns:
                content["body"] = re.sub(pattern, "", content["body"], flags=re.IGNORECASE)
            
            # Remove any remaining placeholder patterns with regex
            placeholder_patterns = [
                r"\[current\s*date\]",
                r"\[start\s*date\]", 
                r"\[date\]",
                r"\[end\s*date\]",
                r"\[today\]",
                r"\[tomorrow\]"
            ]
            
            current_date = datetime.now().strftime('%B %d, %Y')
            for pattern in placeholder_patterns:
                content["body"] = re.sub(pattern, current_date, content["body"], flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            content["body"] = re.sub(r"\n\s*\n\s*\n", "\n\n", content["body"])
            content["body"] = content["body"].strip()
        
        return content
    
    def _generate_clean_emergency_body(self, purpose: str, recipient_name: str, user_input: str = "") -> str:
        """Generate emergency clean email body when normal generation fails"""
        # Get date information if user_input is provided
        if user_input:
            date_info = self._extract_and_format_dates(user_input)
        else:
            current_date = datetime.now()
            date_info = {
                'current_date': current_date.strftime('%B %d, %Y'),
                'start_date': current_date.strftime('%B %d, %Y'),
                'start_date_short': current_date.strftime('%m/%d/%Y')
            }
        
        if purpose == "meeting_request":
            body = f"""Dear {recipient_name},

I hope this email finds you well. I would like to schedule a meeting to discuss an important matter.

Please let me know your availability for the coming week, and I'll send a calendar invitation accordingly.

Best regards,
Assistant"""
        
        elif purpose == "follow_up":
            body = f"""Dear {recipient_name},

I wanted to follow up on our previous discussion.

Please let me know if you need any additional information or if there are any updates on your end.

Best regards,
Assistant"""
        
        elif purpose == "thank_you":
            body = f"""Dear {recipient_name},

Thank you for your assistance and support.

I appreciate your time and help with this matter.

Best regards,
Assistant"""
        
        elif "leave" in purpose.lower() or "vacation" in purpose.lower():
            # Use actual date for leave requests
            start_date = date_info['start_date']
            body = f"""Dear {recipient_name},

I am writing to request time off starting from {start_date} for personal reasons.

I will ensure all my responsibilities are properly handled during my absence.

Thank you for your consideration.

Best regards,
Assistant"""
        
        else:
            body = f"""Dear {recipient_name},

I hope this message finds you well.

I wanted to reach out regarding an important matter that requires your attention.

Please let me know if you have any questions or need additional information.

Best regards,
Assistant"""
        
        # Replace Assistant with configured user name
        return body.replace("Assistant", self.user_name)
    
    def _extract_and_format_dates(self, user_input: str) -> Dict[str, str]:
        """Extract dates from user input and format them properly"""
        current_date = datetime.now()
        
        # Date patterns to look for
        date_patterns = [
            # Relative dates
            r'starting\s+(?:from\s+)?(?:today|tomorrow|next\s+week|next\s+monday|next\s+tuesday|next\s+wednesday|next\s+thursday|next\s+friday)',
            r'from\s+(?:today|tomorrow|next\s+week|next\s+monday|next\s+tuesday|next\s+wednesday|next\s+thursday|next\s+friday)',
            # Specific dates
            r'(?:on\s+|from\s+|starting\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?',
            r'(?:on\s+|from\s+|starting\s+)?\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?',
            r'(?:on\s+|from\s+|starting\s+)?\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:,?\s*\d{4})?'
        ]
        
        extracted_dates = {}
        
        # Look for specific dates in the input
        for pattern in date_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                date_text = matches[0]
                try:
                    # Parse relative dates
                    if 'today' in date_text.lower():
                        extracted_dates['start_date'] = current_date
                    elif 'tomorrow' in date_text.lower():
                        extracted_dates['start_date'] = current_date + timedelta(days=1)
                    elif 'next week' in date_text.lower():
                        days_ahead = 7 - current_date.weekday()
                        extracted_dates['start_date'] = current_date + timedelta(days=days_ahead)
                    elif 'next monday' in date_text.lower():
                        days_ahead = (7 - current_date.weekday()) % 7
                        if days_ahead == 0: days_ahead = 7
                        extracted_dates['start_date'] = current_date + timedelta(days=days_ahead)
                    # Add more specific date parsing as needed
                    
                except Exception as e:
                    logger.warning(f"Could not parse date '{date_text}': {e}")
                
                break
        
        # If no specific date found, use current date
        if 'start_date' not in extracted_dates:
            extracted_dates['start_date'] = current_date
        
        # Format dates for email content
        start_date = extracted_dates['start_date']
        formatted_dates = {
            'current_date': current_date.strftime('%B %d, %Y'),
            'start_date': start_date.strftime('%B %d, %Y'),
            'start_date_short': start_date.strftime('%m/%d/%Y')
        }
        
        return formatted_dates
    
    async def _present_for_confirmation(self, email_content: Dict, email_analysis: Dict, 
                                      original_request: str) -> Dict[str, Any]:
        """Present generated email to user for confirmation (web UI compatible)"""
        
        # For web UI, return the email content for display rather than terminal confirmation
        return {
            "status": "confirmation_needed",
            "message": "Email generated and ready to send",
            "final_output": f"📧 **Email Generated**\n\n**To:** {email_content.get('to', '[NOT SPECIFIED]')}\n**Subject:** {email_content.get('subject', '[NO SUBJECT]')}\n\n**Body:**\n{email_content.get('body', '[NO BODY CONTENT]')}\n\n**Analysis:** {email_analysis.get('analysis_method', 'unknown')} method, {email_content.get('confidence', 0):.0%} confidence\n\n*Note: This email is ready to send through Gmail. Use auto_send=True to send directly.*",
            "generated_content": email_content,
            "email_analysis": email_analysis,
            "original_request": original_request,
            "confidence": email_content.get('confidence', 0),
            "purpose": email_content.get('purpose', 'unknown'),
            "tone": email_content.get('tone', 'professional'),
            "can_send": bool(email_content.get('to')),
            "needs_recipient": not bool(email_content.get('to')),
            "suggestion": "Set auto_send=True in request to send email directly through Gmail" if email_content.get('to') else "Please provide recipient email address"
        }
    
    async def _send_email(self, email_content: Dict) -> Dict[str, Any]:
        """Send the email directly using Gmail API through email controller"""
        try:
            if not self.email_controller.authenticated:
                return {
                    "status": "error",
                    "error": "Gmail not authenticated. Please set up Gmail credentials.",
                    "final_output": "❌ Gmail authentication required. Please set up Gmail credentials to send emails."
                }
            
            # Directly send email through Gmail API
            result = await self.email_controller.send_email(
                to=email_content['to'],
                subject=email_content['subject'],
                body=email_content['body']
            )
            
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "message": f"✅ Email sent successfully to {email_content['to']}",
                    "final_output": f"✅ Email sent successfully to {email_content['to']}\n\nSubject: {email_content['subject']}\nSent at: {result.get('timestamp', 'Unknown')}",
                    "details": result,
                    "email_sent": True
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                return {
                    "status": "error",
                    "error": f"Failed to send email: {error_msg}",
                    "final_output": f"❌ Failed to send email to {email_content['to']}: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {
                "status": "error",
                "error": f"Email sending failed: {str(e)}",
                "final_output": f"❌ Email sending failed: {str(e)}"
            }
    
    def set_user_name(self, name: str):
        """Set the user name for email signatures"""
        self.user_name = name
        logger.info(f"User name set to: {name}")
    
    def get_user_name(self) -> str:
        """Get the current user name"""
        return self.user_name
    
    def add_contact(self, name: str, email: str):
        """Add a contact to the database"""
        self.contact_database[name] = email
        logger.info(f"Added contact: {name} -> {email}")
    
    def get_contacts(self) -> Dict[str, str]:
        """Get all contacts"""
        return self.contact_database.copy()
    
    async def send_quick_email(self, to: str, subject: str, body: str, 
                             auto_confirm: bool = False) -> Dict[str, Any]:
        """Send a quick email with minimal processing"""
        email_content = {
            "to": to,
            "subject": subject,
            "body": body,
            "tone": "professional",
            "purpose": "direct",
            "confidence": 1.0
        }
        
        if auto_confirm:
            return await self._send_email(email_content)
        else:
            return await self._present_for_confirmation(email_content, {}, f"Quick email to {to}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get email automation status"""
        return {
            "email_controller_authenticated": self.email_controller.authenticated,
            "llm_available": self.llm_client is not None,
            "contacts_count": len(self.contact_database),
            "config": self.config,
            "rtx_3050_optimized": True
        }

# Example usage and testing
async def test_email_automation():
    """Test the enhanced email automation"""
    print("Testing Enhanced Email Automation...")
    
    # Initialize (you would pass actual LLM client here)
    automation = EnhancedEmailAutomation()
    
    # Add some test contacts
    automation.add_contact("John Doe", "john.doe@company.com")
    automation.add_contact("Sarah Smith", "sarah.smith@company.com")
    
    # Test various email requests
    test_requests = [
        "Send email to john.doe@company.com about tomorrow's meeting",
        "Email Sarah about the project update",
        "Send thank you email to my manager",
        "Compose email asking for vacation approval"
    ]
    
    for request in test_requests:
        print(f"\n{'='*50}")
        print(f"Testing: {request}")
        print(f"{'='*50}")
        
        result = await automation.process_email_request(request)
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    asyncio.run(test_email_automation())