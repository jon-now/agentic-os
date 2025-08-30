# Enhanced Email Automation System

## Overview
The Enhanced Email Automation System provides intelligent, AI-powered email composition and sending capabilities. It automatically parses natural language requests, identifies recipients, generates professional email content, and handles the complete email workflow.

## Key Features

### 🤖 Intelligent Parsing
- **Natural Language Processing**: Understands requests like "Send email to John about tomorrow's meeting"
- **Recipient Resolution**: Automatically identifies email addresses and resolves contact names
- **Content Generation**: Creates professional email content based on purpose and tone
- **Subject Line Generation**: Automatically generates appropriate subject lines

### 📧 Email Components
- **Automatic Recipient Detection**: Extracts emails from requests or resolves contact names
- **Smart Subject Generation**: Creates context-appropriate subject lines
- **Professional Body Generation**: Generates well-structured email content
- **Tone Adaptation**: Adjusts tone (professional, casual, urgent) based on context

### 🎯 RTX 3050 Optimized
- **GPU-Accelerated LLM**: Uses local Ollama models with RTX 3050 optimizations
- **Memory Efficient**: Designed to work within 6GB VRAM constraints
- **Fallback Support**: Works without GPU when necessary

## Usage Examples

### Basic Email Sending
```
Send email to john.doe@example.com about tomorrow's meeting
```
- **Extracted Recipient**: john.doe@example.com
- **Generated Subject**: Meeting Request - Tomorrow
- **Generated Body**: Professional meeting request with time and agenda placeholders

### Contact Name Resolution
```
Email Sarah about the project update
```
- **Name Resolution**: Looks up "Sarah" in contact database
- **Generated Subject**: Project Update
- **Generated Body**: Professional project status update request

### Purpose-Based Generation
```
Send thank you email to my manager
```
- **Purpose**: Thank you email
- **Generated Subject**: Thank You
- **Generated Body**: Professional gratitude message with specific appreciation

### Meeting Requests
```
Compose email asking for a meeting with the team lead next week
```
- **Purpose**: Meeting request
- **Generated Subject**: Meeting Request - Next Week
- **Generated Body**: Formal meeting request with availability inquiry

### Follow-up Emails
```
Send follow-up email to client@company.com about our discussion
```
- **Purpose**: Follow-up
- **Generated Subject**: Follow-up: Our Discussion
- **Generated Body**: Professional follow-up with action items and next steps

## System Workflow

### 1. Request Analysis
1. **LLM Processing**: Uses local Ollama model to analyze request
2. **Fallback Parsing**: Rule-based parsing if LLM unavailable
3. **Confidence Scoring**: Evaluates understanding confidence

### 2. Recipient Resolution
1. **Direct Email**: Uses email addresses found in request
2. **Contact Lookup**: Resolves names to email addresses
3. **User Prompt**: Asks for email if resolution fails

### 3. Content Generation
1. **Purpose Detection**: Identifies email purpose (meeting, thank you, follow-up, etc.)
2. **Subject Generation**: Creates appropriate subject line
3. **Body Creation**: Generates professional email body
4. **Personalization**: Customizes content for recipient

### 4. User Review
1. **Preview Display**: Shows complete email for review
2. **Edit Options**: Allows editing of subject, body, or recipient
3. **Confirmation**: Requires explicit approval before sending

### 5. Email Sending
1. **Gmail Integration**: Uses Gmail API for sending
2. **Error Handling**: Provides clear error messages
3. **Success Confirmation**: Confirms successful delivery

## Configuration

### Contact Management
```python
# Add contacts programmatically
email_automation.add_contact("John Doe", "john.doe@company.com")
email_automation.add_contact("Sarah Smith", "sarah.smith@company.com")

# View contacts
contacts = email_automation.get_contacts()
```

### Quick Email (Minimal Processing)
```python
result = await email_automation.send_quick_email(
    to="user@example.com",
    subject="Quick Message", 
    body="This is a quick message.",
    auto_confirm=False  # Still shows confirmation
)
```

## Advanced Features

### LLM Integration
- **Local Processing**: Uses Ollama for privacy-focused local AI
- **Model Selection**: Automatically chooses appropriate model size
- **GPU Optimization**: RTX 3050 memory management
- **Fallback Support**: Works without LLM using rule-based parsing

### Error Handling
- **Gmail Authentication**: Clear guidance for setup
- **Network Issues**: Graceful handling of connectivity problems
- **Invalid Requests**: Helpful error messages and suggestions

### Performance
- **Response Time**: < 2 seconds for most requests
- **Memory Usage**: Optimized for RTX 3050 6GB constraints
- **Concurrent Processing**: Single-threaded for stability

## Installation Requirements

### Core Dependencies
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install pydantic fastapi
pip install requests asyncio
```

### Optional (for LLM features)
```bash
# Install Ollama and pull models
ollama pull llama3.2:3b  # Recommended for RTX 3050
ollama pull mistral:7b   # Alternative model
```

### Gmail Setup
1. **Google Cloud Console**: Create project and enable Gmail API
2. **Credentials**: Download credentials.json file
3. **Placement**: Place in `credentials/` folder
4. **Authentication**: Run system to complete OAuth flow

## Examples in Context

### Business Email
```
Input: "Send email to client@company.com about project completion"
Output:
- To: client@company.com
- Subject: Project Completion Update
- Body: Professional project completion notification with next steps
```

### Internal Communication
```
Input: "Email team lead about sick leave tomorrow"
Output:
- To: [resolved from contacts]
- Subject: Sick Leave - Tomorrow
- Body: Professional sick leave notification with coverage plans
```

### Meeting Coordination
```
Input: "Schedule meeting with john.doe@example.com for project review"
Output:
- To: john.doe@example.com
- Subject: Meeting Request: Project Review
- Body: Meeting request with agenda and time options
```

## Troubleshooting

### Common Issues
1. **No LLM Available**: System falls back to rule-based parsing
2. **Gmail Not Authenticated**: Follow authentication setup guide
3. **Recipient Not Found**: System prompts for email address
4. **Low Confidence**: System asks for confirmation before proceeding

### Performance Tips
1. **Use Specific Requests**: More specific requests yield better results
2. **Include Email Addresses**: Direct emails process faster than name resolution
3. **Clear Purpose**: Mention the email purpose for better content generation

## Integration

### With Main System
```python
# In orchestrator or main application
from automation.email_automation import EnhancedEmailAutomation

# Initialize with LLM client
email_automation = EnhancedEmailAutomation(llm_client=ollama_client)

# Process email request
result = await email_automation.process_email_request(user_input)
```

### Standalone Usage
```python
# Direct usage
email_automation = EnhancedEmailAutomation()
result = await email_automation.process_email_request(
    "Send email to user@example.com about meeting"
)
```

This enhanced email automation system provides a significant improvement over basic email sending by understanding user intent, generating appropriate content, and providing a smooth user experience while maintaining privacy through local AI processing.