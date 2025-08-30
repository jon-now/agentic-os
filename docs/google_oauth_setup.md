# Google OAuth Setup Guide for Agentic OS

## Problem
You're seeing "Access blocked: Agentic OS has not completed the Google verification process" because your Google Cloud project needs proper OAuth configuration.

## Solution Steps

### 1. Go to Google Cloud Console
- Visit: https://console.cloud.google.com/
- Sign in with your Google account (srdeepak78@gmail.com)

### 2. Create or Select Project
- If you don't have a project: Click "Create Project"
- Name it: "Agentic OS" or "Practical Agentic System"
- If you have a project: Select it from the dropdown

### 3. Enable Required APIs
Go to "APIs & Services" > "Library" and enable:
- Gmail API
- Google Calendar API
- Google Drive API (if using file operations)

### 4. Configure OAuth Consent Screen
Go to "APIs & Services" > "OAuth consent screen":

#### User Type
- Select "External" (for testing with any Google account)
- Click "Create"

#### OAuth Consent Screen Configuration
Fill out these required fields:
- **App name**: Agentic OS
- **User support email**: srdeepak78@gmail.com
- **Developer contact email**: srdeepak78@gmail.com
- **App domain** (optional): Leave blank for now
- **Authorized domains**: Leave blank for testing

#### Scopes (Add these)
Click "Add or Remove Scopes" and add:
- `https://www.googleapis.com/auth/gmail.readonly`
- `https://www.googleapis.com/auth/gmail.send`
- `https://www.googleapis.com/auth/calendar`
- `https://www.googleapis.com/auth/calendar.events`

#### Test Users (Important!)
Add your email as a test user:
- Click "Add Users"
- Add: srdeepak78@gmail.com
- Add any other emails you want to test with

### 5. Create OAuth Credentials
Go to "APIs & Services" > "Credentials":
- Click "Create Credentials" > "OAuth 2.0 Client IDs"
- Application type: "Desktop application"
- Name: "Agentic OS Desktop Client"
- Click "Create"

### 6. Download Credentials
- Click the download button next to your new OAuth client
- Save the file as `credentials/credentials.json` in your project

### 7. Update Your Code
Make sure your credentials path is correct in your settings:

```python
# config/settings.py
GMAIL_CREDENTIALS_FILE: str = "credentials/credentials.json"
GMAIL_TOKEN_FILE: str = "credentials/gmail_token.pickle"
```

## Testing Phase Notes

### Current Status: Testing
Your app is currently in "Testing" mode, which means:
- ✅ Only test users you added can access it
- ✅ No verification needed for testing
- ✅ Refresh tokens don't expire
- ❌ Limited to 100 users

### For Production (Later)
To make it available to everyone, you'll need to:
1. Submit for Google verification
2. Provide privacy policy
3. Provide terms of service
4. Complete security assessment

## Quick Fix for Current Error

1. **Add yourself as test user** in OAuth consent screen
2. **Make sure app is in Testing mode** (not Production)
3. **Clear browser cache** and try again
4. **Use incognito mode** to test fresh authentication

## Troubleshooting

### If you still get "access_denied":
1. Check that srdeepak78@gmail.com is added as a test user
2. Verify the OAuth consent screen is properly configured
3. Make sure you're using the correct credentials.json file
4. Try deleting any existing token files and re-authenticating

### Common Issues:
- **Wrong project selected**: Make sure you're in the right Google Cloud project
- **APIs not enabled**: Enable Gmail and Calendar APIs
- **Missing test user**: Add your email to test users list
- **Cached tokens**: Delete old token files and re-authenticate

## File Structure
Your credentials folder should look like:
```
credentials/
├── credentials.json          # Downloaded from Google Cloud Console
├── gmail_token.pickle       # Generated after first auth (auto-created)
├── calendar_credentials.json # For calendar (if separate)
└── calendar_token.pickle    # Generated after first auth (auto-created)
```