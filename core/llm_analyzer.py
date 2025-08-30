"""
Enhanced LLM Analyzer for processing user input and generating appropriate actions
Supports both rule-based and ML-based analysis with improved architecture
"""
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Enumeration of available action types"""
    EMAIL = "email"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATION = "file_operation"
    AUTOMATION_WORKFLOW = "automation_workflow"
    CHAT = "chat"
    SYSTEM_INFO = "system_info"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CALENDAR = "calendar"
    REMINDER = "reminder"
    SEARCH = "search"

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class AnalysisResult:
    """Structured analysis result"""
    action_type: ActionType
    intent: str
    parameters: Dict[str, Any]
    confidence: float
    suggestions: List[str]
    timestamp: str
    original_input: str
    processing_time_ms: Optional[float] = None
    alternative_interpretations: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum serialization"""
        result = asdict(self)
        result['action_type'] = self.action_type.value
        return result

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input for security"""
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string")
        
        # Remove potential script injections
        sanitized = re.sub(r'<script.*?</script>', '', user_input, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "..."
        
        return sanitized.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Enhanced email validation"""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract all valid emails from text"""
        pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        emails = re.findall(pattern, text)
        return [email for email in emails if InputValidator.validate_email(email)]

class PatternMatcher:
    """Enhanced pattern matching with fuzzy matching capabilities"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive pattern database"""
        return {
            'greetings': {
                'patterns': [
                    r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening|day))\b',
                    r'\b(what\'s\s+up|how\'s\s+it\s+going|howdy)\b'
                ],
                'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
            },
            'email': {
                'patterns': [
                    r'\b(email|mail|send|compose|message|write\s+to)\b',
                    r'\b(cc|bcc|subject|attach)\b'
                ],
                'keywords': ['email', 'mail', 'send', 'compose', 'message', 'recipient']
            },
            'system_control': {
                'patterns': [
                    r'\b(volume|sound|audio|speaker)\b',
                    r'\b(recycle|trash|clean|temp|temporary)\b',
                    r'\b(open|launch|start|run|execute)\b'
                ],
                'keywords': ['volume', 'sound', 'recycle', 'clean', 'open', 'launch']
            },
            'file_operations': {
                'patterns': [
                    r'\b(file|folder|directory|document)\b',
                    r'\b(copy|move|delete|rename|organize|sort)\b',
                    r'\b(find|search|locate|look\s+for)\b'
                ],
                'keywords': ['file', 'folder', 'copy', 'move', 'delete', 'organize']
            },
            'calendar': {
                'patterns': [
                    r'\b(schedule|appointment|meeting|calendar|remind)\b',
                    r'\b(tomorrow|today|next\s+week|next\s+month)\b'
                ],
                'keywords': ['schedule', 'meeting', 'appointment', 'calendar', 'remind']
            }
        }
    
    def match_category(self, text: str) -> Tuple[str, float]:
        """Match text to category with confidence score"""
        text_lower = text.lower()
        scores = {}
        
        for category, data in self.patterns.items():
            score = 0
            
            # Pattern matching
            for pattern in data['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.3
            
            # Keyword matching with fuzzy logic
            for keyword in data['keywords']:
                if keyword in text_lower:
                    score += 0.2
                elif self._fuzzy_match(keyword, text_lower):
                    score += 0.1
            
            scores[category] = min(score, 1.0)
        
        if not scores or max(scores.values()) < 0.1:
            return 'unknown', 0.0
        
        best_category = max(scores, key=scores.get)
        return best_category, scores[best_category]
    
    def _fuzzy_match(self, keyword: str, text: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for typos and variations"""
        # Simple Levenshtein-like matching
        for word in text.split():
            if len(word) >= 3 and len(keyword) >= 3:
                matches = sum(1 for a, b in zip(keyword, word) if a == b)
                similarity = matches / max(len(keyword), len(word))
                if similarity >= threshold:
                    return True
        return False

class EmailComposer:
    """Enhanced email composition with templates and smart content generation"""
    
    def __init__(self):
        self.templates = self._load_email_templates()
    
    def _load_email_templates(self) -> Dict[str, Dict]:
        """Load email templates for different scenarios"""
        return {
            'job_application_withdrawal': {
                'subject': 'Withdrawal of Job Application - {reason}',
                'body': """Dear {recipient_name},

I hope this email finds you well.

I am writing to inform you that I need to withdraw my application for the {position} position due to {reason_detail}.

I sincerely apologize for any inconvenience this may cause and appreciate the time you have invested in reviewing my application.

Thank you for your understanding.

Best regards,
{sender_name}"""
            },
            'meeting_request': {
                'subject': 'Meeting Request - {topic}',
                'body': """Dear {recipient_name},

I hope you are doing well.

I would like to schedule a meeting to discuss {topic}. The meeting would be approximately {duration} and could take place {timeframe}.

Please let me know your availability, and I will send a calendar invitation accordingly.

Thank you for your time.

Best regards,
{sender_name}"""
            },
            'follow_up': {
                'subject': 'Follow-up: {topic}',
                'body': """Dear {recipient_name},

I wanted to follow up on our previous discussion regarding {topic}.

{follow_up_content}

Please let me know if you have any questions or if there's anything else I can provide.

Best regards,
{sender_name}"""
            },
            'thank_you': {
                'subject': 'Thank You - {topic}',
                'body': """Dear {recipient_name},

Thank you for {specific_reason}. Your {assistance_type} has been greatly appreciated.

{additional_content}

Please don't hesitate to reach out if you need anything from my end.

Best regards,
{sender_name}"""
            },
            'general': {
                'subject': '{subject}',
                'body': """Dear {recipient_name},

I hope this message finds you well.

{content}

Please let me know if you have any questions or need additional information.

Best regards,
{sender_name}"""
            }
        }
    
    def compose_email(self, user_input: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
        """Compose email based on user input and extracted parameters"""
        
        # Determine email type
        email_type = self._determine_email_type(user_input)
        template = self.templates.get(email_type, self.templates['general'])
        
        # Extract and prepare parameters
        params = self._prepare_email_parameters(user_input, extracted_params)
        
        try:
            subject = template['subject'].format(**params)
            body = template['body'].format(**params)
            
            return {
                'subject': subject,
                'body': body,
                'template_used': email_type,
                'to': extracted_params.get('to'),
                'to_name': extracted_params.get('to_name')
            }
        except KeyError as e:
            logger.warning(f"Missing parameter for email template: {e}")
            return self._generate_fallback_email(user_input, extracted_params)
    
    def _determine_email_type(self, user_input: str) -> str:
        """Determine the type of email based on content analysis"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['health', 'medical', 'unable', 'cannot apply', 'withdraw']):
            return 'job_application_withdrawal'
        elif any(word in user_lower for word in ['meeting', 'schedule', 'appointment']):
            return 'meeting_request'
        elif any(word in user_lower for word in ['follow', 'update', 'status', 'check in']):
            return 'follow_up'
        elif any(word in user_lower for word in ['thank', 'appreciate', 'grateful']):
            return 'thank_you'
        else:
            return 'general'
    
    def _prepare_email_parameters(self, user_input: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
        """Prepare parameters for email template formatting"""
        
        # Default parameters
        params = {
            'recipient_name': extracted_params.get('to_name', '[Recipient]'),
            'sender_name': '[Your Name]',
            'topic': 'our discussion',
            'reason': 'personal reasons',
            'reason_detail': 'personal circumstances',
            'position': 'the position',
            'duration': '30-60 minutes',
            'timeframe': 'at your convenience',
            'follow_up_content': 'I wanted to check on the status and see if there are any updates.',
            'specific_reason': 'your assistance',
            'assistance_type': 'support',
            'additional_content': '',
            'content': user_input,
            'subject': self._generate_smart_subject(user_input)
        }
        
        # Context-aware parameter extraction
        if 'health' in user_input.lower() or 'medical' in user_input.lower():
            params['reason'] = 'health reasons'
            params['reason_detail'] = 'some health concerns I am currently experiencing'
        
        return params
    
    def _generate_smart_subject(self, user_input: str) -> str:
        """Generate intelligent email subject based on content"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['urgent', 'asap', 'immediately']):
            return f"Urgent: {self._extract_topic(user_input)}"
        elif 'meeting' in user_lower:
            return f"Meeting Request - {self._extract_topic(user_input)}"
        elif any(word in user_lower for word in ['thank', 'appreciate']):
            return f"Thank You - {self._extract_topic(user_input)}"
        else:
            topic = self._extract_topic(user_input)
            return f"Re: {topic}" if topic != "Important Message" else "Important Message"
    
    def _extract_topic(self, user_input: str) -> str:
        """Extract main topic from user input"""
        # Simple topic extraction - could be enhanced with NLP
        words = user_input.split()
        if len(words) <= 3:
            return " ".join(words).title()
        
        # Look for topic indicators
        topic_indicators = ['about', 'regarding', 'concerning', 'for', 'on']
        for i, word in enumerate(words):
            if word.lower() in topic_indicators and i + 1 < len(words):
                return " ".join(words[i+1:i+4]).title()
        
        return "Important Message"
    
    def _generate_fallback_email(self, user_input: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
        """Generate fallback email when template formatting fails"""
        return {
            'subject': 'Important Message',
            'body': f"""Dear {extracted_params.get('to_name', 'Recipient')},

{user_input}

Best regards,
[Your Name]""",
            'template_used': 'fallback',
            'to': extracted_params.get('to'),
            'to_name': extracted_params.get('to_name')
        }

class SystemController:
    """Enhanced system control operations"""
    
    SUPPORTED_APPLICATIONS = {
        'notepad': ['notepad', 'text editor', 'note pad', 'notes'],
        'calculator': ['calculator', 'calc', 'math'],
        'chrome': ['chrome', 'google chrome', 'browser'],
        'firefox': ['firefox', 'mozilla'],
        'edge': ['edge', 'microsoft edge', 'ie'],
        'word': ['word', 'microsoft word', 'ms word', 'doc'],
        'excel': ['excel', 'spreadsheet', 'microsoft excel', 'xlsx'],
        'powerpoint': ['powerpoint', 'presentation', 'ppt', 'slides'],
        'outlook': ['outlook', 'mail client', 'email client'],
        'teams': ['teams', 'microsoft teams', 'meeting'],
        'zoom': ['zoom', 'video call'],
        'spotify': ['spotify', 'music', 'audio player'],
        'discord': ['discord', 'chat'],
        'slack': ['slack', 'work chat'],
        'cmd': ['cmd', 'command prompt', 'terminal', 'console'],
        'powershell': ['powershell', 'power shell', 'ps'],
        'paint': ['paint', 'mspaint', 'drawing'],
        'explorer': ['explorer', 'file explorer', 'files', 'folders'],
        'settings': ['settings', 'control panel', 'preferences'],
        'task_manager': ['task manager', 'processes', 'performance'],
        'registry': ['registry', 'regedit'],
        'device_manager': ['device manager', 'devices', 'hardware']
    }
    
    @classmethod
    def find_application(cls, user_input: str) -> Tuple[str, float]:
        """Find best matching application with confidence score"""
        user_lower = user_input.lower()
        best_match = None
        best_score = 0
        
        for app, keywords in cls.SUPPORTED_APPLICATIONS.items():
            score = 0
            for keyword in keywords:
                if keyword in user_lower:
                    # Exact match gets higher score
                    score += 1.0 if keyword == user_lower.strip() else 0.7
                elif cls._fuzzy_app_match(keyword, user_lower):
                    score += 0.5
            
            if score > best_score:
                best_score = score
                best_match = app
        
        # Extract from common patterns if no direct match
        if not best_match or best_score < 0.5:
            pattern_match = cls._extract_from_patterns(user_input)
            if pattern_match:
                return pattern_match, 0.6
        
        return best_match or "unknown_application", min(best_score, 1.0)
    
    @staticmethod
    def _fuzzy_app_match(keyword: str, text: str, threshold: float = 0.7) -> bool:
        """Fuzzy matching for application names"""
        for word in text.split():
            if len(word) >= 3 and len(keyword) >= 3:
                matches = sum(1 for a, b in zip(keyword, word) if a == b)
                similarity = matches / max(len(keyword), len(word))
                if similarity >= threshold:
                    return True
        return False
    
    @staticmethod
    def _extract_from_patterns(user_input: str) -> Optional[str]:
        """Extract application name from common command patterns"""
        patterns = [
            r'open\s+([a-zA-Z][a-zA-Z0-9]*)',
            r'launch\s+([a-zA-Z][a-zA-Z0-9]*)',
            r'start\s+([a-zA-Z][a-zA-Z0-9]*)',
            r'run\s+([a-zA-Z][a-zA-Z0-9]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                app_candidate = match.group(1).lower()
                # Check if it's a known app
                for app, keywords in SystemController.SUPPORTED_APPLICATIONS.items():
                    if app_candidate in keywords:
                        return app
                return app_candidate
        
        return None

class ConversationManager:
    """Manage conversation context and history"""
    
    def __init__(self, max_history: int = 50):
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.user_preferences: Dict[str, Any] = {}
        self.session_id = self._generate_session_id()
    
    def add_interaction(self, user_input: str, analysis_result: AnalysisResult):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'analysis_result': analysis_result.to_dict(),
            'session_id': self.session_id
        }
        
        self.conversation_history.append(interaction)
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Update user preferences
        self._update_preferences(analysis_result)
    
    def get_context(self, lookback: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.conversation_history[-lookback:] if self.conversation_history else []
    
    def _update_preferences(self, analysis_result: AnalysisResult):
        """Update user preferences based on interactions"""
        action_type = analysis_result.action_type.value
        
        if action_type not in self.user_preferences:
            self.user_preferences[action_type] = {'count': 0, 'last_used': None}
        
        self.user_preferences[action_type]['count'] += 1
        self.user_preferences[action_type]['last_used'] = datetime.now().isoformat()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

class EnhancedLLMAnalyzer:
    """Enhanced LLM Analyzer with improved architecture and capabilities"""
    
    def __init__(self, enable_ml_features: bool = False):
        self.pattern_matcher = PatternMatcher()
        self.email_composer = EmailComposer()
        self.conversation_manager = ConversationManager()
        self.input_validator = InputValidator()
        self.enable_ml_features = enable_ml_features
        
        self.system_prompt = """
You are an advanced intelligent assistant that analyzes user requests and generates structured actions.
You understand context, maintain conversation history, and provide personalized responses.

Core capabilities:
- Natural language understanding with context awareness
- Multi-step workflow planning
- Personalized response generation
- Error recovery and clarification requests
- Security-conscious input processing

Always prioritize user safety, privacy, and provide helpful, accurate responses.
"""
    
    def analyze_user_input(self, user_input: str) -> AnalysisResult:
        """
        Main analysis method with enhanced processing pipeline
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Input validation and sanitization
            sanitized_input = self.input_validator.sanitize_input(user_input)
            
            # Step 2: Get conversation context
            context = self.conversation_manager.get_context()
            
            # Step 3: Pattern matching and intent classification
            primary_analysis = self._perform_primary_analysis(sanitized_input, context)
            
            # Step 4: Generate alternative interpretations
            alternatives = self._generate_alternatives(sanitized_input)
            
            # Step 5: Create final analysis result
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = AnalysisResult(
                action_type=primary_analysis['action_type'],
                intent=primary_analysis['intent'],
                parameters=primary_analysis['parameters'],
                confidence=primary_analysis['confidence'],
                suggestions=primary_analysis['suggestions'],
                timestamp=datetime.now().isoformat(),
                original_input=user_input,
                processing_time_ms=processing_time,
                alternative_interpretations=alternatives
            )
            
            # Step 6: Update conversation history
            self.conversation_manager.add_interaction(user_input, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing user input: {e}")
            return self._create_error_result(user_input, str(e))
    
    def _perform_primary_analysis(self, user_input: str, context: List[Dict]) -> Dict[str, Any]:
        """Perform primary intent analysis with context awareness"""
        
        # Get pattern matching results
        category, pattern_confidence = self.pattern_matcher.match_category(user_input)
        
        # Context-aware adjustments
        if context:
            last_action = context[-1].get('analysis_result', {}).get('action_type')
            if last_action and pattern_confidence < 0.7:
                # Consider conversation continuity
                pattern_confidence += 0.1
        
        # Route to specific analyzers based on category
        if category == 'greetings':
            return self._analyze_greeting_intent(user_input, pattern_confidence)
        elif category == 'email':
            return self._analyze_email_intent_enhanced(user_input, pattern_confidence)
        elif category == 'system_control':
            return self._analyze_system_intent_enhanced(user_input, pattern_confidence)
        elif category == 'file_operations':
            return self._analyze_file_intent_enhanced(user_input, pattern_confidence)
        elif category == 'calendar':
            return self._analyze_calendar_intent(user_input, pattern_confidence)
        else:
            return self._analyze_fallback_intent(user_input, pattern_confidence)
    
    def _analyze_email_intent_enhanced(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Enhanced email intent analysis"""
        
        # Extract email components
        emails = self.input_validator.extract_emails(user_input)
        to_email = emails[0] if emails else None
        
        # Extract recipient name
        name_match = re.search(r'\(([^)]+)\)', user_input)
        to_name = name_match.group(1) if name_match else None
        
        # Extract subject
        subject_match = re.search(r'subject[:\s]+([^,\n.]+)', user_input, re.IGNORECASE)
        explicit_subject = subject_match.group(1).strip() if subject_match else None
        
        # Compose email
        email_params = {
            'to': to_email,
            'to_name': to_name,
            'explicit_subject': explicit_subject
        }
        
        composed_email = self.email_composer.compose_email(user_input, email_params)
        
        parameters = {
            **composed_email,
            'original_request': user_input,
            'auto_send': False,  # Always require confirmation
            'priority': self._determine_email_priority(user_input)
        }
        
        confidence = base_confidence
        if to_email:
            confidence += 0.1
        if explicit_subject:
            confidence += 0.1
        
        return {
            'action_type': ActionType.EMAIL,
            'intent': f'Compose and send email to {to_email or "[recipient]"}',
            'parameters': parameters,
            'confidence': min(confidence, 1.0),
            'suggestions': [
                'Review the generated email before sending',
                'Add CC/BCC recipients if needed',
                'Attach files if necessary'
            ]
        }
    
    def _analyze_system_intent_enhanced(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Enhanced system control analysis"""
        
        user_lower = user_input.lower()
        
        # Volume control with better extraction
        if any(word in user_lower for word in ['volume', 'sound', 'audio']):
            volume_info = self._extract_volume_info(user_input)
            
            return {
                'action_type': ActionType.SYSTEM_CONTROL,
                'intent': f'Adjust system volume to {volume_info["target"]}%',
                'parameters': {
                    'operation': 'set_volume',
                    'volume': volume_info['target'],
                    'relative': volume_info['relative'],
                    'mute': volume_info['mute']
                },
                'confidence': base_confidence + 0.1,
                'suggestions': ['Volume will be adjusted immediately']
            }
        
        # System cleanup
        elif any(word in user_lower for word in ['clean', 'cleanup', 'temp', 'recycle', 'trash']):
            cleanup_scope = self._determine_cleanup_scope(user_input)
            
            return {
                'action_type': ActionType.SYSTEM_CONTROL,
                'intent': f'Perform system cleanup: {cleanup_scope}',
                'parameters': {
                    'operation': 'cleanup',
                    'scope': cleanup_scope,
                    'safe_mode': True,
                    'confirm_before_delete': True
                },
                'confidence': base_confidence + 0.1,
                'suggestions': ['System will be cleaned safely with confirmation prompts']
            }
        
        # Application launching
        elif any(word in user_lower for word in ['open', 'launch', 'start', 'run']):
            app_name, app_confidence = SystemController.find_application(user_input)
            
            return {
                'action_type': ActionType.SYSTEM_CONTROL,
                'intent': f'Launch application: {app_name}',
                'parameters': {
                    'operation': 'open_application',
                    'application': app_name,
                    'controller_name': app_name,
                    'app_confidence': app_confidence
                },
                'confidence': min(base_confidence + app_confidence * 0.3, 1.0),
                'suggestions': [f'Will attempt to open {app_name}']
            }
        
        # System information
        elif any(word in user_lower for word in ['status', 'info', 'information', 'health']):
            return {
                'action_type': ActionType.SYSTEM_INFO,
                'intent': 'Retrieve system status and information',
                'parameters': {
                    'operation': 'status_check',
                    'include_performance': True,
                    'include_storage': True,
                    'include_network': True
                },
                'confidence': base_confidence + 0.1,
                'suggestions': ['Will provide comprehensive system information']
            }
        
        return {
            'action_type': ActionType.SYSTEM_CONTROL,
            'intent': 'General system operation',
            'parameters': {'operation': 'general', 'description': user_input},
            'confidence': base_confidence,
            'suggestions': ['Please specify what system operation you need']
        }
    
    def _analyze_file_intent_enhanced(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Enhanced file operation analysis"""
        
        user_lower = user_input.lower()
        
        # File organization detection
        organization_indicators = ['organize', 'sort', 'clean up', 'tidy', 'arrange', 'structure']
        file_indicators = ['files', 'documents', 'folders', 'directories', 'downloads', 'desktop']
        
        is_organization = any(indicator in user_lower for indicator in organization_indicators)
        involves_files = any(indicator in user_lower for indicator in file_indicators)
        
        if is_organization and involves_files:
            scope = self._determine_organization_scope(user_input)
            organization_params = self._extract_organization_preferences(user_input)
            
            return {
                'action_type': ActionType.AUTOMATION_WORKFLOW,
                'intent': f'Organize files in {scope}',
                'parameters': {
                    'workflow_type': 'file_organization',
                    'scope': scope,
                    'preferences': organization_params,
                    'analyze_first': True,
                    'backup_before_organize': True,
                    'dry_run_available': True
                },
                'confidence': base_confidence + 0.2,
                'suggestions': [
                    'Files will be analyzed before organization',
                    'Backup will be created for safety',
                    'You can preview changes before applying'
                ]
            }
        
        # Specific file operations
        operations = ['copy', 'move', 'delete', 'rename', 'create', 'backup']
        detected_operation = next((op for op in operations if op in user_lower), None)
        
        if detected_operation:
            file_params = self._extract_file_operation_params(user_input, detected_operation)
            
            return {
                'action_type': ActionType.FILE_OPERATION,
                'intent': f'Perform {detected_operation} operation on files',
                'parameters': {
                    'operation': detected_operation,
                    'source_path': file_params.get('source'),
                    'target_path': file_params.get('target'),
                    'file_pattern': file_params.get('pattern'),
                    'recursive': file_params.get('recursive', False),
                    'confirm_before_action': True
                },
                'confidence': base_confidence + 0.1,
                'suggestions': [f'Will perform {detected_operation} operation with confirmation']
            }
        
        # File search
        elif any(word in user_lower for word in ['find', 'search', 'locate', 'look for']):
            search_params = self._extract_search_parameters(user_input)
            
            return {
                'action_type': ActionType.FILE_OPERATION,
                'intent': 'Search for files matching criteria',
                'parameters': {
                    'operation': 'search',
                    'search_pattern': search_params['pattern'],
                    'search_path': search_params['path'],
                    'file_types': search_params['file_types'],
                    'case_sensitive': False,
                    'include_hidden': False
                },
                'confidence': base_confidence + 0.1,
                'suggestions': ['Will search through specified directories']
            }
        
        # Default file management
        return {
            'action_type': ActionType.FILE_OPERATION,
            'intent': 'General file management operation',
            'parameters': {
                'description': user_input,
                'suggested_operation': 'analyze_file_request',
                'needs_clarification': True
            },
            'confidence': base_confidence,
            'suggestions': ['Please specify what file operation you need']
        }
    
    def _analyze_calendar_intent(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Analyze calendar and scheduling requests"""
        
        user_lower = user_input.lower()
        
        # Extract time information
        time_info = self._extract_time_information(user_input)
        
        # Determine calendar action
        if any(word in user_lower for word in ['schedule', 'book', 'arrange']):
            action = 'schedule_event'
        elif any(word in user_lower for word in ['cancel', 'remove', 'delete']):
            action = 'cancel_event'
        elif any(word in user_lower for word in ['remind', 'reminder', 'alert']):
            action = 'set_reminder'
        else:
            action = 'general_calendar'
        
        return {
            'action_type': ActionType.CALENDAR,
            'intent': f'Calendar operation: {action}',
            'parameters': {
                'action': action,
                'time_info': time_info,
                'description': user_input,
                'auto_invite': False,
                'reminder_minutes': [15, 60] if action == 'schedule_event' else [5]
            },
            'confidence': base_confidence + 0.1,
            'suggestions': ['Calendar event will be created with appropriate reminders']
        }
    
    def _analyze_greeting_intent(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Enhanced greeting analysis with personality"""
        
        time_of_day = self._get_time_of_day()
        user_preferences = self.conversation_manager.user_preferences
        
        return {
            'action_type': ActionType.CHAT,
            'intent': 'User is greeting the assistant',
            'parameters': {
                'conversation_type': 'greeting',
                'time_of_day': time_of_day,
                'user_preferences': user_preferences,
                'personalized': len(self.conversation_manager.conversation_history) > 0
            },
            'confidence': base_confidence + 0.05,
            'suggestions': []
        }
    
    def _analyze_fallback_intent(self, user_input: str, base_confidence: float) -> Dict[str, Any]:
        """Enhanced fallback analysis with smart classification"""
        
        # Check if it's a question
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', '?']
        is_question = any(indicator in user_input.lower() for indicator in question_indicators)
        
        if is_question:
            return {
                'action_type': ActionType.GENERAL_KNOWLEDGE,
                'intent': 'User is asking a knowledge question',
                'parameters': {
                    'topic': user_input,
                    'question_type': self._classify_question_type(user_input),
                    'needs_web_search': self._needs_web_search(user_input)
                },
                'confidence': 0.7,
                'suggestions': ['I can help answer your question']
            }
        
        # Check for automation keywords
        automation_indicators = ['automate', 'workflow', 'task', 'process', 'routine']
        if any(indicator in user_input.lower() for indicator in automation_indicators):
            return {
                'action_type': ActionType.AUTOMATION_WORKFLOW,
                'intent': 'Create custom automation workflow',
                'parameters': {
                    'workflow_type': 'custom',
                    'description': user_input,
                    'complexity': self._assess_workflow_complexity(user_input),
                    'suggested_steps': self._suggest_workflow_steps(user_input)
                },
                'confidence': 0.6,
                'suggestions': ['I can help design a custom workflow for your needs']
            }
        
        # Default to chat for short, conversational inputs
        return {
            'action_type': ActionType.CHAT,
            'intent': 'General conversation or unclear request',
            'parameters': {
                'conversation_type': 'general',
                'needs_clarification': base_confidence < 0.5,
                'input_length': len(user_input),
                'suggested_clarifications': self._generate_clarification_questions(user_input)
            },
            'confidence': base_confidence,
            'suggestions': ['How can I help you with that?']
        }
    
    def _generate_alternatives(self, user_input: str) -> List[Dict[str, Any]]:
        """Generate alternative interpretations of user input"""
        
        alternatives = []
        
        # Try different pattern categories
        for category in ['email', 'system_control', 'file_operations', 'calendar']:
            if category != self.pattern_matcher.match_category(user_input)[0]:
                alt_confidence = max(0.1, self.pattern_matcher.match_category(user_input)[1] - 0.3)
                if alt_confidence > 0.2:
                    alternatives.append({
                        'category': category,
                        'confidence': alt_confidence,
                        'interpretation': f'Could be interpreted as {category} request'
                    })
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    # Helper methods for enhanced functionality
    
    def _extract_volume_info(self, user_input: str) -> Dict[str, Any]:
        """Extract volume control information"""
        
        # Check for mute/unmute
        if any(word in user_input.lower() for word in ['mute', 'silent', 'quiet']):
            return {'target': 0, 'relative': False, 'mute': True}
        elif any(word in user_input.lower() for word in ['unmute', 'sound on']):
            return {'target': 50, 'relative': False, 'mute': False}
        
        # Extract percentage or relative change
        volume_match = re.search(r'(\d+)%?', user_input)
        if volume_match:
            volume = int(volume_match.group(1))
            return {'target': min(max(volume, 0), 100), 'relative': False, 'mute': False}
        
        # Relative adjustments
        if any(phrase in user_input.lower() for phrase in ['turn up', 'increase', 'louder', 'higher']):
            return {'target': 10, 'relative': True, 'mute': False}
        elif any(phrase in user_input.lower() for phrase in ['turn down', 'decrease', 'lower', 'quieter']):
            return {'target': -10, 'relative': True, 'mute': False}
        
        return {'target': 50, 'relative': False, 'mute': False}
    
    def _determine_cleanup_scope(self, user_input: str) -> str:
        """Determine what should be cleaned up"""
        user_lower = user_input.lower()
        
        scopes = []
        if any(word in user_lower for word in ['temp', 'temporary', 'cache']):
            scopes.append('temporary_files')
        if any(word in user_lower for word in ['recycle', 'trash', 'bin']):
            scopes.append('recycle_bin')
        if any(word in user_lower for word in ['log', 'logs']):
            scopes.append('log_files')
        if any(word in user_lower for word in ['download', 'downloads']):
            scopes.append('downloads_cleanup')
        
        return ', '.join(scopes) if scopes else 'general_cleanup'
    
    def _determine_organization_scope(self, user_input: str) -> str:
        """Enhanced organization scope determination - RESTRICTED to Downloads and Documents only"""
        user_lower = user_input.lower()
        
        # RESTRICTED: Only Downloads and Documents folders allowed for safety
        if 'download' in user_lower:
            return 'Downloads folder'
        elif 'document' in user_lower:
            return 'Documents folder'
        else:
            # All other requests map to Downloads + Documents for safety
            return 'common_directories (Downloads, Documents only)'
    
    def _determine_organization_scope_legacy(self, user_input: str) -> str:
        """Legacy organization scope determination - DISABLED for safety"""
        # This method is kept for reference but should not be used
        # All file organization is now restricted to Downloads and Documents only
        return 'common_directories (Downloads, Documents only)'
    
    def _extract_organization_preferences(self, user_input: str) -> Dict[str, Any]:
        """Extract user preferences for file organization"""
        user_lower = user_input.lower()
        
        preferences = {
            'by_type': 'type' in user_lower or 'extension' in user_lower,
            'by_date': any(word in user_lower for word in ['date', 'time', 'recent', 'old']),
            'by_size': 'size' in user_lower,
            'create_subfolders': 'folder' in user_lower or 'subfolder' in user_lower,
            'handle_duplicates': 'duplicate' in user_lower or 'copy' in user_lower,
            'archive_old_files': 'archive' in user_lower or 'old' in user_lower
        }
        
        return preferences
    
    def _extract_file_operation_params(self, user_input: str, operation: str) -> Dict[str, Any]:
        """Extract parameters for file operations"""
        
        # Extract file paths (simple implementation)
        path_pattern = r'[A-Za-z]:[\\\/](?:[^\\\/\n\r]+[\\\/]?)*'
        paths = re.findall(path_pattern, user_input)
        
        # Extract file extensions
        ext_pattern = r'\.(\w+)'
        extensions = re.findall(ext_pattern, user_input)
        
        return {
            'source': paths[0] if paths else None,
            'target': paths[1] if len(paths) > 1 else None,
            'pattern': f"*.{extensions[0]}" if extensions else "*",
            'recursive': 'recursive' in user_input.lower() or 'subfolder' in user_input.lower()
        }
    
    def _extract_search_parameters(self, user_input: str) -> Dict[str, Any]:
        """Enhanced search parameter extraction"""
        
        # Extract quoted search terms
        quoted_match = re.search(r'[\"\']([\s\S]*?)[\"\']', user_input)
        if quoted_match:
            pattern = quoted_match.group(1)
        else:
            # Extract search term after common keywords
            search_match = re.search(r'(?:find|search|locate|look\s+for)\s+([\w\s\.]+)', user_input, re.IGNORECASE)
            pattern = search_match.group(1).strip() if search_match else "*"
        
        # Extract file types
        file_types = []
        type_patterns = {
            'documents': ['.doc', '.docx', '.pdf', '.txt', '.rtf'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
            'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
            'spreadsheets': ['.xls', '.xlsx', '.csv'],
            'presentations': ['.ppt', '.pptx']
        }
        
        user_lower = user_input.lower()
        for type_name, extensions in type_patterns.items():
            if type_name in user_lower:
                file_types.extend(extensions)
        
        # Extract specific extensions
        ext_matches = re.findall(r'\.(\w+)', user_input)
        file_types.extend([f'.{ext}' for ext in ext_matches])
        
        # Determine search path
        path_keywords = {
            'desktop': 'Desktop',
            'downloads': 'Downloads',
            'documents': 'Documents',
            'pictures': 'Pictures',
            'music': 'Music',
            'videos': 'Videos'
        }
        
        search_path = 'user_directories'
        for keyword, path in path_keywords.items():
            if keyword in user_lower:
                search_path = path
                break
        
        return {
            'pattern': pattern,
            'path': search_path,
            'file_types': file_types
        }
    
    def _extract_time_information(self, user_input: str) -> Dict[str, Any]:
        """Extract time and date information from user input"""
        
        time_info = {
            'date': None,
            'time': None,
            'duration': None,
            'relative_time': None,
            'recurring': False
        }
        
        # Relative time patterns
        relative_patterns = {
            'today': 0,
            'tomorrow': 1,
            'next week': 7,
            'next month': 30,
            'in an hour': 1/24,
            'in 2 hours': 2/24
        }
        
        user_lower = user_input.lower()
        for phrase, days_offset in relative_patterns.items():
            if phrase in user_lower:
                time_info['relative_time'] = phrase
                time_info['days_offset'] = days_offset
                break
        
        # Extract specific times
        time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm)?'
        time_match = re.search(time_pattern, user_input, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            period = time_match.group(3)
            
            if period and period.lower() == 'pm' and hour < 12:
                hour += 12
            elif period and period.lower() == 'am' and hour == 12:
                hour = 0
            
            time_info['time'] = f"{hour:02d}:{minute:02d}"
        
        # Extract duration
        duration_pattern = r'(\d+)\s*(hour|minute|hr|min)s?'
        duration_match = re.search(duration_pattern, user_input, re.IGNORECASE)
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2).lower()
            if unit in ['hour', 'hr']:
                time_info['duration'] = amount * 60
            else:
                time_info['duration'] = amount
        
        # Check for recurring patterns
        recurring_keywords = ['daily', 'weekly', 'monthly', 'every', 'recurring', 'repeat']
        time_info['recurring'] = any(keyword in user_lower for keyword in recurring_keywords)
        
        return time_info
    
    def _determine_email_priority(self, user_input: str) -> str:
        """Determine email priority based on content"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['urgent', 'asap', 'immediately', 'critical', 'emergency']):
            return 'high'
        elif any(word in user_lower for word in ['when convenient', 'no rush', 'whenever']):
            return 'low'
        else:
            return 'normal'
    
    def _get_time_of_day(self) -> str:
        """Get current time of day for contextual responses"""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _classify_question_type(self, user_input: str) -> str:
        """Classify the type of question being asked"""
        user_lower = user_input.lower()
        
        if user_input.startswith('what'):
            return 'definition' if any(word in user_lower for word in ['is', 'are', 'means']) else 'factual'
        elif user_input.startswith('how'):
            return 'procedural' if any(word in user_lower for word in ['do', 'to', 'can']) else 'explanatory'
        elif user_input.startswith('why'):
            return 'causal'
        elif user_input.startswith('when'):
            return 'temporal'
        elif user_input.startswith('where'):
            return 'locational'
        elif user_input.startswith('who'):
            return 'personal'
        else:
            return 'general'
    
    def _needs_web_search(self, user_input: str) -> bool:
        """Determine if question needs web search"""
        
        # Current events indicators
        current_indicators = ['today', 'current', 'latest', 'recent', 'now', '2024', '2025']
        
        # Real-time data indicators
        realtime_indicators = ['weather', 'stock', 'news', 'price', 'score']
        
        user_lower = user_input.lower()
        
        return (any(indicator in user_lower for indicator in current_indicators) or
                any(indicator in user_lower for indicator in realtime_indicators))
    
    def _assess_workflow_complexity(self, user_input: str) -> str:
        """Assess the complexity of requested workflow"""
        
        complexity_indicators = {
            'simple': ['one', 'single', 'just', 'only', 'quick'],
            'medium': ['few', 'several', 'multiple', 'some'],
            'complex': ['many', 'complex', 'advanced', 'sophisticated', 'elaborate']
        }
        
        user_lower = user_input.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in user_lower for indicator in indicators):
                return complexity
        
        # Assess by length and keyword density
        word_count = len(user_input.split())
        if word_count < 5:
            return 'simple'
        elif word_count < 15:
            return 'medium'
        else:
            return 'complex'
    
    def _suggest_workflow_steps(self, user_input: str) -> List[str]:
        """Suggest workflow steps based on user input"""
        
        # Basic workflow step suggestions
        steps = ['Analyze requirements', 'Plan execution', 'Execute actions', 'Verify results']
        
        user_lower = user_input.lower()
        
        # Add specific steps based on content
        if any(word in user_lower for word in ['file', 'folder', 'document']):
            steps.insert(1, 'Scan file system')
        
        if any(word in user_lower for word in ['email', 'message', 'send']):
            steps.insert(1, 'Compose messages')
        
        if any(word in user_lower for word in ['schedule', 'calendar', 'meeting']):
            steps.insert(1, 'Check calendar availability')
        
        return steps
    
    def _generate_clarification_questions(self, user_input: str) -> List[str]:
        """Generate helpful clarification questions"""
        
        questions = []
        user_lower = user_input.lower()
        
        if len(user_input.split()) < 3:
            questions.append("Could you provide more details about what you'd like to do?")
        
        if not any(action in user_lower for action in ['send', 'open', 'create', 'find', 'schedule']):
            questions.append("What specific action would you like me to perform?")
        
        if any(word in user_lower for word in ['this', 'that', 'it']) and len(self.conversation_manager.conversation_history) == 0:
            questions.append("What are you referring to specifically?")
        
        return questions[:2]  # Limit to most relevant questions
    
    def _create_error_result(self, user_input: str, error_message: str) -> AnalysisResult:
        """Create error result for exception handling"""
        
        return AnalysisResult(
            action_type=ActionType.CHAT,
            intent='Error processing user input',
            parameters={
                'error': True,
                'error_message': error_message,
                'original_input': user_input
            },
            confidence=0.0,
            suggestions=['Please try rephrasing your request'],
            timestamp=datetime.now().isoformat(),
            original_input=user_input
        )
    
    def format_analysis_for_user(self, analysis: AnalysisResult) -> str:
        """Enhanced formatting for user display"""
        
        # Confidence emoji
        confidence_emoji = {
            'Very High': '🎯',
            'High': '✅',
            'Medium': '🤔',
            'Low': '❓',
            'Very Low': '⚠️'
        }
        
        confidence_level = (
            'Very High' if analysis.confidence >= 0.9 else
            'High' if analysis.confidence >= 0.8 else
            'Medium' if analysis.confidence >= 0.6 else
            'Low' if analysis.confidence >= 0.4 else
            'Very Low'
        )
        
        emoji = confidence_emoji[confidence_level]
        
        output = f"{emoji} **Analysis Results**\n"
        output += f"**Intent:** {analysis.intent}\n"
        output += f"**Action Type:** {analysis.action_type.value.replace('_', ' ').title()}\n"
        output += f"**Confidence:** {confidence_level} ({analysis.confidence:.0%})\n"
        
        if analysis.processing_time_ms:
            output += f"**Processing Time:** {analysis.processing_time_ms:.1f}ms\n"
        
        output += "\n"
        
        # Display key parameters
        if analysis.parameters:
            important_params = {k: v for k, v in analysis.parameters.items() 
                             if k not in ['original_request', 'user_input', 'description'] and v}
            
            if important_params:
                output += "**Key Parameters:**\n"
                for key, value in important_params.items():
                    if isinstance(value, (list, dict)):
                        output += f"  • {key.replace('_', ' ').title()}: {json.dumps(value, indent=2)}\n"
                    else:
                        output += f"  • {key.replace('_', ' ').title()}: {value}\n"
                output += "\n"
        
        # Display suggestions
        if analysis.suggestions:
            output += "**Next Steps:**\n"
            for suggestion in analysis.suggestions:
                output += f"  ✓ {suggestion}\n"
            output += "\n"
        
        # Display alternatives if available
        if analysis.alternative_interpretations:
            output += "**Alternative Interpretations:**\n"
            for alt in analysis.alternative_interpretations:
                output += f"  • {alt['interpretation']} (confidence: {alt['confidence']:.0%})\n"
        
        return output.strip()
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get usage analytics and insights"""
        
        history = self.conversation_manager.conversation_history
        
        if not history:
            return {'message': 'No interaction history available'}
        
        # Calculate statistics
        action_counts = {}
        total_confidence = 0
        
        for interaction in history:
            action_type = interaction['analysis_result']['action_type']
            confidence = interaction['analysis_result']['confidence']
            
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(history)
        most_common_action = max(action_counts, key=action_counts.get)
        
        return {
            'total_interactions': len(history),
            'average_confidence': avg_confidence,
            'most_common_action': most_common_action,
            'action_distribution': action_counts,
            'session_id': self.conversation_manager.session_id,
            'user_preferences': self.conversation_manager.user_preferences
        }
    
    def export_conversation_log(self) -> str:
        """Export conversation history as JSON"""
        
        export_data = {
            'session_id': self.conversation_manager.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'conversation_history': self.conversation_manager.conversation_history,
            'user_preferences': self.conversation_manager.user_preferences,
            'analytics': self.get_analytics()
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)

# Factory class for creating analyzer instances
class AnalyzerFactory:
    """Factory for creating different types of analyzers"""
    
    @staticmethod
    def create_analyzer(analyzer_type: str = 'enhanced', **kwargs) -> EnhancedLLMAnalyzer:
        """Create analyzer instance based on type"""
        
        if analyzer_type == 'enhanced':
            return EnhancedLLMAnalyzer(**kwargs)
        elif analyzer_type == 'basic':
            # Could return basic version for lightweight usage
            return EnhancedLLMAnalyzer(enable_ml_features=False, **kwargs)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

# Backward compatibility alias
LLMAnalyzer = EnhancedLLMAnalyzer

# Example usage and testing
def demo_enhanced_analyzer():
    """Demonstration of enhanced analyzer capabilities"""
    
    analyzer = AnalyzerFactory.create_analyzer('enhanced')
    
    test_inputs = [
        "Hello there!",
        "Send an email to john@example.com about the meeting tomorrow",
        "Open calculator",
        "Organize my downloads folder by file type",
        "Set volume to 75%",
        "Find all PDF files in my documents",
        "Schedule a meeting for next Tuesday at 3 PM",
        "What is machine learning?",
        "Clean up my temporary files and recycle bin"
    ]
    
    print("🚀 Enhanced LLM Analyzer Demo\n")
    print("=" * 50)
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n📝 Test {i}: {test_input}")
        print("-" * 30)
        
        result = analyzer.analyze_user_input(test_input)
        formatted_output = analyzer.format_analysis_for_user(result)
        print(formatted_output)
    
    # Display analytics
    print("\n📊 Session Analytics")
    print("=" * 30)
    analytics = analyzer.get_analytics()
    print(json.dumps(analytics, indent=2))

if __name__ == "__main__":
    demo_enhanced_analyzer()