
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import our AI Safety Models
from abuse_detection_model import AbuseDetectionModel
from escalation_detection_model import EscalationPatternDetector
from crisis_intervention_model import CrisisInterventionModel
from content_filtering_model import ContentFilteringModel

class AISafetySystem:
    """
    Integrated AI Safety System
    Combines all four safety models for comprehensive content moderation and user protection
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initialize all safety models
        self.abuse_detector = AbuseDetectionModel()
        self.abuse_detector.load_model()
        # self.escalation_detector = EscalationPatternDetector()
        # #self.escalation_detector.load_model()   
        # self.crisis_detector = CrisisInterventionModel()
        # #self.crisis_detector.load_model()   
        # self.content_filter = ContentFilteringModel()
        # System configuration
        self.safety_thresholds = self.config.get('safety_thresholds', {
            'abuse_threshold': 0.7,
            'escalation_threshold': 0.8,
            'crisis_threshold': 0.8,
            'content_filter_threshold': 0.6
        })

        # Real-time processing configuration
        self.real_time_buffer = []
        self.conversation_history = {}
        self.user_profiles = {}

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Safety actions and responses
        self.safety_actions = {
            'allow': 'content_approved',
            'warn': 'content_warning_issued',
            'moderate': 'content_flagged_for_review',
            'block': 'content_blocked',
            'escalate': 'human_intervention_required',
            'crisis_response': 'crisis_intervention_activated'
        }

        self.logger.info("AI Safety System initialized successfully")

    def initialize_models(self):
        """Initialize all AI models (load pre-trained weights if available)"""
        try:
            # Try to load pre-trained models
            self.abuse_detector.load_model()
        except:
            self.logger.info("Loading pre-trained abuse detection model...")
            # If no pre-trained model available, initialize with base model
            self.abuse_detector.load_model()

        self.logger.info("All AI Safety models initialized")

    async def process_message(self, message_data: Dict) -> Dict:
        """
        Process a single message through all safety checks

        Args:
            message_data: {
                'user_id': str,
                'text': str,
                'timestamp': datetime,
                'user_age': int,
                'conversation_id': str,
                'platform': str
            }

        Returns:
            Dict containing safety analysis results and recommended actions
        """

        user_id = message_data.get('user_id')
        text = message_data.get('text', '')
        user_age = message_data.get('user_age', 18)
        conversation_id = message_data.get('conversation_id')
        timestamp = message_data.get('timestamp', datetime.now())

        # Initialize results structure
        safety_results = {
            'message_id': f"{user_id}_{int(timestamp.timestamp())}",
            'user_id': user_id,
            'text': text,
            'timestamp': timestamp.isoformat(),
            'safety_checks': {},
            'overall_assessment': {},
            'recommended_actions': [],
            'requires_human_review': False
        }

        # Run all safety checks concurrently
        try:
            # 1. Abuse Language Detection
            abuse_result = await self.run_abuse_detection(text)
            safety_results['safety_checks']['abuse_detection'] = abuse_result

            # 2. Content Filtering (Age-appropriate)
            content_result = await self.run_content_filtering(text, user_age)
            safety_results['safety_checks']['content_filtering'] = content_result

            # 3. Crisis Intervention Detection
            crisis_result = await self.run_crisis_detection(text)
            safety_results['safety_checks']['crisis_detection'] = crisis_result

            # 4. Escalation Pattern Recognition (requires conversation context)
            escalation_result = await self.run_escalation_detection(
                user_id, conversation_id, text, timestamp
            )
            safety_results['safety_checks']['escalation_detection'] = escalation_result

            # Integrate all results and determine final action
            overall_assessment = self.integrate_safety_results(
                abuse_result, content_result, crisis_result, escalation_result
            )
            safety_results['overall_assessment'] = overall_assessment

            # Determine recommended actions
            recommended_actions = self.determine_safety_actions(overall_assessment)
            safety_results['recommended_actions'] = recommended_actions

            # Check if human review is required
            safety_results['requires_human_review'] = self.requires_human_intervention(
                overall_assessment
            )

            # Log safety event if significant
            if overall_assessment['risk_level'] in ['high', 'critical']:
                self.log_safety_event(safety_results)

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            safety_results['error'] = str(e)
            safety_results['recommended_actions'] = ['system_error_review_required']

        return safety_results

    async def run_abuse_detection(self, text: str) -> Dict:
        """Run abuse language detection"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self.abuse_detector.predict, [text]
            )

            if result:
                abuse_result = result[0]
                return {
                    'detected': abuse_result['is_toxic'],
                    'confidence': abuse_result['overall_toxicity_score'],
                    'categories': abuse_result['predictions'],
                    'status': 'completed'
                }
            else:
                return {'status': 'failed', 'error': 'No result from abuse detector'}

        except Exception as e:
            self.logger.error(f"Abuse detection error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def run_content_filtering(self, text: str, user_age: int) -> Dict:
        """Run age-appropriate content filtering"""
        try:
            loop = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                self.executor, 
                self.content_filter.assess_content_appropriateness,
                text, user_age
            )

            filtering_response = self.content_filter.generate_filtering_response(assessment)

            return {
                'should_filter': assessment['should_filter'],
                'user_age': user_age,
                'required_age': assessment['max_required_age'],
                'action': filtering_response['action'],
                'content_issues': assessment['content_issues'],
                'status': 'completed'
            }

        except Exception as e:
            self.logger.error(f"Content filtering error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def run_crisis_detection(self, text: str) -> Dict:
        """Run crisis intervention detection"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self.crisis_detector.predict_crisis, text
            )

            return {
                'crisis_detected': result['crisis_detected'],
                'severity_level': result['severity_assessment']['severity_level'],
                'requires_intervention': result['severity_assessment']['requires_immediate_intervention'],
                'intervention_type': result['intervention_response']['response_type'],
                'risk_factors': result['severity_assessment']['risk_factors'],
                'status': 'completed'
            }

        except Exception as e:
            self.logger.error(f"Crisis detection error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def run_escalation_detection(self, user_id: str, conversation_id: str, 
                                     text: str, timestamp: datetime) -> Dict:
        """Run escalation pattern detection"""
        try:
            # Add message to conversation history
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []

            self.conversation_history[conversation_id].append({
                'text': text,
                'timestamp': timestamp,
                'user_id': user_id
            })

            # Get recent conversation messages for analysis
            recent_messages = self.conversation_history[conversation_id][-5:]  # Last 5 messages

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.escalation_detector.analyze_conversation_sequence,
                recent_messages
            )

            return {
                'escalation_detected': result['escalation_detected'],
                'escalation_score': result['escalation_score'],
                'recommendation': result['recommendation'],
                'pattern_analysis': {
                    'temporal': result['temporal_analysis'],
                    'sentiment': result['sentiment_progression'],
                    'linguistic': result['linguistic_escalation']
                },
                'status': 'completed'
            }

        except Exception as e:
            self.logger.error(f"Escalation detection error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def integrate_safety_results(self, abuse_result: Dict, content_result: Dict, 
                               crisis_result: Dict, escalation_result: Dict) -> Dict:
        """Integrate results from all safety checks to determine overall risk"""

        risk_scores = []
        risk_factors = []
        primary_concerns = []

        # Process abuse detection results
        if abuse_result.get('status') == 'completed':
            if abuse_result['detected']:
                risk_scores.append(abuse_result['confidence'])
                risk_factors.append('abusive_language_detected')
                primary_concerns.append({
                    'type': 'abuse',
                    'severity': 'high' if abuse_result['confidence'] > 0.8 else 'medium',
                    'categories': list(abuse_result['categories'].keys())
                })

        # Process content filtering results
        if content_result.get('status') == 'completed':
            if content_result['should_filter']:
                age_gap = content_result['required_age'] - content_result['user_age']
                content_risk = min(1.0, age_gap / 10.0)  # Normalize age gap to 0-1 scale
                risk_scores.append(content_risk)
                risk_factors.append('age_inappropriate_content')
                primary_concerns.append({
                    'type': 'content_filtering',
                    'severity': 'high' if age_gap > 5 else 'medium',
                    'action_required': content_result['action']
                })

        # Process crisis detection results
        if crisis_result.get('status') == 'completed':
            if crisis_result['crisis_detected']:
                severity_scores = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 0.95}
                crisis_risk = severity_scores.get(crisis_result['severity_level'], 0.5)
                risk_scores.append(crisis_risk)
                risk_factors.append('mental_health_crisis')
                primary_concerns.append({
                    'type': 'crisis',
                    'severity': crisis_result['severity_level'],
                    'intervention_required': crisis_result['requires_intervention']
                })

        # Process escalation detection results
        if escalation_result.get('status') == 'completed':
            if escalation_result['escalation_detected']:
                risk_scores.append(escalation_result['escalation_score'])
                risk_factors.append('conversation_escalation')
                primary_concerns.append({
                    'type': 'escalation',
                    'severity': escalation_result['recommendation']['level'],
                    'pattern': escalation_result['pattern_analysis']
                })

        # Calculate overall risk
        if risk_scores:
            overall_risk_score = max(risk_scores)  # Take the highest risk
            average_risk = np.mean(risk_scores)
        else:
            overall_risk_score = 0.0
            average_risk = 0.0

        # Determine overall risk level
        if overall_risk_score >= 0.9:
            risk_level = 'critical'
        elif overall_risk_score >= 0.7:
            risk_level = 'high'
        elif overall_risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'overall_risk_score': overall_risk_score,
            'average_risk_score': average_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'primary_concerns': primary_concerns,
            'total_checks_completed': sum(1 for r in [abuse_result, content_result, 
                                                    crisis_result, escalation_result] 
                                        if r.get('status') == 'completed')
        }

    def determine_safety_actions(self, overall_assessment: Dict) -> List[Dict]:
        """Determine what actions should be taken based on safety assessment"""

        actions = []
        risk_level = overall_assessment['risk_level']
        primary_concerns = overall_assessment['primary_concerns']

        # Crisis intervention takes highest priority
        crisis_concerns = [c for c in primary_concerns if c['type'] == 'crisis']
        if crisis_concerns:
            crisis_concern = crisis_concerns[0]
            if crisis_concern['intervention_required']:
                actions.append({
                    'action': 'crisis_intervention',
                    'priority': 'critical',
                    'details': 'Immediate crisis support resources needed',
                    'automated': True
                })
            else:
                actions.append({
                    'action': 'provide_support_resources',
                    'priority': 'high',
                    'details': 'Offer mental health support resources',
                    'automated': True
                })

        # Handle escalation
        escalation_concerns = [c for c in primary_concerns if c['type'] == 'escalation']
        if escalation_concerns:
            escalation_concern = escalation_concerns[0]
            if escalation_concern['severity'] in ['high', 'critical']:
                actions.append({
                    'action': 'de_escalation_protocol',
                    'priority': 'high',
                    'details': 'Activate conversation de-escalation measures',
                    'automated': True
                })

        # Handle abusive content
        abuse_concerns = [c for c in primary_concerns if c['type'] == 'abuse']
        if abuse_concerns:
            abuse_concern = abuse_concerns[0]
            if abuse_concern['severity'] == 'high':
                actions.append({
                    'action': 'block_content',
                    'priority': 'high',
                    'details': 'Block abusive content from being posted',
                    'automated': True
                })
            else:
                actions.append({
                    'action': 'flag_for_review',
                    'priority': 'medium',
                    'details': 'Flag content for human moderator review',
                    'automated': True
                })

        # Handle content filtering
        content_concerns = [c for c in primary_concerns if c['type'] == 'content_filtering']
        if content_concerns:
            content_concern = content_concerns[0]
            actions.append({
                'action': content_concern['action_required'],
                'priority': 'medium',
                'details': 'Apply age-appropriate content filtering',
                'automated': True
            })

        # Overall risk-based actions
        if risk_level == 'critical':
            actions.append({
                'action': 'escalate_to_human',
                'priority': 'critical',
                'details': 'Immediate human review required',
                'automated': False
            })
        elif risk_level == 'high':
            actions.append({
                'action': 'enhanced_monitoring',
                'priority': 'high',
                'details': 'Increase monitoring frequency for this user/conversation',
                'automated': True
            })

        # Default action if no specific concerns
        if not actions:
            actions.append({
                'action': 'allow',
                'priority': 'low',
                'details': 'Content passed all safety checks',
                'automated': True
            })

        return actions

    def requires_human_intervention(self, overall_assessment: Dict) -> bool:
        """Determine if human intervention is required"""

        # Critical risk always requires human review
        if overall_assessment['risk_level'] == 'critical':
            return True

        # Crisis situations require human review
        crisis_factors = [f for f in overall_assessment['risk_factors'] 
                         if 'crisis' in f or 'suicide' in f or 'self_harm' in f]
        if crisis_factors:
            return True

        # High-severity abuse requires human review
        abuse_concerns = [c for c in overall_assessment['primary_concerns'] 
                         if c['type'] == 'abuse' and c['severity'] == 'high']
        if abuse_concerns:
            return True

        # Multiple concurrent high-risk factors require human review
        if (overall_assessment['risk_level'] == 'high' and 
            len(overall_assessment['risk_factors']) >= 2):
            return True

        return False

    def log_safety_event(self, safety_results: Dict):
        """Log significant safety events for monitoring and analysis"""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'safety_alert',
            'user_id': safety_results['user_id'],
            'message_id': safety_results['message_id'],
            'risk_level': safety_results['overall_assessment']['risk_level'],
            'risk_factors': safety_results['overall_assessment']['risk_factors'],
            'actions_taken': [action['action'] for action in safety_results['recommended_actions']],
            'human_review_required': safety_results['requires_human_review']
        }

        # In a production system, this would go to a proper logging system
        self.logger.warning(f"SAFETY ALERT: {json.dumps(log_entry)}")

    async def process_batch_messages(self, messages: List[Dict]) -> List[Dict]:
        """Process multiple messages concurrently"""

        tasks = [self.process_message(message) for message in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing message {i}: {str(result)}")
                processed_results.append({
                    'message_id': f"error_{i}",
                    'error': str(result),
                    'status': 'failed'
                })
            else:
                processed_results.append(result)

        return processed_results

    def get_system_stats(self) -> Dict:
        """Get system statistics and health metrics"""

        return {
            'system_status': 'active',
            'models_initialized': {
                'abuse_detection': hasattr(self.abuse_detector, 'model') and self.abuse_detector.model is not None,
                'escalation_detection': self.escalation_detector.is_trained,
                'crisis_detection': self.crisis_detector.is_trained,
                'content_filtering': True  # Rule-based, always available
            },
            'conversation_tracking': {
                'active_conversations': len(self.conversation_history),
                'total_messages_processed': sum(len(conv) for conv in self.conversation_history.values())
            },
            'safety_thresholds': self.safety_thresholds,
            'timestamp': datetime.now().isoformat()
        }

# Demo function to show the integrated system in action
async def demo_ai_safety_system():
    """Demonstrate the AI Safety System with sample conversations"""

    # Initialize the system
    safety_system = AISafetySystem()

    # Sample messages representing different safety concerns
    test_messages = [
        {
            'user_id': 'user123',
            'text': 'Hello, I need help with my account',
            'timestamp': datetime.now(),
            'user_age': 25,
            'conversation_id': 'conv_1',
            'platform': 'web_chat'
        },
        {
            'user_id': 'user456',
            'text': 'You are such an idiot and I hate you!',
            'timestamp': datetime.now(),
            'user_age': 16,
            'conversation_id': 'conv_2',
            'platform': 'social_media'
        },
        {
            'user_id': 'user789',
            'text': "I don't want to live anymore and I'm planning to end it all tonight",
            'timestamp': datetime.now(),
            'user_age': 19,
            'conversation_id': 'conv_3',
            'platform': 'support_chat'
        },
        {
            'user_id': 'child001',
            'text': 'I want to watch that movie with lots of violence and blood',
            'timestamp': datetime.now(),
            'user_age': 10,
            'conversation_id': 'conv_4',
            'platform': 'family_app'
        }
    ]

    print("=== AI Safety System Demo ===\n")

    # Process each message
    for i, message in enumerate(test_messages, 1):
        print(f"Processing Message {i}:")
        print(f"User: {message['user_id']} (Age: {message['user_age']})")
        print(f"Text: '{message['text']}'")

        result = await safety_system.process_message(message)

        print(f"Risk Level: {result['overall_assessment']['risk_level']}")
        print(f"Human Review Required: {result['requires_human_review']}")
        print("Actions Recommended:")
        for action in result['recommended_actions']:
            print(f"  - {action['action']} (Priority: {action['priority']})")

        print("-" * 60)

    # Show system stats
    stats = safety_system.get_system_stats()
    print("\nSystem Statistics:")
    print(f"Active Conversations: {stats['conversation_tracking']['active_conversations']}")
    print(f"Total Messages Processed: {stats['conversation_tracking']['total_messages_processed']}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_ai_safety_system())
