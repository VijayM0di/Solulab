
#!/usr/bin/env python3
"""
AI Safety Models POC - Main Application
Comprehensive AI Safety System for conversational platforms

This application demonstrates a complete AI Safety Models system that includes:
1. Abuse Language Detection
2. Escalation Pattern Recognition  
3. Crisis Intervention
4. Age-Appropriate Content Filtering

Author: AI Safety Models POC
Date: 2024
"""

import asyncio
import argparse
import sys
import json
from datetime import datetime
from typing import Dict, List
import logging

# Import our AI Safety System
from ai_safety_integration import AISafetySystem
from sample_data_generator import SampleDataGenerator

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_safety_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_banner():
    """Print application banner"""

    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    AI SAFETY MODELS POC SYSTEM                      ║
    ║                                                                      ║
    ║  🛡️  Comprehensive AI-Powered Content Moderation & User Protection  ║
    ║                                                                      ║
    ║  Features:                                                           ║
    ║  • Abuse Language Detection (BERT-based)                           ║
    ║  • Escalation Pattern Recognition                                   ║
    ║  • Crisis Intervention Detection                                    ║
    ║  • Age-Appropriate Content Filtering                               ║
    ║                                                                      ║
    ║  Real-time processing with human-in-the-loop escalation            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def run_interactive_demo():
    """Run interactive demo of the AI Safety System"""

    print("\n🔄 Initializing AI Safety System...")
    safety_system = AISafetySystem()

    print("✅ System initialized successfully!")
    print("\n" + "="*60)
    print("INTERACTIVE AI SAFETY DEMO")
    print("="*60)
    print("Enter messages to test the safety system.")
    print("Type 'quit' to exit, 'stats' for system statistics.")
    print("-"*60)

    conversation_id = f"demo_conv_{int(datetime.now().timestamp())}"
    user_id = "demo_user"

    while True:
        try:
            # Get user input
            user_input = input("\n💬 Enter message: ").strip()

            if user_input.lower() == 'quit':
                print("\n👋 Thank you for using AI Safety System Demo!")
                break

            if user_input.lower() == 'stats':
                stats = safety_system.get_system_stats()
                print("\n📊 System Statistics:")
                print(json.dumps(stats, indent=2))
                continue

            if not user_input:
                print("❌ Please enter a message or 'quit' to exit.")
                continue

            # Get user age for content filtering
            try:
                age_input = input("👤 Enter user age (default 18): ").strip()
                user_age = int(age_input) if age_input else 18
            except ValueError:
                user_age = 18
                print("⚠️  Invalid age entered, using default (18)")

            # Create message data
            message_data = {
                'user_id': user_id,
                'text': user_input,
                'timestamp': datetime.now(),
                'user_age': user_age,
                'conversation_id': conversation_id,
                'platform': 'demo_platform'
            }

            # Process message
            print("\n🔍 Analyzing message...")
            result = await safety_system.process_message(message_data)

            # Display results
            print_safety_results(result)

        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def print_safety_results(result: Dict):
    """Print formatted safety analysis results"""

    print("\n" + "="*50)
    print("🛡️  SAFETY ANALYSIS RESULTS")
    print("="*50)

    # Overall Assessment
    assessment = result['overall_assessment']
    risk_level = assessment['risk_level'].upper()
    risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}

    print(f"\n📊 OVERALL RISK LEVEL: {risk_emoji.get(risk_level, '⚪')} {risk_level}")
    print(f"📈 Risk Score: {assessment['overall_risk_score']:.3f}")
    print(f"👁️  Human Review Required: {'✅ YES' if result['requires_human_review'] else '❌ NO'}")

    # Safety Checks Details
    print("\n🔍 SAFETY CHECKS PERFORMED:")

    checks = result['safety_checks']

    # Abuse Detection
    if 'abuse_detection' in checks:
        abuse = checks['abuse_detection']
        if abuse.get('status') == 'completed':
            detected = "✅ DETECTED" if abuse['detected'] else "❌ NOT DETECTED"
            print(f"  🗣️  Abuse Detection: {detected} (Confidence: {abuse['confidence']:.3f})")
            if abuse['detected']:
                categories = [cat for cat, data in abuse['categories'].items() 
                            if data.get('prediction', False)]
                if categories:
                    print(f"      Categories: {', '.join(categories)}")

    # Content Filtering
    if 'content_filtering' in checks:
        content = checks['content_filtering']
        if content.get('status') == 'completed':
            filtered = "🚫 FILTERED" if content['should_filter'] else "✅ APPROVED"
            print(f"  👶 Content Filtering: {filtered}")
            if content['should_filter']:
                print(f"      Required Age: {content['required_age']} (User Age: {content['user_age']})")
                print(f"      Action: {content['action']}")

    # Crisis Detection
    if 'crisis_detection' in checks:
        crisis = checks['crisis_detection']
        if crisis.get('status') == 'completed':
            detected = "🆘 DETECTED" if crisis['crisis_detected'] else "✅ NO CRISIS"
            print(f"  🚨 Crisis Detection: {detected}")
            if crisis['crisis_detected']:
                print(f"      Severity: {crisis['severity_level'].upper()}")
                print(f"      Intervention Required: {'✅ YES' if crisis['requires_intervention'] else '❌ NO'}")

    # Escalation Detection
    if 'escalation_detection' in checks:
        escalation = checks['escalation_detection']
        if escalation.get('status') == 'completed':
            detected = "📈 ESCALATING" if escalation['escalation_detected'] else "✅ STABLE"
            print(f"  ⚡ Escalation Detection: {detected}")
            if escalation['escalation_detected']:
                print(f"      Escalation Score: {escalation['escalation_score']:.3f}")
                print(f"      Recommendation: {escalation['recommendation']['level'].upper()}")

    # Recommended Actions
    if result['recommended_actions']:
        print("\n🎯 RECOMMENDED ACTIONS:")
        for action in result['recommended_actions']:
            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            emoji = priority_emoji.get(action['priority'], '⚪')
            print(f"  {emoji} {action['action'].upper()} (Priority: {action['priority']})")
            print(f"      {action['details']}")

    print("="*50)

async def run_batch_demo():
    """Run batch processing demo with sample data"""

    print("\n🔄 Running Batch Processing Demo...")

    # Initialize system and data generator
    safety_system = AISafetySystem()
    data_generator = SampleDataGenerator()

    # Generate sample conversations
    conversations = data_generator.generate_conversation_data(5)

    # Create test messages from conversations
    test_messages = []
    for i, conv in enumerate(conversations):
        for j, message in enumerate(conv['messages'][:3]):  # Take first 3 messages
            test_messages.append({
                'user_id': f'user_{i}',
                'text': message['text'],
                'timestamp': message['timestamp'],
                'user_age': 15 + (i % 10),  # Vary ages from 15-24
                'conversation_id': conv['conversation_id'],
                'platform': 'demo_batch'
            })

    print(f"\n📊 Processing {len(test_messages)} messages in batch...")

    # Process messages
    results = await safety_system.process_batch_messages(test_messages)

    # Summarize results
    summary = analyze_batch_results(results)
    print_batch_summary(summary)

def analyze_batch_results(results: List[Dict]) -> Dict:
    """Analyze batch processing results"""

    summary = {
        'total_processed': len(results),
        'risk_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
        'human_review_required': 0,
        'safety_issues_detected': {
            'abuse': 0,
            'content_filtering': 0,
            'crisis': 0,
            'escalation': 0
        },
        'processing_errors': 0
    }

    for result in results:
        if 'error' in result:
            summary['processing_errors'] += 1
            continue

        # Count risk levels
        risk_level = result['overall_assessment'].get('risk_level', 'low')
        summary['risk_levels'][risk_level] += 1

        # Count human reviews
        if result['requires_human_review']:
            summary['human_review_required'] += 1

        # Count safety issues
        checks = result['safety_checks']

        if checks.get('abuse_detection', {}).get('detected'):
            summary['safety_issues_detected']['abuse'] += 1

        if checks.get('content_filtering', {}).get('should_filter'):
            summary['safety_issues_detected']['content_filtering'] += 1

        if checks.get('crisis_detection', {}).get('crisis_detected'):
            summary['safety_issues_detected']['crisis'] += 1

        if checks.get('escalation_detection', {}).get('escalation_detected'):
            summary['safety_issues_detected']['escalation'] += 1

    return summary

def print_batch_summary(summary: Dict):
    """Print batch processing summary"""

    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)

    print(f"\n📈 PROCESSING STATISTICS:")
    print(f"  Total Messages Processed: {summary['total_processed']}")
    print(f"  Processing Errors: {summary['processing_errors']}")
    print(f"  Human Review Required: {summary['human_review_required']}")

    print(f"\n🎯 RISK LEVEL DISTRIBUTION:")
    for level, count in summary['risk_levels'].items():
        percentage = (count / summary['total_processed']) * 100 if summary['total_processed'] > 0 else 0
        emoji = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}[level]
        print(f"  {emoji} {level.upper()}: {count} ({percentage:.1f}%)")

    print(f"\n🛡️  SAFETY ISSUES DETECTED:")
    for issue_type, count in summary['safety_issues_detected'].items():
        print(f"  {issue_type.replace('_', ' ').title()}: {count}")

    print("="*60)

def main():
    """Main application entry point"""

    parser = argparse.ArgumentParser(description="AI Safety Models POC System")
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'batch', 'demo'],
        default='demo',
        help='Application mode'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Print banner
    print_banner()

    try:
        if args.mode == 'interactive':
            asyncio.run(run_interactive_demo())
        elif args.mode == 'batch':
            asyncio.run(run_batch_demo())
        else:  # demo mode
            from sys.ai_safety_integration import demo_ai_safety_system
            asyncio.run(demo_ai_safety_system())

    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user.")
    except Exception as e:
        print(f"\n❌ Application error: {str(e)}")
        logging.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
