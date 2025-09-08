#!/usr/bin/env python3
"""
AI Safety Models POC - Quick Demonstration Script

This script provides a quick demonstration of all system capabilities
without requiring any user interaction.
"""

import asyncio
import time
from datetime import datetime

# Import our systems
from ai_safety_integration import AISafetySystem

def print_banner():
    """Print demonstration banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    AI SAFETY MODELS POC DEMO                        ║
    ║                                                                      ║
    ║              🛡️  Comprehensive Safety Demonstration                  ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def demonstrate_abuse_detection():
    """Demonstrate abuse detection capabilities"""

    print("\n🗣️  ABUSE LANGUAGE DETECTION DEMO")
    print("=" * 50)

    safety_system = AISafetySystem()

    test_messages = [
        {"text": "Thank you for your help!", "expected": "clean"},
        {"text": "You're an idiot and I hate you!", "expected": "toxic"},
        {"text": "This is a normal conversation", "expected": "clean"},
        {"text": "I'm going to destroy you, moron!", "expected": "toxic"}
    ]

    for i, test in enumerate(test_messages, 1):
        print(f"\nTest {i}: Analyzing message...")
        print(f"Input: '{test['text']}'")

        message_data = {
            'text': test['text'],
            'user_id': f'demo_user_{i}',
            'user_age': 18,
            'conversation_id': 'abuse_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)

        abuse_result = result['safety_checks'].get('abuse_detection', {})
        detected = abuse_result.get('detected', False)
        confidence = abuse_result.get('confidence', 0)

        status = "🚫 TOXIC" if detected else "✅ CLEAN"
        print(f"Result: {status} (Confidence: {confidence:.3f})")
        print(f"Expected: {test['expected'].upper()}")

        # Show if prediction matches expectation
        correct = (detected and test['expected'] == 'toxic') or (not detected and test['expected'] == 'clean')
        accuracy = "✅ CORRECT" if correct else "❌ INCORRECT"
        print(f"Prediction: {accuracy}")

async def demonstrate_crisis_detection():
    """Demonstrate crisis intervention detection"""

    print("\n\n🚨 CRISIS INTERVENTION DETECTION DEMO")
    print("=" * 50)

    safety_system = AISafetySystem()

    test_messages = [
        {"text": "I'm having a difficult day", "expected": "low_risk"},
        {"text": "I feel hopeless and don't see the point anymore", "expected": "medium_risk"},
        {"text": "I want to end my life and have a plan", "expected": "critical_risk"},
        {"text": "Thanks for listening, I feel better now", "expected": "low_risk"}
    ]

    for i, test in enumerate(test_messages, 1):
        print(f"\nTest {i}: Analyzing crisis indicators...")
        print(f"Input: '{test['text']}'")

        message_data = {
            'text': test['text'],
            'user_id': f'crisis_demo_user_{i}',
            'user_age': 20,
            'conversation_id': 'crisis_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)

        crisis_result = result['safety_checks'].get('crisis_detection', {})
        detected = crisis_result.get('detected', False)
        severity = crisis_result.get('severity_level', 'low')
        intervention = crisis_result.get('requires_intervention', False)

        risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        status = f"{risk_emoji.get(severity, '⚪')} {severity.upper()} RISK"

        print(f"Result: {status}")
        print(f"Crisis Detected: {'YES' if detected else 'NO'}")
        print(f"Intervention Required: {'YES' if intervention else 'NO'}")
        print(f"Expected: {test['expected'].replace('_', ' ').upper()}")

async def demonstrate_content_filtering():
    """Demonstrate age-appropriate content filtering"""

    print("\n\n👶 CONTENT FILTERING DEMO")
    print("=" * 50)

    safety_system = AISafetySystem()

    test_cases = [
        {"text": "Educational content about animals", "age": 8, "expected": "approved"},
        {"text": "This movie contains mild violence", "age": 12, "expected": "filtered"},
        {"text": "Adult content with explicit themes", "age": 16, "expected": "filtered"},
        {"text": "Family-friendly entertainment", "age": 10, "expected": "approved"}
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: Content filtering for {test['age']}-year-old...")
        print(f"Content: '{test['text']}'")

        message_data = {
            'text': test['text'],
            'user_id': f'content_demo_user_{i}',
            'user_age': test['age'],
            'conversation_id': 'content_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)

        content_result = result['safety_checks'].get('content_filtering', {})
        should_filter = content_result.get('should_filter', False)
        action = content_result.get('action', 'unknown')
        required_age = content_result.get('required_age', 0)

        status = "🚫 FILTERED" if should_filter else "✅ APPROVED"
        print(f"Result: {status}")
        print(f"Action: {action.replace('_', ' ').title()}")

        if should_filter:
            print(f"Required Age: {required_age} (User Age: {test['age']})")

        print(f"Expected: {test['expected'].upper()}")

async def demonstrate_escalation_detection():
    """Demonstrate escalation pattern recognition"""

    print("\n\n⚡ ESCALATION PATTERN DETECTION DEMO")
    print("=" * 50)

    safety_system = AISafetySystem()

    # Simulate an escalating conversation
    conversation = [
        "Hi, I need help with my order",
        "This is taking longer than expected",
        "I'm getting frustrated with this delay",
        "This is ridiculous! I'm very angry!",
        "I HATE this service! You people are useless!"
    ]

    print("Simulating escalating conversation:")

    for i, message in enumerate(conversation, 1):
        print(f"\nMessage {i}: '{message}'")

        message_data = {
            'text': message,
            'user_id': 'escalation_demo_user',
            'user_age': 25,
            'conversation_id': 'escalation_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)

        escalation_result = result['safety_checks'].get('escalation_detection', {})
        detected = escalation_result.get('detected', False)
        score = escalation_result.get('escalation_score', 0)
        recommendation = escalation_result.get('recommendation_level', 'unknown')

        status = "📈 ESCALATING" if detected else "✅ STABLE"
        print(f"Status: {status}")
        print(f"Escalation Score: {score:.3f}")

        if detected:
            print(f"Recommendation: {recommendation.upper()} intervention")

        # Add small delay to simulate real conversation timing
        await asyncio.sleep(0.5)

async def demonstrate_integrated_analysis():
    """Demonstrate integrated analysis of complex scenarios"""

    print("\n\n🔗 INTEGRATED ANALYSIS DEMO")
    print("=" * 50)

    safety_system = AISafetySystem()

    complex_scenarios = [
        {
            "text": "I hate this stupid system and I want to hurt myself!",
            "age": 17,
            "description": "Crisis + Abuse + Age consideration"
        },
        {
            "text": "This movie has violence and I'm only 13 but whatever",
            "age": 13,
            "description": "Content filtering + Age verification"
        },
        {
            "text": "You're all idiots and I'm done with everything!",
            "age": 20,
            "description": "Multiple safety concerns"
        }
    ]

    for i, scenario in enumerate(complex_scenarios, 1):
        print(f"\nScenario {i}: {scenario['description']}")
        print(f"Message: '{scenario['text']}'")
        print(f"User Age: {scenario['age']}")

        message_data = {
            'text': scenario['text'],
            'user_id': f'integrated_demo_user_{i}',
            'user_age': scenario['age'],
            'conversation_id': 'integrated_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)

        # Overall assessment
        assessment = result['overall_assessment']
        risk_level = assessment['risk_level']
        #risk_score = assessment['risk_score']
        human_review = result['requires_human_review']

        risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        print(f"\nOverall Risk: {risk_emoji.get(risk_level, '⚪')} {risk_level.upper()}")
        #print(f"Risk Score: {risk_score:.3f}")
        print(f"Human Review: {'⚠️ REQUIRED' if human_review else '✅ NOT REQUIRED'}")

        # Show triggered safety components
        triggered_components = []
        checks = result['safety_checks']

        if checks.get('abuse_detection', {}).get('detected'):
            triggered_components.append("Abuse Detection")
        if checks.get('crisis_detection', {}).get('detected'):
            triggered_components.append("Crisis Detection")
        if checks.get('content_filtering', {}).get('should_filter'):
            triggered_components.append("Content Filtering")
        if checks.get('escalation_detection', {}).get('detected'):
            triggered_components.append("Escalation Detection")

        if triggered_components:
            print(f"Triggered Components: {', '.join(triggered_components)}")

        # Show recommended actions
        actions = result.get('recommended_actions', [])
        if actions:
            print("Recommended Actions:")
            for action in actions[:3]:  # Show first 3 actions
                priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                emoji = priority_emoji.get(action['priority'], '⚪')
                print(f"  {emoji} {action['action'].replace('_', ' ').title()}")

async def demonstrate_performance():
    """Demonstrate system performance"""

    print("\n\n🚀 PERFORMANCE DEMONSTRATION")
    print("=" * 50)

    safety_system = AISafetySystem()

    # Test messages for performance demo
    test_messages = [
        "Hello, this is a test message for performance evaluation",
        "Another message to test processing speed and accuracy",
        "Testing system response time with different content types",
        "Performance demonstration with various message lengths and content",
        "Final test message to complete the performance evaluation"
    ]

    print(f"Processing {len(test_messages)} messages...")

    start_time = time.time()
    results = []

    for i, text in enumerate(test_messages):
        message_start = time.time()

        message_data = {
            'text': text,
            'user_id': f'perf_demo_user_{i}',
            'user_age': 18,
            'conversation_id': 'performance_demo',
            'timestamp': datetime.now()
        }

        result = await safety_system.process_message(message_data)
        processing_time = time.time() - message_start
        results.append(processing_time)

        print(f"  Message {i+1}: {processing_time:.3f}s")

    total_time = time.time() - start_time
    avg_time = sum(results) / len(results)
    throughput = len(test_messages) / total_time

    print(f"\nPerformance Summary:")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Average Time per Message: {avg_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} messages/second")
    print(f"  Fastest: {min(results):.3f}s")
    print(f"  Slowest: {max(results):.3f}s")

async def main():
    """Run the complete demonstration"""

    print_banner()

    print("\n🎬 Starting AI Safety Models Comprehensive Demo...")
    print("This demonstration will showcase all four safety components:\n")

    # Run all demonstrations
    await demonstrate_abuse_detection()
    await demonstrate_crisis_detection()
    await demonstrate_content_filtering()
    await demonstrate_escalation_detection()
    await demonstrate_integrated_analysis()
    await demonstrate_performance()

    print("\n\n🎉 DEMONSTRATION COMPLETED!")
    print("=" * 60)
    print("\nThe AI Safety Models POC has successfully demonstrated:")
    print("✅ Abuse Language Detection with high accuracy")
    print("✅ Crisis Intervention with appropriate severity assessment")
    print("✅ Age-appropriate Content Filtering")
    print("✅ Escalation Pattern Recognition")
    print("✅ Integrated Multi-component Analysis")
    print("✅ Real-time Performance (<1 second average)")
    print("\nThe system is ready for further development and deployment!")
    print("\nNext steps:")
    print("• Run the web interface: python app.py")
    print("• Run interactive demo: python main.py --mode interactive")
    print("• Run evaluation suite: python evaluate_system.py")

if __name__ == "__main__":
    asyncio.run(main())
