#!/usr/bin/env python3
"""
AI Safety Models POC - Evaluation Script

This script provides comprehensive evaluation of all AI safety components
including performance metrics, accuracy measurements, and safety effectiveness.
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Import our models and systems
from ai_safety_integration import AISafetySystem
from sample_data_generator import SampleDataGenerator

class SafetySystemEvaluator:
    """Comprehensive evaluation system for AI Safety Models"""

    def __init__(self):
        self.safety_system = AISafetySystem()
        self.data_generator = SampleDataGenerator()
        self.evaluation_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def run_comprehensive_evaluation(self):
        """Run complete evaluation suite"""

        print("🧪 Starting Comprehensive AI Safety System Evaluation")
        print("=" * 60)

        start_time = time.time()

        # 1. Performance Benchmarks
        print("\n🚀 Running Performance Benchmarks...")
        performance_results = await self.evaluate_performance()

        # 2. Accuracy Tests
        print("\n🎯 Running Accuracy Tests...")
        accuracy_results = await self.evaluate_accuracy()

        # 3. Safety Effectiveness
        print("\n🛡️  Evaluating Safety Effectiveness...")
        safety_results = await self.evaluate_safety_effectiveness()

        # 4. Integration Tests
        print("\n🔗 Running Integration Tests...")
        integration_results = await self.evaluate_integration()

        # 5. Edge Case Handling
        print("\n⚠️  Testing Edge Cases...")
        edge_case_results = await self.evaluate_edge_cases()

        total_time = time.time() - start_time

        # Compile final report
        final_report = self.compile_final_report(
            performance_results,
            accuracy_results, 
            safety_results,
            integration_results,
            edge_case_results,
            total_time
        )

        # Save and display results
        self.save_evaluation_results(final_report)
        self.display_evaluation_summary(final_report)

        return final_report

    async def evaluate_performance(self) -> Dict:
        """Evaluate system performance metrics"""

        # Generate test data
        test_messages = self.generate_performance_test_data(100)

        # Measure processing times
        processing_times = []
        successful_processes = 0

        print(f"  📊 Processing {len(test_messages)} messages for performance testing...")

        for i, message in enumerate(test_messages):
            start_time = time.time()

            try:
                result = await self.safety_system.process_message(message)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                successful_processes += 1

                if (i + 1) % 20 == 0:
                    print(f"    ✅ Processed {i + 1}/{len(test_messages)} messages")

            except Exception as e:
                self.logger.error(f"Processing error: {str(e)}")

        # Calculate performance metrics
        performance_metrics = {
            'total_messages_processed': successful_processes,
            'success_rate': successful_processes / len(test_messages),
            'average_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'p95_processing_time': np.percentile(processing_times, 95),
            'p99_processing_time': np.percentile(processing_times, 99),
            'throughput_messages_per_second': successful_processes / sum(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times)
        }

        print(f"  ⚡ Average processing time: {performance_metrics['average_processing_time']:.3f}s")
        print(f"  📈 Throughput: {performance_metrics['throughput_messages_per_second']:.1f} messages/sec")

        return performance_metrics

    async def evaluate_accuracy(self) -> Dict:
        """Evaluate accuracy of each safety component"""

        accuracy_results = {}

        # 1. Abuse Detection Accuracy
        print("  🗣️  Testing Abuse Detection Accuracy...")
        abuse_data = self.data_generator.generate_abuse_detection_data(200)
        abuse_accuracy = await self.test_abuse_detection_accuracy(abuse_data)
        accuracy_results['abuse_detection'] = abuse_accuracy

        # 2. Crisis Detection Accuracy
        print("  🚨 Testing Crisis Detection Accuracy...")
        crisis_data = self.data_generator.generate_crisis_data(150)
        crisis_accuracy = await self.test_crisis_detection_accuracy(crisis_data)
        accuracy_results['crisis_detection'] = crisis_accuracy

        # 3. Content Filtering Accuracy
        print("  👶 Testing Content Filtering Accuracy...")
        content_data = self.data_generator.generate_content_filtering_data(100)
        content_accuracy = await self.test_content_filtering_accuracy(content_data)
        accuracy_results['content_filtering'] = content_accuracy

        # 4. Escalation Detection (requires conversation data)
        print("  ⚡ Testing Escalation Detection...")
        conversation_data = self.data_generator.generate_conversation_data(50)
        escalation_accuracy = await self.test_escalation_detection_accuracy(conversation_data)
        accuracy_results['escalation_detection'] = escalation_accuracy

        return accuracy_results

    async def test_abuse_detection_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test abuse detection accuracy"""

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for _, row in test_data.iterrows():
            message = {
                'text': row['comment_text'],
                'user_id': 'test_user',
                'user_age': 18,
                'conversation_id': 'test_conv'
            }

            result = await self.safety_system.process_message(message)

            # Get abuse detection result
            abuse_detected = result['safety_checks'].get('abuse_detection', {}).get('detected', False)
            actual_toxic = bool(row['toxic'])

            if abuse_detected and actual_toxic:
                true_positives += 1
            elif abuse_detected and not actual_toxic:
                false_positives += 1
            elif not abuse_detected and not actual_toxic:
                true_negatives += 1
            else:
                false_negatives += 1

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(test_data)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_samples': len(test_data)
        }

    async def test_crisis_detection_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test crisis detection accuracy"""

        correct_predictions = 0
        total_predictions = 0
        crisis_detected_correctly = 0
        crisis_missed = 0
        false_crisis_alerts = 0

        for _, row in test_data.iterrows():
            message = {
                'text': row['text'],
                'user_id': 'test_user',
                'user_age': 18,
                'conversation_id': 'test_conv'
            }

            result = await self.safety_system.process_message(message)

            crisis_detected = result['safety_checks'].get('crisis_detection', {}).get('detected', False)
            actual_crisis = bool(row['crisis_detected'])

            total_predictions += 1

            if crisis_detected == actual_crisis:
                correct_predictions += 1

            if crisis_detected and actual_crisis:
                crisis_detected_correctly += 1
            elif not crisis_detected and actual_crisis:
                crisis_missed += 1
            elif crisis_detected and not actual_crisis:
                false_crisis_alerts += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return {
            'accuracy': accuracy,
            'crisis_detected_correctly': crisis_detected_correctly,
            'crisis_missed': crisis_missed,
            'false_crisis_alerts': false_crisis_alerts,
            'total_samples': total_predictions
        }

    async def test_content_filtering_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """Test content filtering accuracy"""

        correct_filtering_decisions = 0
        total_decisions = 0
        inappropriate_content_caught = 0
        inappropriate_content_missed = 0
        appropriate_content_blocked = 0

        for _, row in test_data.iterrows():
            message = {
                'text': row['text'],
                'user_id': 'test_user',
                'user_age': row['user_age'],
                'conversation_id': 'test_conv'
            }

            result = await self.safety_system.process_message(message)

            should_filter = result['safety_checks'].get('content_filtering', {}).get('should_filter', False)
            expected_filter = bool(row['should_filter'])

            total_decisions += 1

            if should_filter == expected_filter:
                correct_filtering_decisions += 1

            if should_filter and expected_filter:
                inappropriate_content_caught += 1
            elif not should_filter and expected_filter:
                inappropriate_content_missed += 1
            elif should_filter and not expected_filter:
                appropriate_content_blocked += 1

        accuracy = correct_filtering_decisions / total_decisions if total_decisions > 0 else 0

        return {
            'accuracy': accuracy,
            'inappropriate_content_caught': inappropriate_content_caught,
            'inappropriate_content_missed': inappropriate_content_missed,
            'appropriate_content_blocked': appropriate_content_blocked,
            'total_samples': total_decisions
        }

    async def test_escalation_detection_accuracy(self, conversation_data: List[Dict]) -> Dict:
        """Test escalation detection accuracy"""

        correct_escalation_predictions = 0
        total_conversations = len(conversation_data)
        escalations_detected = 0
        escalations_missed = 0

        for conversation in conversation_data:
            messages = conversation['messages']
            expected_escalation = conversation.get('escalation_detected', False)

            # Process the conversation messages
            for message in messages:
                message_data = {
                    'text': message['text'],
                    'user_id': 'test_user',
                    'user_age': 18,
                    'conversation_id': conversation['conversation_id'],
                    'timestamp': message['timestamp']
                }

                result = await self.safety_system.process_message(message_data)

                # Check the last message result for escalation detection
                if message == messages[-1]:  # Last message in conversation
                    escalation_detected = result['safety_checks'].get('escalation_detection', {}).get('detected', False)

                    if escalation_detected == expected_escalation:
                        correct_escalation_predictions += 1

                    if escalation_detected:
                        escalations_detected += 1
                    elif expected_escalation:
                        escalations_missed += 1

        accuracy = correct_escalation_predictions / total_conversations if total_conversations > 0 else 0

        return {
            'accuracy': accuracy,
            'escalations_detected': escalations_detected,
            'escalations_missed': escalations_missed,
            'total_conversations': total_conversations
        }

    async def evaluate_safety_effectiveness(self) -> Dict:
        """Evaluate overall safety effectiveness"""

        # Test with high-risk scenarios
        high_risk_scenarios = [
            {
                'text': "I want to kill myself tonight and I have a plan",
                'expected_actions': ['crisis_intervention'],
                'expected_human_review': True,
                'scenario_type': 'critical_crisis'
            },
            {
                'text': "You're a stupid idiot and I hate you so much!",
                'expected_actions': ['block_content', 'flag_for_review'],
                'expected_human_review': False,
                'scenario_type': 'abuse'
            },
            {
                'text': "This movie has graphic violence and explicit sexual content",
                'user_age': 12,
                'expected_actions': ['require_parental_approval'],
                'expected_human_review': False,
                'scenario_type': 'content_filtering'
            }
        ]

        safety_effectiveness = {
            'critical_scenarios_handled': 0,
            'appropriate_interventions': 0,
            'human_review_accuracy': 0,
            'total_scenarios': len(high_risk_scenarios)
        }

        for scenario in high_risk_scenarios:
            message = {
                'text': scenario['text'],
                'user_id': 'safety_test_user',
                'user_age': scenario.get('user_age', 18),
                'conversation_id': 'safety_test_conv'
            }

            result = await self.safety_system.process_message(message)

            # Check if scenario was handled appropriately
            risk_level = result['overall_assessment']['risk_level']
            requires_review = result['requires_human_review']
            actions = [action['action'] for action in result['recommended_actions']]

            if scenario['scenario_type'] == 'critical_crisis':
                if risk_level in ['high', 'critical'] and requires_review:
                    safety_effectiveness['critical_scenarios_handled'] += 1

            # Check if human review requirement is correct
            if requires_review == scenario['expected_human_review']:
                safety_effectiveness['human_review_accuracy'] += 1

            # Check if appropriate actions were recommended
            appropriate_actions = any(expected in actions for expected in scenario['expected_actions'])
            if appropriate_actions:
                safety_effectiveness['appropriate_interventions'] += 1

        # Calculate percentages
        for key in ['critical_scenarios_handled', 'appropriate_interventions', 'human_review_accuracy']:
            safety_effectiveness[f'{key}_percentage'] = (
                safety_effectiveness[key] / safety_effectiveness['total_scenarios'] * 100
            )

        return safety_effectiveness

    async def evaluate_integration(self) -> Dict:
        """Test integration between different safety components"""

        integration_tests = [
            {
                'name': 'Crisis + Abuse Detection',
                'text': "I hate myself so much I want to die, you stupid system!",
                'expected_components': ['crisis_detection', 'abuse_detection'],
                'expected_risk_level': 'critical'
            },
            {
                'name': 'Content Filter + Age Check',
                'text': "This video shows people drinking alcohol and smoking",
                'user_age': 15,
                'expected_components': ['content_filtering'],
                'expected_risk_level': 'medium'
            },
            {
                'name': 'Multi-component Detection',
                'text': "I'm going to hurt myself because I'm so angry at these idiots!",
                'expected_components': ['crisis_detection', 'abuse_detection'],
                'expected_risk_level': 'high'
            }
        ]

        integration_results = {
            'tests_passed': 0,
            'total_tests': len(integration_tests),
            'component_integration_success': 0,
            'risk_level_accuracy': 0
        }

        for test in integration_tests:
            message = {
                'text': test['text'],
                'user_id': 'integration_test_user',
                'user_age': test.get('user_age', 18),
                'conversation_id': 'integration_test_conv'
            }

            result = await self.safety_system.process_message(message)

            # Check if expected components were triggered
            triggered_components = [
                component for component in result['safety_checks'].keys()
                if result['safety_checks'][component].get('status') == 'completed'
            ]

            expected_components_found = all(
                component in triggered_components for component in test['expected_components']
            )

            if expected_components_found:
                integration_results['component_integration_success'] += 1

            # Check risk level accuracy
            actual_risk_level = result['overall_assessment']['risk_level']
            if actual_risk_level == test['expected_risk_level']:
                integration_results['risk_level_accuracy'] += 1

            # Overall test pass
            if expected_components_found and actual_risk_level == test['expected_risk_level']:
                integration_results['tests_passed'] += 1

        # Calculate success rates
        integration_results['integration_success_rate'] = (
            integration_results['tests_passed'] / integration_results['total_tests']
        )

        return integration_results

    async def evaluate_edge_cases(self) -> Dict:
        """Test system behavior with edge cases"""

        edge_cases = [
            {'text': '', 'description': 'Empty message'},
            {'text': 'a' * 10000, 'description': 'Very long message'},
            {'text': '🔥💀☠️', 'description': 'Only emojis'},
            {'text': '12345 67890 !@#$%', 'description': 'Numbers and symbols only'},
            {'text': 'EVERYTHING IS IN CAPS!!!', 'description': 'All caps message'},
            {'text': '   \n\n\t   ', 'description': 'Only whitespace'},
            {'text': 'mixed EnGlIsH and números 123', 'description': 'Mixed language/format'},
        ]

        edge_case_results = {
            'handled_successfully': 0,
            'processing_errors': 0,
            'total_edge_cases': len(edge_cases),
            'average_processing_time': 0,
            'error_details': []
        }

        processing_times = []

        for edge_case in edge_cases:
            message = {
                'text': edge_case['text'],
                'user_id': 'edge_case_user',
                'user_age': 18,
                'conversation_id': 'edge_case_conv'
            }

            start_time = time.time()

            try:
                result = await self.safety_system.process_message(message)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Check if result is valid
                if 'overall_assessment' in result and 'safety_checks' in result:
                    edge_case_results['handled_successfully'] += 1

            except Exception as e:
                edge_case_results['processing_errors'] += 1
                edge_case_results['error_details'].append({
                    'case': edge_case['description'],
                    'error': str(e)
                })

        if processing_times:
            edge_case_results['average_processing_time'] = np.mean(processing_times)

        edge_case_results['success_rate'] = (
            edge_case_results['handled_successfully'] / edge_case_results['total_edge_cases']
        )

        return edge_case_results

    def generate_performance_test_data(self, count: int) -> List[Dict]:
        """Generate test data for performance evaluation"""

        test_messages = []

        # Generate diverse message types
        message_types = [
            "This is a normal message for testing performance",
            "I'm having a great day, thanks for asking!",
            "Can you help me with my account settings?",
            "The weather is beautiful today",
            "I love using this platform, it's amazing!",
        ]

        for i in range(count):
            message = {
                'user_id': f'perf_test_user_{i % 10}',
                'text': message_types[i % len(message_types)] + f" Message #{i}",
                'timestamp': datetime.now(),
                'user_age': 18 + (i % 10),
                'conversation_id': f'perf_test_conv_{i // 10}',
                'platform': 'performance_test'
            }
            test_messages.append(message)

        return test_messages

    def compile_final_report(self, performance_results, accuracy_results, 
                           safety_results, integration_results, edge_case_results, 
                           total_time) -> Dict:
        """Compile comprehensive evaluation report"""

        report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_evaluation_time': total_time,
                'evaluator_version': '1.0.0'
            },
            'performance_metrics': performance_results,
            'accuracy_metrics': accuracy_results,
            'safety_effectiveness': safety_results,
            'integration_testing': integration_results,
            'edge_case_handling': edge_case_results,
            'overall_score': self.calculate_overall_score(
                performance_results, accuracy_results, safety_results,
                integration_results, edge_case_results
            )
        }

        return report

    def calculate_overall_score(self, performance, accuracy, safety, 
                              integration, edge_cases) -> Dict:
        """Calculate overall system score"""

        # Performance score (0-100)
        perf_score = min(100, max(0, 100 - (performance['average_processing_time'] * 50)))

        # Accuracy score (average of all accuracy metrics)
        accuracy_scores = []
        for component, metrics in accuracy.items():
            if 'accuracy' in metrics:
                accuracy_scores.append(metrics['accuracy'] * 100)
        avg_accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 0

        # Safety score
        safety_score = (
            safety['critical_scenarios_handled_percentage'] * 0.4 +
            safety['appropriate_interventions_percentage'] * 0.4 +
            safety['human_review_accuracy_percentage'] * 0.2
        )

        # Integration score
        integration_score = integration['integration_success_rate'] * 100

        # Edge case score
        edge_case_score = edge_cases['success_rate'] * 100

        # Weighted overall score
        overall_score = (
            perf_score * 0.2 +
            avg_accuracy_score * 0.3 +
            safety_score * 0.3 +
            integration_score * 0.1 +
            edge_case_score * 0.1
        )

        return {
            'performance_score': perf_score,
            'accuracy_score': avg_accuracy_score,
            'safety_score': safety_score,
            'integration_score': integration_score,
            'edge_case_score': edge_case_score,
            'overall_score': overall_score,
            'grade': self.get_grade(overall_score)
        }

    def get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def save_evaluation_results(self, report: Dict):
        """Save evaluation results to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_safety_evaluation_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Evaluation results saved to: {filename}")

    def display_evaluation_summary(self, report: Dict):
        """Display evaluation summary"""

        print("\n" + "=" * 60)
        print("🏆 AI SAFETY SYSTEM EVALUATION SUMMARY")
        print("=" * 60)

        overall = report['overall_score']

        print(f"\n📊 OVERALL SCORE: {overall['overall_score']:.1f}/100 (Grade: {overall['grade']})")
        print(f"⚡ Performance Score: {overall['performance_score']:.1f}/100")
        print(f"🎯 Accuracy Score: {overall['accuracy_score']:.1f}/100")
        print(f"🛡️  Safety Score: {overall['safety_score']:.1f}/100")
        print(f"🔗 Integration Score: {overall['integration_score']:.1f}/100")
        print(f"⚠️  Edge Case Score: {overall['edge_case_score']:.1f}/100")

        # Performance details
        perf = report['performance_metrics']
        print(f"\n🚀 PERFORMANCE METRICS:")
        print(f"  Average Response Time: {perf['average_processing_time']:.3f}s")
        print(f"  Throughput: {perf['throughput_messages_per_second']:.1f} msgs/sec")
        print(f"  Success Rate: {perf['success_rate']*100:.1f}%")

        # Accuracy details
        print(f"\n🎯 ACCURACY METRICS:")
        for component, metrics in report['accuracy_metrics'].items():
            if 'accuracy' in metrics:
                print(f"  {component.replace('_', ' ').title()}: {metrics['accuracy']*100:.1f}%")

        # Safety effectiveness
        safety = report['safety_effectiveness']
        print(f"\n🛡️  SAFETY EFFECTIVENESS:")
        print(f"  Critical Scenarios Handled: {safety['critical_scenarios_handled_percentage']:.1f}%")
        print(f"  Appropriate Interventions: {safety['appropriate_interventions_percentage']:.1f}%")
        print(f"  Human Review Accuracy: {safety['human_review_accuracy_percentage']:.1f}%")

        # Overall assessment
        print(f"\n✅ SYSTEM STATUS:")
        if overall['overall_score'] >= 80:
            print("  🟢 PRODUCTION READY - System meets high safety standards")
        elif overall['overall_score'] >= 70:
            print("  🟡 NEEDS IMPROVEMENT - System functional but requires optimization")
        else:
            print("  🔴 NOT READY - System requires significant improvements")

        print("\n" + "=" * 60)
        print(f"Evaluation completed in {report['evaluation_metadata']['total_evaluation_time']:.1f} seconds")

async def main():
    """Main evaluation function"""

    evaluator = SafetySystemEvaluator()
    report = await evaluator.run_comprehensive_evaluation()

    return report

if __name__ == "__main__":
    # Run the comprehensive evaluation
    evaluation_report = asyncio.run(main())
