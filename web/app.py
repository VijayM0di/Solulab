
from flask import Flask, render_template, request, jsonify, session
import asyncio
import json
from datetime import datetime
import os
import logging

# Import our AI Safety System
from ai_safety_integration import AISafetySystem
from sample_data_generator import SampleDataGenerator

app = Flask(__name__)
app.secret_key = 'ai_safety_demo_key_2024'  # In production, use environment variable

# Initialize the AI Safety System
safety_system = AISafetySystem()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/demo')
def demo_page():
    """Interactive demo page"""
    return render_template('demo.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """API endpoint to analyze a message for safety"""

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # Extract message data
        message_data = {
            'user_id': data.get('user_id', 'web_user'),
            'text': data['text'],
            'timestamp': datetime.now(),
            'user_age': data.get('user_age', 18),
            'conversation_id': data.get('conversation_id', f"web_conv_{session.get('session_id', 'unknown')}"),
            'platform': 'web_interface'
        }

        # Process message asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(safety_system.process_message(message_data))
        loop.close()

        # Format response for web interface
        response = format_web_response(result)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """API endpoint for batch analysis"""

    try:
        data = request.get_json()

        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400

        messages = data['messages']

        if len(messages) > 50:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 50 messages)'}), 400

        # Prepare message data
        message_data_list = []
        for i, msg in enumerate(messages):
            message_data = {
                'user_id': msg.get('user_id', f'batch_user_{i}'),
                'text': msg.get('text', ''),
                'timestamp': datetime.now(),
                'user_age': msg.get('user_age', 18),
                'conversation_id': msg.get('conversation_id', f'batch_conv_{i}'),
                'platform': 'web_batch'
            }
            message_data_list.append(message_data)

        # Process batch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(safety_system.process_batch_messages(message_data_list))
        loop.close()

        # Format results
        formatted_results = [format_web_response(result) for result in results]

        # Generate summary
        summary = generate_batch_summary(formatted_results)

        return jsonify({
            'results': formatted_results,
            'summary': summary,
            'total_processed': len(formatted_results)
        })

    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/generate_sample_data', methods=['POST'])
def generate_sample_data():
    """Generate sample data for testing"""

    try:
        data_generator = SampleDataGenerator()

        # Generate different types of sample data
        abuse_data = data_generator.generate_abuse_detection_data(10)
        crisis_data = data_generator.generate_crisis_data(10)

        sample_messages = []

        # Add abuse samples
        for _, row in abuse_data.head(5).iterrows():
            sample_messages.append({
                'text': row['comment_text'],
                'type': 'abuse_test',
                'expected_toxic': bool(row['toxic'])
            })

        # Add crisis samples
        for _, row in crisis_data.head(5).iterrows():
            sample_messages.append({
                'text': row['text'],
                'type': 'crisis_test',
                'expected_crisis': bool(row['crisis_detected'])
            })

        # Add content filtering samples
        content_samples = [
            {'text': 'This movie contains graphic violence and adult themes', 'user_age': 14},
            {'text': 'A beautiful story about friendship and adventure', 'user_age': 8},
            {'text': 'Educational content about science and nature', 'user_age': 12}
        ]

        for sample in content_samples:
            sample_messages.append({
                'text': sample['text'],
                'type': 'content_filtering_test',
                'user_age': sample['user_age']
            })

        return jsonify({
            'sample_messages': sample_messages,
            'total_samples': len(sample_messages)
        })

    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return jsonify({'error': 'Error generating sample data'}), 500

@app.route('/api/stats')
def get_system_stats():
    """Get system statistics"""

    try:
        stats = safety_system.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Error retrieving statistics'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

def format_web_response(result):
    """Format the safety analysis result for web interface"""

    if 'error' in result:
        return {
            'status': 'error',
            'error': result['error'],
            'message_id': result.get('message_id', 'unknown')
        }

    # Extract key information
    assessment = result['overall_assessment']
    checks = result['safety_checks']

    formatted_response = {
        'status': 'success',
        'message_id': result['message_id'],
        'text_analyzed': result['text'],
        'risk_level': assessment['risk_level'],
        'risk_score': round(assessment['overall_risk_score'], 3),
        'human_review_required': result['requires_human_review'],
        'safety_checks': {},
        'recommended_actions': result['recommended_actions'],
        'timestamp': result['timestamp']
    }

    # Format individual safety checks
    if 'abuse_detection' in checks:
        abuse = checks['abuse_detection']
        formatted_response['safety_checks']['abuse_detection'] = {
            'detected': abuse.get('detected', False),
            'confidence': round(abuse.get('confidence', 0), 3),
            'status': abuse.get('status', 'unknown')
        }

        if abuse.get('detected'):
            categories = abuse.get('categories', {})
            detected_categories = [cat for cat, data in categories.items() 
                                 if isinstance(data, dict) and data.get('prediction')]
            formatted_response['safety_checks']['abuse_detection']['categories'] = detected_categories

    if 'content_filtering' in checks:
        content = checks['content_filtering']
        formatted_response['safety_checks']['content_filtering'] = {
            'should_filter': content.get('should_filter', False),
            'action': content.get('action', 'unknown'),
            'user_age': content.get('user_age', 0),
            'required_age': content.get('required_age', 0),
            'status': content.get('status', 'unknown')
        }

    if 'crisis_detection' in checks:
        crisis = checks['crisis_detection']
        formatted_response['safety_checks']['crisis_detection'] = {
            'detected': crisis.get('crisis_detected', False),
            'severity_level': crisis.get('severity_level', 'low'),
            'requires_intervention': crisis.get('requires_intervention', False),
            'status': crisis.get('status', 'unknown')
        }

    if 'escalation_detection' in checks:
        escalation = checks['escalation_detection']
        formatted_response['safety_checks']['escalation_detection'] = {
            'detected': escalation.get('escalation_detected', False),
            'escalation_score': round(escalation.get('escalation_score', 0), 3),
            'recommendation_level': escalation.get('recommendation', {}).get('level', 'unknown'),
            'status': escalation.get('status', 'unknown')
        }

    return formatted_response

def generate_batch_summary(results):
    """Generate summary statistics for batch results"""

    summary = {
        'risk_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
        'human_review_required': 0,
        'safety_issues': {
            'abuse_detected': 0,
            'content_filtered': 0,
            'crisis_detected': 0,
            'escalation_detected': 0
        },
        'processing_errors': 0
    }

    for result in results:
        if result.get('status') == 'error':
            summary['processing_errors'] += 1
            continue

        # Count risk levels
        risk_level = result.get('risk_level', 'low')
        summary['risk_levels'][risk_level] += 1

        # Count human reviews
        if result.get('human_review_required'):
            summary['human_review_required'] += 1

        # Count safety issues
        checks = result.get('safety_checks', {})

        if checks.get('abuse_detection', {}).get('detected'):
            summary['safety_issues']['abuse_detected'] += 1

        if checks.get('content_filtering', {}).get('should_filter'):
            summary['safety_issues']['content_filtered'] += 1

        if checks.get('crisis_detection', {}).get('detected'):
            summary['safety_issues']['crisis_detected'] += 1

        if checks.get('escalation_detection', {}).get('detected'):
            summary['safety_issues']['escalation_detected'] += 1

    return summary

# Initialize session
@app.before_request
def before_request():
    if 'session_id' not in session:
        session['session_id'] = f"session_{int(datetime.now().timestamp())}"

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create static directories if they don't exist
    for directory in ['static', 'static/css', 'static/js']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    app.run(debug=True, host='0.0.0.0', port=5000)
