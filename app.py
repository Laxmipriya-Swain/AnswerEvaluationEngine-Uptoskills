"""
Answer Evaluate - Semantic Answer Assessment Engine
REST API and Web Dashboard
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from predictor import AnswerPredictor

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Initialize predictor
predictor = AnswerPredictor()


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main web dashboard."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Answer Evaluation Engine',
        'description': 'Semantic Answer Assessment Engine',
        'version': '2.0.0'
    }), 200


@app.route('/api/v1/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate a single question-answer pair.
    
    Request JSON:
    {
        "question": "string",
        "answer": "string"
    }
    
    Response JSON:
    {
        "total_score": float,
        "max_score": int,
        "concept_scores": [...],
        "feedback": "string",
        "coverage_ratio": float,
        "missing_concepts": [...],
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({
                'error': 'Missing required fields: question, answer',
                'status': 'error'
            }), 400
        
        question = data['question'].strip()
        answer = data['answer'].strip()
        
        if not question or not answer:
            return jsonify({
                'error': 'Question and answer cannot be empty',
                'status': 'error'
            }), 400
        
        # Perform evaluation
        result = predictor.evaluate(question, answer)
        
        # Format response
        response = {
            'total_score': result.total_score,
            'max_score': result.max_score,
            'concept_scores': result.concept_scores,
            'feedback': result.feedback,
            'coverage_ratio': round(result.coverage_ratio, 3),
            'missing_concepts': result.missing_concepts,
            'extra_concepts': result.extra_concepts,
            'metadata': result.evaluation_metadata,
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/v1/evaluate-batch', methods=['POST'])
def evaluate_batch():
    """
    Evaluate multiple question-answer pairs.
    
    Request JSON:
    {
        "items": [
            {"question": "string", "answer": "string"},
            ...
        ]
    }
    
    Response JSON:
    {
        "results": [...],
        "batch_size": int,
        "average_score": float,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'items' not in data:
            return jsonify({
                'error': 'Missing required field: items (array of Q&A pairs)',
                'status': 'error'
            }), 400
        
        items = data['items']
        
        if not isinstance(items, list) or len(items) == 0:
            return jsonify({
                'error': 'items must be a non-empty array',
                'status': 'error'
            }), 400
        
        # Extract questions and answers
        questions = [item.get('question', '').strip() for item in items]
        answers = [item.get('answer', '').strip() for item in items]
        
        # Validate
        if any(not q for q in questions) or any(not a for a in answers):
            return jsonify({
                'error': 'All items must have non-empty question and answer',
                'status': 'error'
            }), 400
        
        # Evaluate batch
        results = predictor.evaluate_batch(questions, answers)
        
        # Format results
        formatted_results = [
            {
                'total_score': r.total_score,
                'max_score': r.max_score,
                'concept_scores': r.concept_scores,
                'coverage_ratio': round(r.coverage_ratio, 3),
                'missing_concepts': r.missing_concepts,
                'metadata': r.evaluation_metadata
            }
            for r in results if r is not None
        ]
        
        # Calculate statistics
        average_score = sum(r['total_score'] for r in formatted_results) / len(formatted_results) \
            if formatted_results else 0
        
        response = {
            'results': formatted_results,
            'batch_size': len(formatted_results),
            'average_score': round(average_score, 2),
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/v1/extract-concepts', methods=['POST'])
def extract_concepts():
    """
    Extract concepts from text.
    
    Request JSON:
    {
        "text": "string"
    }
    
    Response JSON:
    {
        "concepts": [...],
        "count": int,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'status': 'error'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Text cannot be empty',
                'status': 'error'
            }), 400
        
        # Extract concepts
        concepts = predictor.extract_concepts_only(text)
        
        # Format concepts
        formatted_concepts = [
            {
                'text': c['text'],
                'type': c.get('type', 'unknown'),
                'importance': round(c['importance'], 3)
            }
            for c in concepts
        ]
        
        response = {
            'concepts': formatted_concepts,
            'count': len(formatted_concepts),
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/v1/compare', methods=['POST'])
def compare_texts():
    """
    Compare question and answer concepts.
    
    Request JSON:
    {
        "question": "string",
        "answer": "string"
    }
    
    Response JSON:
    {
        "covered_concepts": [...],
        "missing_concepts": [...],
        "coverage_ratio": float,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({
                'error': 'Missing required fields: question, answer',
                'status': 'error'
            }), 400
        
        question = data['question'].strip()
        answer = data['answer'].strip()
        
        if not question or not answer:
            return jsonify({
                'error': 'Question and answer cannot be empty',
                'status': 'error'
            }), 400
        
        # Compare
        comparison = predictor.compare_texts(question, answer)
        
        response = {
            'covered_concepts': [c['text'] for c in comparison['covered_concepts']],
            'missing_concepts': [c['text'] for c in comparison['missing_concepts']],
            'extra_concepts': [c['text'] for c in comparison['extra_concepts']],
            'coverage_ratio': round(comparison['coverage_ratio'], 3),
            'total_question_concepts': len(comparison['question_concepts']),
            'status': 'success'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/v1/config', methods=['GET', 'POST'])
def config():
    """Get or update evaluator configuration."""
    if request.method == 'GET':
        from predictor import SCORING_WEIGHTS
        return jsonify({
            'default_weights': SCORING_WEIGHTS,
            'status': 'success'
        }), 200
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'weights' in data:
                predictor.set_weights(data['weights'])
                return jsonify({
                    'message': 'Configuration updated',
                    'weights': data['weights'],
                    'status': 'success'
                }), 200
            else:
                return jsonify({
                    'error': 'No configuration to update',
                    'status': 'error'
                }), 400
                
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Starting Answer Evaluation Engine...")
    print("Listening on http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /api/v1/evaluate")
    print("  POST /api/v1/evaluate-batch")
    print("  POST /api/v1/extract-concepts")
    print("  POST /api/v1/compare")
    print("  GET  /api/v1/config")
    print("  POST /api/v1/config")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
