"""
Flask API Server for REAL IVRS/IRT Neural Network
Connects your TensorFlow AI to the web interface
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from datetime import datetime

# Import your REAL AI model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_backend import create_real_ai_model
    print("✅ Successfully imported REAL IVRS/IRT AI with TensorFlow!")
    ai_model = create_real_ai_model()
    print("🧠 Neural Network loaded and trained!")
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")
    print("Install dependencies: pip install tensorflow scikit-learn flask flask-cors")
    ai_model = None

app = Flask(__name__)
CORS(app)

@app.route('/heal_element', methods=['POST'])
def heal_element():
    """Use REAL neural network to heal IVRS/IRT elements"""
    try:
        if not ai_model:
            return jsonify({'error': 'Neural network not loaded'}), 500
        
        data = request.json
        element_info = data.get('element_info', {})
        html_content = data.get('html_content', '')
        clinical_context = data.get('clinical_context', '')
        
        print(f"🔧 Neural Network healing: {element_info.get('text_hint', 'Unknown')}")
        
        # Use REAL TensorFlow neural network
        result = ai_model.heal_element(element_info, html_content, clinical_context)
        
        print(f"✅ Neural Network confidence: {result['ai_suggestion']['confidence']:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Neural network error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    """Train neural network with feedback"""
    try:
        if not ai_model:
            return jsonify({'error': 'Neural network not loaded'}), 500
        
        data = request.json
        healing_id = data.get('healing_id', 0)
        actual_success = data.get('actual_success', False)
        
        print(f"📚 Training neural network: {'SUCCESS' if actual_success else 'FAILURE'}")
        
        # Retrain REAL neural network
        ai_model.provide_feedback(healing_id, actual_success)
        
        return jsonify({
            'status': 'success',
            'message': 'TensorFlow neural network retrained'
        })
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get REAL neural network statistics"""
    try:
        if not ai_model:
            return jsonify({
                'server_status': 'offline',
                'model_loaded': False,
                'neural_network_active': False
            })
        
        # Get REAL statistics from TensorFlow model
        stats = ai_model.get_neural_network_stats()
        stats.update({
            'server_status': 'online',
            'model_loaded': True,
            'last_updated': datetime.now().isoformat()
        })
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for neural network"""
    return jsonify({
        'status': 'healthy',
        'ai_model_loaded': ai_model is not None,
        'neural_network_active': ai_model is not None,
        'focus': 'IVRS/IRT Self-Healing with TensorFlow',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 IVRS/IRT Neural Network Server Starting...")
    print("🎯 Focus: Clinical Trial Element Healing")
    print("🧠 AI: TensorFlow Deep Neural Network")
    print("🌐 Server: http://localhost:8000")
    
    if ai_model:
        print("✅ Neural Network: TRAINED & READY")
        print("🏥 IVRS/IRT Contexts: LOADED")
        print("📊 Model Parameters:", ai_model.neural_network.get_model_info().get('parameters', 'N/A'))
    else:
        print("❌ Neural Network: FAILED TO LOAD")
        print("   Run: pip install tensorflow scikit-learn flask flask-cors")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
