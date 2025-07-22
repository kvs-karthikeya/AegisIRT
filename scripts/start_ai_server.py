"""
Start the REAL IVRS/IRT AI Server
This will run your TensorFlow neural network and serve it via Flask
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Flask and CORS
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    print("✅ Flask imported successfully")
except ImportError:
    print("❌ Flask not installed. Run: pip install flask flask-cors")
    sys.exit(1)

# Import your REAL AI model
try:
    from ai_backend import create_real_ai_model
    print("✅ REAL AI model imported successfully!")
    
    # Create your actual neural network
    print("🧠 Loading TensorFlow neural network...")
    ai_model = create_real_ai_model()
    print("🚀 Neural network loaded and trained!")
    
except Exception as e:
    print(f"❌ Failed to load AI model: {e}")
    print("Install required packages:")
    print("pip install tensorflow scikit-learn flask flask-cors numpy")
    ai_model = None

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Next.js

@app.route('/heal_element', methods=['POST'])
def heal_element():
    """Use your REAL TensorFlow neural network to heal IVRS elements"""
    try:
        if not ai_model:
            return jsonify({
                'error': 'Neural network not loaded',
                'details': 'TensorFlow model failed to initialize'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        element_info = data.get('element_info', {})
        html_content = data.get('html_content', '')
        clinical_context = data.get('clinical_context', '')
        
        print(f"🔧 Neural Network healing: {element_info.get('text_hint', 'Unknown')}")
        
        # Call your REAL neural network
        result = ai_model.heal_element(element_info, html_content, clinical_context)
        
        print(f"✅ Neural Network confidence: {result['ai_suggestion']['confidence']:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Healing error: {e}")
        return jsonify({
            'error': 'Neural network prediction failed',
            'details': str(e)
        }), 500

@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    """Train your neural network with feedback"""
    try:
        if not ai_model:
            return jsonify({'error': 'Neural network not loaded'}), 500
        
        data = request.get_json()
        healing_id = data.get('healing_id', 0)
        actual_success = data.get('actual_success', False)
        
        print(f"📚 Training neural network: {'SUCCESS' if actual_success else 'FAILURE'}")
        
        # Retrain your REAL neural network
        ai_model.provide_feedback(healing_id, actual_success)
        
        return jsonify({
            'status': 'success',
            'message': 'TensorFlow neural network retrained with feedback'
        })
        
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        return jsonify({
            'error': 'Failed to train neural network',
            'details': str(e)
        }), 500

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get your REAL neural network statistics"""
    try:
        if not ai_model:
            return jsonify({
                'server_status': 'offline',
                'model_loaded': False,
                'neural_network_active': False,
                'total_predictions': 0
            })
        
        # Get REAL statistics from your TensorFlow model
        stats = ai_model.get_neural_network_stats()
        stats.update({
            'server_status': 'online',
            'model_loaded': True,
            'last_updated': datetime.now().isoformat()
        })
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return jsonify({
            'error': 'Failed to get neural network statistics',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for your neural network"""
    return jsonify({
        'status': 'healthy',
        'ai_model_loaded': ai_model is not None,
        'neural_network_active': ai_model is not None,
        'focus': 'IVRS/IRT Self-Healing with TensorFlow',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'IVRS/IRT Self-Healing AI Server',
        'neural_network': 'TensorFlow Deep Learning',
        'status': 'online' if ai_model else 'offline',
        'endpoints': ['/heal_element', '/provide_feedback', '/get_stats', '/health']
    })

if __name__ == '__main__':
    print("🚀 IVRS/IRT Self-Healing AI Server Starting...")
    print("🎯 Focus: Clinical Trial Element Healing")
    print("🧠 AI: TensorFlow Deep Neural Network")
    print("🌐 Server: http://localhost:8000")
    print("=" * 50)
    
    if ai_model:
        print("✅ Neural Network: TRAINED & READY")
        print("🏥 IVRS/IRT Contexts: LOADED")
        model_info = ai_model.get_neural_network_stats()
        print(f"📊 Model Parameters: {model_info.get('parameters', 'N/A')}")
        print(f"🎯 Model Accuracy: {model_info.get('final_accuracy', 0):.1%}")
        print(f"🔄 Training Epochs: {model_info.get('training_epochs', 'N/A')}")
    else:
        print("❌ Neural Network: FAILED TO LOAD")
        print("   Install: pip install tensorflow scikit-learn flask flask-cors")
    
    print("=" * 50)
    print("🔗 Ready to connect to Next.js interface!")
    
    # Start the server
    app.run(host='0.0.0.0', port=8000, debug=False)
