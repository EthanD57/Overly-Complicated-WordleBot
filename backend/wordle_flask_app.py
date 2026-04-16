"""
Flask server wrapper for headless Wordle bot.
Exposes /play endpoint that accepts GET requests with query parameters.
"""

from flask import Flask, request, jsonify
import subprocess
import json
import os

app = Flask(__name__)

# Valid models
VALID_MODELS = [
    "entropy_maximization",
    "random_forest_classifier",
    "random_forest_regressor",
    "neural_network_classifier",
    "deep_q_network"
]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway."""
    return jsonify({"status": "healthy"}), 200


@app.route('/play', methods=['GET'])
def play_game():
    """
    Play a Wordle game with specified parameters.
    
    Query parameters:
    - word (optional): The word to guess. If not provided, random word is chosen.
    - model (optional): Model to use. Default: entropy_maximization
    
    Example: GET /play?word=crane&model=entropy_maximization
    """
    try:
        word = request.args.get('word', default=None)
        model = request.args.get('model', default='entropy_maximization', type=str)

        # Validate model
        if model not in VALID_MODELS:
            model = 'entropy_maximization'

        cmd = ['python', 'headless_main.py', '--model', model]

        if word is not None:
            cmd.extend(['--word', word.lower()])

        # Run the headless bot
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )

        # Check for errors
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Bot execution failed: {result.stderr}"
            }), 500

        #Results parsing
        try:
            game_result = json.loads(result.stdout)
            return jsonify(game_result), 200
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "error": f"Invalid output from bot: {result.stdout}"
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Game took too long to complete (30 second timeout)"
        }), 504

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    return jsonify({
        "models": VALID_MODELS
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
