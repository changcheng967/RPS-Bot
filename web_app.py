from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
import numpy as np
import random
from collections import defaultdict
import time

app = Flask(__name__)
app.secret_key = 'super_secret_key_123'  # Needed for session tracking

# Load AI model (only once on startup)
model = tf.keras.models.load_model('rps_gpu_model.h5')

# Analytics tracking
class GameStats:
    def __init__(self):
        self.reset_stats()
        
    def reset_stats(self):
        self.total_games = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.ties = 0
        self.player_move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.ai_move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.move_history = []
        self.win_streak = 0
        self.max_win_streak = 0
        self.last_update = time.time()
    
    def update_stats(self, player_move, ai_move, result):
        self.total_games += 1
        self.player_move_counts[player_move] += 1
        self.ai_move_counts[ai_move] += 1
        self.move_history.append((player_move, ai_move, result))
        
        if result == 'win':
            self.player_wins += 1
            self.win_streak += 1
            if self.win_streak > self.max_win_streak:
                self.max_win_streak = self.win_streak
        elif result == 'loss':
            self.ai_wins += 1
            self.win_streak = 0
        else:
            self.ties += 1
        
        self.last_update = time.time()

stats = GameStats()

@app.route('/')
def home():
    if 'stats' not in session:
        session['stats'] = {
            'total_games': 0,
            'player_wins': 0,
            'ai_wins': 0,
            'ties': 0
        }
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    data = request.json
    player_move = data.get('move', random.choice(['rock', 'paper', 'scissors']))
    move_history = data.get('move_history', [])
    
    # Predict AI move (using last 5 moves if available)
    ai_move = predict_ai_move(move_history)
    
    # Determine result
    result = determine_result(player_move, ai_move)
    
    # Update stats
    stats.update_stats(player_move, ai_move, result)
    session['stats'] = {
        'total_games': stats.total_games,
        'player_wins': stats.player_wins,
        'ai_wins': stats.ai_wins,
        'ties': stats.ties
    }
    
    # Prepare response
    response = {
        'ai_move': ai_move,
        'result': result,
        'stats': {
            'total_games': stats.total_games,
            'player_wins': stats.player_wins,
            'ai_wins': stats.ai_wins,
            'ties': stats.ties,
            'win_rate': round(stats.player_wins / max(1, stats.total_games - stats.ties) * 100, 1),
            'ai_win_rate': round(stats.ai_wins / max(1, stats.total_games - stats.ties) * 100, 1),
            'player_move_distribution': stats.player_move_counts,
            'ai_move_distribution': stats.ai_move_counts,
            'current_streak': stats.win_streak,
            'max_streak': stats.max_win_streak
        }
    }
    
    return jsonify(response)

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    stats.reset_stats()
    session['stats'] = {
        'total_games': 0,
        'player_wins': 0,
        'ai_wins': 0,
        'ties': 0
    }
    return jsonify({'status': 'stats reset'})

def predict_ai_move(move_history):
    """
    Predicts the AI move based on the last 5 player moves using the neural network.
    If less than 5 moves are available, a random choice is made.
    """
    if len(move_history) >= 5:
        last_five = move_history[-5:]
        input_data = np.array([[
            [1,0,0] if m == 'rock' else [0,1,0] if m == 'paper' else [0,0,1] 
            for m in last_five
        ]])
        # Normalize the input shape for model prediction
        input_data = input_data.reshape((1, 5, 3))  # Adjust input shape for LSTM if needed
        prediction = model.predict(input_data, verbose=0)[0]
        ai_move = ['rock', 'paper', 'scissors'][np.argmax(prediction)]
    else:
        ai_move = random.choice(['rock', 'paper', 'scissors'])
    
    return ai_move

def determine_result(player_move, ai_move):
    """
    Determines the result of the game based on the player's and AI's moves.
    """
    if player_move == ai_move:
        return 'tie'
    elif (player_move == 'rock' and ai_move == 'scissors') or \
         (player_move == 'paper' and ai_move == 'rock') or \
         (player_move == 'scissors' and ai_move == 'paper'):
        return 'win'
    else:
        return 'loss'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
