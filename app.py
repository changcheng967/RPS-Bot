import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
from collections import defaultdict
import pickle

app = Flask(__name__, static_folder='templates')
CORS(app)

# Initialize game data
DATA_FILE = '/tmp/rps_data.pkl'

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        game_data = pickle.load(f)
else:
    game_data = {
        'games_played': 0,
        'move_history': [],
        'outcome_history': [],
        'transition_counts': defaultdict(lambda: defaultdict(int)),
        'move_probabilities': {'rock': 0.34, 'paper': 0.33, 'scissors': 0.33}
    }

def update_model():
    """Train the model based on collected data"""
    if len(game_data['move_history']) < 5:
        return

    transition_counts = game_data['transition_counts']
    last_move = None
    
    for move in game_data['move_history']:
        if last_move is not None:
            transition_counts[last_move][move] += 1
        last_move = move
    
    transition_probs = {}
    for from_move in ['rock', 'paper', 'scissors']:
        total = sum(transition_counts[from_move].values())
        if total > 0:
            transition_probs[from_move] = {
                to_move: count/total 
                for to_move, count in transition_counts[from_move].items()
            }
        else:
            transition_probs[from_move] = {
                'rock': 0.33,
                'paper': 0.33,
                'scissors': 0.33
            }
    
    if game_data['move_history']:
        last_opponent_move = game_data['move_history'][-1]
        next_move_probs = transition_probs.get(last_opponent_move, {
            'rock': 0.33,
            'paper': 0.33,
            'scissors': 0.33
        })
        
        expected_opponent_move = max(next_move_probs.items(), key=lambda x: x[1])[0]
        
        if expected_opponent_move == 'rock':
            game_data['move_probabilities'] = {'rock': 0.1, 'paper': 0.6, 'scissors': 0.3}
        elif expected_opponent_move == 'paper':
            game_data['move_probabilities'] = {'rock': 0.3, 'paper': 0.1, 'scissors': 0.6}
        else:
            game_data['move_probabilities'] = {'rock': 0.6, 'paper': 0.3, 'scissors': 0.1}
    
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(game_data, f)

def determine_winner(user_choice, bot_choice):
    if user_choice == bot_choice:
        return 'draw'
    elif (user_choice == 'rock' and bot_choice == 'scissors') or \
         (user_choice == 'scissors' and bot_choice == 'paper') or \
         (user_choice == 'paper' and bot_choice == 'rock'):
        return 'user'
    else:
        return 'bot'

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/play', methods=['POST'])
def play():
    user_choice = request.json['choice']
    
    choices = ['rock', 'paper', 'scissors']
    probs = [game_data['move_probabilities'][c] for c in choices]
    bot_choice = random.choices(choices, weights=probs, k=1)[0]
    
    outcome = determine_winner(user_choice, bot_choice)
    
    game_data['games_played'] += 1
    game_data['move_history'].append(user_choice)
    game_data['outcome_history'].append(outcome)
    
    if game_data['games_played'] % 10 == 0:
        update_model()
    
    return jsonify({
        'bot_choice': bot_choice,
        'outcome': outcome,
        'games_played': game_data['games_played'],
        'move_probs': game_data['move_probabilities'],
        'user_wins': game_data['outcome_history'].count('user'),
        'bot_wins': game_data['outcome_history'].count('bot'),
        'draws': game_data['outcome_history'].count('draw')
    })

@app.route('/api/stats')
def stats():
    return jsonify({
        'games_played': game_data['games_played'],
        'user_wins': game_data['outcome_history'].count('user'),
        'bot_wins': game_data['outcome_history'].count('bot'),
        'draws': game_data['outcome_history'].count('draw'),
        'move_probs': game_data['move_probabilities'],
        'last_10_moves': game_data['move_history'][-10:] if game_data['move_history'] else [],
        'win_rate': game_data['outcome_history'].count('bot') / game_data['games_played'] if game_data['games_played'] > 0 else 0
    })

if __name__ == '__main__':
    app.run()
