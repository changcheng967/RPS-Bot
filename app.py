from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from collections import defaultdict
import pickle
import os

app = Flask(__name__)

# Initialize game data
if os.path.exists('rps_data.pkl'):
    with open('rps_data.pkl', 'rb') as f:
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
        return  # Not enough data to train
    
    # Analyze opponent patterns
    transition_counts = game_data['transition_counts']
    last_move = None
    
    # Count transitions between moves
    for move in game_data['move_history']:
        if last_move is not None:
            transition_counts[last_move][move] += 1
        last_move = move
    
    # Calculate transition probabilities
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
    
    # Predict next move based on opponent's last move
    if game_data['move_history']:
        last_opponent_move = game_data['move_history'][-1]
        next_move_probs = transition_probs.get(last_opponent_move, {
            'rock': 0.33,
            'paper': 0.33,
            'scissors': 0.33
        })
        
        # Choose move that beats most likely opponent move
        expected_opponent_move = max(next_move_probs.items(), key=lambda x: x[1])[0]
        
        if expected_opponent_move == 'rock':
            game_data['move_probabilities'] = {'rock': 0.1, 'paper': 0.6, 'scissors': 0.3}
        elif expected_opponent_move == 'paper':
            game_data['move_probabilities'] = {'rock': 0.3, 'paper': 0.1, 'scissors': 0.6}
        else:  # scissors
            game_data['move_probabilities'] = {'rock': 0.6, 'paper': 0.3, 'scissors': 0.1}
    
    # Save updated model
    with open('rps_data.pkl', 'wb') as f:
        pickle.dump(game_data, f)

def determine_winner(user_choice, bot_choice):
    """Determine the winner of a RPS game"""
    if user_choice == bot_choice:
        return 'draw'
    elif (user_choice == 'rock' and bot_choice == 'scissors') or \
         (user_choice == 'scissors' and bot_choice == 'paper') or \
         (user_choice == 'paper' and bot_choice == 'rock'):
        return 'user'
    else:
        return 'bot'

@app.route('/')
def index():
    """Render the main game page"""
    return render_template('index.html', 
                         games_played=game_data['games_played'],
                         move_probs=game_data['move_probabilities'])

@app.route('/play', methods=['POST'])
def play():
    """Handle a game round"""
    user_choice = request.json['choice']
    
    # Bot makes choice based on current probabilities
    choices = ['rock', 'paper', 'scissors']
    probs = [game_data['move_probabilities'][c] for c in choices]
    bot_choice = random.choices(choices, weights=probs, k=1)[0]
    
    # Determine outcome
    outcome = determine_winner(user_choice, bot_choice)
    
    # Update game data
    game_data['games_played'] += 1
    game_data['move_history'].append(user_choice)
    game_data['outcome_history'].append(outcome)
    
    # Train model every 10 games
    if game_data['games_played'] % 10 == 0:
        update_model()
    
    # Prepare response
    response = {
        'bot_choice': bot_choice,
        'outcome': outcome,
        'games_played': game_data['games_played'],
        'move_probs': game_data['move_probabilities'],
        'user_wins': game_data['outcome_history'].count('user'),
        'bot_wins': game_data['outcome_history'].count('bot'),
        'draws': game_data['outcome_history'].count('draw')
    }
    
    return jsonify(response)

@app.route('/stats')
def stats():
    """Return game statistics"""
    stats = {
        'games_played': game_data['games_played'],
        'user_wins': game_data['outcome_history'].count('user'),
        'bot_wins': game_data['outcome_history'].count('bot'),
        'draws': game_data['outcome_history'].count('draw'),
        'move_probs': game_data['move_probabilities'],
        'last_10_moves': game_data['move_history'][-10:] if game_data['move_history'] else [],
        'win_rate': game_data['outcome_history'].count('bot') / game_data['games_played'] if game_data['games_played'] > 0 else 0
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
