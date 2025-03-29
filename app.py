import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify, session
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session tracking

# Load the trained model
model = load_model('rps_gpu_model.h5')

# Define Rock-Paper-Scissors move mapping
MOVES = ['Rock', 'Paper', 'Scissors']

# Initialize session stats
def init_session():
    session.setdefault('wins', 0)
    session.setdefault('losses', 0)
    session.setdefault('ties', 0)
    session.setdefault('total_games', 0)
    session.setdefault('win_streak', 0)
    session.setdefault('max_streak', 0)

# Function to predict the next move using the trained model
def predict_move(user_move):
    user_move_one_hot = np.zeros(3)
    user_move_one_hot[user_move] = 1  # One-hot encoding
    prediction_input = np.expand_dims(user_move_one_hot, axis=0)
    
    prediction = model.predict(prediction_input)
    predicted_move = np.argmax(prediction)
    return predicted_move

@app.route('/')
def index():
    init_session()
    return render_template('index.html', stats=session)

@app.route('/play', methods=['POST'])
def play():
    init_session()
    user_move = int(request.form['move'])  

    bot_move = predict_move(user_move)

    if user_move == bot_move:
        result = 'It\'s a tie!'
        session['ties'] += 1
        session['win_streak'] = 0
    elif (user_move == 0 and bot_move == 2) or (user_move == 1 and bot_move == 0) or (user_move == 2 and bot_move == 1):
        result = 'You win!'
        session['wins'] += 1
        session['win_streak'] += 1
        session['max_streak'] = max(session['max_streak'], session['win_streak'])
    else:
        result = 'You lose!'
        session['losses'] += 1
        session['win_streak'] = 0

    session['total_games'] += 1
    win_rate = (session['wins'] / session['total_games']) * 100 if session['total_games'] > 0 else 0

    return jsonify({
        'user_move': MOVES[user_move],
        'bot_move': MOVES[bot_move],
        'result': result,
        'wins': session['wins'],
        'losses': session['losses'],
        'ties': session['ties'],
        'win_rate': f"{win_rate:.2f}%",
        'total_games': session['total_games'],
        'max_streak': session['max_streak']
    })

if __name__ == '__main__':
    app.run(debug=True)
