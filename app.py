import os
import random
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "rps_model.h5"
DATA_PATH = "game_data.json"
TRAIN_INTERVAL = 10

# Initialize or load game history
def load_game_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return {"history": [], "wins": 0, "losses": 0, "ties": 0, "games_played": 0}

game_data = load_game_data()

def save_game_data():
    with open(DATA_PATH, "w") as f:
        json.dump(game_data, f)

# Load or create model
def create_lightweight_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Output for rock, paper, scissors
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_lightweight_model()

def get_ai_move(player_move):
    if len(game_data["history"]) < 10:
        return random.choice(["rock", "paper", "scissors"])

    moves = {"rock": 0, "paper": 1, "scissors": 2}
    x_input = np.zeros((1, 3))
    x_input[0, moves[player_move]] = 1
    prediction = model.predict(x_input)[0]
    counter_moves = {0: "paper", 1: "scissors", 2: "rock"}
    return counter_moves[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html', stats=game_data)

@app.route('/play', methods=['POST'])
def play():
    player_move = request.json['move']
    ai_move = get_ai_move(player_move)
    
    outcomes = {("rock", "scissors"): "win", ("scissors", "paper"): "win", ("paper", "rock"): "win"}
    outcome = "tie" if player_move == ai_move else "win" if (player_move, ai_move) in outcomes else "loss"
    
    game_data["history"].append({"player": player_move, "ai": ai_move})
    game_data["games_played"] += 1
    if outcome == "win":
        game_data["wins"] += 1
    elif outcome == "loss":
        game_data["losses"] += 1
    else:
        game_data["ties"] += 1
    
    if len(game_data["history"]) % TRAIN_INTERVAL == 0:
        train_model()
    
    save_game_data()
    return jsonify({"player_move": player_move, "ai_move": ai_move, "outcome": outcome, "stats": game_data})

def train_model():
    moves = {"rock": 0, "paper": 1, "scissors": 2}
    X, y = [], []
    for game in game_data["history"]:
        X.append([1 if game["player"] == key else 0 for key in moves])
        y.append([1 if game["ai"] == key else 0 for key in moves])
    
    X, y = np.array(X), np.array(y)
    
    # Train with a small batch size
    model.fit(X, y, epochs=5, batch_size=2, verbose=0)
    model.save(MODEL_PATH)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
