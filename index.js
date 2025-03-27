const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
const db = new sqlite3.Database(':memory:'); // Use ':memory:' or 'rps.db' for persistence

// Initialize database
db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS global_stats (
    total_games INTEGER DEFAULT 0,
    ai_wins INTEGER DEFAULT 0,
    human_wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0
  )`);
  
  db.run(`INSERT OR IGNORE INTO global_stats DEFAULT VALUES`);
  
  db.run(`CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    model_data TEXT,
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )`);
});

// AI Model Functions
function createNewModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [9]}));
  model.add(tf.layers.dense({units: 16, activation: 'relu'}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});
  return model;
}

// Game Logic
function predictNextMove(model, history) {
  if (!history || history.length < 3) return [0.33, 0.33, 0.33];
  
  const moveMap = {rock: 0, paper: 1, scissors: 2};
  const input = history.slice(-3).flatMap(move => {
    const encoded = [0, 0, 0];
    encoded[moveMap[move.humanMove]] = 1;
    return encoded;
  });
  
  return Array.from(model.predict(tf.tensor2d([input])).dataSync();
}

function determineOutcome(human, ai) {
  if (human === ai) return 'tie';
  const wins = {rock: 'scissors', paper: 'rock', scissors: 'paper'};
  return wins[human] === ai ? 'human' : 'ai';
}

// API Endpoints
app.use(express.json());
app.use(express.static('public'));

app.post('/api/user', (req, res) => {
  const { userId } = req.body;
  db.get("SELECT 1 FROM users WHERE user_id = ?", [userId], (err, row) => {
    if (!row) {
      const model = createNewModel();
      model.save(tf.io.withSaveHandler(artifacts => {
        const modelData = JSON.stringify(artifacts);
        db.run(`INSERT INTO users (user_id, model_data) VALUES (?, ?)`, [userId, modelData]);
        db.run(`UPDATE global_stats SET active_users = active_users + 1`);
        res.json({userId});
      });
    } else {
      res.json({userId});
    }
  });
});

app.post('/api/move', (req, res) => {
  const { userId, humanMove, history } = req.body;
  db.get("SELECT model_data FROM users WHERE user_id = ?", [userId], (err, row) => {
    if (!row) return res.status(404).send('User not found');
    
    tf.loadLayersModel(tf.io.fromMemory(JSON.parse(row.model_data))).then(model => {
      const prediction = predictNextMove(model, history);
      const aiMove = ['rock', 'paper', 'scissors'][prediction.indexOf(Math.max(...prediction))];
      const outcome = determineOutcome(humanMove, aiMove);
      
      res.json({aiMove, prediction, outcome});
      tf.dispose(model);
    });
  });
});

app.post('/api/save', (req, res) => {
  const { userId, humanMove, aiMove, outcome, history } = req.body;
  
  db.serialize(() => {
    db.run(`INSERT INTO game_history (user_id, human_move, ai_move, outcome) 
           VALUES (?, ?, ?, ?)`, [userId, humanMove, aiMove, outcome]);
    
    db.run(`UPDATE users SET 
            games_played = games_played + 1,
            ${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'} = 
            ${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'} + 1,
            last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?`, [userId]);
    
    db.run(`UPDATE global_stats SET 
            total_games = total_games + 1,
            ${outcome === 'ai' ? 'ai_wins' : outcome === 'human' ? 'human_wins' : 'ties'} = 
            ${outcome === 'ai' ? 'ai_wins' : outcome === 'human' ? 'human_wins' : 'ties'} + 1`);
    
    if (history?.length >= 10) {
      db.get("SELECT model_data FROM users WHERE user_id = ?", [userId], (err, row) => {
        tf.loadLayersModel(tf.io.fromMemory(JSON.parse(row.model_data))).then(model => {
          // Training logic here
          model.save(tf.io.withSaveHandler(artifacts => {
            db.run("UPDATE users SET model_data = ? WHERE user_id = ?", 
                  [JSON.stringify(artifacts), userId]);
          });
        });
      });
    }
  });
  
  res.json({success: true});
});

// Frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
module.exports = app;
