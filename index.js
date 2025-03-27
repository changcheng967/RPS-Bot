const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
const db = new sqlite3.Database(':memory:'); // Switch to 'rps.db' for persistence

// Initialize database
db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS global_stats (
    total_games INTEGER DEFAULT 0,
    ai_wins INTEGER DEFAULT 0,
    human_wins INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    training_cycles INTEGER DEFAULT 0
  )`);
  
  db.run(`INSERT OR IGNORE INTO global_stats DEFAULT VALUES`);
  
  db.run(`CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    model_data TEXT,
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    training_cycles INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )`);
});

// Enhanced AI Model with Learning
function createNewModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [9]}));
  model.add(tf.layers.dense({units: 16, activation: 'relu'}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

async function trainModel(model, history) {
  if (history.length < 10) return model;
  
  const moveMap = {rock: 0, paper: 1, scissors: 2};
  const {inputs, labels} = history.slice(-50).reduce((acc, _, i, arr) => {
    if (i < arr.length - 3) {
      const seq = arr.slice(i, i + 3);
      const next = arr[i + 3];
      acc.inputs.push(seq.flatMap(m => {
        const e = [0, 0, 0]; e[moveMap[m.humanMove]] = 1; return e;
      }));
      const l = [0, 0, 0]; l[moveMap[next.humanMove]] = 1;
      acc.labels.push(l);
    }
    return acc;
  }, {inputs: [], labels: []});

  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(labels);
  
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 8,
    validationSplit: 0.2
  });
  
  tf.dispose([xs, ys]);
  return model;
}

// Fixed predictNextMove function
function predictNextMove(model, history) {
  if (!history || history.length < 3) return [0.33, 0.33, 0.33];
  
  const moveMap = {rock: 0, paper: 1, scissors: 2};
  const input = history.slice(-3).flatMap(move => {
    const encoded = [0, 0, 0];
    encoded[moveMap[move.humanMove]] = 1;
    return encoded;
  });
  
  const prediction = model.predict(tf.tensor2d([input]));
  const result = Array.from(prediction.dataSync());
  tf.dispose(prediction);
  return result;
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
      model.save(tf.io.withSaveHandler(async artifacts => {
        const modelData = JSON.stringify(artifacts);
        db.run(`INSERT INTO users (user_id, model_data) VALUES (?, ?)`, [userId, modelData]);
        db.run(`UPDATE global_stats SET active_users = active_users + 1`);
        res.json({userId, trainingCycles: 0});
      }));
    } else {
      db.get("SELECT training_cycles FROM users WHERE user_id = ?", [userId], (err, row) => {
        res.json({userId, trainingCycles: row.training_cycles});
      });
    }
  });
});

app.post('/api/move', async (req, res) => {
  const { userId, humanMove } = req.body;
  
  db.get("SELECT model_data, training_cycles FROM users WHERE user_id = ?", [userId], async (err, row) => {
    if (!row) return res.status(404).send('User not found');
    
    try {
      const model = await tf.loadLayersModel(tf.io.fromMemory(JSON.parse(row.model_data)));
      const history = await new Promise(resolve => {
        db.all("SELECT human_move as humanMove FROM game_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50", [userId], (err, rows) => {
          resolve(rows);
        });
      });
      
      const prediction = predictNextMove(model, history);
      const aiMove = ['rock', 'paper', 'scissors'][prediction.indexOf(Math.max(...prediction))];
      const outcome = determineOutcome(humanMove, aiMove);
      
      res.json({
        aiMove,
        prediction,
        outcome,
        trainingStats: {
          userCycles: row.training_cycles,
          ...await getGlobalStats()
        }
      });
      
      tf.dispose(model);
    } catch (err) {
      res.status(500).json({error: err.message});
    }
  });
});

// ... (rest of your endpoints remain the same)

// Frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

module.exports = app;
