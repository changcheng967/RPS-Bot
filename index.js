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

app.post('/api/save', async (req, res) => {
  const { userId, humanMove, aiMove, outcome } = req.body;
  
  db.serialize(async () => {
    // Save game record
    db.run(`INSERT INTO game_history (user_id, human_move, ai_move, outcome) 
           VALUES (?, ?, ?, ?)`, [userId, humanMove, aiMove, outcome]);
    
    // Update user stats
    db.run(`UPDATE users SET 
            games_played = games_played + 1,
            ${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'} = 
            ${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'} + 1,
            last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?`, [userId]);
    
    // Update global stats
    db.run(`UPDATE global_stats SET 
            total_games = total_games + 1,
            ${outcome === 'ai' ? 'ai_wins' : outcome === 'human' ? 'human_wins' : 'ties'} = 
            ${outcome === 'ai' ? 'ai_wins' : outcome === 'human' ? 'human_wins' : 'ties'} + 1`);
    
    // Train model periodically
    db.get(`SELECT COUNT(*) as count FROM game_history WHERE user_id = ?`, [userId], async (err, row) => {
      if (row.count % 10 === 0) {
        const history = await new Promise(resolve => {
          db.all(`SELECT human_move as humanMove FROM game_history 
                WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50`, [userId], (err, rows) => {
            resolve(rows);
          });
        });
        
        const userData = await new Promise(resolve => {
          db.get(`SELECT model_data FROM users WHERE user_id = ?`, [userId], (err, row) => {
            resolve(row);
          });
        });
        
        const model = await tf.loadLayersModel(tf.io.fromMemory(JSON.parse(userData.model_data)));
        await trainModel(model, history);
        
        model.save(tf.io.withSaveHandler(async artifacts => {
          db.run(`UPDATE users SET 
                model_data = ?,
                training_cycles = training_cycles + 1
                WHERE user_id = ?`, [JSON.stringify(artifacts), userId]);
          
          db.run(`UPDATE global_stats SET training_cycles = training_cycles + 1`);
        }));
      }
    });
  });
  
  res.json({success: true});
});

// Stats Endpoint
app.get('/api/stats', async (req, res) => {
  res.json(await getGlobalStats());
});

async function getGlobalStats() {
  return new Promise(resolve => {
    db.get(`SELECT 
            total_games,
            ai_wins,
            human_wins,
            ties,
            active_users,
            training_cycles
          FROM global_stats`, (err, row) => {
      resolve(row || {});
    });
  });
}

// Frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

module.exports = app;
