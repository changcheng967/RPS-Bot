const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');

const app = express();
app.use(bodyParser.json({ limit: '10mb' }));
app.use(cors());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/rpsAI', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

// Database Models
const UserModel = mongoose.model('User', new mongoose.Schema({
  userId: String,
  modelData: Object,
  stats: {
    gamesPlayed: { type: Number, default: 0 },
    wins: { type: Number, default: 0 },
    losses: { type: Number, default: 0 },
    ties: { type: Number, default: 0 }
  },
  history: [{
    humanMove: String,
    aiMove: String,
    outcome: String,
    timestamp: { type: Date, default: Date.now }
  }],
  lastUpdated: { type: Date, default: Date.now }
}));

const GlobalStatsModel = mongoose.model('GlobalStats', new mongoose.Schema({
  totalGames: { type: Number, default: 0 },
  aiWins: { type: Number, default: 0 },
  humanWins: { type: Number, default: 0 },
  ties: { type: Number, default: 0 },
  activeUsers: { type: Number, default: 0 }
}));

// Initialize global stats if they don't exist
async function initGlobalStats() {
  const stats = await GlobalStatsModel.findOne();
  if (!stats) {
    await GlobalStatsModel.create({});
  }
}

// AI Model Management
async function createNewModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    inputShape: [9] // 3 previous moves encoded as 3x one-hot
  }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// API Endpoints

// Get or create user
app.post('/api/user', async (req, res) => {
  const { userId } = req.body;
  let user = await UserModel.findOne({ userId });
  
  if (!user) {
    const model = await createNewModel();
    const modelData = await model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
    
    user = await UserModel.create({
      userId,
      modelData,
      stats: { gamesPlayed: 0, wins: 0, losses: 0, ties: 0 }
    });
    
    // Update global active users count
    await GlobalStatsModel.updateOne({}, { $inc: { activeUsers: 1 } });
  }
  
  res.json({
    userId: user.userId,
    stats: user.stats,
    modelData: user.modelData
  });
});

// Make a move
app.post('/api/move', async (req, res) => {
  const { userId, humanMove, history } = req.body;
  
  // Load user and model
  const user = await UserModel.findOne({ userId });
  if (!user) return res.status(404).send('User not found');
  
  const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
  
  // Predict next move
  const prediction = await predictNextMove(model, history);
  const aiMove = selectAIMove(prediction);
  
  // Determine outcome
  const outcome = determineOutcome(humanMove, aiMove);
  
  // Prepare response
  res.json({
    aiMove,
    prediction: Array.from(prediction),
    outcome
  });
});

// Save game result
app.post('/api/save', async (req, res) => {
  const { userId, humanMove, aiMove, outcome, history } = req.body;
  
  // Update user stats
  const update = { 
    $inc: { 
      'stats.gamesPlayed': 1,
      [`stats.${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'}`]: 1
    },
    $push: { 
      history: { humanMove, aiMove, outcome } 
    },
    lastUpdated: new Date()
  };
  
  // Update global stats
  await GlobalStatsModel.updateOne({}, {
    $inc: {
      totalGames: 1,
      [outcome === 'ai' ? 'aiWins' : outcome === 'human' ? 'humanWins' : 'ties']: 1
    }
  });
  
  // Retrain model if enough data
  if (history.length >= 10) {
    const user = await UserModel.findOne({ userId });
    const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
    
    const { inputs, labels } = prepareTrainingData(history);
    await model.fit(inputs, labels, {
      epochs: 10,
      batchSize: 8,
      verbose: 0
    });
    
    // Save updated model
    const updatedModelData = await model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
    update.modelData = updatedModelData;
    
    tf.dispose([inputs, labels]);
  }
  
  await UserModel.updateOne({ userId }, update);
  res.json({ success: true });
});

// Get global stats
app.get('/api/global', async (req, res) => {
  const stats = await GlobalStatsModel.findOne();
  res.json(stats);
});

// Helper functions
async function predictNextMove(model, history) {
  if (history.length < 3) {
    return tf.tensor1d([0.33, 0.33, 0.33]);
  }
  
  const input = prepareInputFromHistory(history.slice(-3));
  return tf.tidy(() => {
    return model.predict(tf.tensor2d([input])).squeeze();
  });
}

function prepareInputFromHistory(history) {
  const moveMap = { rock: 0, paper: 1, scissors: 2 };
  const input = [];
  
  for (const move of history) {
    const encoded = [0, 0, 0];
    encoded[moveMap[move.humanMove]] = 1;
    input.push(...encoded);
  }
  
  return input;
}

function prepareTrainingData(history) {
  const sequenceLength = 3;
  const moveMap = { rock: 0, paper: 1, scissors: 2 };
  const inputs = [];
  const labels = [];
  
  for (let i = 0; i < history.length - sequenceLength; i++) {
    const sequence = history.slice(i, i + sequenceLength);
    const nextMove = history[i + sequenceLength].humanMove;
    
    const encodedInput = prepareInputFromHistory(sequence);
    const encodedLabel = [0, 0, 0];
    encodedLabel[moveMap[nextMove]] = 1;
    
    inputs.push(encodedInput);
    labels.push(encodedLabel);
  }
  
  return {
    inputs: tf.tensor2d(inputs),
    labels: tf.tensor2d(labels)
  };
}

function selectAIMove(prediction) {
  const predictionArray = Array.isArray(prediction) ? prediction : Array.from(prediction.dataSync());
  const moves = ['rock', 'paper', 'scissors'];
  const predictedIndex = predictionArray.indexOf(Math.max(...predictionArray));
  const counters = { rock: 'paper', paper: 'scissors', scissors: 'rock' };
  return counters[moves[predictedIndex]];
}

function determineOutcome(human, ai) {
  if (human === ai) return 'tie';
  if (
    (human === 'rock' && ai === 'scissors') ||
    (human === 'paper' && ai === 'rock') ||
    (human === 'scissors' && ai === 'paper')
  ) return 'human';
  return 'ai';
}

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, async () => {
  await initGlobalStats();
  console.log(`Server running on port ${PORT}`);
});
