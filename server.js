require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

const app = express();

// Middleware
app.use(bodyParser.json({ limit: '10mb' }));
app.use(cors({
  origin: [
    process.env.FRONTEND_URL || 'http://localhost:3000',
    'https://rps-bot-1.onrender.com/' // Update with your Render frontend URL
  ]
}));
app.use(express.static(path.join(__dirname, 'public')));

// Database Models
const GlobalStatsSchema = new mongoose.Schema({
  totalGames: { type: Number, default: 0 },
  aiWins: { type: Number, default: 0 },
  humanWins: { type: Number, default: 0 },
  ties: { type: Number, default: 0 },
  activeUsers: { type: Number, default: 0 }
});
const GlobalStatsModel = mongoose.model('GlobalStats', GlobalStatsSchema);

const UserSchema = new mongoose.Schema({
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
});
const UserModel = mongoose.model('User', UserSchema);

// MongoDB Connection with Retry
const connectWithRetry = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI, {
      serverSelectionTimeoutMS: 30000,
      socketTimeoutMS: 45000,
      maxPoolSize: 5
    });
    console.log('MongoDB connected successfully');
    await initializeCollections();
  } catch (err) {
    console.error('MongoDB connection error, retrying in 5 seconds...', err);
    setTimeout(connectWithRetry, 5000);
  }
};

// Collection Initialization
async function initializeCollections() {
  try {
    const db = mongoose.connection.db;
    const collections = await db.listCollections().toArray();
    const collectionNames = collections.map(c => c.name);

    if (!collectionNames.includes('globalstats')) {
      await db.createCollection('globalstats');
      await GlobalStatsModel.create({});
      console.log('Created globalstats collection');
    }

    if (!collectionNames.includes('users')) {
      await db.createCollection('users');
      console.log('Created users collection');
    }
  } catch (err) {
    console.error('Error initializing collections:', err);
  }
}

// AI Model Management
async function createNewModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    inputShape: [9]
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
app.post('/api/user', async (req, res) => {
  try {
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
      
      await GlobalStatsModel.updateOne({}, { $inc: { activeUsers: 1 } });
    }
    
    res.json({
      userId: user.userId,
      stats: user.stats,
      modelData: user.modelData
    });
  } catch (err) {
    console.error('Error in /api/user:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/move', async (req, res) => {
  try {
    const { userId, humanMove, history } = req.body;
    const user = await UserModel.findOne({ userId });
    if (!user) return res.status(404).json({ error: 'User not found' });
    
    const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
    const prediction = await predictNextMove(model, history);
    const aiMove = selectAIMove(prediction);
    
    res.json({
      aiMove,
      prediction: Array.from(prediction),
      outcome: determineOutcome(humanMove, aiMove)
    });
  } catch (err) {
    console.error('Error in /api/move:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/save', async (req, res) => {
  try {
    const { userId, humanMove, aiMove, outcome, history } = req.body;
    
    const update = { 
      $inc: { 
        'stats.gamesPlayed': 1,
        [`stats.${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'}`]: 1
      },
      $push: { history: { humanMove, aiMove, outcome } },
      lastUpdated: new Date()
    };
    
    await GlobalStatsModel.updateOne({}, {
      $inc: {
        totalGames: 1,
        [outcome === 'ai' ? 'aiWins' : outcome === 'human' ? 'humanWins' : 'ties']: 1
      }
    });
    
    if (history.length >= 10) {
      const user = await UserModel.findOne({ userId });
      const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
      
      const { inputs, labels } = prepareTrainingData(history);
      await model.fit(inputs, labels, { epochs: 10, batchSize: 8 });
      
      const updatedModelData = await model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
      update.modelData = updatedModelData;
      
      tf.dispose([inputs, labels]);
    }
    
    await UserModel.updateOne({ userId }, update);
    res.json({ success: true });
  } catch (err) {
    console.error('Error in /api/save:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/global', async (req, res) => {
  try {
    const stats = await GlobalStatsModel.findOne();
    res.json(stats || {});
  } catch (err) {
    console.error('Error in /api/global:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Health Check
app.get('/health', (req, res) => {
  const status = {
    status: 'up',
    db: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
    timestamp: new Date()
  };
  res.status(status.db === 'connected' ? 200 : 503).json(status);
});

// Helper Functions
async function predictNextMove(model, history) {
  if (history.length < 3) return tf.tensor1d([0.33, 0.33, 0.33]);
  
  const input = prepareInputFromHistory(history.slice(-3));
  return tf.tidy(() => model.predict(tf.tensor2d([input])).squeeze());
}

function prepareInputFromHistory(history) {
  const moveMap = { rock: 0, paper: 1, scissors: 2 };
  return history.flatMap(move => {
    const encoded = [0, 0, 0];
    encoded[moveMap[move.humanMove]] = 1;
    return encoded;
  });
}

function prepareTrainingData(history) {
  const sequenceLength = 3;
  const moveMap = { rock: 0, paper: 1, scissors: 2 };
  const inputs = [];
  const labels = [];
  
  for (let i = 0; i < history.length - sequenceLength; i++) {
    const sequence = history.slice(i, i + sequenceLength);
    const nextMove = history[i + sequenceLength].humanMove;
    
    inputs.push(prepareInputFromHistory(sequence));
    
    const encodedLabel = [0, 0, 0];
    encodedLabel[moveMap[nextMove]] = 1;
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
  if ((human === 'rock' && ai === 'scissors') ||
      (human === 'paper' && ai === 'rock') ||
      (human === 'scissors' && ai === 'paper')) return 'human';
  return 'ai';
}

// Error Handling
process.on('unhandledRejection', (err) => {
  console.error('Unhandled rejection:', err);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
});

// Start Server
const PORT = process.env.PORT || 10000;
connectWithRetry().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
