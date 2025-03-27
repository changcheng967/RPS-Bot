require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();

// ============================================
// Enhanced Configuration
// ============================================

// Security Middleware
app.use(helmet());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate Limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Static Files with Cache Control
app.use(express.static(path.join(__dirname, 'public'), {
  maxAge: '1d',
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache');
    }
  }
}));

// ============================================
// Database Setup
// ============================================

const GlobalStatsSchema = new mongoose.Schema({
  totalGames: { type: Number, default: 0 },
  aiWins: { type: Number, default: 0 },
  humanWins: { type: Number, default: 0 },
  ties: { type: Number, default: 0 },
  activeUsers: { type: Number, default: 0 }
}, { timestamps: true });

const UserSchema = new mongoose.Schema({
  userId: { type: String, unique: true },
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
    outcome: String
  }]
}, { timestamps: true });

const GlobalStatsModel = mongoose.model('GlobalStats', GlobalStatsSchema);
const UserModel = mongoose.model('User', UserSchema);

// ============================================
// Database Connection
// ============================================

const connectWithRetry = async () => {
  const options = {
    serverSelectionTimeoutMS: 30000,
    socketTimeoutMS: 45000,
    maxPoolSize: 5,
    retryWrites: true,
    retryReads: true
  };

  try {
    await mongoose.connect(process.env.MONGODB_URI, options);
    console.log('MongoDB connected successfully');
    await initializeCollections();
  } catch (err) {
    console.error('MongoDB connection error:', err.message);
    console.log('Retrying connection in 5 seconds...');
    setTimeout(connectWithRetry, 5000);
  }
};

async function initializeCollections() {
  try {
    const collections = await mongoose.connection.db.listCollections().toArray();
    const collectionNames = collections.map(c => c.name);

    if (!collectionNames.includes('globalstats')) {
      await GlobalStatsModel.create({});
      console.log('Initialized globalstats collection');
    }

    if (!collectionNames.includes('users')) {
      await UserModel.createCollection();
      console.log('Initialized users collection');
    }
  } catch (err) {
    console.error('Collection initialization error:', err.message);
  }
}

// ============================================
// AI Model Management
// ============================================

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

// ============================================
// API Endpoints
// ============================================

app.post('/api/user', async (req, res, next) => {
  try {
    const { userId } = req.body;
    if (!userId) return res.status(400).json({ error: 'User ID required' });

    let user = await UserModel.findOne({ userId });
    
    if (!user) {
      const model = await createNewModel();
      const modelData = await model.save(tf.io.withSaveHandler(artifacts => artifacts));
      
      user = await UserModel.create({
        userId,
        modelData,
        stats: { gamesPlayed: 0, wins: 0, losses: 0, ties: 0 }
      });
      
      await GlobalStatsModel.updateOne(
        {}, 
        { $inc: { activeUsers: 1 } },
        { upsert: true }
      );
    }
    
    res.json({
      userId: user.userId,
      stats: user.stats
    });
  } catch (err) {
    next(err);
  }
});

app.post('/api/move', async (req, res, next) => {
  try {
    const { userId, humanMove, history } = req.body;
    if (!userId || !humanMove) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const user = await UserModel.findOne({ userId });
    if (!user) return res.status(404).json({ error: 'User not found' });
    
    const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
    const prediction = await predictNextMove(model, history || []);
    const aiMove = selectAIMove(prediction);
    
    res.json({
      aiMove,
      prediction: Array.from(prediction.dataSync()),
      outcome: determineOutcome(humanMove, aiMove)
    });

    tf.dispose(model);
  } catch (err) {
    next(err);
  }
});

app.post('/api/save', async (req, res, next) => {
  try {
    const { userId, humanMove, aiMove, outcome, history } = req.body;
    if (!userId || !humanMove || !aiMove || !outcome) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const update = { 
      $inc: { 
        'stats.gamesPlayed': 1,
        [`stats.${outcome === 'ai' ? 'losses' : outcome === 'human' ? 'wins' : 'ties'}`]: 1
      },
      $push: { 
        history: { 
          humanMove, 
          aiMove, 
          outcome 
        } 
      }
    };

    // Update global stats
    await GlobalStatsModel.updateOne(
      {},
      { $inc: { 
        totalGames: 1,
        [outcome === 'ai' ? 'aiWins' : outcome === 'human' ? 'humanWins' : 'ties']: 1 
      }},
      { upsert: true }
    );

    // Retrain model if enough data
    if (history && history.length >= 10) {
      const user = await UserModel.findOne({ userId });
      const model = await tf.loadLayersModel(tf.io.fromMemory(user.modelData));
      
      const { inputs, labels } = prepareTrainingData(history);
      await model.fit(inputs, labels, { 
        epochs: 10, 
        batchSize: 8,
        verbose: 0
      });
      
      const updatedModelData = await model.save(tf.io.withSaveHandler(artifacts => artifacts));
      update.modelData = updatedModelData;
      
      tf.dispose([inputs, labels, model]);
    }
    
    await UserModel.updateOne({ userId }, update);
    res.json({ success: true });
  } catch (err) {
    next(err);
  }
});

app.get('/api/global', async (req, res, next) => {
  try {
    const stats = await GlobalStatsModel.findOne({});
    res.json(stats || {
      totalGames: 0,
      aiWins: 0,
      humanWins: 0,
      ties: 0,
      activeUsers: 0
    });
  } catch (err) {
    next(err);
  }
});

// ============================================
// Frontend Handling
// ============================================

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ============================================
// Helper Functions
// ============================================

async function predictNextMove(model, history) {
  if (!history || history.length < 3) {
    return tf.tensor1d([0.33, 0.33, 0.33]);
  }
  
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
  const predictionArray = Array.isArray(prediction) ? 
    prediction : 
    Array.from(prediction.dataSync());
  
  const moves = ['rock', 'paper', 'scissors'];
  const predictedIndex = predictionArray.indexOf(Math.max(...predictionArray));
  const counters = { rock: 'paper', paper: 'scissors', scissors: 'rock' };
  return counters[moves[predictedIndex]];
}

function determineOutcome(human, ai) {
  if (human === ai) return 'tie';
  const winConditions = {
    rock: 'scissors',
    paper: 'rock',
    scissors: 'paper'
  };
  return winConditions[human] === ai ? 'human' : 'ai';
}

// ============================================
// Error Handling
// ============================================

app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({ 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

process.on('unhandledRejection', (err) => {
  console.error('Unhandled rejection:', err.stack);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err.stack);
  process.exit(1);
});

// ============================================
// Server Startup
// ============================================

const PORT = process.env.PORT || 10000;

connectWithRetry().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`Database: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Disconnected'}`);
  });
});
