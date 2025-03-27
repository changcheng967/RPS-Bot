// Game state with deep learning integration
const gameState = {
    model: null,
    gamesPlayed: 0,
    aiWins: 0,
    humanWins: 0,
    ties: 0,
    history: [],
    training: false,
    moveMap: { rock: 0, paper: 1, scissors: 2 },
    reverseMoveMap: ['rock', 'paper', 'scissors']
};

// DOM elements
const resultEl = document.getElementById('result');
const humanMoveEl = document.getElementById('humanMove');
const aiMoveEl = document.getElementById('aiMove');
const outcomeEl = document.getElementById('outcome');
const gameCountEl = document.getElementById('gameCount');
const aiWinsEl = document.getElementById('aiWins');
const humanWinsEl = document.getElementById('humanWins');
const tiesEl = document.getElementById('ties');
const winRateEl = document.getElementById('winRate');
const trainingStatusEl = document.getElementById('trainingStatus');
const rockProbEl = document.getElementById('rockProb');
const paperProbEl = document.getElementById('paperProb');
const scissorsProbEl = document.getElementById('scissorsProb');

// Chart setup
const ctx = document.getElementById('historyChart').getContext('2d');
const historyChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'AI Win Rate',
                data: [],
                borderColor: '#4285F4',
                backgroundColor: 'rgba(66, 133, 244, 0.1)',
                tension: 0.1,
                fill: true
            },
            {
                label: 'Your Win Rate',
                data: [],
                borderColor: '#0F9D58',
                backgroundColor: 'rgba(15, 157, 88, 0.1)',
                tension: 0.1,
                fill: true
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

// Initialize game
async function initGame() {
    // Load game history from localStorage
    loadGameState();
    
    // Initialize TensorFlow.js model
    await initModel();
    
    // Set up event listeners
    document.getElementById('rock').addEventListener('click', () => playGame('rock'));
    document.getElementById('paper').addEventListener('click', () => playGame('paper'));
    document.getElementById('scissors').addEventListener('click', () => playGame('scissors'));
    
    updateUI();
    trainingStatusEl.textContent = 'Ready to play!';
}

// Initialize TensorFlow model
async function initModel() {
    trainingStatusEl.textContent = 'Initializing neural network...';
    
    // Check if we have a saved model in localStorage
    const savedModel = localStorage.getItem('rpsModel');
    
    if (savedModel) {
        try {
            gameState.model = await tf.loadLayersModel('localstorage://rpsModel');
            trainingStatusEl.textContent = 'Loaded trained model from memory';
            return;
        } catch (e) {
            console.log('Failed to load saved model', e);
        }
    }
    
    // Create new model if no saved model exists
    gameState.model = tf.sequential();
    
    // Input layer - expecting last 3 moves encoded as one-hot vectors
    gameState.model.add(tf.layers.dense({
        units: 64,
        activation: 'relu',
        inputShape: [9] // 3 moves * 3 possibilities (rock/paper/scissors)
    }));
    
    // Hidden layer
    gameState.model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }));
    
    // Output layer - probabilities for rock/paper/scissors
    gameState.model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));
    
    // Compile the model
    gameState.model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    trainingStatusEl.textContent = 'New model created. Start playing!';
}

// Main game function
async function playGame(humanChoice) {
    if (gameState.training) {
        trainingStatusEl.textContent = 'AI is currently learning... please wait';
        return;
    }
    
    // Get AI prediction
    const aiPrediction = await predictNextMove();
    const aiChoice = selectAIMove(aiPrediction);
    
    // Show AI thinking process
    updateAIConfidence(aiPrediction);
    
    // Determine outcome
    const outcome = determineWinner(humanChoice, aiChoice);
    
    // Update game state
    gameState.gamesPlayed++;
    if (outcome === 'ai') gameState.aiWins++;
    if (outcome === 'human') gameState.humanWins++;
    if (outcome === 'tie') gameState.ties++;
    
    // Add to history
    gameState.history.push({
        human: humanChoice,
        ai: aiChoice,
        outcome: outcome,
        timestamp: new Date().toISOString()
    });
    
    // Update UI
    displayResult(humanChoice, aiChoice, outcome);
    updateUI();
    
    // Train model asynchronously
    setTimeout(async () => {
        await trainModel();
        saveGameState();
    }, 0);
}

// Predict next human move using neural network
async function predictNextMove() {
    if (gameState.history.length < 3) {
        // Not enough data yet, return equal probabilities
        return [0.33, 0.33, 0.33];
    }
    
    // Prepare input data (last 3 moves)
    const lastThree = gameState.history.slice(-3);
    const input = [];
    
    for (const move of lastThree) {
        // One-hot encode each move
        const encoded = [0, 0, 0];
        encoded[gameState.moveMap[move.human]] = 1;
        input.push(...encoded);
    }
    
    // Predict
    const prediction = tf.tidy(() => {
        const inputTensor = tf.tensor2d([input]);
        const output = gameState.model.predict(inputTensor);
        return output.dataSync();
    });
    
    return Array.from(prediction);
}

// Select AI move based on prediction probabilities
function selectAIMove(prediction) {
    // Counter the most likely human move
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    const predictedHumanMove = gameState.reverseMoveMap[maxIndex];
    
    // Counter that move (with some randomness)
    if (Math.random() < 0.85) { // 85% chance to counter predicted move
        return counterMove(predictedHumanMove);
    }
    
    // 15% chance to make a random move (to avoid being too predictable)
    return randomChoice();
}

// Train the model with current history
async function trainModel() {
    if (gameState.history.length < 10) {
        return; // Not enough data to train
    }
    
    gameState.training = true;
    trainingStatusEl.textContent = 'AI is learning from recent games...';
    
    try {
        // Prepare training data
        const {inputs, labels} = prepareTrainingData();
        
        // Train the model
        await gameState.model.fit(inputs, labels, {
            epochs: 20,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    trainingStatusEl.textContent = 
                        `Training... Epoch ${epoch + 1} - Loss: ${logs.loss.toFixed(4)}`;
                }
            }
        });
        
        // Save the trained model
        await gameState.model.save('localstorage://rpsModel');
        
        trainingStatusEl.textContent = 'AI has learned from recent games!';
    } catch (error) {
        console.error('Training error:', error);
        trainingStatusEl.textContent = 'Training failed - continuing with current knowledge';
    } finally {
        gameState.training = false;
    }
}

// Prepare training data from game history
function prepareTrainingData() {
    // We'll use sequences of 3 moves to predict the 4th
    const sequenceLength = 3;
    const inputs = [];
    const labels = [];
    
    for (let i = 0; i < gameState.history.length - sequenceLength; i++) {
        // Get sequence of moves
        const sequence = gameState.history.slice(i, i + sequenceLength);
        const nextMove = gameState.history[i + sequenceLength].human;
        
        // Encode input sequence
        const encodedInput = [];
        for (const move of sequence) {
            const encoded = [0, 0, 0];
            encoded[gameState.moveMap[move.human]] = 1;
            encodedInput.push(...encoded);
        }
        
        // Encode label (next move)
        const encodedLabel = [0, 0, 0];
        encodedLabel[gameState.moveMap[nextMove]] = 1;
        
        inputs.push(encodedInput);
        labels.push(encodedLabel);
    }
    
    // Convert to tensors
    const inputTensor = tf.tensor2d(inputs);
    const labelTensor = tf.tensor2d(labels);
    
    return {inputs: inputTensor, labels: labelTensor};
}

// Helper functions
function counterMove(move) {
    const counters = {
        rock: 'paper',
        paper: 'scissors',
        scissors: 'rock'
    };
    return counters[move];
}

function randomChoice() {
    return gameState.reverseMoveMap[Math.floor(Math.random() * 3)];
}

function determineWinner(human, ai) {
    if (human === ai) return 'tie';
    if (
        (human === 'rock' && ai === 'scissors') ||
        (human === 'paper' && ai === 'rock') ||
        (human === 'scissors' && ai === 'paper')
    ) {
        return 'human';
    }
    return 'ai';
}

// UI update functions
function displayResult(human, ai, outcome) {
    const emoji = {
        rock: '✊',
        paper: '✋',
        scissors: '✌️'
    };
    
    humanMoveEl.textContent = emoji[human];
    aiMoveEl.textContent = emoji[ai];
    
    let outcomeText = '';
    if (outcome === 'tie') {
        outcomeText = "It's a tie!";
    } else if (outcome === 'human') {
        outcomeText = "You win!";
    } else {
        outcomeText = "AI wins!";
    }
    
    outcomeEl.textContent = outcomeText;
}

function updateAIConfidence(prediction) {
    const rockPct = Math.round(prediction[0] * 100);
    const paperPct = Math.round(prediction[1] * 100);
    const scissorsPct = Math.round(prediction[2] * 100);
    
    rockProbEl.style.width = `${rockPct}%`;
    rockProbEl.textContent = `Rock: ${rockPct}%`;
    paperProbEl.style.width = `${paperPct}%`;
    paperProbEl.textContent = `Paper: ${paperPct}%`;
    scissorsProbEl.style.width = `${scissorsPct}%`;
    scissorsProbEl.textContent = `Scissors: ${scissorsPct}%`;
}

function updateUI() {
    gameCountEl.textContent = gameState.gamesPlayed;
    aiWinsEl.textContent = gameState.aiWins;
    humanWinsEl.textContent = gameState.humanWins;
    tiesEl.textContent = gameState.ties;
    
    const winRate = gameState.gamesPlayed > 0 
        ? Math.round((gameState.aiWins / gameState.gamesPlayed) * 100) 
        : 0;
    winRateEl.textContent = winRate;
    
    updateChart();
}

function updateChart() {
    if (gameState.history.length < 2) return;
    
    const labels = [];
    const aiWinRates = [];
    const humanWinRates = [];
    
    // Calculate running win rates
    let aiWins = 0;
    let humanWins = 0;
    
    for (let i = 0; i < gameState.history.length; i++) {
        const game = gameState.history[i];
        if (game.outcome === 'ai') aiWins++;
        if (game.outcome === 'human') humanWins++;
        
        if (i % Math.max(1, Math.floor(gameState.history.length / 20)) === 0 || i === gameState.history.length - 1) {
            labels.push(`Game ${i + 1}`);
            aiWinRates.push(Math.round((aiWins / (i + 1)) * 100));
            humanWinRates.push(Math.round((humanWins / (i + 1)) * 100));
        }
    }
    
    historyChart.data.labels = labels;
    historyChart.data.datasets[0].data = aiWinRates;
    historyChart.data.datasets[1].data = humanWinRates;
    historyChart.update();
}

// Save/load game state
function saveGameState() {
    const saveData = {
        gamesPlayed: gameState.gamesPlayed,
        aiWins: gameState.aiWins,
        humanWins: gameState.humanWins,
        ties: gameState.ties,
        history: gameState.history
    };
    
    localStorage.setItem('rpsGameState', JSON.stringify(saveData));
}

function loadGameState() {
    const savedData = localStorage.getItem('rpsGameState');
    if (savedData) {
        try {
            const parsed = JSON.parse(savedData);
            gameState.gamesPlayed = parsed.gamesPlayed || 0;
            gameState.aiWins = parsed.aiWins || 0;
            gameState.humanWins = parsed.humanWins || 0;
            gameState.ties = parsed.ties || 0;
            gameState.history = parsed.history || [];
            
            trainingStatusEl.textContent = 'Loaded previous game state';
        } catch (e) {
            console.log('Failed to load game state', e);
        }
    }
}

// Initialize the game when page loads
window.addEventListener('load', initGame);
