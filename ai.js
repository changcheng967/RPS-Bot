// Game state
const state = {
    gamesPlayed: 0,
    aiWins: 0,
    humanWins: 0,
    ties: 0,
    humanHistory: [],
    movePatterns: {},
    lastHumanMove: null,
    moveCounts: { rock: 0, paper: 0, scissors: 0 }
};

// DOM elements
const resultEl = document.getElementById('result');
const gameCountEl = document.getElementById('gameCount');
const winRateEl = document.getElementById('winRate');
const rockProgress = document.getElementById('rockProgress');
const paperProgress = document.getElementById('paperProgress');
const scissorsProgress = document.getElementById('scissorsProgress');

// Initialize buttons
document.getElementById('rock').addEventListener('click', () => playGame('rock'));
document.getElementById('paper').addEventListener('click', () => playGame('paper'));
document.getElementById('scissors').addEventListener('click', () => playGame('scissors'));

// Main game function
function playGame(humanChoice) {
    // Record human move
    state.humanHistory.push(humanChoice);
    state.moveCounts[humanChoice]++;
    updateMoveStats();
    
    // AI makes its move (learning from patterns)
    const aiChoice = makeAIChoice(humanChoice);
    
    // Determine winner
    const outcome = determineWinner(humanChoice, aiChoice);
    
    // Update game state
    state.gamesPlayed++;
    if (outcome === 'ai') state.aiWins++;
    if (outcome === 'human') state.humanWins++;
    if (outcome === 'tie') state.ties++;
    
    // Update UI
    displayResult(humanChoice, aiChoice, outcome);
    updateStats();
    
    // Store the current move for pattern recognition
    state.lastHumanMove = humanChoice;
}

// AI decision making with learning
function makeAIChoice(currentHumanMove) {
    // First few moves are random while building history
    if (state.humanHistory.length < 5) {
        return randomChoice();
    }
    
    // Analyze patterns
    const predictedMove = predictHumanMove();
    
    // Choose counter to predicted move (with some randomness for variability)
    if (predictedMove && Math.random() > 0.2) { // 80% chance to use prediction
        return counterMove(predictedMove);
    }
    
    // Fallback to random or frequency-based choice
    return weightedRandomChoice();
}

// Predict human move based on patterns
function predictHumanMove() {
    // Check for immediate repeats
    if (state.humanHistory.length >= 2) {
        const lastMove = state.humanHistory[state.humanHistory.length - 1];
        const secondLastMove = state.humanHistory[state.humanHistory.length - 2];
        
        // If player repeated last move, they might do it again
        if (lastMove === secondLastMove) {
            return lastMove;
        }
    }
    
    // Check for common sequences (e.g., rock-paper-scissors cycle)
    if (state.humanHistory.length >= 3) {
        const sequence = state.humanHistory.slice(-3).join('-');
        if (state.movePatterns[sequence]) {
            return state.movePatterns[sequence];
        }
    }
    
    // Fallback to most frequent move
    return getMostFrequentMove();
}

// Get counter to a move
function counterMove(move) {
    const counters = {
        rock: 'paper',
        paper: 'scissors',
        scissors: 'rock'
    };
    return counters[move];
}

// Get random move
function randomChoice() {
    const choices = ['rock', 'paper', 'scissors'];
    return choices[Math.floor(Math.random() * 3)];
}

// Weighted random choice based on human's move frequencies
function weightedRandomChoice() {
    const total = state.moveCounts.rock + state.moveCounts.paper + state.moveCounts.scissors;
    const rockProb = state.moveCounts.rock / total;
    const paperProb = state.moveCounts.paper / total;
    const rand = Math.random();
    
    if (rand < rockProb) return counterMove('rock');
    if (rand < rockProb + paperProb) return counterMove('paper');
    return counterMove('scissors');
}

// Get human's most frequent move
function getMostFrequentMove() {
    const counts = state.moveCounts;
    if (counts.rock > counts.paper && counts.rock > counts.scissors) return 'rock';
    if (counts.paper > counts.rock && counts.paper > counts.scissors) return 'paper';
    if (counts.scissors > counts.rock && counts.scissors > counts.paper) return 'scissors';
    return randomChoice(); // if equal, return random
}

// Determine game outcome
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

// Update move statistics display
function updateMoveStats() {
    const total = state.moveCounts.rock + state.moveCounts.paper + state.moveCounts.scissors;
    const rockPct = Math.round((state.moveCounts.rock / total) * 100) || 0;
    const paperPct = Math.round((state.moveCounts.paper / total) * 100) || 0;
    const scissorsPct = Math.round((state.moveCounts.scissors / total) * 100) || 0;
    
    rockProgress.style.width = `${rockPct}%`;
    rockProgress.textContent = `Rock: ${rockPct}%`;
    paperProgress.style.width = `${paperPct}%`;
    paperProgress.textContent = `Paper: ${paperPct}%`;
    scissorsProgress.style.width = `${scissorsPct}%`;
    scissorsProgress.textContent = `Scissors: ${scissorsPct}%`;
}

// Update game statistics
function updateStats() {
    gameCountEl.textContent = state.gamesPlayed;
    const winRate = state.gamesPlayed > 0 
        ? Math.round((state.aiWins / state.gamesPlayed) * 100) 
        : 0;
    winRateEl.textContent = winRate;
}

// Display game result
function displayResult(human, ai, outcome) {
    const emoji = {
        rock: '✊',
        paper: '✋',
        scissors: '✌️'
    };
    
    let resultText = `You chose ${emoji[human]}, AI chose ${emoji[ai]}. `;
    
    if (outcome === 'tie') {
        resultText += "It's a tie!";
    } else if (outcome === 'human') {
        resultText += "You win!";
    } else {
        resultText += "AI wins!";
    }
    
    resultEl.textContent = resultText;
}

// Initialize pattern recognition
function initializePatterns() {
    // Simple pattern: if player did A then B, they might do C next
    const possibleMoves = ['rock', 'paper', 'scissors'];
    
    possibleMoves.forEach(a => {
        possibleMoves.forEach(b => {
            const sequence = `${a}-${b}`;
            // Default prediction is random until we see patterns
            state.movePatterns[sequence] = randomChoice();
        });
    });
}

// Start the game
initializePatterns();
