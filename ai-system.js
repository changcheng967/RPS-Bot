// Configuration
const MODEL_CONFIG = {
    layers: [
        {units: 16, activation: 'relu', inputShape: [9]},
        {units: 8, activation: 'relu'},
        {units: 3, activation: 'softmax'}
    ],
    optimizer: 'adam',
    loss: 'categoricalCrossentropy'
};

class AISystem {
    constructor() {
        this.model = null;
        this.trainingData = {
            global: {},
            user: []
        };
        this.userHistory = [];
        this.gameStats = {
            totalGames: 0,
            aiWins: 0,
            userWins: 0,
            ties: 0
        };
        this.init();
    }

    async init() {
        await this.loadModel();
        await this.loadTrainingData();
        this.setupEventListeners();
    }

    async loadModel() {
        this.model = tf.sequential();
        MODEL_CONFIG.layers.forEach(layer => {
            this.model.add(tf.layers.dense(layer));
        });
        this.model.compile({
            optimizer: MODEL_CONFIG.optimizer,
            loss: MODEL_CONFIG.loss
        });
    }

    async loadTrainingData() {
        try {
            const response = await fetch('https://raw.githubusercontent.com/yourusername/yourrepo/main/training-data.json');
            this.trainingData.global = await response.json();
            this.updateStatsDisplay();
        } catch (error) {
            console.error("Error loading training data:", error);
            this.trainingData.global = { sequences: [] };
        }
    }

    predictNextMove() {
        if (this.userHistory.length < 3) {
            return Math.floor(Math.random() * 3);
        }

        const moveMap = {rock: 0, paper: 1, scissors: 2};
        const input = this.userHistory.slice(-3).flatMap(move => {
            const encoded = [0, 0, 0];
            encoded[moveMap[move]] = 1;
            return encoded;
        });

        const prediction = this.model.predict(tf.tensor2d([input]));
        const result = Array.from(prediction.dataSync());
        tf.dispose(prediction);
        return result;
    }

    determineOutcome(userMove, aiMove) {
        if (userMove === aiMove) return 'tie';
        const winConditions = {rock: 'scissors', paper: 'rock', scissors: 'paper'};
        return winConditions[userMove] === aiMove ? 'user' : 'ai';
    }

    async play(userMove) {
        // Get AI move
        const prediction = this.predictNextMove();
        const moves = ['rock', 'paper', 'scissors'];
        const aiMoveIndex = prediction.indexOf(Math.max(...prediction));
        const aiMove = moves[aiMoveIndex];

        // Determine outcome
        const outcome = this.determineOutcome(userMove, aiMove);

        // Update game stats
        this.updateGameStats(outcome);

        // Update history
        this.userHistory.push(userMove);
        if (this.userHistory.length > 50) this.userHistory.shift();

        // Update display
        this.updateMoveDisplay(userMove, aiMove, outcome);
        this.updateConfidenceDisplay(prediction);

        // Retrain periodically
        if (this.userHistory.length % 10 === 0) {
            await this.retrainModel();
        }

        // Update stats display
        this.updateStatsDisplay();
    }

    updateGameStats(outcome) {
        this.gameStats.totalGames++;
        if (outcome === 'ai') this.gameStats.aiWins++;
        if (outcome === 'user') this.gameStats.userWins++;
        if (outcome === 'tie') this.gameStats.ties++;
    }

    updateMoveDisplay(userMove, aiMove, outcome) {
        document.getElementById('user-move').textContent = `You: ${userMove}`;
        document.getElementById('ai-move').textContent = `AI: ${aiMove}`;
        
        const outcomeElement = document.getElementById('outcome');
        outcomeElement.textContent = 
            outcome === 'tie' ? "It's a tie!" :
            outcome === 'user' ? "You win!" : "AI wins!";
        
        outcomeElement.className = 'outcome ' + outcome;
    }

    updateConfidenceDisplay(prediction) {
        const moves = ['rock', 'paper', 'scissors'];
        moves.forEach((move, index) => {
            const confidence = Math.round(prediction[index] * 100);
            const bar = document.getElementById(`${move}-confidence`);
            const percent = document.getElementById(`${move}-percent`);
            
            bar.style.width = `${confidence}%`;
            percent.textContent = `${confidence}%`;
        });
    }

    updateStatsDisplay() {
        // Update global stats
        document.getElementById('global-cycles').textContent = 
            this.trainingData.global.trainingCycles || 0;
        
        // Update win rate
        const winRate = this.gameStats.totalGames > 0 ?
            Math.round((this.gameStats.aiWins / this.gameStats.totalGames) * 100) : 0;
        
        document.getElementById('ai-winrate').textContent = `${winRate}%`;
        document.getElementById('ai-winrate-bar').style.width = `${winRate}%`;
    }

    async retrainModel() {
        if (this.userHistory.length < 10) return;
        
        // Prepare training data
        const moveMap = {rock: 0, paper: 1, scissors: 2};
        const {inputs, labels} = this.userHistory.reduce((acc, _, i, arr) => {
            if (i < arr.length - 3) {
                const seq = arr.slice(i, i + 3);
                const next = arr[i + 3];
                
                // Encode sequence
                const input = seq.flatMap(move => {
                    const encoded = [0, 0, 0];
                    encoded[moveMap[move]] = 1;
                    return encoded;
                });
                
                // Encode label
                const label = [0, 0, 0];
                label[moveMap[next]] = 1;
                
                acc.inputs.push(input);
                acc.labels.push(label);
            }
            return acc;
        }, {inputs: [], labels: []});

        // Train model
        await this.model.fit(
            tf.tensor2d(inputs),
            tf.tensor2d(labels),
            {
                epochs: 10,
                batchSize: 8,
                verbose: 0
            }
        );

        // Update training cycles
        if (!this.trainingData.global.trainingCycles) {
            this.trainingData.global.trainingCycles = 0;
        }
        this.trainingData.global.trainingCycles++;
        
        // Update display
        this.updateStatsDisplay();
    }

    setupEventListeners() {
        document.querySelectorAll('.choice-btn').forEach(button => {
            button.addEventListener('click', () => {
                this.play(button.dataset.move);
            });
        });
    }
}

// Initialize the AI system
const aiSystem = new AISystem();