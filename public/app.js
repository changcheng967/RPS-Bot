const { createApp, ref, computed, onMounted } = Vue;

createApp({
  setup() {
    const userId = ref(localStorage.getItem('rpsUserId') || '');
    const userStats = ref({ gamesPlayed: 0, wins: 0, losses: 0, ties: 0 });
    const globalStats = ref({ totalGames: 0, aiWins: 0, humanWins: 0, ties: 0, activeUsers: 0 });
    const gameHistory = ref([]);
    const aiPrediction = ref([0.33, 0.33, 0.33]);
    
    // Computed properties
    const winRate = computed(() => {
      return userStats.value.gamesPlayed > 0 
        ? Math.round((userStats.value.wins / userStats.value.gamesPlayed) * 100) 
        : 0;
    });
    
    const globalWinRate = computed(() => {
      return globalStats.value.totalGames > 0
        ? Math.round((globalStats.value.aiWins / globalStats.value.totalGames) * 100)
        : 0;
    });
    
    // Methods
    const createUser = async () => {
      const newUserId = 'user_' + Math.random().toString(36).substr(2, 9);
      const response = await fetch('/api/user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: newUserId })
      });
      
      const data = await response.json();
      userId.value = data.userId;
      userStats.value = data.stats;
      localStorage.setItem('rpsUserId', userId.value);
    };
    
    const playGame = async (humanMove) => {
      // Get AI move
      const moveResponse = await fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          userId: userId.value,
          humanMove,
          history: gameHistory.value.slice(-10) // Send last 10 moves
        })
      });
      
      const moveData = await moveResponse.json();
      aiPrediction.value = moveData.prediction;
      
      // Determine outcome
      const outcome = determineOutcome(humanMove, moveData.aiMove);
      
      // Save game
      await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: userId.value,
          humanMove,
          aiMove: moveData.aiMove,
          outcome,
          history: gameHistory.value
        })
      });
      
      // Update local state
      gameHistory.value.push({
        humanMove,
        aiMove: moveData.aiMove,
        outcome,
        timestamp: new Date().toISOString()
      });
      
      userStats.value.gamesPlayed++;
      if (outcome === 'human') userStats.value.wins++;
      if (outcome === 'ai') userStats.value.losses++;
      if (outcome === 'tie') userStats.value.ties++;
      
      // Refresh global stats
      loadGlobalStats();
    };
    
    const loadGlobalStats = async () => {
      const response = await fetch('/api/global');
      globalStats.value = await response.json();
    };
    
    const determineOutcome = (human, ai) => {
      if (human === ai) return 'tie';
      if (
        (human === 'rock' && ai === 'scissors') ||
        (human === 'paper' && ai === 'rock') ||
        (human === 'scissors' && ai === 'paper')
      ) return 'human';
      return 'ai';
    };
    
    // Lifecycle hooks
    onMounted(async () => {
      if (userId.value) {
        // Load user data
        const response = await fetch('/api/user', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId: userId.value })
        });
        
        const data = await response.json();
        userStats.value = data.stats;
        
        // Load global stats
        await loadGlobalStats();
      }
    });
    
    return {
      userId,
      userStats,
      globalStats,
      winRate,
      globalWinRate,
      aiPrediction,
      createUser,
      playGame
    };
  }
}).mount('#app');
