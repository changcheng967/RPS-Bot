#include <emscripten.h>
#include <vector>
#include <map>

// Game state
std::vector<int> history;
std::map<std::pair<int, int>, int> transitions;

EMSCRIPTEN_KEEPALIVE
int calculateBestMove(int userMove, int* historyPtr, int length) {
    // Update history
    history.assign(historyPtr, historyPtr + length);
    
    // 1. Check for immediate spam (3+ repeats)
    if (history.size() >= 3) {
        int last = history.back();
        if (last == userMove && 
            history[history.size()-2] == last && 
            history[history.size()-3] == last) {
            return (userMove + 1) % 3; // Counter the spam
        }
    }
    
    // 2. Markov chain prediction (order 2)
    if (history.size() >= 2) {
        int last1 = history[history.size()-1];
        int last2 = history[history.size()-2];
        auto key = std::make_pair(last2, last1);
        
        transitions[key]++;
        
        // Find most likely next move
        int predicted = 0;
        int maxCount = 0;
        for (int i = 0; i < 3; i++) {
            if (transitions[std::make_pair(last1, i)] > maxCount) {
                maxCount = transitions[std::make_pair(last1, i)];
                predicted = i;
            }
        }
        
        if (maxCount > 3) { // Enough data
            return (predicted + 1) % 3; // Counter predicted move
        }
    }
    
    // 3. Fallback to frequency analysis
    int counts[3] = {0};
    for (int move : history) {
        counts[move]++;
    }
    
    // Counter most frequent move
    int mostFrequent = 0;
    if (counts[1] > counts[mostFrequent]) mostFrequent = 1;
    if (counts[2] > counts[mostFrequent]) mostFrequent = 2;
    
    return (mostFrequent + 1) % 3;
}
