#include <emscripten.h>
#include <vector>
#include <map>
#include <utility>

extern "C" {

EMSCRIPTEN_KEEPALIVE
int* createHistoryBuffer(int size) {
    return (int*)malloc(size * sizeof(int));
}

EMSCRIPTEN_KEEPALIVE
void freeHistoryBuffer(int* ptr) {
    free(ptr);
}

EMSCRIPTEN_KEEPALIVE
int calculateBestMove(int userMove, int* historyPtr, int length) {
    // Convert to vector for easier processing
    std::vector<int> history(historyPtr, historyPtr + length);
    static std::map<std::pair<int, int>, int> transitions;
    
    // 1. Anti-spam (2+ repeats)
    if (history.size() >= 2 && 
        history.back() == userMove && 
        history[history.size()-2] == userMove) {
        return (userMove + 1) % 3;
    }
    
    // 2. Markov chain (order 2)
    if (history.size() >= 2) {
        auto key = std::make_pair(history[history.size()-2], history.back());
        transitions[key]++;
        
        // Predict next move
        int predicted = 0;
        int maxCount = 0;
        for (int i = 0; i < 3; i++) {
            auto currentKey = std::make_pair(history.back(), i);
            if (transitions[currentKey] > maxCount) {
                maxCount = transitions[currentKey];
                predicted = i;
            }
        }
        
        if (maxCount > 2) return (predicted + 1) % 3;
    }
    
    // 3. Frequency analysis
    int counts[3] = {0};
    for (int move : history) counts[move]++;
    int mostFrequent = (counts[1] > counts[0]) ? 1 : 0;
    mostFrequent = (counts[2] > counts[mostFrequent]) ? 2 : mostFrequent;
    
    return (mostFrequent + 1) % 3;
}

}