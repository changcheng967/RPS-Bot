// Import WebAssembly module
let wasmExports;

async function initWasm() {
    const response = await fetch('rps_ai.wasm');
    const bytes = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(bytes);
    wasmExports = module.instance.exports;
}

// Initialize WASM
initWasm();

// Message handler
self.onmessage = function(e) {
    if (e.data.type === 'calculate') {
        const { userMove, history } = e.data;
        
        // Use WASM for heavy calculations
        const aiMove = wasmExports.calculateBestMove(userMove, history);
        
        // Send back to main thread
        self.postMessage({
            aiMove: aiMove,
            analysis: "WASM-powered prediction"
        });
    }
};
