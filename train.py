import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

# Configure GPU (CUDA)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to allow TensorFlow to allocate only necessary GPU memory
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

class RPSModelTrainer:
    def __init__(self):
        self.move_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.reverse_map = {v: k for k, v in self.move_map.items()}
        self.model = self._build_gpu_model()

    def _build_gpu_model(self):
        """Build LSTM model optimized for CUDA acceleration (via TensorFlow GPU)"""
        model = Sequential([
            LSTM(256, input_shape=(5, 3)),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _generate_synthetic_data(self, num_samples):
        moves = ['rock', 'paper', 'scissors']
        X, y = [], []
        
        for _ in range(num_samples):
            # Generate random sequences with some patterns
            if random.random() < 0.7:
                seq = [random.choice(moves) for _ in range(5)]
            else:
                # Add common human patterns
                seq = random.choice([
                    ['rock', 'rock', 'paper', 'scissors', 'rock'],
                    ['paper', 'paper', 'scissors', 'rock', 'paper'],
                    ['scissors', 'scissors', 'rock', 'paper', 'scissors']
                ])
            
            X.append([self._one_hot_encode(m) for m in seq])
            y.append(self._one_hot_encode(self._get_winning_move(seq[-1])))
        
        return np.array(X), np.array(y)

    def _one_hot_encode(self, move):
        return [1 if i == self.move_map[move] else 0 for i in range(3)]

    def _get_winning_move(self, move):
        if move == 'rock': return 'paper'
        if move == 'paper': return 'scissors'
        return 'rock'

    def train(self, epochs=100):
        X, y = self._generate_synthetic_data(50000)
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=1024,
            validation_split=0.2,
            callbacks=[TensorBoard(log_dir='./logs')]
        )
        self.model.save('rps_gpu_model.h5')
        print(f"Model saved with {'GPU' if gpus else 'CPU'} acceleration")

if __name__ == "__main__":
    trainer = RPSModelTrainer()
    trainer.train()
