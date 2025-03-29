import tensorflow as tf
import numpy as np
import os

class RPSModelTrainer:
    def __init__(self):
        self.move_map = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.reverse_map = {v: k for k, v in self.move_map.items()}
        self.model = self.build_model()

    # Build the LSTM model
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(256, input_shape=(5, 3)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # Generate synthetic data for training
    def generate_synthetic_data(self, num_samples):
        moves = ['rock', 'paper', 'scissors']
        X = []
        y = []

        for _ in range(num_samples):
            seq = self.get_human_patterns() if np.random.rand() < 0.7 else [moves[np.random.randint(3)] for _ in range(5)]
            X.append([self.one_hot_encode(m) for m in seq])
            y.append(self.one_hot_encode(self.get_winning_move(seq[-1])))

        return np.array(X), np.array(y)

    # One-hot encode a move
    def one_hot_encode(self, move):
        encoding = [0, 0, 0]
        encoding[self.move_map[move]] = 1
        return encoding

    # Get the winning move
    def get_winning_move(self, move):
        if move == 'rock':
            return 'paper'
        elif move == 'paper':
            return 'scissors'
        return 'rock'

    # Get predefined human patterns
    def get_human_patterns(self):
        patterns = [
            ['rock', 'rock', 'paper', 'scissors', 'rock'],
            ['paper', 'paper', 'scissors', 'rock', 'paper'],
            ['scissors', 'scissors', 'rock', 'paper', 'scissors']
        ]
        return patterns[np.random.randint(len(patterns))]

    # Train the model
    def train(self, epochs=100):
        X, y = self.generate_synthetic_data(50000)

        # Split data into training and validation sets
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=1024, validation_data=(X_val, y_val))

        print('Training complete')
        print('Final loss:', history.history['loss'][-1])
        print('Final accuracy:', history.history['accuracy'][-1])

        # Save the model
        self.model.save('rps_model')
        print('Model saved')

if __name__ == "__main__":
    trainer = RPSModelTrainer()
    trainer.train()
