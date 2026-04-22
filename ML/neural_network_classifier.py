import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ML.base_model import BaseWordleModel
from Utilities.game_state import GameState
from Utilities.shared_utils import FEATURE_SIZE

_TRAINING_EPOCHS = 1000
_LOG_INTERVAL = 100


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class NeuralNetworkClassifier(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="Neural Network Classifier", word_list=word_list)
        self._model = NeuralNetwork()
        self.model_path = Path('ML/saved_models/neural_network.pkl')

    def train(self):
        if self.model_path.exists():
            saved_bot = self.load(self.model_path)
            self._model = saved_bot._model
            self.is_trained = True
            return

        print("Training Model... this might take a bit")
        try:
            with open('ML/training_data/wordle_training.pkl', 'rb') as f:
                training_data = pickle.load(f)
        except FileNotFoundError:
            print("Error: Training data not found. Please generate training data first.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred while loading the training data: {e}")
            exit()

        x = np.array([example[0] for example in training_data])
        y = np.array([example[1] for example in training_data])
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

        for i in range(_TRAINING_EPOCHS):
            loss = self._train_epoch(optimizer, criterion, x_tensor, y_tensor)
            if i % _LOG_INTERVAL == 0:
                print(f"Loss: {loss} at epoch {i}")

        self.is_trained = True
        self._model.eval()
        self.save(self.model_path, False)

    def _train_epoch(self, optimizer, criterion, x, y):
        optimizer.zero_grad()
        loss = criterion(self._model(x), y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, game_state: GameState) -> np.ndarray:
        features = self.engineer_features(game_state).reshape(1, -1)
        x = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = self._model(x)
        return output.squeeze().numpy()
