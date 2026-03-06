import pickle
from pathlib import Path

import numpy as np
from numpy import ndarray

from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from Utilities.game_state import GameState


class RandomForestClassifierModel(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest_classifier", word_list=word_list)

        # Initialize sklearn RandomForestClassifier with defaults
        self._model = RandomForestClassifier()
        self.model_path = Path('ML/saved_models/random_forest_classifier.pkl')

    def train(self) -> None:
        """
        Loads training data from the disk and trains the model off of it.
        """

        if self.model_path.exists():
            saved_bot = self.load(self.model_path)
            self._model = saved_bot._model
            self.is_trained = True
            return

        with open('ML/training_data/wordle_training.pkl', 'rb') as f:
            training_data = pickle.load(f)

        print("This bot isn't trained yet! Training...")
        x = np.array([example[0] for example in training_data])  # Features
        y = np.array([example[1] for example in training_data])  # Labels

        y_binary = (y > 0.35).astype(int)

        #Use parallel jobs ONLY for fit(). Can't have anything over n=1 when using multipool/other parallelization
        self._model = MultiOutputClassifier(RandomForestClassifier(), n_jobs=4)
        self._model.fit(x, y_binary)
        self._model.n_jobs = 1
        self.is_trained = True

        self.save(self.model_path, True)


    def predict(self, game_state: GameState) -> ndarray:
        """
        Predict letter probabilities for a single game state.

        Args:
            game_state (GameState): The current state of the game

        Returns:
            np.ndarray: Shape (26,) with probabilities for letters A-Z
                        probs[0] = P(letter 'A' in next guess)
                        probs[25] = P(letter 'Z' in next guess)
        """

        features = self.engineer_features(game_state).reshape(1, -1)
        proba_list = self._model.predict_proba(features)

        letter_probs = np.zeros(26)
        for i, proba in enumerate(proba_list):
            if proba.shape[1] == 2:
                letter_probs[i] = proba[0, 1]
            else:
                # Classifier only saw one class during training
                # Q seemingly just sucks at being high entropy, so this catches it
                letter_probs[i] = 0.0

        return letter_probs