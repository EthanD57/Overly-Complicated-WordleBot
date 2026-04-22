import pickle
from pathlib import Path

import numpy as np
from numpy import ndarray

from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from Utilities.game_state import GameState

# Labels are continuous letter-frequency values; binarize above this threshold.
_LABEL_THRESHOLD = 0.35


class RandomForestClassifierModel(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest_classifier", word_list=word_list)
        self._model = RandomForestClassifier()
        self.model_path = Path('ML/saved_models/random_forest_classifier.pkl')

    def train(self) -> None:
        if self.model_path.exists():
            saved_bot = self.load(self.model_path)
            self._model = saved_bot._model
            self.is_trained = True
            return

        with open('ML/training_data/wordle_training.pkl', 'rb') as f:
            training_data = pickle.load(f)

        print("This bot isn't trained yet! Training...")
        x = np.array([example[0] for example in training_data])
        y = np.array([example[1] for example in training_data])

        y_binary = (y > _LABEL_THRESHOLD).astype(int)

        # n_jobs > 1 only during fit — reset to 1 before saving to avoid conflicts
        # when this model is later used inside a multiprocessing Pool.
        self._model = MultiOutputClassifier(RandomForestClassifier(), n_jobs=4)
        self._model.fit(x, y_binary)
        self._model.n_jobs = 1
        self.is_trained = True
        self.save(self.model_path, True)

    def predict(self, game_state: GameState) -> ndarray:
        features = self.engineer_features(game_state).reshape(1, -1)
        proba_list = self._model.predict_proba(features)

        letter_probs = np.zeros(26)
        for i, proba in enumerate(proba_list):
            if proba.shape[1] == 2:
                letter_probs[i] = proba[0, 1]
            else:
                # Only one class seen during training for this letter (effectively absent).
                letter_probs[i] = 0.0

        return letter_probs
