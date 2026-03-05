import pickle
from pathlib import Path

import numpy as np
from numpy import ndarray

from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestRegressor
from Utilities.game_state import GameState


class RandomForestRegressorModel(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest_regressor", word_list=word_list)

        self.model_path = Path('ML/saved_models/random_forest_regressor.pkl')
        self._model = RandomForestRegressor(n_estimators=100)


    def train(self) -> None:
        """
        Loads training data from the disk and trains the model off of it.
        """

        if self.model_path.exists():
            saved_bot = self.load(self.model_path)
            self._model = saved_bot._model
            self.is_trained = True
            return

        print("Training model...")
        with open('ML/training_data/wordle_training.pkl', 'rb') as f:
            training_data = pickle.load(f)

        x = np.array([example[0] for example in training_data])  # Features
        y = np.array([example[1] for example in training_data])  # Labels

        #Use parallel jobs ONLY for fit(). Can't have anything over n=1 when using multipool/other parallelization
        self._model = RandomForestRegressor(n_estimators=100, n_jobs=4)
        self._model.fit(x, y)
        self._model.n_jobs = 1
        self.is_trained = True

        self.save(self.model_path, True)


    def make_guess(self) -> str:
        """Get candidates, score by probability, return best word"""
        letter_probs = self.predict(self.game_state)

        best_word = None
        best_score = -1
        remaining_count = len(self.game_state.remaining_words)

        if remaining_count == 1: return self.game_state.remaining_words[0]

        if remaining_count > 20:
            candidate_pool = self.game_state.master_list
        else:
            candidate_pool = self.game_state.remaining_words

        for word in candidate_pool:
            score = sum(letter_probs[ord(letter) - ord('a')] for letter in set(word))

            # Prefer words that could actually be the answer
            if word in self.game_state.remaining_words:
                score += 0.01

            if score > best_score:
                best_word = word
                best_score = score

        return best_word


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
        letter_probs = self._model.predict(features)[0]

        return letter_probs