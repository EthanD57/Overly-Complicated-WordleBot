import pickle
import numpy as np
from numpy import ndarray

from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from Utilities.game_state import GameState


class RandomForestBot(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest", word_list=word_list)

        # Initialize sklearn RandomForestClassifier with defaults for now
        #IF the performance is bad, I'll adjust this
        self._model = RandomForestClassifier()


    def train(self) -> None:
        """
        Loads training data from the disk and trains the model off of it.
        """

        with open('ML/training_data/wordle_training.pkl', 'rb') as f:
            training_data = pickle.load(f)

        x = np.array([example[0] for example in training_data])  # Features
        y = np.array([example[1] for example in training_data])  # Labels

        y_binary = (y > 0.30).astype(int)

        self._model = MultiOutputClassifier(RandomForestClassifier())
        self._model.fit(x, y_binary)
        self.is_trained = True


    def make_guess(self) -> str:
        """Get candidates, score by probability, return best word"""

        letter_probs = self.predict(self.game_state)

        best_word = None
        best_score = -1

        if len(self.game_state.remaining_words) == 1: return self.game_state.remaining_words[0]

        for word in self.game_state.remaining_words:
            score = sum(letter_probs[ord(letter) - ord('a')] for letter in word)
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
