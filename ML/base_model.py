import joblib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from Utilities.game_state import GameState
from Utilities.shared_utils import extract_features, SACRIFICIAL_THRESHOLD, ANSWER_BONUS


class BaseWordleModel(ABC):
    """
    Abstract base class for supervised Wordle ML models.

    Subclasses implement train() and predict(); this class provides the shared
    make_guess() loop, feature engineering, and save/load helpers.
    """

    def __init__(self, model_name: str, word_list: list[str]) -> None:
        self.model_name = model_name
        self.is_trained = False
        self._model = None
        self.game_state = GameState(word_list)

    @staticmethod
    def engineer_features(game_state: GameState) -> np.ndarray:
        """
        Convert a game state into a feature vector of shape (314,).

        Changing the feature layout requires retraining all models.
        This is always called after filter_words() updates the game state.
        """
        return extract_features(game_state)

    @abstractmethod
    def train(self) -> None:
        """Load training data from disk and fit the model."""
        pass

    @abstractmethod
    def predict(self, game_state: GameState) -> np.ndarray:
        """
        Predict letter affinity scores for the current game state.

        Returns:
            np.ndarray: Shape (26,) — score for each letter A-Z.
        """
        pass

    def make_guess(self) -> str:
        """
        Score every candidate word by summing predicted letter affinities,
        then return the highest-scoring word.

        When few words remain the candidate pool shrinks to remaining_words only,
        avoiding expensive full-vocabulary scoring and reducing trap risk.
        """
        letter_probs = self.predict(self.game_state)
        self.game_state.guess_count += 1

        if len(self.game_state.remaining_words) == 1:
            return self.game_state.remaining_words[0]

        if len(self.game_state.remaining_words) > SACRIFICIAL_THRESHOLD:
            candidate_pool = self.game_state.master_list
        else:
            candidate_pool = self.game_state.remaining_words

        best_word = None
        best_score = -1

        for word in candidate_pool:
            score = sum(letter_probs[ord(letter) - ord('a')] for letter in set(word))
            if word in self.game_state.remaining_words:
                score += ANSWER_BONUS
            if score > best_score:
                best_word = word
                best_score = score

        return best_word

    def save(self, filepath: Path, keep_game_state: bool) -> None:
        """Save trained model to disk using joblib."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        if not keep_game_state:
            self.game_state.reset()
        with open(filepath, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(filepath: Path) -> 'BaseWordleModel':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            return joblib.load(f)
