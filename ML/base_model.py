import joblib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from Utilities.game_state import GameState
from Utilities.shared_utils import extract_features


class BaseWordleModel(ABC):
    """
    Abstract base class for Wordle ML models.

    All models must implement:
    - train(): Learn from training data
    - predict(): Make predictions on new game states

    All models share:
    - engineer_features(): Convert game state → feature vector
    - save()/load(): Persistence with joblib
    """

    def __init__(self, model_name: str, word_list: list[str]) -> None:
        self.model_name = model_name
        self.is_trained = False
        # Actual model (RandomForest, Neural Net, etc.) goes here
        self._model = None
        self.game_state = GameState(word_list)


    @staticmethod
    def engineer_features(game_state: GameState) -> np.ndarray:
        """
        Convert a game state into a feature vector.

        This is the SAME for all models. If you change features,
        you must retrain all models.

        This method is ALWAYS called after filter_words() is called.

        Args:
            game_state (GameState): The current state of the game including remaining words,
                                    guess count, and the master word list.

        Returns:
            np.ndarray: Shape (313,) with engineered features

        Features include:
        - Letter frequencies in remaining words
        - Positional constraints (green/yellow/gray letters)
        - Remaining word count
        - Guess number
        """

        return extract_features(game_state)


    @abstractmethod
    def train(self) -> None:
        """
        Loads training data from the disk and trains the model off of it.
        """
        pass


    @abstractmethod
    def predict(self, game_state: GameState) -> np.ndarray:
        """
        Predict letter probabilities for a single game state.

        Args:
            game_state (GameState): The current state of the game

        Returns:
            np.ndarray: Shape (26,) with probabilities for letters A-Z
                        probs[0] = P(letter 'A' in next guess)
                        probs[25] = P(letter 'Z' in next guess)
        """
        pass


    def make_guess(self) -> str:
        """
        This method houses the main logic for the bot to make a guess.

        Returns:
            str: The guessed answer

        """
        letter_probs = self.predict(self.game_state)
        self.game_state.guess_count += 1

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


    def save(self, filepath: Path, keep_game_state: bool) -> None:
        """Save trained model to disk using joblib."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        if not keep_game_state: self.game_state.reset()
        with open(filepath, 'wb') as f:
            joblib.dump(self, f)


    @staticmethod
    def load(filepath: Path) -> 'BaseWordleModel':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            return joblib.load(f)




