from abc import ABC, abstractmethod
import numpy as np
import joblib

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
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: Shape (num_samples, 313) - feature matrices
            y: Shape (num_samples, 26) - soft labels for each letter
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


    @abstractmethod
    def make_guess(self) -> str:
        """
        This method houses the main logic for the bot to make a guess.

        Returns:
            str: The guessed answer

        """
    pass


    def save(self, filepath: str) -> None:
        """Save trained model to disk using joblib."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self, filepath)


    @staticmethod
    def load(filepath: str) -> 'BaseWordleModel':
        """Load trained model from disk."""
        return joblib.load(filepath)




