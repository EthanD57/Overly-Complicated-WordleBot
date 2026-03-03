import numpy as np
from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestClassifier
from Utilities.game_state import GameState


class RandomForestBot(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest", word_list=word_list)

        # Initialize sklearn RandomForestClassifier with defaults for now
        #IF the performance is bad, I'll adjust this
        self._model = RandomForestClassifier()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: Shape (num_samples, 313) - feature matrices
            y: Shape (num_samples, 26) - soft labels for each letter
        """
        pass

    def make_guess(self) -> str:
        """Get candidates, score by probability, return best word"""
        pass

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