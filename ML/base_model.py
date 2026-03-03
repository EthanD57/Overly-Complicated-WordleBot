from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import numpy as np
import joblib
from pathlib import Path
from Utilities.game_state import GameState

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

        This method is ALWAYS called after the update_features() method in GameState

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
        pass

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

    def filter_words(self, guess: str, result: list[int]):
        """
        Filters the words based off the score response from the game.

        Uses a two-pass algorithm that first collects all the requirements from the score.

        Gray indicates a hard-limit of a given character in answer.
        Yellow indicates a hard-minimum of a given character in an answer.
        Green indicates the exact position that a letter must appear.

        The second pass filters the remaining word list to only contain words that match all
        the rules from the first pass.

        Args:
            guess (str): The guessed answer
            result (list[int]): The score response from the game

        Returns:
            None

        """
        # First pass: collect all requirements from the guess
        letter_min_count = defaultdict(int)  # Minimum times a letter must appear
        letter_max_count = {}  # Maximum times a letter can appear
        position_requirements = {}  # pos -> letter (green: must be at this position)
        position_exclusions = defaultdict(set)  # pos -> {letters} (yellow: can't be at this position)

        for pos, (letter, score) in enumerate(zip(guess, result)):
            if score == 2:  # Green - letter is in the correct position
                letter_min_count[letter] += 1
                position_requirements[pos] = letter
            elif score == 1:  # Yellow - letter is in word but wrong position
                letter_min_count[letter] += 1
                position_exclusions[pos].add(letter)
            else:  # Gray - letter is not in word, OR we've found all instances
                # Count how many times this letter appears as green/yellow in the entire guess
                green_yellow_count = sum(1 for l, s in zip(guess, result) if l == letter and s in [1, 2])
                if green_yellow_count > 0:
                    # Letter appears exactly this many times (no more)
                    letter_max_count[letter] = green_yellow_count
                else:
                    # Letter not in word at all
                    letter_max_count[letter] = 0

        # Second pass: filter words based on all requirements
        filtered_words = []
        for word in self.game_state.remaining_words:
            # Check position requirements (green letters must be in correct spots)
            if not all(word[pos] == letter for pos, letter in position_requirements.items()):
                continue

            # Check position exclusions (yellow letters can't be in certain positions)
            if any(word[pos] in excluded_letters for pos, excluded_letters in position_exclusions.items()):
                continue

            # Check minimum letter counts (green + yellow letters must appear at least this many times)
            valid = True
            for letter, min_count in letter_min_count.items():
                if word.count(letter) < min_count:
                    valid = False
                    break

            if not valid:
                continue

            # Check maximum letter counts (gray letters limit the count)
            for letter, max_count in letter_max_count.items():
                if word.count(letter) > max_count:
                    valid = False
                    break

            if valid:
                filtered_words.append(word)

        self.game_state.remaining_words = filtered_words

    def save(self, filepath: str) -> None:
        """Save trained model to disk using joblib."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'BaseWordleModel':
        """Load trained model from disk."""
        return joblib.load(filepath)

    def update_game_state(self) -> None:
        """Update game state constraints after filtering words."""
        self.game_state.update_constraints()

    def get_high_frequency_candidates(self, top_n=300) -> list:
        """
        Get words with the highest letter frequency in remaining words

        Args:
            top_n (int): The amount of words the function should return

        Returns:
            List: The list of words composed of the most common letters

        """
        # Count letter frequencies in remaining words
        letter_freq = Counter()
        for word in self.game_state.remaining_words:
            for letter in set(word):
                letter_freq[letter] += 1

        # Score each candidate word by how many high-frequency letters it has
        scored_candidates = []
        for word in self.game_state.remaining_words:
            score = sum(letter_freq[letter] for letter in set(word))
            scored_candidates.append((score, word))

        scored_candidates.sort(reverse=True)
        return [word for _, word in scored_candidates[:top_n]]
