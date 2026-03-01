import numpy as np
from collections import defaultdict, Counter
from Bots import simple_bot
from Utilities import score_guess
import pickle


class TrainingDataCollector:
    def __init__(self, word_list: list[str]):
        self.word_list = word_list
        self.training_data = []

    def extract_features(self, bot: simple_bot.WordleBot, guess_count: int) -> np.array:
        """
        Extract features from current game state.

        Returns:
            np.array: Feature vector representing the current state
        """
        features = []

        features[0]= self.get_high_frequency_candidates()

        # TODO: Add green letter features (5 × 26 one-hot)
        for guess,score in zip(bot.scored_rounds.keys(), bot.scored_rounds.values()):

        # TODO: Add yellow letter features (5 × 26 binary)

        # TODO: Add gray letter features (26 binary)

        # TODO: Add remaining word count (normalized)

        # TODO: Add guess number

        return np.array(features)

    def get_high_frequency_candidates(self) -> list:
        """
        Get words with the highest letter frequency in remaining words

        Returns:
            List: The list of words composed of the most common letters

        """
        # Count letter frequencies in remaining words
        letter_freq = Counter()
        for word in self.word_list:
            for letter in set(word):
                letter_freq[letter] += 1

        # Score each candidate word by how many high-frequency letters it has
        scored_candidates = []
        for word in self.word_list:
            score = sum(letter_freq[letter] for letter in set(word)) / len(self.word_list)
            scored_candidates.append((score, word))

        scored_candidates.sort(reverse=True)
        return scored_candidates