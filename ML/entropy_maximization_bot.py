from pathlib import Path

import numpy as np
import pickle as pkl

from Utilities.game_state import GameState
from Utilities.shared_utils import get_high_frequency_candidates

class EntropyBot:
    def __init__(self, word_list: list[str], pattern_table: np.ndarray) -> None:
        self.game_state = GameState(word_list)
        self.pattern_table = pattern_table


    def calculate_entropy(self, guess: str) -> float:
        """
        Incredibly fast entropy calculation using NumPy broadcasting and bincount.

        Args:
            guess: The string to calculate entropy for

        """
        guess_idx = self.game_state.word_to_index[guess]

        # Slice out the scores for this guess against ONLY the remaining words
        possible_patterns = self.pattern_table[guess_idx, self.game_state.remaining_words_indices]

        counts = np.bincount(possible_patterns)
        # Filter out patterns that didn't happen (where count is 0)
        active_counts = counts[counts > 0]
        probabilities = active_counts / len(self.game_state.remaining_words_indices)

        # Vectorized entropy math: -sum(P * log2(P))
        return -1.0 * np.sum(probabilities * np.log2(probabilities))

    def make_guess(self) -> str:
        """
        Make a Guess That Maximizes Entropy in Order to Shrink the Remaining Word List as Much as Possible
        First Guess is Always "Crane" Due to it Being Optimal

        Returns:
            str: The Bot's guess for that round

        """

        # if self.game_state.guess_count == 0:
        #     self.game_state.guess_count += 1  # Increment the GameState guess count
        #     return "crane"
        #
        self.game_state.guess_count += 1  # Increment the GameState guess count
        remaining_words_length = len(self.game_state.remaining_words)

        if remaining_words_length == 1:  #No entropy calculations for just 1 word
            return self.game_state.remaining_words[0]

        best_word = ""
        max_entropy = -1

        #If we have a lot of possible words, we just check the top 300 words with high-frequency letters
        if remaining_words_length > 20:
            guess_candidates = get_high_frequency_candidates(self.game_state, 300, self.game_state.master_list)
        else:
            #Below 20 remaining words is dangerous because we can get stuck in traps like LIGHT, MIGHT, SIGHT, etc.
            #To combat this, we allow the bot to make a sacrificial guess like "MILES" to rule out LIGHT, MIGHT, and SIGHT
            guess_candidates = self.game_state.master_list

        for word in guess_candidates:
            entropy = self.calculate_entropy(word)

            if word in self.game_state.remaining_words:  #This acts as a tiebreaker because we would PREFER to guess a word
                entropy += 0.01  #that could actually be the answer. So if both are high-entropy, pick one
                #that COULD actually be the answer.

            if entropy > max_entropy:
                max_entropy = entropy
                best_word = word

        return best_word
