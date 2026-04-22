import numpy as np

from Utilities.game_state import GameState
from Utilities.shared_utils import (get_high_frequency_candidates,
                                    SACRIFICIAL_THRESHOLD, HIGH_FREQ_TOP_N, ANSWER_BONUS)


class EntropyBot:
    def __init__(self, word_list: list[str], pattern_table: np.ndarray) -> None:
        self.game_state = GameState(word_list)
        self.pattern_table = pattern_table

    def calculate_entropy(self, guess: str) -> float:
        """
        Compute the Shannon entropy of the pattern distribution for a given guess
        over the current remaining words.

        Args:
            guess: Word to evaluate.

        Returns:
            float: Expected bits of information gained by guessing this word.
        """
        guess_idx = self.game_state.word_to_index[guess]
        possible_patterns = self.pattern_table[guess_idx, self.game_state.remaining_words_indices]

        counts = np.bincount(possible_patterns)
        active_counts = counts[counts > 0]
        probabilities = active_counts / len(self.game_state.remaining_words_indices)

        return np.sum(-probabilities * np.log2(probabilities))

    def make_guess(self) -> str:
        """
        Choose the word that maximizes entropy over the remaining word list.

        When few words remain, the full master list is searched so the bot can
        make a sacrificial guess (e.g. "miles") that rules out several rhyming
        candidates at once rather than guessing blindly from the short list.

        Returns:
            str: The best guess.
        """
        self.game_state.guess_count += 1
        remaining = self.game_state.remaining_words

        if len(remaining) == 1:
            return remaining[0]

        if len(remaining) > SACRIFICIAL_THRESHOLD:
            candidates = get_high_frequency_candidates(self.game_state, HIGH_FREQ_TOP_N,
                                                       self.game_state.master_list)
        else:
            candidates = self.game_state.master_list

        best_word = ""
        max_entropy = -1

        for word in candidates:
            entropy = self.calculate_entropy(word)
            if word in remaining:
                entropy += ANSWER_BONUS
            if entropy > max_entropy:
                max_entropy = entropy
                best_word = word

        return best_word
