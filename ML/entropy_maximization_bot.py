import math
from collections import defaultdict

from Utilities.game_state import GameState
from Utilities.shared_utils import score_guess, get_high_frequency_candidates


class EntropyBot:
    def __init__(self, word_list: list[str]):
        self.game_state = GameState(word_list)
        self.seen_scores = {}


    def _get_score(self, answer: str, guess: str) -> tuple:
        """
        Cached version of score_guess for speed

        Args:
            answer (str): The answer (or simulated answer) for the game
            guess (str): The guessed answer

        Result:
            tuple: The score result for the (answer,guess) pair
        """
        pair = (answer, guess)
        if pair not in self.seen_scores:
            self.seen_scores[pair] = tuple(score_guess(answer, guess))
        return self.seen_scores[pair]


    def calculate_entropy(self, word: str) -> float:
        """
        Calculates the entropy of a given word

        Args:
            word (str): The word to calculate the entropy for against all remaining words

        Returns:
            float: The entropy for the given word

        """
        patterns = defaultdict(int)

        for possible_answer in self.game_state.remaining_words:
            res = self._get_score(possible_answer, word)
            patterns[res] += 1

        entropy = 0.0
        total_remaining_words = len(self.game_state.remaining_words)

        for count in patterns.values():
            p = count / total_remaining_words
            entropy -= p * math.log(p, 2)

        return entropy

    def make_guess(self) -> str:
        """
        Make a Guess That Maximizes Entropy in Order to Shrink the Remaining Word List as Much as Possible
        First Guess is Always "Crane" Due to it Being Optimal

        Returns:
            str: The Bot's guess for that round

        """
        self.game_state.guess_count += 1 #Increment the GameState guess count

        if self.game_state.guess_count == 0: return "crane"
        num_words = len(self.game_state.remaining_words)

        if num_words == 1:  #No entropy calculations for just 1 word
            return self.game_state.remaining_words[0]

        best_word = ""
        max_entropy = -1

        if num_words > 20:
            candidates_to_check = min(300, len(self.game_state.remaining_words) * 2)
            guess_candidates = get_high_frequency_candidates(self.game_state, candidates_to_check)
        else:
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

