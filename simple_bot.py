import math
from collections import defaultdict
from utilities import score_guess

class WordleBot:
    def __init__(self, word_list: list[str]):
        self.master_list = word_list
        self.remaining_words = word_list
        self.seen_scores = {}


    def get_score(self, answer: str, guess: str) -> tuple:
        """Cached version of score_guess for speed."""
        pair = (answer, guess)
        if pair not in self.seen_scores:
            self.seen_scores[pair] = tuple(score_guess(answer, guess))
        return self.seen_scores[pair]


    def calculate_entropy(self, word:str) -> float:
        """ Calculates the entropy of a given word"""

        patterns = defaultdict(int)

        for possible_answer in self.remaining_words:
            res = self.get_score(possible_answer, word)
            patterns[res] += 1

        entropy = 0.0
        total_remaining_words = len(self.remaining_words)

        for count in patterns.values():
            p = count / total_remaining_words
            entropy -= p * math.log(p, 2)

        return entropy

    def make_guess(self, guess_count: int):
        """
        Make a First Guess Based Off of The Most Common
        Letters in The word_list

        Returns:
            str: The Bot's First Guess

        """
        if guess_count == 0: return "crane"
        num_words = len(self.remaining_words)

        if num_words == 1:  #No entropy calculations for just 1 word
            return self.remaining_words[0]

        best_word = ""
        max_entropy = -1

        if num_words > 20:
            guess_candidates = self.remaining_words
        else:
            guess_candidates = self.master_list

        for word in guess_candidates:
            entropy = self.calculate_entropy(word)

            if word in self.remaining_words:  #This acts as a tie breaker because we would PREFER to guess a word
                entropy += 0.01               #that could actually be the answer. So if both are high-entropy, pick one
                                              #that COULD actually be the answer.

            if entropy > max_entropy:
                max_entropy = entropy
                best_word = word

        return best_word

    def filter_words(self, guess: str, result: list[int]):

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
            else:  # Grey - letter is not in word, OR we've found all instances
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
        for word in self.remaining_words:
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

            # Check maximum letter counts (grey letters limit the count)
            for letter, max_count in letter_max_count.items():
                if word.count(letter) > max_count:
                    valid = False
                    break

            if valid:
                filtered_words.append(word)

        self.remaining_words = filtered_words
