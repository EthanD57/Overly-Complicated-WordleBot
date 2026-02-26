import math
from collections import defaultdict, Counter
from utilities import score_guess


class WordleBot:
    def __init__(self, word_list: list[str]):
        self.master_list = word_list
        self.remaining_words = word_list
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

    def _get_high_frequency_candidates(self, top_n=300) -> list:
        """
        Get words with the highest letter frequency in remaining words

        Args:
            top_n (int): The amount of words the function should return

        Returns:
            List: The list of words composed of the most common letters

        """
        # Count letter frequencies in remaining words
        letter_freq = Counter()
        for word in self.remaining_words:
            for letter in set(word):
                letter_freq[letter] += 1

        # Score each candidate word by how many high-frequency letters it has
        scored_candidates = []
        for word in self.master_list:
            score = sum(letter_freq[letter] for letter in set(word))
            scored_candidates.append((score, word))

        scored_candidates.sort(reverse=True)
        return [word for _, word in scored_candidates[:top_n]]

    def _calculate_entropy(self, word: str) -> float:
        """
        Calculates the entropy of a given word

        Args:
            word (str): The word to calculate the entropy for against all remaining words

        Returns:
            float: The entropy for the given word

        """
        patterns = defaultdict(int)

        for possible_answer in self.remaining_words:
            res = self._get_score(possible_answer, word)
            patterns[res] += 1

        entropy = 0.0
        total_remaining_words = len(self.remaining_words)

        for count in patterns.values():
            p = count / total_remaining_words
            entropy -= p * math.log(p, 2)

        return entropy

    def make_guess(self, guess_count: int) -> str:
        """
        Make a First Guess Based Off of The Most Common
        Letters in The word_list

        Args:
            guess_count (int): Tells the bot how many guesses it has already made

        Returns:
            str: The Bot's guess for that round

        """
        if guess_count == 0: return "crane"
        num_words = len(self.remaining_words)

        if num_words == 1:  #No entropy calculations for just 1 word
            return self.remaining_words[0]

        best_word = ""
        max_entropy = -1

        if num_words > 20:
            guess_candidates = self._get_high_frequency_candidates(top_n=500)
        else:
            guess_candidates = self.master_list

        for word in guess_candidates:
            entropy = self._calculate_entropy(word)

            if word in self.remaining_words:  #This acts as a tiebreaker because we would PREFER to guess a word
                entropy += 0.01  #that could actually be the answer. So if both are high-entropy, pick one
                #that COULD actually be the answer.

            if entropy > max_entropy:
                max_entropy = entropy
                best_word = word

        return best_word

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

            # Check maximum letter counts (gray letters limit the count)
            for letter, max_count in letter_max_count.items():
                if word.count(letter) > max_count:
                    valid = False
                    break

            if valid:
                filtered_words.append(word)

        self.remaining_words = filtered_words
