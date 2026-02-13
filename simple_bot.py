class WordleBot:
    def __init__(self, word_list: list[str]):
        self.remaining_words = word_list
        self.guesses = []
        self.word_list_letter_freq = {}

        for word in self.remaining_words: #Make a dictionary containing the frequency of all letters
            for character in word:
                if character in self.word_list_letter_freq:
                    self.word_list_letter_freq[character] += 1
                else:
                    self.word_list_letter_freq[character] = 1


    def make_first_guess(self):
        """
        Make a First Guess Based Off of The Most Common
        Letters in The word_list

        Returns:
            str: The Bot's First Guess

        """
        best_guess = ""
        best_guess_score = 0
        current_guess = ""
        current_guess_score = 0
        for word in self.remaining_words:
            used_chars = []
            current_guess = word
            current_guess_score = 0
            for character in word:
                current_guess_score += self.word_list_letter_freq[character] \
                    if character not in used_chars else 0

                used_chars.append(character)
            if current_guess_score > best_guess_score:
                best_guess = current_guess
                best_guess_score = current_guess_score
        return best_guess

    def filter_words(self, guess:str, result:list[int]):
        list_of_bad_letters = []
        list_of_good_letters = []
        for letter, score in zip(guess, result):
            if score == 2: #Mark safe letters so  duplicates are skipped like "Spoon"
                list_of_good_letters.append(letter)
            elif score == 0:
                list_of_bad_letters.append(letter)
            else:
                continue

        for word in self.remaining_words[:]:
            if any(letter in word for letter in list_of_bad_letters):
                if not any(letter in word for letter in list_of_good_letters):
                    self.remaining_words.remove(word)

        pass

    def make_next_guess(self):
        # Use self.remaining_words
        pass
