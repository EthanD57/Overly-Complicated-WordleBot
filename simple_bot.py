from operator import contains


class WordleBot:
    def __init__(self, word_list: list[str]):
        self.master_list = word_list
        self.remaining_words = word_list
        self.guesses = []
        self.word_list_letter_freq = {}

        for word in self.remaining_words:
            for char in word:
                self.word_list_letter_freq[char] = self.word_list_letter_freq.get(char, 0) + 1

    def make_first_guess(self):
        """
        Make a First Guess Based Off of The Most Common
        Letters in The word_list

        Returns:
            str: The Bot's First Guess

        """
        best_guess = ""
        best_guess_score = 0
        for word in self.remaining_words:
            used_chars = []
            current_guess = word
            current_guess_score = 0
            for character in word:
                current_guess_score += self.word_list_letter_freq[character] \
                    if character not in used_chars else self.word_list_letter_freq[character] * 0.5
                used_chars.append(character)
            if current_guess_score > best_guess_score:
                best_guess = current_guess
                best_guess_score = current_guess_score
        return best_guess

    def filter_words(self, guess:str, result:list[int]):
        position = [0,1,2,3,4]
        good_letters = set()
        for letter, score, pos in zip(guess, result, position):
            if score == 2:     #Green character
                good_letters.add(letter)
                for word in self.remaining_words[:]:
                    if word[pos] != letter:     #MUST be in the same position to remain in the list.
                        self.remaining_words.remove(word)
            elif score == 1: #Yellow character
                good_letters.add(letter)
                for word in self.remaining_words[:]:
                    if word[pos] == letter or letter not in word:     #Character can NOT be in the same position to remain in the list.
                        self.remaining_words.remove(word)
            else:  #TODO: Fix this because it ignores grey letter positions and change how letters are checked. If a grey letter is checked before it appears as green in the same word, it is removed (GREY -> ERASE <- GREEN)
                if letter in good_letters: continue
                for word in self.remaining_words[:]:
                    if letter in word:
                        self.remaining_words.remove(word)

            for word in self.remaining_words:
                for char in word:
                    self.word_list_letter_freq[char] = self.word_list_letter_freq.get(char, 0) + 1


    # def make_next_guess(self):
    #     best_guess = ""
    #     best_guess_score = 0
    #     for word in self.remaining_words:
    #         used_chars = []
    #         current_guess = word
    #         current_guess_score = 0
    #         for character in word:
    #             current_guess_score += self.word_list_letter_freq[character] \
    #                 if character not in used_chars else 0
    #
    #             used_chars.append(character)
    #         if current_guess_score > best_guess_score:
    #             best_guess = current_guess
    #             best_guess_score = current_guess_score
    #     return best_guess
