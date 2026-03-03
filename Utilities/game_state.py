class GameState:
    def __init__(self, word_list: list[str]) -> None:
        self.remaining_words = word_list
        self.guess_count = 0
        self.master_list = word_list
        self.gray_letters = set()
        self.green_letters = {}
        self.yellow_letters = set()
        self.scored_rounds = dict()

    def update_constraints(self):
        self.gray_letters = set()
        self.green_letters = {}
        self.yellow_letters = set()

        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if all(letter not in word for word in self.remaining_words):
                self.gray_letters.add(letter)
            else:
                for pos in range(5):
                    if all(letter == word[pos] for word in self.remaining_words):
                        self.green_letters[pos] = letter
                    else:
                        self.yellow_letters.add(letter)
