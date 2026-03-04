class GameState:
    def __init__(self, word_list: list[str]) -> None:
        self.remaining_words = word_list
        self.guess_count = 0
        self.master_list = word_list
        self.gray_letters = set()
        self.green_letters = {}
        self.yellow_letters = set()
        self.scored_rounds = dict()

    def reset(self) -> None:
        self.remaining_words = self.master_list
        self.guess_count = 0
        self.gray_letters = set()
        self.green_letters = {}
        self.yellow_letters = set()
        self.scored_rounds = dict
