class Wordle:
    def __init__(self) -> None:
        try:
            with open("words.txt", "r", encoding="utf-8") as f:
                self.word_list = set(line.strip() for line in f)
        except FileNotFoundError:
            print("Error: 'words.txt' not found. Please ensure the word list file is in the correct directory.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred while loading words.txt: {e}")
            exit()

