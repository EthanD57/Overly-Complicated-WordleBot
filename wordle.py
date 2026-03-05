from pathlib import Path

class Wordle:
    def __init__(self, filepath: Path) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.word_list = list(line.strip() for line in f)
        except FileNotFoundError:
            print("Error: 'words.txt' not found. Please ensure the word list file is in the correct directory.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred while loading words.txt: {e}")
            exit()

        self.needRecompute = True

