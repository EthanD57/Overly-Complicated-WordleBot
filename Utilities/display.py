from colorama import Fore, Back, Style, init

# autoreset prevents color bleed across print calls on Windows
init(autoreset=True)


def print_menu(model: int, model_options: list[str]):
    print("\nMain Menu\n"
          "---------------------------------------------\n"
          f"Current Model: {model_options[model-1]}\n\n"
          "Please Choose an Option:\n"
          "1. User-Chosen Word\n"
          "2. Randomly-Chosen Word\n"
          "3. Test The Bot\n"
          "4. Generate Training Data\n"
          "5. Pick Which Model to Use\n"
          "To Quit, Enter 'q'\n")


def print_wordle_result(word: str, result: list[int]):
    """
    Print a single Wordle guess with colored tiles.

    Args:
        word: The guessed word.
        result: Per-letter scores — 2 green, 1 yellow, 0 gray.
    """
    output = ""
    for letter, status in zip(word, result):
        if status == 2:
            output += Back.GREEN + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        elif status == 1:
            output += Back.YELLOW + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        else:
            output += Back.WHITE + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        output += " "
    print(output)


def print_game_state(guesses: list[tuple[str, list[int]]]):
    print("\nWordle Game State:")
    print("-" * 30)
    for word, result in guesses:
        print_wordle_result(word, result)
    print("-" * 30)


def print_game_start():
    print("\n\nThe Bot Will Make Guesses to Find the Correct Word.")
    print("\nThe Game Ends With 6 Unsuccessful Guesses or a Correct Guess.")
    print("\nWordle Game Start:")
    print("-" * 30)


def print_end_screen(correct_word: str, guess_count: int):
    print(f"You guessed the word, {correct_word}, in {guess_count} guesses!")
