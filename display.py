from colorama import Fore, Back, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

def print_wordle_result(word: str, result: list[int]):
    """
    Print a Wordle guess with colored output.

    Args:
        word: The guessed word (5 letters)
        result: List of integers where:
                0 = wrong letter (gray)
                1 = correct letter, wrong position (yellow)
                2 = correct letter, correct position (green)
    """
    output = ""
    for letter, status in zip(word, result):
        if status == 2:  # Correct position
            output += Back.GREEN + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        elif status == 1:  # Wrong position
            output += Back.YELLOW + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        else:  # Wrong letter
            output += Back.WHITE + Fore.BLACK + f" {letter.upper()} " + Style.RESET_ALL
        output += " "  # Space between letters
    print(output)


def print_game_state(guesses: list[tuple[str, list[int]]]):
    """
    Print multiple guesses with their results.

    Args:
        guesses: List of (word, result) tuples
    """
    print("\nWordle Game State:")
    print("-" * 30)
    for word, result in guesses:
        print_wordle_result(word, result)
    print("-" * 30)

def print_game_start():
    print("\nTo Play, Enter a 5-Letter Word to Make a Guess.")
    print("\nThe Game Ends With 6 Unsuccessful Guesses or a Correct Guess.")
    print("\nWordle Game Start:")
    print("-" * 30)


def print_end_screen(correct_word:str, guess_count:int):
    print(f"You guessed the word, {correct_word}, in {guess_count} guesses!")