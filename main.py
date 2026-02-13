from random import choice

import wordle
import display

def print_menu():
    print("Welcome to The Overly-Complicated Wordle Bot!\n"
            "---------------------------------------------\n\n"
            "Please Choose an Option:\n"
            "1. User-Chosen Word\n"
            "2. Randomly-Chosen Word\n"
            "3. To Quit, Press 'q'\n")

def startup(game_instance: wordle.Wordle):
    """
    Initialize The Wordle Game According to User Input
    The User Has the Option to Choose a Word or Have one
    Randomly be Decided.

    TODO: They May Then Choose Which Bot to Use.

    Args:
        game_instance: Wordle instance

    Returns:
        None

    """
    print_menu()
    usr_input = input()

    while usr_input != "q":
        if usr_input == "1":
            usr_word = handle_user_word(game_instance.word_list)
            play_game(game_instance.word_list, usr_word)
        elif usr_input == "2":
            rnd_word = rand_word(game_instance.word_list)
            play_game(game_instance.word_list, rnd_word)
        else:
            print_menu()
            usr_input = input()

def handle_user_word(words: set[str]):
    """
    Takes In User Input for Word Selection and
    Ensures it is Within the Word List

    Args:
        words (set[str]): A set containing all valid words for the game

    Returns:
        str: The word chosen by the user

    """
    while True:
        word = str(input("Enter A 5-Letter Word, or Enter 'q' to Quit\n")).lower()
        if word == "q":
            exit()
        elif word not in words:
            print("Invalid Word. Please Enter A New Word, or Enter 'q' to Quit\n")
            continue
        else:
            return word

def rand_word(words: set[str]):
    """
    Returns a Random Word From The Word List

    Args:
        words (set[str]): A set containing all valid words for the game

    Returns:
        str: The randomly chosen word

    """
    return choice(tuple(words))

def score_guess(correct_word: str, guess:str) -> list[int]:
    """
    Scores the Guess For the Current Round.
    Uses a two-pass algorithm to properly handle duplicate letters:
    1. First pass: Mark exact matches (green/2)
    2. Second pass: Mark wrong positions (yellow/1) only if letters remain available

    This ensures that if a letter appears multiple times in the guess but fewer
    times in the answer, only the appropriate number of instances get marked as
    yellow/green (matching real Wordle behavior).

    Args:
        correct_word (str): The Correct Word for the Wordle Game
        guess (str): The Guess From the Bot

    Returns:
        list[int]: A list containing the Score of the Correct Word
                   2 = correct position (green)
                   1 = wrong position (yellow)
                   0 = not in word (gray)

    """
    result = [0] * len(guess)
    answer_chars = list(correct_word)

    # First pass: Mark exact matches and remove them from available pool
    for i, char in enumerate(guess):
        if char == correct_word[i]:
            result[i] = 2
            answer_chars[i] = None  # Mark as used

    # Second pass: Mark wrong positions for remaining letters
    for i, char in enumerate(guess):
        if result[i] == 0:  # Not already an exact match
            if char in answer_chars:
                result[i] = 1
                answer_chars[answer_chars.index(char)] = None  # Mark as used

    return result

def play_game(words:set[str], word="Apple"):
    """
    The Main Game Loop Logic.
    The Bot Plays the Game and the Results of Each
    Round is Shown in The Console For the
    User to Follow Along.

    Args:
        words (set[str]): A set containing all valid words for the game
        word (str): The Word Chosen For the Game

    Returns:
        None

    """
    display.print_game_start()
    guess_count = 0
    guesses = []
    while guess_count < 5:
        guess = str(input().lower())
        if guess == word:
            guess_count += 1
            guesses.append([guess, score_guess(word, guess)])
            display.print_game_state(guesses)
            display.print_end_screen(word, guess_count)
            break
        elif guess not in words:
            print("Not a valid Word")
        else:
            guesses.append([guess, score_guess(word, guess)])
            guess_count += 1
        display.print_game_state(guesses)
    return "Game Finished. Returning to Home Screen.\n"

if __name__ == '__main__':
    game = wordle.Wordle()
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    startup(game)


