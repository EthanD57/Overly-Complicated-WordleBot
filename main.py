from random import choice

import wordle
import display
import simple_bot
from utilities import score_guess

TESTING_MODE = False

def _startup(game_instance: wordle.Wordle):
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
    usr_input = ""
    while usr_input != "q":
        display.print_menu()
        usr_input = input()
        if usr_input == "1":
            usr_word = _handle_user_word(game_instance.word_list)
            print(_play_game(game_instance.word_list, usr_word))
        elif usr_input == "2":
            rnd_word = _rand_word(game_instance.word_list)
            print(_play_game(game_instance.word_list, rnd_word))
        elif usr_input == "3":
            global TESTING_MODE
            TESTING_MODE = True
            testing_range = int(input("How many games should be ran to test the bot?\n"))
            _test_bot(game_instance.word_list, testing_range, game_instance)
        elif usr_input == 'q': exit()
        else:
            display.print_menu()
            usr_input = input()


def _handle_user_word(words: set[str]):
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


def _rand_word(words: set[str]):
    """
    Returns a Random Word From The Word List

    Args:
        words (set[str]): A set containing all valid words for the game

    Returns:
        str: The randomly chosen word

    """
    word = choice(tuple(words))
    if not TESTING_MODE: print(f"Random Word Chosen is {word}")
    return word


def _play_game(words: set[str], word=""):
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
    bot = simple_bot.WordleBot(list(words))
    guess_count = 0
    guesses = []
    while guess_count < 6:
        guess = bot.make_guess(guess_count)
        if guess == word:  ##Correct Word Guessed
            guess_count += 1
            guesses.append([guess, score_guess(word, guess)])
            display.print_game_state(guesses)
            display.print_end_screen(word, guess_count)
            return ""
        else:  ##Incorrect Word Guessed. Update Game State and Send Score
            score = score_guess(word, guess)
            guesses.append([guess, score])
            bot.filter_words(guess, score)  ##Give the Bot Its Score for the Round
            guess_count += 1
        display.print_game_state(guesses)
    return "Word Not Guessed :("


def _test_bot(words: set[str], testing_runs: int, game_instance: wordle.Wordle):
    """
    The Main Game Loop Logic.
    The Bot Plays the Game and the Results of Each
    Round is Logged For a Final Score Breakdown.

    Args:
        words (set[str]): A set containing all valid words for the game
        testing_runs (int): The number of testing runs the bot should do
        game_instance (wordle.Wordle): The game instance used in testing

    Returns:
        None

    """
    correct_games = 0
    incorrect_games = 0
    guess_counts = []
    for x in range(testing_runs):
        word = _rand_word(game_instance.word_list)
        bot = simple_bot.WordleBot(list(words))
        guess_count = 0
        guesses = []
        solved = False
        while guess_count < 6:
            guess = bot.make_guess(guess_count)
            if guess == word:  ##Correct Word Guessed
                guess_count += 1
                guesses.append([guess, score_guess(word, guess)])
                solved = True
            else:  ##Incorrect Word Guessed. Update Game State and Send Score
                score = score_guess(word, guess)
                guesses.append([guess, score])
                bot.filter_words(guess, score)  ##Give the Bot Its Score for the Round
                guess_count += 1
        if solved: correct_games += 1
        else: incorrect_games += 1
        guess_counts.append(guess_count)


    print(f"\n\nCorrect Games Percentage: {(correct_games / testing_runs) * 100}%")
    print(f"Incorrect Games Percentage: {(incorrect_games / testing_runs) * 100}%")
    print("Average Number of Guesses: ", round(sum(guess_counts) / len(guess_counts), 2))


if __name__ == '__main__':
    game = wordle.Wordle()
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    _startup(game)
