from random import choice

import wordle
import display
import simple_bot
from utilities import score_guess
from multiprocessing import Pool

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
            usr_word = _handle_user_word(game_instance)
            print(_play_game(game_instance.word_list, usr_word))
        elif usr_input == "2":
            rnd_word = _rand_word(game_instance.word_list)
            print(_play_game(game_instance.word_list, rnd_word))
        elif usr_input == "3":
            global TESTING_MODE
            TESTING_MODE = True
            testing_range = int(input("How many games should be ran to test the bot?\n"))
            _test_bot_parallel(game_instance.word_list, testing_range)
            exit()
        elif usr_input == 'q': exit()
        else:
            display.print_menu()
            usr_input = input()


def _handle_user_word(instance: wordle.Wordle):
    """
    Takes In User Input for Word Selection and
    Ensures it is 5 Characters Long

    Args:
        instance (wordle.Wordle): The game instance to pull the word list from

    Returns:
        str: The word chosen by the user

    """
    while True:
        word = str(input("Enter A 5-Letter Word, or Enter 'q' to Quit\n")).lower().strip()
        if word == "q":
            exit()
        elif len(word) > 5 or len(word) < 5:
            continue
        elif word not in instance.word_list:
            instance.word_list.add(word)
            return word
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


def _test_bot_parallel(words: set[str], testing_runs: int):
    correct_games = 0
    incorrect_games = 0
    guess_counts = []
    with Pool(processes=4) as pool:
        args = [(_rand_word(words), words) for _ in range(testing_runs)]
        results = pool.map(_run_single_game, args)
        for result in results:
            if result > 5:
                incorrect_games += 1
            else:
                correct_games += 1
                guess_counts.append(result)
    print(f"\n\nCorrect Games Percentage: {(correct_games / testing_runs) * 100}%")
    print(f"Incorrect Games Percentage: {(incorrect_games / testing_runs) * 100}%")
    print("Average Number of Guesses: ", round(sum(guess_counts) / len(guess_counts), 2))


def _run_single_game(args):
    word, word_list = args
    bot = simple_bot.WordleBot(list(word_list))
    guess_count = 0
    while guess_count < 6:
        guess = bot.make_guess(guess_count)
        if guess == word:
            return guess_count
        score = score_guess(word, guess)
        bot.filter_words(guess, score)
        guess_count += 1
    return guess_count


if __name__ == '__main__':
    game = wordle.Wordle()
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    _startup(game)
