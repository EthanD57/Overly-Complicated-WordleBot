import time
from pathlib import Path
from random import choice
import wordle
from multiprocessing import Pool
import click

from Utilities.data_collector import TrainingDataCollector
from Utilities.shared_utils import filter_words, score_guess
from ML import entropy_maximization_bot, random_forest
from Utilities import display


TESTING_MODE = False


def _startup(game_instance: wordle.Wordle):
    """
    Initialize The Wordle Game According to User Input
    The User Has the Option to Choose a Word or Have one
    Randomly be Decided.

    Args:
        game_instance: Wordle instance

    Returns:
        None

    """
    model = 1
    while True:
        display.print_menu()
        usr_input = click.prompt("Please Choose an Option", type = click.Choice(["1", "2", "3", "4", "5", "q"]), show_choices=False)
        if usr_input == "1":        #User-Chosen Word
            usr_word = _handle_user_word(game_instance)
            print(_play_game(game_instance.word_list, model, usr_word))
        elif usr_input == "2":      #Random Word
            rnd_word = _rand_word(game_instance.word_list)
            print(_play_game(game_instance.word_list, model, rnd_word))
        elif usr_input == "3":      #Test the Model/Bot
            global TESTING_MODE
            TESTING_MODE = True
            testing_range = click.prompt("Enter the Number of Tests You Would Like to Run",
                                         type = click.IntRange(1,), show_choices=False)
            processes = click.prompt("How Many Parallel Processes Should be Used",
                                     type = click.IntRange(1,12), show_choices=True)
            _test_bot_parallel(game_instance.word_list, testing_range, processes, model)
            print("Testing Complete! Returning To Main Menu...")
        elif usr_input == "4":      #Collect Training Data
            testing_range = click.prompt("Enter the Number of Games to Collect Data From",
                                         type = click.IntRange(1,), show_choices=False)
            processes = click.prompt("How Many Parallel Processes Should be Used",
                                     type=click.IntRange(1, 12), show_choices=True)
            _gather_testing_data(game_instance, testing_range, processes)
            print("Training Data Collected! Returning To Main Menu...")
        elif usr_input == "5":      #Choose Model/Bot to Use
            print("Model Options:\n"
                  "1. Entropy Maximization\n"
                  "2. Random Forest Classifier")
            model = click.prompt("Enter the Model You Would Like to Use",
                                 type=click.IntRange(1,2), show_choices=False)
        elif usr_input == 'q': exit()
        else:
            continue


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
        word = click.prompt("Please Enter a 5-Character String or Enter 'q' to Exit", type=str)
        if word == "q":
            exit()
        elif len(word) > 5 or len(word) < 5:
            continue
        elif word in instance.word_list:
            return word
        else:
            instance.word_list.add(word)
            instance.needRecompute = True


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


def _play_game(words: set[str], model: int, word=""):
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
    bot = entropy_maximization_bot.EntropyBot(list(words)) if model == 1 else random_forest.RandomForestBot(list(words))
    if model != 1 and not bot.is_trained: bot.train()
    guess_count = 0
    guesses = []
    while guess_count < 6:
        guess = bot.make_guess()
        if guess == word:  ##Correct Word Guessed
            guess_count += 1
            guesses.append([guess, score_guess(word, guess)])
            display.print_game_state(guesses)
            display.print_end_screen(word, guess_count)
            return ""
        else:  ##Incorrect Word Guessed. Update Game State and Send Score
            score = score_guess(word, guess)
            guesses.append([guess, score])
            ##Give the Bot Its Score for the Round
            filter_words(guess, score, bot.game_state)
            guess_count += 1
        display.print_game_state(guesses)
    return "Word Not Guessed :("


def _test_bot_parallel(words: set[str], testing_runs: int, processes=2, model: int = 1):
    correct_games = 0
    incorrect_games = 0
    guess_counts = []
    with Pool(processes) as pool:
        args = [(_rand_word(words), words, model) for _ in range(testing_runs)]
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
    word, word_list, model = args
    bot = entropy_maximization_bot.EntropyBot(list(word_list)) if model == 1 else random_forest.RandomForestBot(list(word_list))
    if model != 1 and not bot.is_trained: bot.train()
    guess_count = 0
    while guess_count < 6:
        guess = bot.make_guess()
        if guess == word:
            return guess_count
        score = score_guess(word, guess)
        filter_words(guess, score, bot.game_state)
        guess_count += 1
    return guess_count

def _gather_testing_data(game_instance: wordle.Wordle, game_count, process_count):
    collector = TrainingDataCollector(list(game_instance.word_list), game_instance.needRecompute)
    game_instance.needRecompute = False

    start_time = time.time()
    print(f"Collecting data from {game_count} games...")
    collector.collect_training_data_parallel(num_games= game_count, k= 10, processes= process_count)
    stop_time = time.time()

    print(f"Time taken: {stop_time - start_time}")
    print(f"Collected {len(collector.training_data)} training examples")
    print(f"Feature shape: {collector.training_data[0][0].shape}")
    print(f"Label shape: {collector.training_data[0][1].shape}")


if __name__ == '__main__':
    word_list_path = Path("words.txt")
    game = wordle.Wordle(word_list_path)
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    _startup(game)
