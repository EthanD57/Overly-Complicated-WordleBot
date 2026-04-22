import time
from pathlib import Path
from random import choice
import wordle
from multiprocessing import Pool
import click
import pickle as pkl

from Utilities.data_collector import TrainingDataCollector
from Utilities.shared_utils import filter_words, score_guess, calculate_entropy_pattern_table, MAX_GUESSES
from ML import (entropy_maximization_bot, random_forest_classifier,
                random_forest_regressor, deep_q_network, neural_network_classifier)
from Utilities import display

TESTING_MODE = False

# Assigned lazily on first use; always set before any worker reads it.
worker_pattern_table = None
model_options = ["Entropy Maximization", "Random Forest Classifier", "Random Forest Regressor",
                 "Neural Network Classifier", "Deep Q-Network"]


def _startup(game_instance: wordle.Wordle):
    model = 1
    while True:
        display.print_menu(model, model_options)
        usr_input = click.prompt("Please Choose an Option", type=click.Choice(["1", "2", "3", "4", "5", "q"]),
                                 show_choices=False)
        if usr_input == "1":
            usr_word = _handle_user_word(game_instance)
            print(_play_game(game_instance, model, usr_word))
        elif usr_input == "2":
            rnd_word = _rand_word(game_instance.word_list)
            print(_play_game(game_instance, model, rnd_word))
        elif usr_input == "3":
            global TESTING_MODE
            TESTING_MODE = True
            testing_range = click.prompt("Enter the Number of Tests You Would Like to Run",
                                         type=click.IntRange(1, ), show_choices=False)
            if model < 4:
                processes = click.prompt("How Many Parallel Processes Should be Used",
                                         type=click.IntRange(1, 20), show_choices=True)
                _test_bot(game_instance, testing_range, processes, model)
            else:
                _test_non_parallel_models(game_instance, testing_range, model)
            print("Testing Complete! Returning To Main Menu...")
        elif usr_input == "4":
            testing_range = click.prompt("Enter the Number of Games to Collect Data From",
                                         type=click.IntRange(1, ), show_choices=False)
            processes = click.prompt("How Many Parallel Processes Should be Used",
                                     type=click.IntRange(1, 20), show_choices=True)
            _gather_testing_data(game_instance, testing_range, processes)
            print("Training Data Collected! Returning To Main Menu...")
        elif usr_input == "5":
            print("Model Options:\n"
                  "1. Entropy Maximization\n"
                  "2. Random Forest Classifier\n"
                  "3. Random Forest Regressor\n"
                  "4. Neural Network Classifier\n"
                  "5. Deep Q-Network\n")
            model = click.prompt("Enter the Model You Would Like to Use",
                                 type=click.IntRange(1, 5), show_choices=False)
        elif usr_input == 'q':
            exit()


def _handle_user_word(instance: wordle.Wordle) -> str:
    """
    Prompt the user for a 5-letter answer word.

    Words not in the standard list are added to word_list and flagged for
    pattern table recomputation so the entropy bot remains consistent.
    """
    while True:
        word = click.prompt("Please Enter a 5-Character String or Enter 'q' to Exit", type=str)
        if word == "q":
            exit()
        if len(word) != 5:
            continue
        if word not in game.word_list:
            instance.needRecompute = True
            instance.word_list.append(word)
        return word


def _rand_word(words: list[str]) -> str:
    word = choice(tuple(words))
    if not TESTING_MODE:
        print(f"Random Word Chosen is {word}")
    return word


def _play_game(game_instance: wordle.Wordle, model: int, word: str = "") -> str:
    display.print_game_start()
    bot = initialize_bot(game_instance, model)

    guess_count = 0
    guesses = []
    while guess_count < MAX_GUESSES:
        guess = bot.make_guess()
        score = score_guess(word, guess)
        guesses.append([guess, score])
        guess_count += 1

        if guess == word:
            display.print_game_state(guesses)
            display.print_end_screen(word, guess_count)
            return ""

        filter_words(guess, score, bot.game_state)
        display.print_game_state(guesses)

    return "Word Not Guessed :("


def init_worker(pattern_table):
    global worker_pattern_table
    worker_pattern_table = pattern_table


def _test_bot(game_instance: wordle.Wordle, testing_runs: int, processes: int = 2, model: int = 1):
    correct_games = 0
    incorrect_games = 0
    guess_counts = []
    pattern_table = None

    if model != 1:
        initialize_bot(game_instance, model)
    else:
        pattern_table = get_pattern_table(game_instance)

    with Pool(processes, initializer=init_worker, initargs=(pattern_table,)) as pool:
        args = [(_rand_word(game_instance.word_list), game_instance.word_list, model) for _ in range(testing_runs)]
        results = pool.map(_run_single_game, args)
        for result in results:
            if result > MAX_GUESSES - 1:
                incorrect_games += 1
            else:
                correct_games += 1
                guess_counts.append(result)

    print(f"\n\nCorrect Games Percentage: {round((correct_games / testing_runs) * 100, 2)}%")
    print(f"Incorrect Games Percentage: {round((incorrect_games / testing_runs) * 100, 2)}%")
    print("Average Number of Guesses: ", round(sum(guess_counts) / len(guess_counts), 2))


def _test_non_parallel_models(game_instance: wordle.Wordle, testing_runs: int, model: int = 1):
    correct_games = 0
    incorrect_games = 0
    guess_counts = []
    if model != 1:
        initialize_bot(game_instance, model)
    for i in range(testing_runs):
        guess_count = _run_single_game((_rand_word(game_instance.word_list), game_instance.word_list, model))
        if guess_count < MAX_GUESSES:
            guess_counts.append(guess_count)
            correct_games += 1
        else:
            incorrect_games += 1
    print(f"\n\nCorrect Games Percentage: {round((correct_games / testing_runs) * 100, 2)}%")
    print(f"Incorrect Games Percentage: {round((incorrect_games / testing_runs) * 100, 2)}%")
    print("Average Number of Guesses: ", round(sum(guess_counts) / len(guess_counts), 2))


def _run_single_game(args) -> int:
    word, word_list, model = args

    if model == 1:
        bot = entropy_maximization_bot.EntropyBot(word_list, worker_pattern_table)
    elif model == 2:
        bot = random_forest_classifier.RandomForestClassifierModel(word_list)
    elif model == 3:
        bot = random_forest_regressor.RandomForestRegressorModel(word_list)
    elif model == 4:  # Neural net cannot run in parallel; should never reach here via _test_bot
        bot = neural_network_classifier.NeuralNetworkClassifier(word_list)
    else:
        bot = deep_q_network.DQNBot(word_list)

    if model != 1 and not bot.is_trained:
        bot.train()

    guess_count = 0
    while guess_count < MAX_GUESSES:
        guess = bot.make_guess()
        if guess == word:
            return guess_count
        score = score_guess(word, guess)
        filter_words(guess, score, bot.game_state)
        guess_count += 1
    return guess_count


def _gather_testing_data(game_instance: wordle.Wordle, game_count: int, process_count: int):
    pattern_table = get_pattern_table(game_instance)
    collector = TrainingDataCollector(game_instance.word_list, pattern_table)

    start_time = time.time()
    print(f"Collecting data from {game_count} games...")
    collector.collect_training_data_parallel(num_games=game_count, k=10, processes=process_count)
    elapsed = time.time() - start_time

    print(f"Time taken: {elapsed:.1f}s")
    print(f"Collected {len(collector.training_data)} training examples")
    print(f"Feature shape: {collector.training_data[0][0].shape}")
    print(f"Label shape: {collector.training_data[0][1].shape}")


def get_pattern_table(game_instance: wordle.Wordle):
    global worker_pattern_table
    path = Path("ML/saved_models/pattern_table.pkl")

    if path.exists():
        with open(path, 'rb') as f:
            worker_pattern_table = pkl.load(f)

    if worker_pattern_table is None or game_instance.needRecompute:
        worker_pattern_table = calculate_entropy_pattern_table(game_instance.word_list)
        game_instance.needRecompute = False

    with open(path, 'wb') as f:
        pkl.dump(worker_pattern_table, f)

    return worker_pattern_table


def initialize_bot(game_instance: wordle.Wordle, model: int = 1):
    if model == 1:
        return entropy_maximization_bot.EntropyBot(game_instance.word_list, get_pattern_table(game_instance))
    elif model == 2:
        bot = random_forest_classifier.RandomForestClassifierModel(game_instance.word_list)
        bot.train()
        return bot
    elif model == 3:
        bot = random_forest_regressor.RandomForestRegressorModel(game_instance.word_list)
        bot.train()
        return bot
    elif model == 4:
        bot = neural_network_classifier.NeuralNetworkClassifier(game_instance.word_list)
        bot.train()
    else:
        bot = deep_q_network.DQNBot(game_instance.word_list)
        bot.train()
    return bot


if __name__ == '__main__':
    word_list_path = Path("words.txt")
    game = wordle.Wordle(word_list_path)
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    _startup(game)
