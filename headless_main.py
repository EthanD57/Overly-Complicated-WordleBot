from pathlib import Path
from random import choice
import wordle
import pickle as pkl
import argparse

from Utilities.shared_utils import filter_words, score_guess, calculate_entropy_pattern_table
from ML import entropy_maximization_bot, random_forest_classifier, random_forest_regressor, deep_q_network, \
    neural_network_classifier


def main(game_instance: wordle.Wordle):
    """
    Initialize The Wordle Game According to User Input
    The User Has the Option to Choose a Word or Have one
    Randomly be Decided.

    Args:
        game_instance: Wordle instance

    Returns:
        None

    """
    parser = argparse.ArgumentParser(description="Wordle Bot Runner")

    parser.add_argument('--word', type=str, required=False, help='The word to guess')
    parser.add_argument('--model', type=str, default='entropy_maximization',
                        help='Which model to use')

    args = parser.parse_args()




def _play_game(game_instance: wordle.Wordle, model: int, word=""):
    """
    The Main Game Loop Logic.
    The Bot Plays the Game and the Results of Each
    Round is Shown in The Console For the
    User to Follow Along.

    Args:

    Returns:
        None

    """

    bot = _initialize_bot(game_instance, model)

    guess_count = 0
    guesses = []
    while guess_count < 6:
        guess = bot.make_guess()
        if guess == word:  ##Correct Word Guessed
            guess_count += 1
            guesses.append([guess, score_guess(word, guess)])
            break
        else:  ##Incorrect Word Guessed. Update Game State and Send Score
            score = score_guess(word, guess)
            guesses.append([guess, score])
            ##Give the Bot Its Score for the Round
            filter_words(guess, score, bot.game_state)
            guess_count += 1
    return guesses


def _load_pattern_table():
    path = Path("ML/saved_models/pattern_table.pkl")

    if path.exists():
        with open(path, 'rb') as f:
            worker_pattern_table = pkl.load(f)
    else:
        print("No pattern_table File Exists!")
        print("Crashing for safety!")
        exit()

    return worker_pattern_table


def _initialize_bot(game_instance: wordle.Wordle, model: int = 1):
    if model == 1:
        return entropy_maximization_bot.EntropyBot(game_instance.word_list, _load_pattern_table())
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
    main(game)
