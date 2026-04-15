from pathlib import Path
from random import choice
import wordle
import pickle as pkl

from Utilities.shared_utils import filter_words, score_guess, calculate_entropy_pattern_table
from ML import entropy_maximization_bot, random_forest_classifier, random_forest_regressor

TESTING_MODE = False

#I know I'm going to get IDE warnings about this, but leaving it as None for now is fine. It will always get assigned
worker_pattern_table = None
model_options = ["Entropy Maximization", "Random Forest Classifier", "Random Forest Regressor", "Neural Network Classifier"]\


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


def _rand_word(words: list[str]):
    """
    Returns a Random Word From The Word List

    Args:
        words (list[str]): A list containing all valid words for the game

    Returns:
        str: The randomly chosen word

    """
    word = choice(tuple(words))
    if not TESTING_MODE: print(f"Random Word Chosen is {word}")
    return word


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

    bot = initialize_bot(game_instance, model)

    guess_count = 0
    guesses = []
    while guess_count < 6:
        guess = bot.make_guess()
        if guess == word:  ##Correct Word Guessed
            guess_count += 1
            guesses.append([guess, score_guess(word, guess)])
            return ""
        else:  ##Incorrect Word Guessed. Update Game State and Send Score
            score = score_guess(word, guess)
            guesses.append([guess, score])
            ##Give the Bot Its Score for the Round
            filter_words(guess, score, bot.game_state)
            guess_count += 1
    return "Word Not Guessed :("


if __name__ == '__main__':
    word_list_path = Path("words.txt")
    game = wordle.Wordle(word_list_path)
    print(f"Successfully Loaded {len(game.word_list)} Words Into The Game!\n\n")
    _startup(game)
