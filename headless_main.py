import json
import sys
from pathlib import Path
from random import choice
import pickle as pkl
import argparse

from Utilities.shared_utils import filter_words, score_guess
from ML import entropy_maximization_bot, random_forest_classifier, random_forest_regressor, deep_q_network, \
    neural_network_classifier
import wordle


def main():
    """
    Headless Wordle bot runner that outputs JSON for web consumption.
    """
    models = {
        "entropy_maximization": 1,
        "random_forest_classifier": 2,
        "random_forest_regressor": 3,
        "neural_network_classifier": 4,
        "deep_q_network": 5
    }

    parser = argparse.ArgumentParser(description="Wordle Bot Runner")
    parser.add_argument('--word', type=str, required=False, help='The word to guess')
    parser.add_argument('--model', type=str, default='entropy_maximization',
                        help='Which model to use', choices=list(models.keys()))

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse calls sys.exit on error, we need to catch and return JSON
        print(json.dumps({
            "success": False,
            "error": "Invalid arguments provided"
        }))
        return

    try:
        # Load word list
        word_list_path = Path("words.txt")
        if not word_list_path.exists():
            print(json.dumps({
                "success": False,
                "error": "Word list not found"
            }))
            return

        game = wordle.Wordle(word_list_path)

        # Get the word (random if not specified)
        if args.word and args.word.lower() in game.word_list:
            target_word = args.word.lower()
        else:
            target_word = choice(game.word_list)
        # Checking the word length is NOT required here because this will NOT allow users
        # to enter their own chosen words for performance reasons. If the word list doesn't contain
        # the word sent my the website, it simply picks a random word. This sanitizes input from the website.

        # Run the game
        model_id = models[args.model]
        guesses = play_game(game, model_id, target_word)

        # Format response
        result = {
            "success": True,
            "word": target_word,
            "model": args.model,
            "guesses": guesses,
            "won": len(guesses) < 7,  # Won if guessed in < 6 indexed attempts (indexes 0-5)
            "num_guesses": len(guesses)
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))


def play_game(game_instance: wordle.Wordle, model: int, word: str) -> list:
    """
    Run a single Wordle game.

    Args:
        game_instance: Wordle instance with word list
        model: Model ID (1-5)
        word: Target word to guess

    Returns:
        List of guesses with scores
    """
    bot = initialize_bot(game_instance, model)

    guess_count = 0
    guesses = []

    while guess_count < 6:
        guess = bot.make_guess()
        score = score_guess(word, guess)

        # Format: [guess_word, [score_array]]
        guesses.append({
            "guess": guess,
            "score": score  # [0=wrong, 1=wrong position, 2=correct position]
        })

        if guess == word:
            break

        filter_words(guess, score, bot.game_state)
        guess_count += 1

    return guesses


def load_pattern_table() -> any:
    """Load pre-computed entropy pattern table for entropy bot."""
    path = Path("ML/saved_models/pattern_table.pkl")

    if path.exists():
        with open(path, 'rb') as f:
            return pkl.load(f)
    else:
        raise FileNotFoundError(
            "Pattern table not found. Run entropy bot training first."
        )


def initialize_bot(game_instance: wordle.Wordle, model: int = 1):
    """
    Initialize the appropriate bot based on model ID.

    Args:
        game_instance: Wordle instance
        model: Model ID (1-5)

    Returns:
        Initialized bot instance
    """
    if model == 1:
        return entropy_maximization_bot.EntropyBot(
            game_instance.word_list,
            load_pattern_table()
        )
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
        return bot
    elif model == 5:
        bot = deep_q_network.DQNBot(game_instance.word_list)
        bot.train()
        return bot
    else:
        raise ValueError(f"Unknown model ID: {model}")


if __name__ == '__main__':
    main()