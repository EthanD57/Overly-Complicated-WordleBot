from multiprocessing import Pool
from random import choice
import pickle
import numpy as np

from ML.entropy_maximization_bot import EntropyBot
from ML import entropy_maximization_bot
from Utilities.shared_utils import (calculate_normalized_letter_freq, score_guess,
                                    get_high_frequency_candidates, filter_words,
                                    extract_features, SACRIFICIAL_THRESHOLD, MAX_GUESSES)

# When remaining words > SACRIFICIAL_THRESHOLD, cap entropy evaluation to this
# many high-frequency candidates to keep label generation tractable.
_ENTROPY_LABEL_POOL_SIZE = 200

worker_pattern_table = None


def create_training_labels(bot: entropy_maximization_bot.EntropyBot, k: int) -> np.ndarray:
    """
    Compute the normalized letter-frequency vector of the top-k highest-entropy
    words from the current game state. Used as the training label for a given state.

    Args:
        bot: EntropyBot with the current remaining word list.
        k: Number of top-entropy words to include in the label.

    Returns:
        np.ndarray: Shape (26,) normalized letter frequencies.
    """
    if len(bot.game_state.remaining_words) > SACRIFICIAL_THRESHOLD:
        pool_size = min(_ENTROPY_LABEL_POOL_SIZE, len(bot.game_state.remaining_words) * 2)
        candidates = get_high_frequency_candidates(bot.game_state, pool_size)
    else:
        candidates = bot.game_state.master_list

    entropy_pairs = [(word, bot.calculate_entropy(word)) for word in candidates]
    entropy_pairs.sort(key=lambda x: x[1], reverse=True)
    top_k_words = [word for word, _ in entropy_pairs[:k]]

    return calculate_normalized_letter_freq(top_k_words)


def _collect_games_worker(args: tuple) -> list:
    """
    Simulate games with the entropy bot and collect (features, labels) pairs.

    Args:
        args: (num_games, k, word_list)

    Returns:
        list of (feature_vector, label_vector) tuples.
    """
    num_games, k, word_list = args
    training_data = []

    for _ in range(num_games):
        bot = EntropyBot(word_list, worker_pattern_table)
        target_word = choice(word_list)
        guess_count = 0

        while guess_count < MAX_GUESSES:
            training_data.append((
                extract_features(bot.game_state),
                create_training_labels(bot, k),
            ))
            bot_guess = bot.make_guess()
            if bot_guess == target_word:
                break
            score = score_guess(target_word, bot_guess)
            filter_words(bot_guess, score, bot.game_state)
            guess_count += 1

    return training_data


def init_worker(pattern_table):
    global worker_pattern_table
    worker_pattern_table = pattern_table


class TrainingDataCollector:
    def __init__(self, word_list: list[str], pattern_table: np.ndarray):
        self.word_list = word_list
        self.training_data = []
        self.entropy_pattern_table = pattern_table

    def collect_training_data_parallel(self, num_games: int, k: int = 10, processes: int = 4):
        """
        Simulate games across multiple processes and aggregate training data.

        Args:
            num_games: Total number of games to simulate.
            k: Number of top-entropy words used to construct each label.
            processes: Worker process count.
        """
        games_per_process = num_games // processes
        args = [(games_per_process, k, self.word_list) for _ in range(processes)]

        with Pool(processes=processes, initializer=init_worker,
                  initargs=(self.entropy_pattern_table,)) as pool:
            results = pool.map(_collect_games_worker, args)

        for process_data in results:
            self.training_data.extend(process_data)

        with open('ML/training_data/wordle_training.pkl', 'wb') as f:
            pickle.dump(self.training_data, f)
