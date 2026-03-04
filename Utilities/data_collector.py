
from multiprocessing import Pool
from random import choice
import numpy as np
import pickle

from ML.entropy_maximization_bot import EntropyBot
from ML import entropy_maximization_bot
from Utilities.shared_utils import calculate_normalized_letter_freq, score_guess, get_high_frequency_candidates, filter_words


def create_training_labels(bot: entropy_maximization_bot.EntropyBot, k: int):
    """
    This function looks at the current state of the bot, find the top K-highest entropy
    words in the remaining words and then returns label weights for each letter in those words

    Args:
        k: The number of words to pull from the entropy list
        bot (entropy_maximization_bot.EntropyBot): The bot that contains the remaining word list and round history

    Returns:
        np.array: An array of letter labels weights floats
    """

    if len(bot.game_state.remaining_words) > 20:    #The entropy function is super costly, so we're guess_candidates
                                                    #20 is arbitrary
        candidates_to_check = min(300, len(bot.game_state.remaining_words) * 2)    #300 is arbitrary
        guess_candidates = get_high_frequency_candidates(bot.game_state, candidates_to_check)
    else:
        guess_candidates = bot.game_state.master_list

    word_entropies = []
    for word in guess_candidates:   #Get the word entropies from the bot
        word_entropies.append(bot.calculate_entropy(word))

    entropy_guess_pairs = list(zip(guess_candidates, word_entropies))
    entropy_guess_pairs.sort(key=lambda x: x[1], reverse=True)

    top_k_words = [pair[0] for pair in entropy_guess_pairs[:k]]

    return calculate_normalized_letter_freq(top_k_words)


class TrainingDataCollector:
    def __init__(self, word_list: list[str]):
        self.word_list = word_list
        self.training_data = []

    def extract_features(self, bot: entropy_maximization_bot.EntropyBot, guess_count: int):
        """
        Extract features from current game state.

        Returns:
            np.array: Feature vector representing the current state
        """

        letter_frequencies= calculate_normalized_letter_freq(bot.game_state.remaining_words)
        green_letters = np.zeros((5, 26), dtype=int)
        yellow_letters = np.zeros((5, 26), dtype=int)
        gray_letters = np.zeros(26, dtype=int)

        for guess, score in bot.game_state.scored_rounds.items():
            for pos, (letter, result) in enumerate(zip(guess, score)):
                letter_idx = ord(letter) - ord('a')
                if result == 2:  #If the letter is green, mark that position as green in the character's array
                    green_letters[pos, letter_idx] = 1
                if result == 1:  #If the letter is yellow, mark that position is yellow in that character's array
                    yellow_letters[pos, letter_idx] = 1
                if result == 0:  #If the letter is gray, mark that position as gray in the letter array.
                    gray_letters[letter_idx] = 1

        features = np.concatenate([
            letter_frequencies,  # 26 values
            green_letters.flatten(),  # 130 values (5×26)
            yellow_letters.flatten(),  # 130 values (5×26)
            gray_letters,  # 26 values
            [len(bot.game_state.remaining_words) / len(self.word_list)],  # 1 value
            [guess_count]  # 1 value
        ])

        return features


    def _collect_games_worker(self, args: tuple):
        """
        Run games and collect (features, labels) pairs.

        Args:
            args (tuple): Contains the number of games to run in this process, the number of top entropy words
            to use (k), and the word list to use.
        """
        training_data = []
        num_games, k, word_list = args

        for x in range(num_games):
            bot = EntropyBot(word_list)
            target_word = choice(bot.game_state.remaining_words)
            guess_count = 0

            while guess_count < 6:
                training_data.append((
                    self.extract_features(bot, guess_count),
                    create_training_labels(bot, k)
                ))

                bot_guess = bot.make_guess(guess_count)

                if bot_guess == target_word:
                    break

                score = score_guess(target_word, bot_guess)
                filter_words(bot_guess, score, bot.game_state)
                guess_count += 1

        return training_data


    def collect_training_data_parallel(self, num_games: int, k: int = 10, processes: int = 4):
        """
        Run games and collect (features, labels) pairs.

        Args:
            processes: The number of parallel processes to use
            num_games: Number of games to simulate
            k: Number of top entropy words to use for labels
        """

        # Split games across processes
        games_per_process = num_games // processes

        with Pool(processes=processes) as pool:
            args = [(games_per_process, k, self.word_list) for _ in range(processes)]
            results = pool.map(self._collect_games_worker, args)

        # Combine results from all processes
        for process_data in results:
            self.training_data.extend(process_data)

        # Save
        with open('../ML/training_data/wordle_training.pkl', 'wb') as f:
            pickle.dump(self.training_data, f)


if __name__ == '__main__':
    main()