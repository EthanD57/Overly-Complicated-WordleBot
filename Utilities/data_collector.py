from pathlib import Path
from random import choice
import numpy as np
from collections import defaultdict, Counter
import wordle
from Utilities.score_guess import score_guess
from ML import entropy_maximization_bot
import pickle

def main():
    game = wordle.Wordle()
    collector = TrainingDataCollector(list(game.word_list))

    print("Collecting data from 10 games...")
    collector.collect_training_data(num_games=10000, k=10)

    print(f"Collected {len(collector.training_data)} training examples")
    print(f"Feature shape: {collector.training_data[0][0].shape}")
    print(f"Label shape: {collector.training_data[0][1].shape}")

class TrainingDataCollector:
    def __init__(self, word_list: list[str]):
        self.word_list = word_list
        self.training_data = []

    def extract_features(self, bot: entropy_maximization_bot.EntropyBot, guess_count: int) -> np.array:
        """
        Extract features from current game state.

        Returns:
            np.array: Feature vector representing the current state
        """
        features = []

        letter_frequencies= self.calculate_normalized_letter_freq(bot.remaining_words)
        green_letters = np.zeros((5, 26), dtype=int)
        yellow_letters = np.zeros((5, 26), dtype=int)
        gray_letters = np.zeros(26, dtype=int)

        for guess, score in bot.scored_rounds.items():
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
            [len(bot.remaining_words) / len(self.word_list)],  # 1 value
            [guess_count]  # 1 value
        ])

        return features

    def calculate_normalized_letter_freq(self, remaining_words: list[str]):
        """
            Extract normalized letter frequency from remaining words

        Returns:
            np.array: An array containing normalized frequencies of all letters

        """
        # Count letter frequencies in remaining words
        letter_freq = Counter()
        for word in remaining_words:
            for letter in set(word):
                letter_freq[letter] += 1

        # Create array for a-z, normalized
        frequencies = np.zeros(26)
        total_words = len(remaining_words) if remaining_words else 1
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            frequencies[ord(letter) - ord('a')] = letter_freq[letter] / total_words

        return frequencies

    def create_training_labels(self, bot: entropy_maximization_bot.EntropyBot, k: int):
        """
        This function looks at the current state of the bot, find the top K-highest entropy
        words in the remaining words and then returns label weights for each letter in those words

        Args:
            k: The number of words to pull from the entropy list
            bot (entropy_maximization_bot.EntropyBot): The bot that contains the remaining word list and round history

        Returns:
            np.array: An array of letter labels weights floats
        """

        if len(bot.remaining_words) > 20:   #The entropy function is super costly, so we're guess_candidates
            candidates_to_check = min(300, len(bot.remaining_words) * 2)
            guess_candidates = bot.get_high_frequency_candidates(candidates_to_check)
        else:
            guess_candidates = bot.master_list

        word_entropies = []
        for word in guess_candidates:   #Get the word entropies from the bot
            word_entropies.append(bot.calculate_entropy(word))

        entropy_guess_pairs = list(zip(guess_candidates, word_entropies))
        entropy_guess_pairs.sort(key=lambda x: x[1], reverse=True)

        top_k_words = [pair[0] for pair in entropy_guess_pairs[:k]]

        return self.calculate_normalized_letter_freq(top_k_words)

    def collect_training_data(self, num_games: int, k: int = 10):
        """
        Run games and collect (features, labels) pairs.

        Args:
            num_games: Number of games to simulate
            k: Number of top entropy words to use for labels
        """
        for x in range(num_games):
            bot = entropy_maximization_bot.EntropyBot(self.word_list)
            target_word = choice(bot.remaining_words)
            guess_count = 0

            while guess_count < 6:
                self.training_data.append((self.extract_features(bot, guess_count),
                                           self.create_training_labels(bot, k)))

                bot_guess = bot.make_guess(guess_count)

                if bot_guess == target_word:
                    break

                score = score_guess(target_word, bot_guess)
                bot.filter_words(bot_guess, score)
                guess_count += 1

        training_data_path = Path("../ML/training_data/wordle_training_data.txt")
        with open(training_data_path, 'wb') as f:
            pickle.dump(self.training_data, f)

