from collections import Counter, defaultdict

import numpy as np

from Utilities.game_state import GameState


def calculate_normalized_letter_freq(remaining_words: list[str]):
    """
    Extract normalized letter frequency from remaining words

    Args:
        remaining_words (list[str]): List of remaining words

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


def score_guess(correct_word: str, guess: str) -> list[int]:
    """
    Scores the Guess For the Current Round.
    Uses a two-pass algorithm to properly handle duplicate letters:
    1. First pass: Mark exact matches (green/2)
    2. Second pass: Mark wrong positions (yellow/1) only if letters remain available

    This ensures that if a letter appears multiple times in the guess but fewer
    times in the answer, only the appropriate number of instances get marked as
    yellow/green (matching real Wordle behavior).

    Args:
        correct_word (str): The Correct Word for the Wordle Game
        guess (str): The Guess From the Bot

    Returns:
        list[int]: A list containing the Score of the Correct Word
                   2 = correct position (green)
                   1 = wrong position (yellow)
                   0 = not in word (gray)

    """
    result = [0] * len(guess)
    answer_chars = list(correct_word)

    # First pass: Mark exact matches and remove them from available pool
    for i, char in enumerate(guess):
        if char == correct_word[i]:
            result[i] = 2
            answer_chars[i] = None  # Mark as used

    # Second pass: Mark wrong positions for remaining letters
    for i, char in enumerate(guess):
        if result[i] == 0:  # Not already an exact match
            if char in answer_chars:
                result[i] = 1
                answer_chars[answer_chars.index(char)] = None  # Mark as used

    return result


def filter_words(guess: str, result: list[int], game_state: GameState):
    """
    Filters the words based off the score response from the game.

    Uses a two-pass algorithm that first collects all the requirements from the score.

    Gray indicates a hard-limit of a given character in answer.
    Yellow indicates a hard-minimum of a given character in an answer.
    Green indicates the exact position that a letter must appear.

    The second pass filters the remaining word list to only contain words that match all
    the rules from the first pass.

    Args:
        game_state (GameState): GameState object representing the current game state for the bot
        guess (str): The guessed answer
        result (list[int]): The score response from the game

    Returns:
        None

    """
    # First pass: collect all requirements from the guess
    letter_min_count = defaultdict(int)  # Minimum times a letter must appear
    letter_max_count = {}  # Maximum times a letter can appear
    position_exclusions = defaultdict(set)  # pos -> {letters} (yellow: can't be at this position)
    game_state.scored_rounds[guess] = result

    for pos, (letter, score) in enumerate(zip(guess, result)):
        if score == 2:  # Green - letter is in the correct position
            letter_min_count[letter] += 1
            game_state.green_letters[pos] = letter
        elif score == 1:  # Yellow - letter is in word but wrong position
            letter_min_count[letter] += 1
            position_exclusions[pos].add(letter)
            game_state.yellow_letters.add(letter)
        else:  # Gray - letter is not in word, OR we've found all instances
            # Count how many times this letter appears as green/yellow in the entire guess
            green_yellow_count = sum(1 for l, s in zip(guess, result) if l == letter and s in [1, 2])
            if green_yellow_count > 0:
                # Letter appears exactly this many times (no more)
                letter_max_count[letter] = green_yellow_count
            else:
                # Letter not in word at all
                letter_max_count[letter] = 0
                game_state.gray_letters.add(letter)

    # Second pass: filter words based on all requirements
    filtered_words = []
    for word in game_state.remaining_words:
        # Check position requirements (green letters must be in correct spots)
        if not all(word[pos] == letter for pos, letter in game_state.green_letters.items()):
            continue

        # Check position exclusions (yellow letters can't be in certain positions)
        if any(word[pos] in excluded_letters for pos, excluded_letters in position_exclusions.items()):
            continue

        # Check minimum letter counts (green + yellow letters must appear at least this many times)
        valid = True
        for letter, min_count in letter_min_count.items():
            if word.count(letter) < min_count:
                valid = False
                break

        if not valid:
            continue

        # Check maximum letter counts (gray letters limit the count)
        for letter, max_count in letter_max_count.items():
            if word.count(letter) > max_count:
                valid = False
                break

        if valid:
            filtered_words.append(word)

    game_state.remaining_words = filtered_words


def get_high_frequency_candidates(game_state: GameState, top_n=300) -> list:
    """
    Get words with the highest letter frequency in remaining words

    Args:
        game_state (GameState): GameState object representing the current game state for the bot
        top_n (int): The amount of words the function should return

    Returns:
        List: The list of words composed of the most common letters

    """
    # Count letter frequencies in remaining words
    letter_freq = Counter()
    for word in game_state.remaining_words:
        for letter in set(word):
            letter_freq[letter] += 1

    # Score each candidate word by how many high-frequency letters it has
    scored_candidates = []
    for word in game_state.remaining_words:
        score = sum(letter_freq[letter] for letter in set(word))
        scored_candidates.append((score, word))

    scored_candidates.sort(reverse=True)
    return [word for _, word in scored_candidates[:top_n]]


def extract_features(game_state: GameState):
    """
    Extract features from current game state.

    Returns:
        np.array: Feature vector representing the current state
    """

    letter_frequencies= calculate_normalized_letter_freq(game_state.remaining_words)
    green_letters = np.zeros((5, 26), dtype=int)
    yellow_letters = np.zeros((5, 26), dtype=int)
    gray_letters = np.zeros(26, dtype=int)

    for guess, score in game_state.scored_rounds.items():
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
        [len(game_state.remaining_words) / len(game_state.master_list)],  # 1 value
        [game_state.guess_count]  # 1 value
    ])

    return features