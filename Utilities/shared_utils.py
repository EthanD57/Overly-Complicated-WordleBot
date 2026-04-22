from collections import Counter, defaultdict
from Utilities.game_state import GameState
import numpy as np

# Feature vector length produced by extract_features()
FEATURE_SIZE = 314

MAX_GUESSES = 6

# Entropy bot / base model guess-selection tuning
SACRIFICIAL_THRESHOLD = 20   # Below this many remaining words, allow sacrificial guesses to break traps
HIGH_FREQ_TOP_N = 300        # Candidate pool size when remaining words exceeds SACRIFICIAL_THRESHOLD
ANSWER_BONUS = 0.01          # Score tiebreaker for words that are still valid answers


def calculate_normalized_letter_freq(remaining_words: list[str]) -> np.ndarray:
    """
    Return a (26,) array of normalized letter frequencies across remaining_words.
    Each value is the fraction of words containing that letter (index 0 = 'a').
    """
    letter_freq = Counter()
    for word in remaining_words:
        for letter in set(word):
            letter_freq[letter] += 1

    total_words = len(remaining_words) if remaining_words else 1
    frequencies = np.zeros(26)
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        frequencies[ord(letter) - ord('a')] = letter_freq[letter] / total_words

    return frequencies


def score_guess(correct_word: str, guess: str) -> list[int]:
    """
    Score a guess against the correct word using Wordle rules.

    Uses a two-pass algorithm to correctly handle duplicate letters:
    pass 1 — mark exact matches (green/2) and remove them from the available pool.
    pass 2 — mark wrong-position letters (yellow/1) only for letters still available.

    Returns:
        list[int]: per-letter scores — 2 green, 1 yellow, 0 gray.
    """
    result = [0] * len(guess)
    answer_chars = list(correct_word)

    for i, char in enumerate(guess):
        if char == correct_word[i]:
            result[i] = 2
            answer_chars[i] = None  # consume so duplicate yellows are bounded correctly

    for i, char in enumerate(guess):
        if result[i] == 0 and char in answer_chars:
            result[i] = 1
            answer_chars[answer_chars.index(char)] = None

    return result


def filter_words(guess: str, result: list[int], game_state: GameState) -> None:
    """
    Narrow game_state.remaining_words based on the scored guess.

    Gray with duplicates is handled carefully: if a letter appears green/yellow
    elsewhere in the same guess, gray means "no additional copies", not "absent".

    Args:
        guess: The guessed word.
        result: Per-letter scores from score_guess().
        game_state: Mutated in place — remaining_words and indices are updated.
    """
    letter_min_count = defaultdict(int)
    letter_max_count = {}
    position_exclusions = defaultdict(set)
    game_state.scored_rounds[guess] = result

    for pos, (letter, score) in enumerate(zip(guess, result)):
        if score == 2:
            letter_min_count[letter] += 1
            game_state.green_letters[pos] = letter
        elif score == 1:
            letter_min_count[letter] += 1
            position_exclusions[pos].add(letter)
            game_state.yellow_letters.add(letter)
        else:
            # Count green+yellow appearances of this letter in the same guess to
            # determine whether it's fully absent (max=0) or capped at that count.
            green_yellow_count = sum(1 for l, s in zip(guess, result) if l == letter and s in [1, 2])
            letter_max_count[letter] = green_yellow_count
            if green_yellow_count == 0:
                game_state.gray_letters.add(letter)

    filtered_words = []
    for word in game_state.remaining_words:
        if not all(word[pos] == letter for pos, letter in game_state.green_letters.items()):
            continue
        if any(word[pos] in excluded for pos, excluded in position_exclusions.items()):
            continue

        valid = True
        for letter, min_count in letter_min_count.items():
            if word.count(letter) < min_count:
                valid = False
                break
        if not valid:
            continue

        for letter, max_count in letter_max_count.items():
            if word.count(letter) > max_count:
                valid = False
                break

        if valid:
            filtered_words.append(word)

    game_state.remaining_words = filtered_words
    game_state.remaining_words_indices = [game_state.word_to_index[word] for word in filtered_words]


def get_high_frequency_candidates(game_state: GameState, top_n: int = HIGH_FREQ_TOP_N,
                                   candidate_pool: list = None) -> list:
    """
    Return the top_n words from candidate_pool ranked by coverage of high-frequency letters.

    Letter frequencies are computed from game_state.remaining_words, not from candidate_pool,
    so the ranking always reflects what letters are most informative right now.

    Args:
        game_state: Current game state (remaining_words drives frequency counts).
        top_n: Maximum number of candidates to return.
        candidate_pool: Words to rank. Defaults to game_state.remaining_words.
    """
    if candidate_pool is None:
        candidate_pool = game_state.remaining_words

    letter_freq = Counter()
    for word in game_state.remaining_words:
        for letter in set(word):
            letter_freq[letter] += 1

    scored_candidates = [(sum(letter_freq[l] for l in set(word)), word) for word in candidate_pool]
    scored_candidates.sort(reverse=True)
    return [word for _, word in scored_candidates[:top_n]]


def extract_features(game_state: GameState) -> np.ndarray:
    """
    Encode the current game state as a flat feature vector of length FEATURE_SIZE (314).

    Layout:
        [0:26]    normalized letter frequencies in remaining words
        [26:156]  green letter flags, shape (5, 26) flattened
        [156:286] yellow letter flags, shape (5, 26) flattened
        [286:312] gray letter flags, shape (26,)
        [312]     fraction of words still remaining
        [313]     current guess count
    """
    letter_frequencies = calculate_normalized_letter_freq(game_state.remaining_words)
    green_letters = np.zeros((5, 26), dtype=int)
    yellow_letters = np.zeros((5, 26), dtype=int)
    gray_letters = np.zeros(26, dtype=int)

    for guess, score in game_state.scored_rounds.items():
        for pos, (letter, result) in enumerate(zip(guess, score)):
            letter_idx = ord(letter) - ord('a')
            if result == 2:
                green_letters[pos, letter_idx] = 1
            elif result == 1:
                yellow_letters[pos, letter_idx] = 1
            else:
                gray_letters[letter_idx] = 1

    return np.concatenate([
        letter_frequencies,
        green_letters.flatten(),
        yellow_letters.flatten(),
        gray_letters,
        [len(game_state.remaining_words) / len(game_state.master_list)],
        [game_state.guess_count],
    ])


def calculate_entropy_pattern_table(word_list: list[str]) -> np.ndarray:
    """
    Precompute an N×N matrix where entry [i, j] encodes the Wordle score for
    guessing word i when the answer is word j, as a base-3 integer in [0, 242].
    """
    n = len(word_list)
    pattern_matrix = np.zeros((n, n), dtype=np.uint8)
    # Each score is a 5-digit base-3 number; these are the place values (3^4 … 3^0).
    powers_of_3 = np.array([81, 27, 9, 3, 1], dtype=np.uint8)

    print(f"Precomputing {n}x{n} pattern table (this might take a minute, but only happens once)...")

    for i, guess in enumerate(word_list):
        for j, answer in enumerate(word_list):
            score_int = np.sum(np.array(score_guess(answer, guess)) * powers_of_3)
            pattern_matrix[i, j] = score_int

    return pattern_matrix
