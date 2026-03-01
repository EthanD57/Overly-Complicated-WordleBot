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