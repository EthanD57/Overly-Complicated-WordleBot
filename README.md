# Wordle ML Implementations

A collection of machine-learning approaches to solving Wordle, ranging from a classical information-theoretic bot to reinforcement learning with a Deep Q-Network.

## Models

| # | Model | Strategy | Needs Pre-generated Training Data |
|---|-------|----------|-----------------------------------|
| 1 | **Entropy Maximization** | Picks the guess that maximizes Shannon entropy over the remaining word distribution | No |
| 2 | **Random Forest Classifier** | Predicts which letters are most likely to appear next, trained on entropy-bot demonstrations | Yes |
| 3 | **Random Forest Regressor** | Same as above but regresses letter frequencies rather than classifying them | Yes |
| 4 | **Neural Network Classifier** | Feedforward network (sigmoid output) trained on entropy-bot demonstrations | Yes |
| 5 | **Deep Q-Network** | Learns a Q-value over the full vocabulary via self-play | No (trains itself) |

## Prerequisites

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

The first run of this program, <b>especially the entropy maximization bot</b>, will take a long time to complete. The 
pattern table takes a very long time to generate and is directly correlated to the word list length. Approximately 1.7
billion operations are completed for this. <br><br>
You might be asking yourself: "Why did you code the bot this way?" 
<br> The generation of thousands of testing runs (I did 10,000) was taking far too long. ~5 minutes on the first 
run is better than an hour for generating testing data. 
<br><br>
If you are interested in seeing each bot in action but do not have the time to commit to training, feel free to check out
my personal portfolio where the bot is hosted: www.defilippi.dev
<br>
You can contact me with questions about the bot from there as well. 

## Project Structure

```
├── main.py                          # CLI entry point
├── wordle.py                        # Wordle game — loads word list
├── words.txt                        # Word list
├── requirements.txt
├── ML/
│   ├── base_model.py                # Abstract base for sklearn / torch models
│   ├── entropy_maximization_bot.py
│   ├── random_forest_classifier.py
│   ├── random_forest_regressor.py
│   ├── neural_network_classifier.py
│   ├── deep_q_network.py
│   ├── saved_models/                # Cached pattern table + trained model files
│   └── training_data/               # Generated training data (wordle_training.pkl)
└── Utilities/
    ├── game_state.py                # Game state container
    ├── shared_utils.py              # Scoring, filtering, feature extraction, constants
    ├── display.py                   # Colorized terminal output (colorama)
    └── data_collector.py            # Parallel training data generation
```

## Running

```bash
python main.py
```

The interactive menu offers five options:

| Option | Description |
|--------|-------------|
| 1 | Enter a 5-letter word for the bot to solve |
| 2 | Let the bot solve a randomly chosen word |
| 3 | Run a batch test and report accuracy / average guesses |
| 4 | Generate training data for the supervised models |
| 5 | Switch the active model |

## Training the Supervised Models

The Random Forest and Neural Network models require training data produced by the entropy bot:

1. Select **option 4** from the menu and choose how many games to simulate (1,000+ recommended).
2. Training data is saved to `ML/training_data/wordle_training.pkl`.
3. Select a supervised model and play or test — it trains automatically on first use and caches the result to `ML/saved_models/`.

The DQN trains itself through self-play when first used; no separate data collection step is needed.

## How It Works

### Feature Vector (314 dimensions)

Every game state is encoded as a flat 314-element vector by `extract_features()`:

| Slice | Size | Description |
|-------|------|-------------|
| `[0:26]` | 26 | Normalized letter frequencies in remaining words |
| `[26:156]` | 130 (5×26) | Positional green-letter flags |
| `[156:286]` | 130 (5×26) | Positional yellow-letter flags |
| `[286:312]` | 26 | Gray letter flags |
| `[312]` | 1 | Fraction of words still in play |
| `[313]` | 1 | Current guess count |

### Entropy Maximization

Selects the word that maximizes Shannon entropy over the Wordle pattern distribution, collapsing the remaining word list as quickly as possible. A precomputed N×N pattern table (cached to `ML/saved_models/pattern_table.pkl`) encodes every (guess, answer) pair as a base-3 integer, making entropy calculation a fast NumPy operation.

When fewer than 20 words remain, the bot searches the entire master list instead of just the top high-frequency candidates — this allows a sacrificial guess (e.g. "miles") to rule out several rhyming traps like LIGHT / MIGHT / SIGHT / TIGHT at once.

### Supervised Models (RF Classifier, RF Regressor, Neural Net)

Trained on `(game_state_features, letter_frequency_labels)` pairs collected from entropy-bot games. The label for each state is the normalized letter frequency of the top-k highest-entropy candidate words — a soft target encoding what letters the entropy bot considers most valuable.

At inference, the model predicts a letter score for each of the 26 letters and the word with the highest total score across its unique letters is returned.

### Deep Q-Network

Maps the 314-dimensional state to a Q-value for every word in the vocabulary. Uses:
- **Experience replay** (buffer size 10,000) to decorrelate training samples
- **Separate target network** updated every 1,000 steps for training stability
- **ε-greedy exploration** decaying from 0.95 → 0.03 over training
- **Bellman update**: `Q(s,a) ← r + γ · max_a' Q_target(s', a')`
- Gradient clipping (max norm 1.0) to prevent exploding gradients

## Testing Results

The table below contains testing results for each bot for your convenience. Each testing run consists of 1000 tests.

| Model | Correct Games | Incorrect Games | Avg Number of Guesses | 
|-|---------------|-----------------|-----------------------|
| **Entropy Maximization** | 99.2%         | 0.8%            | 3.19                  |
| **Random Forest Classifier** | 87.2%         | 12.8%           | 3.37                  |
| **Random Forest Regressor** | 82.5%         | 17.5%           | 3.3                   |
| **Neural Network Classifier** | 83.3%         | 16.7%           | 3.49                  |
| **Deep Q-Network** | 85.9%         | 14.1%           | 3.53                  |
