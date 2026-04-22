from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from Utilities.game_state import GameState
from Utilities.shared_utils import extract_features, filter_words, score_guess, FEATURE_SIZE, MAX_GUESSES


def calculate_reward(won: bool, done: bool, guess_count: int,
                     words_before: int, words_after: int) -> float:
    """
    Compute the reward signal for a single guess.

    Rewards winning early and eliminating many words; penalizes losing.
    Clipped to [-10, 10] to stabilise training.

    Args:
        won: Whether the guess was correct.
        done: Whether the episode ended (win or max guesses reached).
        guess_count: Current guess number (1-indexed).
        words_before: Remaining word count before the guess.
        words_after: Remaining word count after the guess.
    """
    reward = 0.0
    if won:
        reward += 100.0 - (guess_count * 5.0)
    elif done:
        reward -= 50.0
    if words_before > 0:
        reward += (words_before - words_after) / words_before * 10.0
    return np.clip(reward, -10, 10)


class QNetwork(nn.Module):
    """Estimates Q-values for every word in the vocabulary given a game state."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNBot:
    def __init__(self, word_list: list[str]):
        self.game_state = GameState(word_list)
        self.model_path = Path('ML/saved_models/dqn_bot.pth')
        self.is_trained = False

        self.state_size = FEATURE_SIZE
        self.action_size = len(word_list)
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.gamma = 0.97
        self.epsilon = 0.95
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.target_update_frequency = 1000
        self.training_steps = 0
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

    def choose_action(self, epsilon: float) -> str:
        if random.random() < epsilon:
            return random.choice(self.game_state.remaining_words)

        state_tensor = torch.FloatTensor(extract_features(self.game_state).reshape(1, -1))
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Only consider words still in play; map back to master_list index.
        remaining_q = q_values[0, self.game_state.remaining_words_indices]
        word_idx = self.game_state.remaining_words_indices[torch.argmax(remaining_q).item()]
        return self.game_state.master_list[word_idx]

    def store_experience(self, state: np.ndarray, action_idx: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def train_step(self) -> float:
        """
        Sample a minibatch from the replay buffer and perform one gradient update.

        Returns:
            float: MSE loss for monitoring.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions))
        rewards_t = torch.FloatTensor(np.array(rewards))
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones))

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Bellman target: r + γ * max_a Q_target(s', a) * (1 - done)
            max_next_q = self.target_network(next_states_t).max(1)[0]
        target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def make_guess(self) -> str:
        guess = self.choose_action(epsilon=0)
        self.game_state.guess_count += 1
        return guess

    def train(self, num_episodes: int = 1000) -> None:
        """
        Train via self-play: play num_episodes games, storing experiences and
        updating the Q-network after each guess.

        Args:
            num_episodes: Number of games to simulate.
        """
        if self.model_path.exists():
            self.load(self.model_path)
            return

        for i in range(num_episodes):
            self.game_state.reset()
            answer = random.choice(self.game_state.master_list)
            loss = 0

            while self.game_state.guess_count < MAX_GUESSES:
                pre_state = extract_features(self.game_state)
                words_before = len(self.game_state.remaining_words)
                guess = self.choose_action(self.epsilon)
                self.game_state.guess_count += 1

                score = score_guess(answer, guess)
                filter_words(guess, score, self.game_state)
                words_after = len(self.game_state.remaining_words)
                post_state = extract_features(self.game_state)

                won = guess == answer
                lost = not won and self.game_state.guess_count >= MAX_GUESSES
                done = won or lost
                reward = calculate_reward(won, done, self.game_state.guess_count,
                                          words_before, words_after)
                self.store_experience(pre_state, self.game_state.word_to_index[guess],
                                      reward, post_state, done)
                loss = self.train_step()
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (i + 1) % 100 == 0:
                print(f"Episode {i + 1}/{num_episodes} | Epsilon: {self.epsilon:.3f} | Loss: {loss:.4f}")

        self.is_trained = True
        self.save(self.model_path)
        print("Training complete!")

    def save(self, filepath: Path) -> None:
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'is_trained': self.is_trained,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.is_trained = checkpoint.get('is_trained', True)
