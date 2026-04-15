from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from Utilities.game_state import GameState
from Utilities.shared_utils import extract_features, filter_words, score_guess


def calculate_reward(won: bool, done: bool, guess_count: int,
                     words_before: int, words_after: int) -> float:
    """
    Calculate reward for a guess.

    Args:
        won: Whether the game was won
        done: Whether the episode is done or not
        guess_count: Current guess number (1-6)
        words_before: Number of possible words before guess
        words_after: Number of possible words after guess

    Returns:
        float: The reward value
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
    """Neural network that estimates Q-values for all possible actions"""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        layer_1_output = torch.relu(self.fc1(x))
        layer_2_output = torch.relu(self.fc2(layer_1_output))
        return self.fc3(layer_2_output)


class DQNBot:
    def __init__(self, word_list: list[str]):
        self.game_state = GameState(word_list)
        self.model_path = Path('ML/saved_models/dqn_bot.pth')
        self.is_trained = False

        # Network setup
        self.state_size = 314  # from extract_features()
        self.action_size = len(word_list)
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Hyperparameters
        self.gamma = 0.97  # Discount factor
        self.epsilon = 0.95  # Current exploration rate
        self.epsilon_min = 0.03  # Minimum exploration
        self.epsilon_decay = 0.995   # How fast to reduce exploration
        self.learning_rate = 0.0001
        self.target_update_frequency = 1000  # Update target network every 1000 training steps
        self.training_steps = 0
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)  # Stores experiences
        self.batch_size = 64

    def choose_action(self, epsilon: float) -> str:
        random_number = random.random()
        if random_number < epsilon:
            return random.choice(self.game_state.remaining_words)

        state = extract_features(self.game_state).reshape(1, -1)
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        remaining_q_values = q_values[0, self.game_state.remaining_words_indices]
        best_idx = torch.argmax(remaining_q_values).item()
        word_idx = self.game_state.remaining_words_indices[best_idx]

        return self.game_state.master_list[word_idx]

    def store_experience(self, state: np.ndarray, action_idx: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """
        Store experience tuple in replay buffer.

        Args:
            state: State features before action (314,)
            action_idx: Index of word that was guessed
            reward: Reward received
            next_state: State features after action (314,)
            done: Whether episode ended (won or lost)
        """
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def train_step(self) -> float:
        """
        Sample batch from replay buffer and perform one training step.

        Returns:
            float: Loss value (for monitoring)
        """
        # Check if we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        states_float = torch.FloatTensor(states)
        actions_long = torch.LongTensor(actions)  # Not FloatTensor
        rewards_float = torch.FloatTensor(rewards)
        next_states_float = torch.FloatTensor(next_states)
        dones_float = torch.FloatTensor(dones)

        #Get current Q-values for actions taken
        current_q_values  = self.q_network(states_float)
        current_q = current_q_values.gather(1, actions_long.unsqueeze(1)).squeeze(1)

        with torch.no_grad():  # Don't need gradients for target network
            next_q_values = self.target_network(next_states_float)  # Shape: (batch_size, action_size)
            max_next_q = next_q_values.max(1)[0]  # Shape: (batch_size,)

        #Calculate target Q-values using Bellman equation
        target_q = rewards_float + self.gamma * max_next_q * (1 - dones_float)

        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()  # Update weights

        # Track training steps and update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()  # Return loss value for monitoring

    def make_guess(self) -> str:
        guess = self.choose_action(epsilon=0)
        self.game_state.guess_count += 1
        return guess

    def train(self, num_episodes: int = 1000) -> None:
        """
        Train the DQN by playing games and learning from experience.

        Args:
            num_episodes: Number of games to play during training
        """
        if self.model_path.exists():
            self.load(self.model_path)
            return

        for i in range(num_episodes):
            self.game_state.reset()
            answer = random.choice(self.game_state.master_list)
            loss = 0
            while self.game_state.guess_count < 6:
                pre_state = extract_features(self.game_state)
                word_count_before = len(self.game_state.remaining_words)
                guess = self.choose_action(self.epsilon)
                self.game_state.guess_count += 1
                score = score_guess(answer, guess)
                filter_words(guess, score, self.game_state)
                word_count_after = len(self.game_state.remaining_words)
                post_state = extract_features(self.game_state)
                if guess == answer:
                    reward = calculate_reward(True, False, self.game_state.guess_count,
                                              word_count_before, word_count_after)
                    done = True
                else:
                    lost = (self.game_state.guess_count >= 6)
                    reward = calculate_reward(False, lost, self.game_state.guess_count,
                                              word_count_before, word_count_after)
                    done = lost  # Episode ends if we lost
                self.store_experience(pre_state, self.game_state.word_to_index[guess], reward, post_state, done)
                loss = self.train_step()
                if done: break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (i + 1) % 100 == 0:  # Print every 100 episodes
                print(f"Episode {i + 1}/{num_episodes} | Epsilon: {self.epsilon:.3f} | Loss: {loss:.4f}")

        self.is_trained = True
        self.save(self.model_path)
        print("Training complete!")

    def save(self, filepath: Path) -> None:
        """Save trained model to disk."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'game_state': self.game_state.reset(),
            'is_trained': self.is_trained,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load trained model from disk."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']