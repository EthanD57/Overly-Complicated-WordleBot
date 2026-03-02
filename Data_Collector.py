import wordle
from Utilities.ML_trainer import TrainingDataCollector

game = wordle.Wordle()
collector = TrainingDataCollector(list(game.word_list))

print("Collecting data from 10 games...")
collector.collect_training_data(num_games=1, k=10)

print(f"Collected {len(collector.training_data)} training examples")
print(f"Feature shape: {collector.training_data[0][0].shape}")
print(f"Label shape: {collector.training_data[0][1].shape}")