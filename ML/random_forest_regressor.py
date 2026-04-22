import pickle
from pathlib import Path

import numpy as np
from numpy import ndarray

from ML.base_model import BaseWordleModel
from sklearn.ensemble import RandomForestRegressor
from Utilities.game_state import GameState

_N_ESTIMATORS = 100


class RandomForestRegressorModel(BaseWordleModel):
    def __init__(self, word_list: list[str]):
        super().__init__(model_name="random_forest_regressor", word_list=word_list)
        self.model_path = Path('ML/saved_models/random_forest_regressor.pkl')
        self._model = RandomForestRegressor(n_estimators=_N_ESTIMATORS)

    def train(self) -> None:
        if self.model_path.exists():
            saved_bot = self.load(self.model_path)
            self._model = saved_bot._model
            self.is_trained = True
            return

        print("Training model...")
        with open('ML/training_data/wordle_training.pkl', 'rb') as f:
            training_data = pickle.load(f)

        x = np.array([example[0] for example in training_data])
        y = np.array([example[1] for example in training_data])

        # n_jobs > 1 only during fit — reset to 1 before saving to avoid conflicts
        # when this model is later used inside a multiprocessing Pool.
        self._model = RandomForestRegressor(n_estimators=_N_ESTIMATORS, n_jobs=4)
        self._model.fit(x, y)
        self._model.n_jobs = 1
        self.is_trained = True
        self.save(self.model_path, True)

    def predict(self, game_state: GameState) -> ndarray:
        features = self.engineer_features(game_state).reshape(1, -1)
        return self._model.predict(features)[0]
