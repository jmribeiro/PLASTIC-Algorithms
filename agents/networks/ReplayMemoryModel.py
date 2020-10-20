import random
from abc import ABC, abstractmethod
from collections import deque

import pathlib

import numpy as np

from environment.PursuitState import PursuitState
from environment.utils import agent_directions


class ReplayMemoryModel(ABC):

    def __init__(self, name, trainable, layers, replay_batch_size, replay_memory_size):
        self._name = name
        self._replay_memory = deque(maxlen=replay_memory_size)
        self._replay_batch_size = replay_batch_size
        self._trainable = trainable
        self._input_layers = layers

    @property
    def name(self):
        return self._name

    @property
    def trainable(self):
        return self._trainable

    def remember(self, datapoint):
        self._replay_memory.append(datapoint)

    def replay_fit(self, datapoint):
        self.remember(datapoint)
        info = self.replay()
        info["replay buffer samples"] = len(self._replay_memory)
        return info

    def replay(self):
        if not self._trainable: return
        batch = random.sample(self._replay_memory, k=min(len(self._replay_memory), self._replay_batch_size))
        info = self.fit_batch(batch, False)
        return info

    @abstractmethod
    def fit_batch(self, batch, verbose):
        raise NotImplementedError()

    @abstractmethod
    def save(self, directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        if len(self._replay_memory) > 0:
            self.save_dataset(list(self._replay_memory), f"{directory}/memory.npy")

    def save_replay_memory(self, directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        if len(self._replay_memory) > 0:
            self.save_dataset(list(self._replay_memory), f"{directory}/memory.npy")

    def load_replay_memory(self, directory):
        try:
            memory = self.load_dataset(f"{directory}/memory.npy")
        except Exception:
            print(f"WARN: No previous replay memory found in {directory}", flush=True)
            memory = []
        [self.remember(datapoint) for datapoint in memory]

    def load(self, directory, load_memory=True):
        if load_memory:
            try: memory = self.load_dataset(f"{directory}/memory.npy")
            except Exception: memory = []
            [self.remember(datapoint) for datapoint in memory]
        self._load(directory)

    @abstractmethod
    def _load(self, directory):
        raise NotImplementedError()

    @staticmethod
    def parse_layer_blueprints(layer_blueprints):
        layers = []
        for bp in layer_blueprints:
            units, activation = bp.split(", ")
            layers.append((int(units), activation))
        return layers

    @staticmethod
    def save_dataset(dataset, file):

        m = len(dataset)
        D = np.zeros((m, 32))
        world_size = dataset[0][0].world_size

        for i, datapoint in enumerate(dataset):
            S, A, R, S_, T = datapoint

            s = S.features()
            A = [agent_directions()[a] for a in A] #FIXME
            a = tuple([a_coor for a in A for a_coor in a])
            s_ = S_.features()

            D[i, 0] = world_size[0]
            D[i, 1] = world_size[1]
            D[i, 2:12] = s
            D[i, 12:20] = a
            D[i, 20] = R
            D[i, 21: 31] = s_
            D[i, 31] = 1 if T else 0

        np.save(file, D)

    @staticmethod
    def load_dataset(file):
        D = np.load(file)

        assert D.shape[1] == 32

        world_size = (int(D[0][0]), int(D[0][1]))

        dataset = []

        for d in D:

            s = d[2:12]
            a = d[12:20]
            R = d[20]
            s_ = d[21: 31]
            T = True if d[31] == 1 else 0

            S = PursuitState.from_features(s, world_size)
            S_ = PursuitState.from_features(s_, world_size)

            A = []
            for i in range(4):
                direction = (int(a[i * 2]), int(a[i * 2 + 1]))
                action = agent_directions().index(direction)
                A.append(action)

            A = tuple(A)

            datapoint = S, A, R, S_, T
            dataset.append(datapoint)

        return dataset

    """
        def new_buffer(self):
            [self._old_replay_memory.append(x) for x in self._replay_memory]
            self._replay_memory = deque(maxlen=self._replay_memory.maxlen)

        @staticmethod
        def double_experience_replay(replay_memory, older_replay_memory, batch_size):

            if len(replay_memory) >= batch_size:
                # If we can sample from new one, sample from new one
                batch = random.sample(replay_memory, batch_size)
            else:
                # Take everything we can
                batch_part1 = list(replay_memory)
                missing = batch_size - len(batch_part1)
                batch_part2 = random.sample(older_replay_memory, min(len(older_replay_memory), missing))
                batch = batch_part1 + batch_part2
                random.shuffle(batch)

            return batch
        """
