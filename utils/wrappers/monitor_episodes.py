from collections import deque
from time import perf_counter
from typing import Any, Deque, Optional, SupportsFloat, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env, Wrapper, utils

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MonitorEpisodes(
    Wrapper[ObsType, ActType, ObsType, ActType], utils.RecordConstructorArgs
):
    """This wrapper keeps track of
        - observations
        - actions
        - costs/rewards
        - episode length
        - episode execution time
    that the environment is subject to during the learning process.
    """

    def __init__(
        self, env: Env[ObsType, ActType], deque_size: Optional[int] = None
    ) -> None:
        """This wrapper will keep track of observations, actions and rewards as well as
        episode length and execution time.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to apply the wrapper to.
        deque_size : int, optional
            The maximum number of episodes to hold as historical data in the internal
            deques. By default, `None`, i.e., unlimited.
        """
        utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        Wrapper.__init__(self, env)
        # long-term storages
        self.observations: Deque[npt.NDArray[ObsType]] = deque(  # type: ignore
            maxlen=deque_size
        )
        self.actions: Deque[npt.NDArray[ActType]] = deque(  # type: ignore
            maxlen=deque_size
        )
        self.rewards: Deque[npt.NDArray[np.floating]] = deque(maxlen=deque_size)
        self.episode_lengths: Deque[int] = deque(maxlen=deque_size)
        self.exec_times: Deque[float] = deque(maxlen=deque_size)
        # current-episode-storages
        self.ep_observations: list[ObsType] = []
        self.ep_actions: list[ActType] = []
        self.ep_rewards: list[SupportsFloat] = []
        self.t0: float = perf_counter()
        self.ep_length: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment and resets the current data accumulators."""
        observation, info = super().reset(seed=seed, options=options)
        self._clear_ep_data()
        self.ep_observations.append(observation)
        return observation, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # sourcery skip: extract-method
        """Steps through the environment, accumulating the episode data."""
        output = super().step(action)
        obs, reward, terminated, truncated, info = output

        # accumulate data
        self.ep_observations.append(obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_length += 1

        # if episode is done, save the current data to history
        if terminated or truncated:
            # append data
            self.observations.append(np.asarray(self.ep_observations))
            self.actions.append(np.asarray(self.ep_actions))
            self.rewards.append(np.asarray(self.ep_rewards))
            self.episode_lengths.append(self.ep_length)
            self.exec_times.append(perf_counter() - self.t0)

            # clear this episode's data
            self._clear_ep_data()

        return obs, reward, terminated, truncated, info

    def _clear_ep_data(self) -> None:
        # clear this episode's lists and reset counters
        self.ep_observations.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()
        self.t0 = perf_counter()
        self.ep_length = 0
