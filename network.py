from collections import deque, namedtuple
import numpy as np
import torch.nn as nn

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently.

    Parameters
    ----------
    capacity : int
        The maximum number of transitions that can be stored in the memory."""

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(
        self, batch_size: int, np_random: np.random.Generator
    ) -> list[Transition]:
        index_samples = np_random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in index_samples]

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    """A deep recurrent Q-network (DRQN) that maps state sequences to Q-values for
    gear shifts. The architecture is a (bidirectional) RNN followed by a fully connected
    layer.

    Parameters
    ----------
    input_size : int
        The size of the input state vector.
    hidden_size : int
        The number of features in the hidden state of the RNN.
    num_actions : int, optional
        The number of actions that can be taken, by default 3.
        Three actions : downshift, no shift, upshift.
    num_layers : int, optional
        The number of recurrent layers, by default 1."""

    def __init__(
        self, input_size, hidden_size, num_actions=3, num_layers=1, bidirectional=False
    ):
        super(DRQN, self).__init__()
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_actions
        )
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        drqn_out, _ = self.rnn(x)
        q_values = self.fc(drqn_out)
        return q_values
