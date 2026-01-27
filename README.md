# Reinforcement Learning with Distributed MPC for Fuel-Efficient Platoon Control with Discrete Gear Transitions

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/mpcrl-vehicle-gears/blob/main/LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Reinforcement Learning with Distributed MPC for Fuel-Efficient Platoon Control with Discrete Gear Transitions](https://arxiv.org/abs/2601.18294) submitted to [IEEE Transactions on Intelligent Transportation Systems (T-ITS)](https://ieee-itss.org/pub/t-its/).

In this work we propose a learning-based  distributed model predictive controller for co-optimization of vehicle speed and gear-shift schedule in vehicle platoons.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2025learning,
  title={Reinforcement Learning with Distributed MPC for Fuel-Efficient Platoon Control with Discrete Gear Transitions},
  author={Mallick, Samuel and Battocletti, Gianpietro and Boskos, Dimitris and Dabiri, Azita and De Schutter, Bart},
  journal={arXiv preprint arXiv:2601.18294},
  year={2026}
}
```

---

## Installation

The code was created with `Python 3.13`. To access it, clone the repository

```bash
git clone -branch paper-2026 --single-branch https://github.com/SamuelMallick/mpcrl-vehicle-gears
cd mpcrl-vehicle-gears
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`agents`** contains classes for various agents, each managing the gear-shift schedule and MPC controller in different ways, e.g., deep Q-learning agent.
- **`config_files`** contains files that describe different test configurations.
- **`mpcs`** contains the classes for all mpc controllers.
- **`run_platoon`** contains launching scripts for each multi-vehicle scenario.
- **`run_single_vehicle`** contains launching scripts for each single-vehicle scenario.
- **`utils`** contains utility functions.
- **`visualisation`** contains scripts for creating the plots used in the paper.
- **`env.py`** is the class that simulates the underlying physical system.
- **`network.py`** contains the classes for building the deep Q-networks used in reinforcement learning.
- **`vehicle.py`** is the class that describes vehicle dynamics.

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/mpcrl-vehicle-gears/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

[Gianpietro Battocletti](https://www.tudelft.nl/staff/g.battocletti/), PhD Candidate [g.battocletti@tudelft.nl]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2026 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-vehicle-gearse” (Reinforcement Learning with Distributed MPC for Fuel-Efficient Platoon Control with Discrete Gear Transitions) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.