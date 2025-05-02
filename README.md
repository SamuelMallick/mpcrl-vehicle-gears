# Learning-Based MPC for Fuel Efficient Control of Autonomous Vehicles with Discrete Gear Selection

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/mpcrl-vehicle-gears/blob/main/LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Learning-Based MPC for Fuel Efficient Control of Autonomous Vehicles with Discrete Gear Selection](https://arxiv.org/abs/2503.11359) submitted to [IEEE Control Systems Letters (L-CSS)](https://ieee-cssletters.dei.unipd.it/index.php).

In this work we propose a learning-based model predictive controller for co-optimization of vehicle speed and gear-shift schedule of an autonomous vehicle.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2025learning,
  title={Learning-Based MPC for Fuel Efficient Control of Autonomous Vehicles with Discrete Gear Selection},
  author={Mallick, Samuel and Battocletti, Gianpietro and Dong, Qizhang and Dabiri, Azita and De Schutter, Bart},
  journal={arXiv preprint arXiv:2503:11359},
  year={2025}
}
```

---

## Installation

The code was created with `Python 3.13`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/mpcrl-vehicle-gears
cd mpcrl-vehicle-gears
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`run.py`** is the main python file to start and configure simulations.
- **`vehicle.py`** contains the vehicle model.
- **`network.py`** contains the classes required for neural network function approximators.
- **`mpc.py`** contains the classes for all mpc controllers.
- **`env.py`** contains the class for simulating the base model/environment.
- **`agents.py`** contains the classes for (learning) agents who use various MPCs to act in the environment.
- **`visualisation`** contains scripts for generating the figures used in Learning-Based Model Predictive Control for Efficient Control of Autonomous Vehicles.
- **`utils`** contains auxillary scrips and classes.
- **`results`** contains .pkl files for the data used in Learning-Based Model Predictive Control for Efficient Control of Autonomous Vehicles. Supervised learning data is not available due to file size limits, but is available upon request.
- **`config_files`** contains configuration files used for launching different simulations.

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/mpcrl-vehicle-gears/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2024 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-vehicle-gearse” (Learning-Based MPC for Fuel Efficient Control of Autonomous Vehicles with Discrete Gear Selection) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.