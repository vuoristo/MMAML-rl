# Multimodal Model-Agnostic Meta-Learning for Reinforcement Learning

This project is an implementation of [**Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation**](https://arxiv.org/abs/1910.13616), which is published in [**NeurIPS 2019**](https://neurips.cc/Conferences/2019/). Visit [project page](https://vuoristo.github.io/MMAML/) for more information. Code for classification can be found at [MMAML-Classification](https://github.com/shaohua0116/MMAML-Classification).

Model-agnostic meta-learners aim to acquire meta-prior parameters from a distribution of tasks and adapt to novel tasks with few gradient updates. Yet, seeking a common initialization shared across the entire task distribution substantially limits the diversity of the task distributions that they are able to learn from. We propose a multimodal MAML (MMAML) framework, which is able to modulate its meta-learned prior according to the identified mode, allowing more efficient fast adaptation. An illustration of the proposed framework is as follows.

<p align="center">
    <img src="asset/model.png" width="360"/>
</p>

This implementation is based on and includes code from [ProMP](https://github.com/jonasrothfuss/ProMP).


## Installation / Dependencies
The code can be run in Anaconda or Virtualenv environments. For other installation methods refer to the [ProMP repository](https://github.com/jonasrothfuss/ProMP).

### Using Anaconda or Virtualenv

##### 1. Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions).

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

##### 2. Create either venv or conda environment and activate it

###### Virtualenv
```
pip install --upgrade virtualenv
virtualenv <venv-name>
source <venv-name>/bin/activate
```

###### Anaconda
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda create -n <env-name> python=3.6
source activate <env-name>
```

##### B.3. Install the required python dependencies
```
pip install -r requirements.txt
```

##### B.4. Set up the Mujoco physics engine and mujoco-py
For running the majority of the provided Meta-RL environments, the Mujoco physics engine as well as a
corresponding python wrapper are required.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py),
please follow the instructions [here](https://github.com/openai/mujoco-py).


## Running
Use the following commands to execute MMAML-rl algorithm.

```bash
python run_scripts/mumo_run_point_mass.py --config_file configs/point_env_momentum_dense.json
```

```bash
python run_scripts/mumo_run_mujoco.py --config_file configs/reacher.json
```

```bash
python run_scripts/mumo_run_mujoco.py --config configs/ant_rand_goal_mode.json
```

Use the following commands to execute ProMP algorithm.

```bash
python run_scripts/pro-mp_run_point_mass.py --config_file configs/point_env_momentum_dense.json
```

```bash
python run_scripts/pro-mp_run_mujoco.py --config_file configs/reacher.json
```

```bash
python run_scripts/pro-mp_run_mujoco.py --config configs/ant_rand_goal_mode.json
```

## Results

Please check out [our paper](https://arxiv.org/abs/1910.13616) for comprehensive results.

## Related work
- [ProMP: Proximal Meta-Policy Search](https://arxiv.org/abs/1810.06784) in ICLR 2019
- \[MAML\] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) in ICML 2017

## Cite the paper
If you find this useful, please cite
```
@inproceedings{vuorio2019multimodal,
  title={Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation},
  author={Vuorio, Risto and Sun, Shao-Hua and Hu, Hexiang and Lim, Joseph J.},
  booktitle={Neural Information Processing Systems},
  year={2019},
}
```