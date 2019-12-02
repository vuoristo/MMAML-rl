import os

import numpy as np
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

from meta_policy_search.envs.base import MetaEnv
from meta_policy_search.utils import logger


class AntRandGoalMode1ModesEnv(MetaEnv, gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        self.goal_radius = 4.0
        self.noise_scale = 0.8
        self.mode_locations = []
        max_range = np.pi
        for i in range(1):
            ang = i * max_range / 3 - np.pi / 2
            self.mode_locations.append(
                [self.goal_radius * np.sin(ang),
                self.goal_radius * np.cos(ang)])
        self.mode_locations = np.array(self.mode_locations)

        self.set_task(self.sample_tasks(1)[0])
        asset_path = os.path.join(
            os.getcwd(),
            'meta_policy_search/envs/mujoco_envs/assets/ant_target.xml')
        MujocoEnv.__init__(self, asset_path, 2)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        goal_idx = np.arange(n_tasks) % len(self.mode_locations)
        goals = self.mode_locations[goal_idx]

        stddev = self.noise_scale

        goals = np.stack([
            np.random.normal(goals[:, 0], stddev, n_tasks),
            np.random.normal(goals[:, 1], stddev, n_tasks),
            np.ones((n_tasks)) * 0.03,
        ]).T

        return goals

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_pos = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_pos

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        goal_reward = -np.sum(np.abs(xposafter[:3] - self.goal_pos))
        ctrl_cost = .1 * np.square(a).sum()
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost + survive_reward
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_survive=survive_reward,
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        geom_idx = self.model.geom_name2id("target")
        self.model.geom_pos[geom_idx][0:3] = self.goal_pos
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        progs = [np.mean(path["env_infos"]["reward_forward"]) for path in paths]
        ctrl_cost = [-np.mean(path["env_infos"]["reward_ctrl"]) for path in paths]

        logger.logkv(prefix + 'AverageForwardReturn', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardReturn', np.max(progs))
        logger.logkv(prefix + 'MinForwardReturn', np.min(progs))
        logger.logkv(prefix + 'StdForwardReturn', np.std(progs))

        logger.logkv(prefix + 'AverageCtrlCost', np.mean(ctrl_cost))


if __name__ == "__main__":
    env = AntRandGoalMode1ModesEnv()
    while True:
        env.set_task(env.sample_tasks(1)[0])
        env.reset_model()
        for _ in range(50):
            env.render()
            obs, reward, _, _ = env.step(env.action_space.sample())  # take a random action