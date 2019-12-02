import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_policy_search.envs.base import MetaEnv


class Reacher6ModeEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        a = np.sqrt(0.125 ** 2 + 0.125 ** 2) * 1.2
        self.corners = np.array([
            [a, 0],
            [a / 2, np.sqrt(3) * a / 2],
            [-a / 2, np.sqrt(3) * a / 2],
            [-a / 2, -np.sqrt(3) * a / 2],
            [a / 2, -np.sqrt(3) * a / 2],
            [-a, 0],
        ], dtype=np.float32)
        self.noise_scale = 0.025
        self.max_norm = 0.25
        self.set_task(self.sample_tasks(1)[0])
        asset_path = os.path.join(
            os.getcwd(),
            'meta_policy_search/envs/mujoco_envs/assets/reacher.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 2)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        # goal_idx = np.arange(n_tasks) % len(self.corners)
        goal_idx = np.random.randint(0, 6, n_tasks)
        goals = self.corners[goal_idx]
        goals += np.random.normal(
            0, self.noise_scale, n_tasks * 2).reshape((-1, 2))
        # clip by norm
        goal_norm = np.linalg.norm(goals, axis=1)
        goals[goal_norm > self.max_norm] = (
            goals[goal_norm > self.max_norm]
            * (self.max_norm / goal_norm[goal_norm>self.max_norm]
                ).reshape((-1, 1))
        )
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
        vec = self.get_body_com("fingertip")[:2] - self.goal_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
        ])


if __name__ == "__main__":
    env = Reacher6ModeEnv()
    while True:
        env.set_task(env.sample_tasks(1)[0])
        env.reset_model()
        for _ in range(50):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action