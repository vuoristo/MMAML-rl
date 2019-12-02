import os

import numpy as np
from gym import utils
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_policy_search.envs.base import MetaEnv


class Reacher3Link2ModeEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.mode_locations = np.array(
            [[-0.225, -0.225], [0.225, 0.225]],
            dtype=np.float32)
        self.noise_scale = 0.10
        self.max_norm = 0.30
        self.set_task(self.sample_tasks(1)[0])
        asset_path = os.path.join(
            os.getcwd(),
            'meta_policy_search/envs/mujoco_envs/assets/reacher_3link.xml')
        MujocoEnv.__init__(self, asset_path, 2)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        goal_idx = np.random.randint(0, 2, n_tasks)
        goals = self.mode_locations[goal_idx]

        stddev = self.noise_scale

        goals = np.stack([
            np.random.normal(goals[:, 0], stddev, n_tasks),
            np.random.normal(goals[:, 1], stddev, n_tasks),
            np.ones((n_tasks)) * 0.03,
        ]).T

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
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("fingertip")
        goal_reward = -np.sum(np.abs(xposafter[:3] - self.goal_pos))
        ctrl_cost = .1 * np.square(a).sum()
        reward = goal_reward - ctrl_cost
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        geom_idx = self.model.geom_name2id("target")
        self.model.geom_pos[geom_idx][0:3] = self.goal_pos
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
    env = Reacher3Link2ModeEnv()
    while True:
        env.set_task(env.sample_tasks(1)[0])
        env.reset_model()
        for _ in range(100):
            env.render()
            obs, reward, _, _ = env.step(env.action_space.sample())  # take a random action