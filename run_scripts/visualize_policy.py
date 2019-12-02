import argparse
import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import sys
import joblib
import tensorflow as tf
from PIL import Image
import moviepy.editor as mpy

from meta_policy_search.utils import logger
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline

from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_target_loc import HalfCheetahTargetLocationEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_target_speed import HalfCheetahTargetSpeedEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal_mode import AntRandGoalModeEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal_mode_2_modes import AntRandGoalMode2ModesEnv
from meta_policy_search.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from meta_policy_search.envs.point_envs.point_env_2d_walls import MetaPointEnvWalls
from meta_policy_search.envs.point_envs.point_env_2d_momentum import MetaPointEnvMomentum
from meta_policy_search.envs.mujoco_envs.walker2d_target_location import Walker2DTargetLocation
from meta_policy_search.envs.mujoco_envs.reacher import ReacherEnv
from meta_policy_search.envs.mujoco_envs.reacher_2_mode import Reacher2ModeEnv
from meta_policy_search.envs.mujoco_envs.reacher_6_mode import Reacher6ModeEnv
from meta_policy_search.envs.mujoco_envs.reacher_3_link import Reacher3LinkEnv
from meta_policy_search.envs.mujoco_envs.reacher_3_link_2_mode import Reacher3Link2ModeEnv
from meta_policy_search.envs.mujoco_envs.reacher_3_link_6_mode import Reacher3Link6ModeEnv
from meta_policy_search.envs.mujoco_envs.reacher_3_link_1_mode import Reacher3Link1ModeEnv

from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.policies.mumo_meta_gaussian_mlp_policy import MumoMetaGaussianMLPPolicy
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_algos.mumo_pro_mp import MumoProMP

WIDTH = 600
HEIGHT = 600

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_path', type=str, default=None,
                        help='path to checkpoint')
    parser.add_argument(
        '--overrides', type=json.loads, default={},
        help='accepts json for overriding training parameters')
    parser.add_argument('--video_filename', default=None)
    parser.add_argument('--num_trajs', type=int, default=10)
    args = parser.parse_args(sys.argv[1:])

    params_path = os.path.join(
        os.path.split(args.restore_path)[0], 'params.json')

    with open(params_path, 'r') as f:
        params = json.load(f)

    params.update(args.overrides)

    baseline = LinearFeatureBaseline()

    env = globals()[params['env']]() # instantiate env
    env = normalize(env) # apply normalize wrapper to env

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True # pylint: disable=E1101
    sess = tf.Session(config=gpu_config)

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=params['meta_batch_size'],
        hidden_sizes=params['hidden_sizes'],
        cell_size=params['cell_size'],
        rollouts_per_meta_task=params['rollouts_per_meta_task'],
        max_path_length=params['max_path_length'],
        use_betas=params['use_betas'],
        shift_gammas=params['shift_gammas'],
    )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        # This batch_size is confusing
        rollouts_per_meta_task=params['rollouts_per_meta_task'],
        meta_batch_size=params['meta_batch_size'],
        max_path_length=params['max_path_length'],
        parallel=params['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=params['discount'],
        gae_lambda=params['gae_lambda'],
        normalize_adv=params['normalize_adv'],
    )

    algo = ProMP(
        policy=policy,
        inner_lr=params['inner_lr'],
        meta_batch_size=params['meta_batch_size'],
        num_inner_grad_steps=params['num_inner_grad_steps'],
        learning_rate=params['learning_rate'],
        num_ppo_steps=params['num_promp_steps'],
        clip_eps=params['clip_eps'],
        target_inner_step=params['target_inner_step'],
        init_inner_kl_penalty=params['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=params['adaptive_inner_kl_penalty'],
    )

    saver = tf.train.Saver()

    if args.restore_path is not None:
        logger.log('Restoring parameters from {}'.format(args.restore_path))
        saver.restore(sess, args.restore_path)
        logger.log('Restored')

    uninit_vars = [
        var for var in tf.global_variables()
        if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))

    wrapped_env = env
    while hasattr(wrapped_env, '_wrapped_env'):
        wrapped_env = wrapped_env._wrapped_env

    frame_skip = wrapped_env.frame_skip if hasattr(wrapped_env, 'frame_skip') else 1
    assert hasattr(wrapped_env, 'dt'), 'environment must have dt attribute that specifies the timestep'
    timestep = wrapped_env.dt
    speedup = 1

    with sess.as_default() as sess:
        policy.switch_to_pre_update()

        # Preupdate:
        tasks = env.sample_tasks(params['meta_batch_size'])
        sampler.vec_env.set_tasks(tasks)

        # Preupdate:
        for i in range(params['num_inner_grad_steps']):
            paths = sampler.obtain_samples(log=False)
            samples_data = sample_processor.process_samples(
                paths, log=True, log_prefix='%i_' % i)
            env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % i)
            algo._adapt(samples_data)

        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(
            paths, log=True, log_prefix='%i_' % params['num_inner_grad_steps'])
        env.log_diagnostics(
            sum(list(paths.values()), []),
            prefix='%i_' % params['num_inner_grad_steps'])
        logger.dumpkvs()
        images = []

        # Postupdate:
        for _ in range(args.num_trajs):
            task_i = np.random.choice(range(params['meta_batch_size']))
            env.set_task(tasks[task_i])
            print(tasks[task_i])
            obs = env.reset()
            for _ in range(params['max_path_length']):
                action, _ = policy.get_action(obs, task_i)
                obs, reward, done, _ = env.step(action)
                time.sleep(0.001)
                if done:
                    break

                if args.video_filename is not None:
                    image = env.sim.render(WIDTH, HEIGHT)
                    time.sleep(timestep*frame_skip / speedup)
                    pil_image = Image.frombytes('RGB', (WIDTH, HEIGHT), image)
                    images.append(np.flipud(np.array(pil_image)))

    if args.video_filename is not None:
        fps = int(speedup/timestep * frame_skip)
        clip = mpy.ImageSequenceClip(images, fps=fps)
        if args.video_filename[-3:] == 'gif':
            clip.write_gif(args.video_filename, fps=fps)
        else:
            clip.write_videofile(args.video_filename, fps=fps)