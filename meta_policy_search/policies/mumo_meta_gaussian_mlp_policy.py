import numpy as np
import tensorflow as tf

from meta_policy_search.policies.base import MetaPolicy
from meta_policy_search.policies.mumo_gaussian_mlp_policy import MumoGaussianMLPPolicy
from meta_policy_search.policies.networks.mumo_mlp import (
    forward_mlp,
    forward_modulated_mlp,
)
from meta_policy_search.utils import flat_to_padded_sequences


class MumoMetaGaussianMLPPolicy(MumoGaussianMLPPolicy, MetaPolicy):
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())  # store init arguments for serialization

        self.pre_update_action_var = None
        self.pre_update_mean_var = None
        self.pre_update_log_std_var = None

        self.mod_action_var = None
        self.mod_mean_var = None
        self.mod_log_std_var = None

        self.modulation_vector = None

        super(MumoMetaGaussianMLPPolicy, self).__init__(*args, **kwargs)

    def build_graph(self):
        """
        Builds computational graph for policy
        """

        # Create pre-update policy by calling build_graph of the super class
        super(MumoMetaGaussianMLPPolicy, self).build_graph()
        self.pre_update_action_var = tf.split(self.action_var, self.meta_batch_size)
        self.pre_update_mean_var = tf.split(self.mean_var, self.meta_batch_size)
        self.pre_update_log_std_var = [self.log_std_var for _ in range(self.meta_batch_size)]

        # Create lightweight policy graph that takes the policy parameters as placeholders
        with tf.variable_scope(self.name + "_ph_graph"):
            mean_network_phs_meta_batch, log_std_network_phs_meta_batch = [], []

            self.mod_action_var = []
            self.mod_mean_var = []
            self.mod_log_std_var = []

            # build meta_batch_size graphs for post-update policies --> thereby the policy parameters are placeholders
            obs_var_per_task = tf.split(
                self.obs_var, self.meta_batch_size, axis=0)

            for idx in range(self.meta_batch_size):
                with tf.variable_scope("task_%i" % idx):
                    # create mean network parameter placeholders
                    mean_network_phs = self._create_placeholders_for_vars(
                        scope=self.name + "/mean_network")  # -> returns ordered dict
                    mean_network_phs_meta_batch.append(mean_network_phs)

                    with tf.variable_scope("modulated_mean_network"):
                        # forward pass through the mean mpl
                        _, mod_mean_var = forward_modulated_mlp(
                            output_dim=self.action_dim,
                            hidden_sizes=self.hidden_sizes,
                            hidden_nonlinearity=self.hidden_nonlinearity,
                            output_nonlinearity=self.output_nonlinearity,
                            input_var=obs_var_per_task[idx],
                            mod_vars=self.mod_var[idx],
                            mlp_params=mean_network_phs,
                            use_betas=self.use_betas,
                            shift_gammas=self.shift_gammas,
                            mod_size=self.modulation_vector_size
                        )

                    with tf.variable_scope("log_std_network"):
                        # create log_stf parameter placeholders
                        log_std_network_phs = self._create_placeholders_for_vars(
                            scope=self.name + "/log_std_network")  # -> returns ordered dict
                        log_std_network_phs_meta_batch.append(
                            log_std_network_phs)

                        # weird stuff since log_std_network_phs is ordered dict
                        log_std_var = list(log_std_network_phs.values())[0]

                    mod_action_var = mod_mean_var + tf.random_normal(
                        shape=tf.shape(mod_mean_var)) * tf.exp(log_std_var)

                    self.mod_action_var.append(mod_action_var)
                    self.mod_mean_var.append(mod_mean_var)
                    self.mod_log_std_var.append(log_std_var)

            # merge mean_network_phs and log_std_network_phs into policies_params_phs
            self.policies_params_phs = []
            # Mutate mean_network_ph here
            for idx, odict in enumerate(mean_network_phs_meta_batch):
                odict.update(log_std_network_phs_meta_batch[idx])
                self.policies_params_phs.append(odict)

            self.policy_params_keys = list(self.policies_params_phs[0].keys())

    def switch_to_pre_update(self):
        self.modulation_vector = None
        super(MumoMetaGaussianMLPPolicy, self).switch_to_pre_update()

    def compute_modulation(self, data):
        """Computes modulation vectors for experience collection time and stores
        them for later use.
        """
        assert len(data) == self.meta_batch_size

        observations = []
        actions = []
        rewards = []

        rollouts_per_meta_task = [len(task["path_lengths"]) for task in data]
        max_rollouts_per_meta_task = max(rollouts_per_meta_task)

        for task in data:
            observations.append(
                flat_to_padded_sequences(
                    task["observations"],
                    task["path_lengths"],
                    max_rollouts=max_rollouts_per_meta_task,
                    max_path_length=self.max_path_length
                ))
            actions.append(
                flat_to_padded_sequences(
                    task["actions"],
                    task["path_lengths"],
                    max_rollouts=max_rollouts_per_meta_task,
                    max_path_length=self.max_path_length
                ))
            rewards.append(
                flat_to_padded_sequences(
                    np.expand_dims(task["rewards"], axis=2),
                    task["path_lengths"],
                    max_rollouts=max_rollouts_per_meta_task,
                    max_path_length=self.max_path_length))

        mod_data = np.concatenate([
                np.stack(observations),
                np.stack(actions),
                np.stack(rewards)],
            axis=3)

        sess = tf.get_default_session()
        mods = sess.run(self.mod_var, {
            self.mod_input_var: mod_data,
            self.num_paths_var: rollouts_per_meta_task})
        self.modulation_vector = mods
        self.rollouts_per_meta_task = rollouts_per_meta_task

        return mod_data

    def get_action(self, observation, task=0):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.repeat(np.expand_dims(
            np.expand_dims(observation, axis=0),
            axis=0),
            self.meta_batch_size, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[task][0], dict(
            mean=agent_infos[task][0]['mean'],
            log_std=agent_infos[task][0]['log_std'])
        return action, agent_infos

    def get_actions(self, observations):
        """
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        Returns:
            (tuple) : A tuple containing a list of numpy arrays of action, and a list of list of dicts of agent infos
        """
        assert len(observations) == self.meta_batch_size

        actions, agent_infos = self._get_actions(
            observations, pre_update=self._pre_update_mode)

        assert len(actions) == self.meta_batch_size
        return actions, agent_infos

    def _get_actions(self, observations, pre_update):
        assert self.policies_params_vals is not None

        obs_stack = np.concatenate(observations, axis=0)
        feed_dict = {self.obs_var: obs_stack}

        if self.modulation_vector is None:
            inputs = [
                self.pre_update_action_var,
                self.pre_update_mean_var,
                self.pre_update_log_std_var,
            ]
        else:
            inputs = [
                self.mod_action_var,
                self.mod_mean_var,
                self.mod_log_std_var,
            ]
            feed_dict.update({
                self.mod_var: self.modulation_vector,
                **self.policies_params_feed_dict})

        sess = tf.get_default_session()
        actions, means, log_stds = sess.run(inputs, feed_dict=feed_dict)
        # Get rid of fake batch size dimension (would be better to do this
        # in tf, if we can match batch sizes)
        log_stds = np.concatenate(log_stds)
        agent_infos = [
            [dict(mean=mean, log_std=log_stds[idx]) for mean in means[idx]]
            for idx in range(self.meta_batch_size)]
        return actions, agent_infos
