from meta_policy_search.utils import (
    logger,
    tensorboard_util,
    create_feed_dict,
)
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer

import tensorflow as tf
import numpy as np
from collections import OrderedDict

class MumoProMP(MAMLAlgo):
    """
    ProMP Algorithm

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for optimizing the meta-objective
        num_ppo_steps (int): number of ProMP steps (without re-sampling)
        num_minibatches (int): number of minibatches for computing the ppo gradient steps
        clip_eps (float): PPO clip range
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for annealing clip_eps. If anneal_factor < 1, clip_eps <- anneal_factor * clip_eps at each iteration
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable

    """
    def __init__(
            self,
            *args,
            name="ppo_maml",
            learning_rate=1e-3,
            num_ppo_steps=5,
            num_minibatches=1,
            clip_eps=0.2,
            target_inner_step=0.01,
            init_inner_kl_penalty=1e-2,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            summary_writer=None,
            **kwargs
            ):
        super(MumoProMP, self).__init__(*args, **kwargs)

        self.optimizer = MAMLPPOOptimizer(learning_rate=learning_rate, max_epochs=num_ppo_steps, num_minibatches=num_minibatches)
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = init_inner_kl_penalty * np.ones(self.num_inner_grad_steps)
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps
        self.summary_writer = summary_writer
        self.log_step = 0

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                   dist_info_old_sym, dist_info_new_sym)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
        return surr_obj_adapt

    def _make_modulation_phs(self, prefix):
        modulation_vector_phs = []
        for task_id in range(self.meta_batch_size):
            ph = tf.placeholder(
                dtype=tf.float32,
                shape=[None, sum(self.policy.modulation_vector_size)],
                name='modulation_vector' + '_' + prefix + '_' + str(task_id))
            # all_phs_dict['%s_task%i_%s'%(prefix, task_id, 'meta_input')] = ph
            modulation_vector_phs.append(ph)

        return modulation_vector_phs

    def _build_inner_adaption(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        (obs_phs, action_phs, adv_phs, dist_info_old_phs,
         adapt_input_ph_dict) = self._make_input_placeholders('adapt')

        adapted_policies_params = []
        mod_adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    distribution_info_new = self.policy.distribution_info_sym(
                        mod_var=None,
                        obs_var=obs_phs[i],
                        params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    surr_obj_adapt = self._adapt_objective_sym(
                        action_phs[i], adv_phs[i], dist_info_old_phs[i],
                        distribution_info_new)

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("adapt_step"):
                    adapted_policy_param = self._adapt_sym(
                        surr_obj_adapt, self.policy.policies_params_phs[i])
                adapted_policies_params.append(adapted_policy_param)

                with tf.variable_scope("mod_adapt_objective"):
                    distribution_info_new = self.policy.distribution_info_sym(
                        self.modulation_ph_dict[i],
                        obs_phs[i],
                        params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    surr_obj_adapt = self._adapt_objective_sym(
                        action_phs[i], adv_phs[i], dist_info_old_phs[i],
                        distribution_info_new)

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("mod_adapt_step"):
                    mod_adapted_policy_param = self._adapt_sym(
                        surr_obj_adapt, self.policy.policies_params_phs[i])
                mod_adapted_policies_params.append(mod_adapted_policy_param)

        return (
            adapted_policies_params,
            mod_adapted_policies_params,
            adapt_input_ph_dict,
        )

    def build_graph(self):
        """
        Creates the computation graph
        """

        """ Create Variables """
        with tf.variable_scope(self.name):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.modulation_ph_dict = self._make_modulation_phs('adapt')
            (self.adapted_policies_params,
             self.mod_adapted_policies_params,
             self.adapt_input_ph_dict) = self._build_inner_adaption()

            """ ----- Build graph for the meta-update ----- """
            self.meta_op_phs_dict = OrderedDict()
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step0')
            self.meta_op_phs_dict.update(all_phs_dict)

            distribution_info_vars, current_policy_params = [], []
            all_surr_objs, all_inner_kls = [], []

        """ Pre step. This constructs the unmodulated, pre-update graph. """
        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(
                mod_var=None, obs_var=obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)
            current_policy_params.append(self.policy.policy_params)

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, kls, adapted_policy_params = [], [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])
                    kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))

                    adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

                    adapted_policy_params.append(adapted_params_var)
                    kls.append(kl_loss)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                all_inner_kls.append(kls)

                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [
                    self.policy.distribution_info_sym(
                        self.policy.mod_var[i], obs_phs[i],
                        params=adapted_policy_params[i])
                    for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params

            # per step: compute mean of kls over tasks
            mean_inner_kl_per_step = tf.stack([tf.reduce_mean(tf.stack(inner_kls)) for inner_kls in all_inner_kls])

            """ Outer objective """
            surr_objs, outer_kls = [], []

            # Create placeholders
            inner_kl_coeff = tf.placeholder(tf.float32, shape=[self.num_inner_grad_steps], name='inner_kl_coeff')
            self.meta_op_phs_dict['inner_kl_coeff'] = inner_kl_coeff

            clip_eps_ph = tf.placeholder(tf.float32, shape=[], name='clip_eps')
            self.meta_op_phs_dict['clip_eps'] = clip_eps_ph

            # meta-objective
            for i in range(self.meta_batch_size):
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_phs[i], dist_info_old_phs[i],
                                                                                 distribution_info_vars[i])
                outer_kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))

                # clipped likelihood ratio
                clipped_obj = tf.minimum(likelihood_ratio * adv_phs[i],
                                         tf.clip_by_value(likelihood_ratio,
                                                          1 - clip_eps_ph,
                                                          1 + clip_eps_ph) * adv_phs[i])
                surr_obj = - tf.reduce_mean(clipped_obj)


                surr_objs.append(surr_obj)
                outer_kls.append(outer_kl)

            mean_outer_kl = tf.reduce_mean(tf.stack(outer_kls))
            inner_kl_penalty = tf.reduce_mean(inner_kl_coeff * mean_inner_kl_per_step)

            """ Mean over meta tasks """
            meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0)) + inner_kl_penalty

            self.optimizer.build_graph(
                loss=meta_objective,
                target=self.policy,
                input_ph_dict=self.meta_op_phs_dict,
                inner_kl=mean_inner_kl_per_step,
                outer_kl=mean_outer_kl,
            )

    def optimize_policy(
        self, all_samples_data, mod_samples_data, num_paths_per_rollout,
        log=True):
        """
        Performs MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        meta_op_input_dict = self._extract_input_dict_meta_op(
            all_samples_data, self._optimization_keys)

        extra_feed_dict = {
            self.policy.mod_input_var: mod_samples_data,
            self.policy.num_paths_var: num_paths_per_rollout,
        }

        # add kl_coeffs / clip_eps to meta_op_input_dict
        meta_op_input_dict['inner_kl_coeff'] = self.inner_kl_coeff

        meta_op_input_dict['clip_eps'] = self.clip_eps

        if log: logger.log("Optimizing")

        loss_before, grad_norms = self.optimizer.optimize(
            input_val_dict=meta_op_input_dict,
            extra_feed_dict=extra_feed_dict)
        if self.summary_writer is not None:
            for name, norm in grad_norms.items():
                tensorboard_util.log_scalar(
                    self.summary_writer, 'grads/' + name, norm, self.log_step)
            self.log_step += 1

        if log: logger.log("Computing statistics")
        loss_after, inner_kls, outer_kl = self.optimizer.compute_stats(
            input_val_dict=meta_op_input_dict,
            extra_feed_dict=extra_feed_dict)

        if self.adaptive_inner_kl_penalty:
            if log: logger.log("Updating inner KL loss coefficients")
            self.inner_kl_coeff = self.adapt_kl_coeff(
                self.inner_kl_coeff,
                inner_kls,
                self.target_inner_step)

        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('KLInner', np.mean(inner_kls))
            logger.logkv('KLCoeffInner', np.mean(self.inner_kl_coeff))

    def adapt_kl_coeff(self, kl_coeff, kl_values, kl_target):
        if hasattr(kl_values, '__iter__'):
            assert len(kl_coeff) == len(kl_values)
            return np.array([_adapt_kl_coeff(kl_coeff[i], kl, kl_target) for i, kl in enumerate(kl_values)])
        else:
            return _adapt_kl_coeff(kl_coeff, kl_values, kl_target)

    def _adapt(self, samples, extra_feed_dict={}):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(samples) == self.meta_batch_size
        assert [sample_dict.keys() for sample_dict in samples]
        sess = tf.get_default_session()

        # prepare feed dict
        input_dict = self._extract_input_dict(
            samples, self._optimization_keys, prefix='adapt')
        input_ph_dict = self.adapt_input_ph_dict

        feed_dict_inputs = create_feed_dict(
            placeholder_dict=input_ph_dict, value_dict=input_dict)
        feed_dict_params = self.policy.policies_params_feed_dict

        feed_dict = {**feed_dict_inputs, **feed_dict_params, **extra_feed_dict}

        if len(extra_feed_dict) == 0:
            param_phs = self.adapted_policies_params
        else:
            param_phs = self.mod_adapted_policies_params

        # compute the post-update / adapted policy parameters
        adapted_policies_params_vals = sess.run(
            param_phs, feed_dict=feed_dict)

        # store the new parameter values in the policy
        self.policy.update_task_parameters(adapted_policies_params_vals)


def _adapt_kl_coeff(kl_coeff, kl, kl_target):
    if kl < kl_target / 1.5:
        kl_coeff /= 2

    elif kl > kl_target * 1.5:
        kl_coeff *= 2
    return kl_coeff
