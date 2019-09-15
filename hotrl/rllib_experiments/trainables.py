import time
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count

import ray.rllib.optimizers as optimizers
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print

from hotrl.rllib_experiments.CustomSyncReplayOptimizer import CustomSyncReplayOptimizer


# Define a Trainable (executed on a separate thread)
def dqn_train(config, reporter):
    # Modify default optimizer to return the batch after each step
    config["optimizer_class"] = "CustomSyncReplayOptimizer"
    setattr(optimizers, "CustomSyncReplayOptimizer", CustomSyncReplayOptimizer)
    
    # Instantiate a trainer
    cfg = {
        # "n_step"                    : 3,
        # "buffer_size"               : 100000,
        # "sample_batch_size"         : 32,
        # "train_batch_size"          : 128,
        # "learning_starts"           : 5000,
        # "target_network_update_freq": 5000,
        "timesteps_per_iteration"   : 100,
        # "num_workers"               : cpu_count(),
        # "per_worker_exploration"    : True,
        # "worker_side_prioritization": True,
        # "min_iter_time_s"           : 1,
    }
    trainer = DQNTrainer(config={**config, **cfg}, env="House")
    
    # Modify training loop to receive batches from the optimizer
    # and return custom info in the training result dict
    def _custom_train(self):
        start_timestep = self.global_timestep
        
        # Update worker explorations
        exp_vals = [self.exploration0.value(self.global_timestep)]
        self.local_evaluator.foreach_trainable_policy(
            lambda p, _: p.set_epsilon(exp_vals[0]))
        for i, e in enumerate(self.remote_evaluators):
            exp_val = self.explorations[i].value(self.global_timestep)
            e.foreach_trainable_policy.remote(
                lambda p, _: p.set_epsilon(exp_val))
            exp_vals.append(exp_val)
        
        # Do optimization steps
        start = time.time()
        extra_metrics = defaultdict(lambda: defaultdict(list))
        metrics = ['extreme_penalty', 'comfort_penalty', 'cost']
        metrics += [f'{r}_temperature' for r in self.local_evaluator.env.rooms]
        while (
            self.global_timestep - start_timestep <
            self.config["timesteps_per_iteration"]
        ) or time.time() - start < self.config["min_iter_time_s"]:
            info_dict = self.optimizer.step()
            info_dict = info_dict.policy_batches['default_policy'].data
            for metric in metrics:
                for episode_id, info in zip(info_dict['eps_id'],
                                            info_dict['infos']):
                    extra_metrics[metric][str(episode_id)].append(
                        info[metric])
            
            self.update_target_if_needed()
        
        if self.config["per_worker_exploration"]:
            # Only collect metrics from the third of workers with lowest eps
            result = self.collect_metrics(
                selected_evaluators=self.remote_evaluators[
                                    -len(self.remote_evaluators) // 3:])
        else:
            result = self.collect_metrics()
        
        result.update(
            timesteps_this_iter=self.global_timestep - start_timestep,
            info=dict({
                "min_exploration"   : min(exp_vals),
                "max_exploration"   : max(exp_vals),
                "num_target_updates": self.num_target_updates,
            }, **self.optimizer.stats()))
        
        result['extra_metrics'] = extra_metrics
        
        return result
    
    trainer._train = partial(_custom_train, trainer)
    
    while True:
        result = trainer.train()  # Executes one training step
        # print(pretty_print(result))
        reporter(**result)  # notifies TrialRunner
