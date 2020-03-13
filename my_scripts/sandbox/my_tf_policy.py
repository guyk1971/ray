import ray
from ray import tune
from ray.tune.logger import pretty_print


import tensorflow as tf
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

def postprocess_advantages(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    return compute_advantages(
        sample_batch, 0.0, policy.config["gamma"], use_gae=False, use_critic=False)



def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    print("====================== in policy_gradient_loss ============================")
    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) *
        train_batch[Postprocessing.ADVANTAGES])

def postprocess_advantages_(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    print("in postprocess_adv_")
    return sample_batch[sample_batch.REWARDS]


def policy_gradient_loss_(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) * train_batch[SampleBatch.REWARDS])

    # return -tf.reduce_mean(
    #     action_dist.logp(train_batch["actions"]) * train_batch["rewards"])


if __name__=='__main__':
    MyTFPolicy = build_tf_policy(
        name="MyTFPolicy",
        loss_fn=policy_gradient_loss,
        postprocess_fn=postprocess_advantages)

    MyTrainer = build_trainer(
        name="MyCustomTrainer",
        default_policy=MyTFPolicy)
    ray.init()
    tune.run(MyTrainer, config={"env": "CartPole-v0", "num_workers": 0, "num_gpus":1})
