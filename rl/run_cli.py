from rl.trainer import trainer
from rl.trainer import gradient_policy_trainer as gp_trainer
import click
import os

PRJ_DIR = os.getcwd()


@click.group()
def entry_point():
    pass


@click.command()
@click.option('--net', '-n', 'net',
              type=click.Choice(['sarsa', 'dqn', 'double_dqn', 'per_dqn', 'q-learning',
                                 'duel_dqn', 'duel_double_dqn', 'dqn_cnn', 'noise_dqn',
                                 'dp_dqn', 'rainbow_dq']),
              default='sarsa',
              show_default=True,
              help="The net arch")
@click.option('--mode', '-m', 'mode',
              type=click.Choice(['train', 'test', 'test_env']),
              default='train',
              show_default=True,
              help="The run mode")
@click.option('--env', '-e', 'env',
              show_default=True,
              type=click.Choice(['CartPole-v0', 'LunarLander-v2', 'MsPacman-v0', 'Taxi-v3']),
              default='CartPole-v0')
def run(net, mode, env):
    if net == 'q-learning':
        dqn_trainer = trainer.QLearningTrainer(env=env)
    elif net == 'sarsa':
        dqn_trainer = trainer.SarsaTrainer(env=env)
    elif net == 'dqn':
        dqn_trainer = trainer.DQNTrainer(env=env)
    elif net == 'double_dqn':
        dqn_trainer = trainer.DoubleDQNTrainer(env=env)
    elif net == 'per_dqn':
        dqn_trainer = trainer.PerDQNTrainer(env=env)
    elif net == 'duel_dqn':
        dqn_trainer = trainer.DuelDQNTrainer(env=env)
    elif net == 'duel_double_dqn':
        dqn_trainer = trainer.DuelDoubleDQNTrainer(env=env)
    elif net == 'noise_dqn':
        dqn_trainer = trainer.NoiseDQNTrainer(env=env)
    elif net == 'dp_dqn':
        dqn_trainer = trainer.DistPerspectiveDQNTrainer(env=env)
    elif net == 'rainbow_dq':
        dqn_trainer = trainer.RainBowTrainer(env=env)
    else:
        dqn_trainer = trainer.DQNCNNTrainer(env=env)
    dqn_trainer.initialize()
    if mode == 'train':
        dqn_trainer.train()
    elif mode == 'test':
        dqn_trainer.test()
    elif mode == 'test_env':
        dqn_trainer.test_env()
    else:
        raise ValueError(f"mode {mode} not support!!")


@click.command()
@click.option('--net', '-n', 'net',
              type=click.Choice(['reinforce', 'reinforce_baseline', 'a2c']),
              default='a2c',
              show_default=True,
              help="The net arch")
@click.option('--mode', '-m', 'mode',
              type=click.Choice(['train', 'test', 'test_env']),
              default='train',
              show_default=True,
              help="The run mode")
@click.option('--env', '-e', 'env',
              show_default=True,
              type=click.Choice(['CartPole-v0', 'LunarLander-v2', 'MsPacman-v0', 'Taxi-v3']),
              default='CartPole-v0')
def run2(net, mode, env):
    if net == 'reinforce':
        dqn_trainer = gp_trainer.GradientPolicyTrainer(env=env)
    elif net == 'reinforce_baseline':
        dqn_trainer = gp_trainer.ReinforceBaselineTrainer(env=env)
    elif net == 'a2c':
        dqn_trainer = gp_trainer.A2CTrainer(env=env)
    else:
        raise NotImplementedError
    dqn_trainer.initialize()
    if mode == 'train':
        dqn_trainer.train()
    elif mode == 'test':
        dqn_trainer.test()
    elif mode == 'test_env':
        dqn_trainer.test_env()
    else:
        raise ValueError(f"mode {mode} not support!!")


entry_point.add_command(run)
entry_point.add_command(run2)

if __name__ == "__main__":
    run()
