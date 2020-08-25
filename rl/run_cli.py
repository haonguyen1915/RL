from rl.trainer.trainer import DQNTrainer
import click
import os

PRJ_DIR = os.getcwd()


@click.group()
def entry_point():
    pass


@click.command()
@click.option('--mode', '-m', 'mode',
              type=click.Choice(['train', 'test', 'infer']),
              default='train',
              help="The run mode")
@click.option('--env', '-e', 'env',
              type=click.Choice(['CartPole-v0', 'LunarLander-v2']),
              default='CartPole-v0')
def run(mode, env):
    dqn_trainer = DQNTrainer(env=env)
    dqn_trainer.initialize()
    if mode == 'train':
        dqn_trainer.train()


entry_point.add_command(run)

if __name__ == "__main__":
    run()
