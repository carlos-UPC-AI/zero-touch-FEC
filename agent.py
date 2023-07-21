# DEPENDENCIES
import os
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3 import DQN, PPO, A2C
from environment import Environment
from torch.optim import Adam
from torch.nn import ReLU
from config import (
    ENV_SMALL, ENV_MEDIUM, ENV_LARGE, BACKGROUND_VEHICLES, SAVE_FREQ,
    EVAL_EPISODES, PROJECT_DIR, LOGS, TRAINING_ROUNDS,
    CHECKPOINTS, MODELS, TIME_STEPS
)

# GLOBALS
ENV = ENV_SMALL # select from: ENV_SMALL, ENV_MEDIUM or ENV_LARGE
# Select from the three scenarios (small, medium, large)
ENVIRONMENT = Environment(scenario=ENV['scenario'],
                          edges_cost=ENV['edges_cost'],
                          network=ENV['network'],
                          fec_coverage=ENV['fec_coverage'],
                          state_features=ENV['state_features'],
                          no_up_nodes=ENV['no_up_nodes'],
                          no_down_nodes=ENV['no_down_nodes'],
                          no_left_nodes=ENV['no_left_nodes'],
                          no_right_nodes=ENV['no_right_nodes'],
                          nodes=ENV['nodes'],
                          route='random',  # Select from random: [[1, 16], [9, 12], [4, 13]], any_route or specific
                          # route: [source, target]
                          vehicles_num=BACKGROUND_VEHICLES)


#  FUNCTIONS
def training_fn(training_round, dqn_model):
    # Save a checkpoint every 50000 steps
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ,
                                             save_path=PROJECT_DIR + CHECKPOINTS,
                                             name_prefix="DQN_r" + str(training_round),
                                             save_replay_buffer=False,
                                             save_vecnormalize=False)

    # Training DQN model
    if not os.path.isfile(PROJECT_DIR + MODELS + 'DQN.zip'):
        dqn_model.learn(total_timesteps=TIME_STEPS, log_interval=500, callback=checkpoint_callback, progress_bar=True)
        dqn_model.save(PROJECT_DIR + MODELS + 'DQN.zip')
    # Retrieve model to keep training
    else:
        dqn_model = DQN.load(PROJECT_DIR + MODELS + 'DQN.zip', ENVIRONMENT)
        dqn_model.learn(total_timesteps=TIME_STEPS, log_interval=500, callback=checkpoint_callback, progress_bar=True)
        dqn_model.save(PROJECT_DIR + MODELS + 'DQN.zip')
        dqn_model.save(PROJECT_DIR + MODELS + 'DQN_r' + str(training_round))

    ENVIRONMENT.close()


def evaluate_fn():
    eval_model = DQN.load(PROJECT_DIR + MODELS + 'DQN.zip', ENVIRONMENT)
    for episode in range(EVAL_EPISODES):
        obs = ENVIRONMENT.reset()
        print(f"\n\nEPISODE: {episode}\n\n*** INITIAL STATE ***")
        print('\nVNF:', obs.tolist())
        ENVIRONMENT.render()
        done = False
        step = 0
        while not done:
            print('\n-- STEP:', step, '--')
            step += 1
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, done, info = ENVIRONMENT.step(action)
            print('obs:', obs.tolist())
            ENVIRONMENT.render()
            print('\nreward:', reward, '\ndone:', done, '\ninfo', info, '\n ')
    ENVIRONMENT.close()


# DQN model
model = DQN(
    policy="MlpPolicy",
    env=ENVIRONMENT,
    learning_rate=0.0003,
    buffer_size=1000000,
    tensorboard_log=PROJECT_DIR + LOGS,
    batch_size=128,
    gamma=0.99,
    device='auto',
    verbose=0,
    target_update_interval=500,
    exploration_fraction=0.5,
    exploration_final_eps=0.0,
    train_freq=4,
    policy_kwargs=dict(net_arch=[256, 128],
                       activation_fn=ReLU,
                       optimizer_class=Adam,
                       )
)

# Train DQN model
for tr in range(TRAINING_ROUNDS):
    # Train model
    training_fn(tr, model)
    # Evaluate model
    evaluate_fn()
