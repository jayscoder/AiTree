from __future__ import annotations
from typing import SupportsFloat

from gymnasium.core import ActType, ObsType

from core.nodes import *
import datetime
import utils
import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from utils import ACModel
import numpy
import multiprocessing
import gymnasium as gym


@register(
        props=[
            {
                'name'    : 'algo',
                'type'    : str,
                'default' : 'ppo',
                'required': False,
                'desc'    : 'The algorithm',
                'options' : ['ppo', 'a2c']
            },
            {
                'name'    : 'model',
                'type'    : str,
                'default' : '',
                'required': True,
                'desc'    : 'Model Name'
            },
            {
                'name'    : 'recurrence',
                'type'    : int,
                'default' : 1,
                'required': False,
                'desc'    : 'The recurrence'
            },
            {
                'name'    : 'log_interval',
                'type'    : int,
                'default' : 1,
                'required': False,
                'desc'    : 'number of updates between two logs (default: 1)'
            },
            {
                'name'    : 'save_interval',
                'type'    : int,
                'default' : 10,
                'required': False,
                'desc'    : 'number of updates between two saves (default: 10, 0 means no saving)'
            },
            {
                'name'    : 'procs',
                'type'    : int,
                'default' : 16,
                'required': False,
                'desc'    : 'number of processes (default: 16)'
            },
            {
                'name'    : 'epochs',
                'type'    : int,
                'default' : 4,
                'required': False,
                'desc'    : 'number of epochs for PPO (default: 4)'
            },
            {
                'name'    : 'batch_size',
                'type'    : int,
                'default' : 256,
                'required': False,
                'desc'    : 'batch size for PPO (default: 256)'
            },
            {
                'name'    : 'frames_per_proc',
                'type'    : int,
                'default' : None,
                'required': False,
                'desc'    : 'number of frames per process before update (default: 5 for A2C and 128 for PPO)'
            },
            {
                'name'    : 'discount',
                'type'    : float,
                'default' : 0.99,
                'required': False,
                'desc'    : 'discount factor (default: 0.99)'
            },
            {
                'name'    : 'lr',
                'type'    : float,
                'default' : 0.001,
                'required': False,
                'desc'    : 'learning rate (default: 0.001)'
            },
            {
                'name'    : 'gae_lambda',
                'type'    : float,
                'default' : 0.95,
                'required': False,
                'desc'    : 'lambda coefficient in GAE formula (default: 0.95, 1 means no gae)'
            },
            {
                'name'    : 'entropy_coef',
                'type'    : float,
                'default' : 0.01,
                'required': False,
                'desc'    : 'entropy term coefficient (default: 0.01)'
            },
            {
                'name'    : 'value_loss_coef',
                'type'    : float,
                'default' : 0.5,
                'required': False,
                'desc'    : 'value loss term coefficient (default: 0.5)'
            },
            {
                'name'    : 'max_grad_norm',
                'type'    : float,
                'default' : 0.5,
                'required': False,
                'desc'    : 'maximum norm of gradient (default: 0.5)'
            },
            {
                'name'    : 'optim_eps',
                'type'    : float,
                'default' : 1e-8,
                'required': False,
                'desc'    : 'Adam and RMSprop optimizer epsilon (default: 1e-8)'
            },
            {
                'name'    : 'optim_alpha',
                'type'    : float,
                'default' : 0.99,
                'required': False,
                'desc'    : 'RMSprop optimizer alpha (default: 0.99)'
            },
            {
                'name'    : 'clip_eps',
                'type'    : float,
                'default' : 0.2,
                'required': False,
                'desc'    : 'clipping epsilon for PPO (default: 0.2)'
            },
            {
                'name'    : 'recurrence',
                'type'    : int,
                'default' : 1,
                'required': False,
                'desc'    : 'number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.'
            },
            {
                'name'    : 'text',
                'type'    : bool,
                'default' : False,
                'required': False,
                'desc'    : 'add a GRU to the model to handle text input'
            },
            {
                'name'    : 'pause',
                'type'    : float,
                'default' : 0.1,
                'required': False,
                'desc'    : 'pause duration between two consequent actions of the agent (default: 0.1)'
            },
            {
                'name'    : 'argmax',
                'type'    : bool,
                'default' : False,
                'required': False,
                'desc'    : 'select the action with highest probability (default: False)'
            },
            {
                'name'    : 'shift',
                'type'    : int,
                'default' : 0,
                'required': False,
                'desc'    : 'number of times the environment is reset at the beginning (default: 0)'
            },
        ]
)
class RlNode(Node, gym.Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gym.Env.__init__(self)
        self.algo = self.get_prop('algo')
        self.recurrence = self.get_prop('recurrence')
        self.save_interval = self.get_prop('save_interval')
        self.log_interval = self.get_prop('log_interval')

        self.mem = self.recurrence > 1

        self.custom_actions = []
        self.custom_goal = self.find_node(RlGoal)
        rl_action = self.find_node(RlAction)
        if rl_action is not None:
            self.custom_actions = rl_action.children

    def init_env(self):
        self.observation_space = self.simulation.observation_space  # 观察空间
        if len(self.custom_actions) > 0:
            self.action_space = gym.spaces.Discrete(len(self.custom_actions))  # 动作空间
        else:
            self.action_space = self.simulation.action_space

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        if len(self.custom_actions) > 0:
            start_i = len(self.simulation.step_results)
            action_node = self.custom_actions[action]
            self.status = action_node.tick()
            for i in range(start_i, len(self.simulation.step_results)):
                reward += self.simulation.step_results[i].reward
            if self.status == FAILURE:
                reward -= 0.01
            last_step_result = self.simulation.step_results[-1]
            obs = last_step_result.obs
            terminated = last_step_result.terminated
            truncated = last_step_result.truncated
            info = last_step_result.info
        else:
            obs, reward, terminated, truncated, info = self.simulation.step(action)
            if info['is_changed']:
                self.status = SUCCESS
            else:
                self.status = FAILURE

        if self.custom_goal is not None:
            goal_status = self.custom_goal.execute()
            info['goal_status'] = goal_status
            if goal_status == SUCCESS:
                reward += 1
            else:
                reward -= 0.01
        return obs, reward, terminated, truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return self.simulation.reset(seed=seed, options=options)

    def lazy_init(self):
        # 定义环境的观察空间和动作空间
        self.init_env()

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        default_model_name = f"{self.simulation.env_id}_{self.algo}_seed{self.simulation.seed}_{date}"

        self.model_name = self.get_prop('model') or default_model_name
        self.model_dir = os.path.join(self.simulation.workspace, self.model_name)

        # Load loggers and Tensorboard writer

        self.txt_logger = utils.get_txt_logger(self.model_dir)
        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir)
        self.tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        # Load environments

        self.envs = self.get_rl_envs()

        if self.simulation.train:
            self.init_train()
        else:
            self.init_evaluate()

    def get_rl_envs(self):
        return [self]

    def init_evaluate(self):
        self.agent = utils.Agent(
                self.envs[0].observation_space,
                self.envs[0].action_space,
                self.model_dir,
                argmax=self.get_prop('argmax'),
                use_memory=self.mem,
                use_text=self.get_prop('text'))

    def init_train(self):
        try:
            self.train_status = utils.get_status(self.model_dir)
        except OSError:
            self.train_status = { "num_frames": 0, "update": 0 }
        self.txt_logger.info("Training status loaded\n")

        # Load observations preprocessor

        self.obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.envs[0].observation_space)
        if "vocab" in self.train_status:
            self.preprocess_obss.vocab.load_vocab(self.train_status["vocab"])
        self.txt_logger.info("Observations preprocessor loaded")

        # Load model

        self.acmodel = ACModel(self.obs_space, self.envs[0].action_space, self.mem, self.get_prop('text'))
        if "model_state" in self.train_status:
            self.acmodel.load_state_dict(self.train_status["model_state"])
        self.acmodel.to(device)
        self.txt_logger.info("Model loaded\n")
        self.txt_logger.info("{}\n".format(self.acmodel))

        # Load algo

        if self.algo == "a2c":
            self.algo = torch_ac.A2CAlgo(self.envs,
                                         self.acmodel, device,
                                         self.get_prop('frames_per_proc'),
                                         self.get_prop('discount'),
                                         self.get_prop('lr'),
                                         self.get_prop('gae_lambda'),
                                         self.get_prop('entropy_coef'),
                                         self.get_prop('value_loss_coef'),
                                         self.get_prop('max_grad_norm'),
                                         self.get_prop('recurrence'),
                                         self.get_prop('optim_alpha'),
                                         self.get_prop('optim_eps'),
                                         self.preprocess_obss)
        elif self.algo == "ppo":
            self.algo = torch_ac.PPOAlgo(self.envs,
                                         self.acmodel,
                                         device,
                                         self.get_prop('frames_per_proc'),
                                         self.get_prop('discount'),
                                         self.get_prop('lr'),
                                         self.get_prop('gae_lambda'),
                                         self.get_prop('entropy_coef'),
                                         self.get_prop('value_loss_coef'),
                                         self.get_prop('max_grad_norm'),
                                         self.get_prop('recurrence'),
                                         self.get_prop('optim_eps'),
                                         self.get_prop('clip_eps'),
                                         self.get_prop('epochs'),
                                         self.get_prop('batch_size'),
                                         self.preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(self.algo))

        if "optimizer_state" in self.train_status:
            self.algo.optimizer.load_state_dict(self.train_status["optimizer_state"])
        self.txt_logger.info("Optimizer loaded\n")

        # Train model

        self.num_frames = self.train_status["num_frames"]
        self.update = self.train_status["update"]
        self.start_time = time.time()

    def execute_train(self):
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = self.algo.collect_experiences()
        logs2 = self.algo.update_parameters(exps)
        logs = { **logs1, **logs2 }
        update_end_time = time.time()

        self.num_frames += logs["num_frames"]
        self.update += 1

        # Print logs

        if self.update % self.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - self.start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [self.update, self.num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            self.txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if self.train_status["num_frames"] == 0:
                self.csv_logger.writerow(header)
            self.csv_logger.writerow(data)
            self.csv_file.flush()

            for field, value in zip(header, data):
                self.tb_writer.add_scalar(field, value, self.num_frames)

        # Save status
        if self.save_interval > 0 and self.update % self.save_interval == 0:
            status = {
                "num_frames" : self.num_frames, "update": self.update,
                "model_state": self.acmodel.state_dict(), "optimizer_state": self.algo.optimizer.state_dict()
            }
            if hasattr(self.preprocess_obss, "vocab"):
                status["vocab"] = self.preprocess_obss.vocab.vocab
            utils.save_status(status, self.model_dir)
            self.txt_logger.info("Status saved")

        return self.status

    def execute_evaluate(self):
        obs = self.simulation.step_results[-1].obs
        action = self.agent.get_action(obs)
        obs, reward, terminated, truncated, info = self.envs[0].step(action)
        self.agent.analyze_feedback(reward, self.simulation.done)

        if len(self.custom_actions) > 0:
            return self.status
        else:
            if info['is_changed']:
                return SUCCESS
            return FAILURE

    def execute(self) -> NODE_STATUS:
        if self.simulation.train:
            return self.execute_train()
        else:
            return self.execute_evaluate()


@register()
class RlGoal(And):
    pass


@register()
class RlAction(Parallel):
    pass
