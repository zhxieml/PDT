import argparse
import os
from pathlib import Path
import pickle
import random
import time

import d4rl
import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import create_dataloader
from src.models.decision_transformer import DecisionTransformer
from src.evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from src.lamb import Lamb
from src.logger import Logger
from src.replay_buffer import ReplayBuffer
from src.trainer import SequenceTrainer
from src import utils


MAX_EPISODE_LEN = 1000


def get_env_builder(seed, env_name, target_goal=None):
    def make_env_fn():
        env = gym.make(env_name)
        env.seed(seed)
        if hasattr(env.env, "wrapped_env"):
            env.env.wrapped_env.seed(seed)
        elif hasattr(env.env, "seed"):
            env.env.seed(seed)
        else:
            pass
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        if target_goal:
            env.set_target_goal(target_goal)
            print(f"Set the target goal to be {env.target_goal}")
        return env

    return make_env_fn

def get_target_goal(env_name):
    if "antmaze" in env_name:
        env = gym.make(env_name)
        target_goal = env.target_goal
        env.close()
        print(f"Generated the fixed target goal: {target_goal}")
    else:
        target_goal = None

    return target_goal

class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["data_dir"],
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], [])

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
            kl_div_weight=variant["kl_div_weight"],
            num_future_samples=variant["num_future_samples"],
            sample_topk=variant["sample_topk"],
            mask_future=variant["mask_future"]
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        ) if self.model.stochastic_policy else None

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0

        if variant["model_path_prefix"] is not None:
            self._load_model(variant["model_path_prefix"], variant["model_name"])

        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant) if not self.variant["disable_log"] else None

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict() if self.log_temperature_optimizer is not None else None,
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix, name):
        name = "model" if name is None else name

        if Path(f"{path_prefix}/{name}.pt").exists():
            with open(f"{path_prefix}/{name}.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                print("Optimizer state dict not loaded")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/{name}.pt")
        else:
            raise ValueError(f"Checkpoint {path_prefix}/{name}.pt not found!")

    def _load_dataset(self, data_dir, env_name):
        dataset_path = os.path.join(data_dir, f"{env_name}.pkl")
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        pretrain=True,
    ):
        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                pretrain=pretrain,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        buffer_returns = [np.sum(traj["rewards"]) for traj in self.replay_buffer.trajectories]

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
            "aug_traj/buffer_return_mean": np.mean(buffer_returns),
            "aug_traj/buffer_return_std": np.std(buffer_returns),
        }

    def pretrain(self, eval_envs):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                pretrain=True,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            pretrain=True
        )

        writer = (
            SummaryWriter(self.logger.log_path) if not self.variant["disable_log"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                max_future_len=self.variant["future_K"],
                weighted_by="length"
            )

            train_outputs = trainer.train_iteration(
                loss_fn=trainer.pretrain_loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)

            if not self.variant["disable_log"]:
                self.logger.log_metrics(
                    outputs,
                    iter_num=self.pretrain_iter,
                    total_transitions_sampled=self.total_transitions_sampled,
                    writer=writer,
                )
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=True,
                )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"] \
            if "evaluation/return_mean_gm" in outputs \
            else outputs["evaluation/return_mean"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs):
        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            pretrain=False
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                pretrain=False,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if not self.variant["disable_log"] else None
        )

        # collecting warmup trajectories
        warmup_envs = SubprocVecEnv(
            [
                get_env_builder(
                    i + 100, env_name=self.variant["env"],
                    target_goal=get_target_goal(self.variant["env"]),
                )
                for i in range(10)
            ]
        )
        while self.total_transitions_sampled < self.variant["online_warmup_samples"]:
            self._augment_trajectories(
                warmup_envs,
                pretrain=True
            )
            print(
                f"{self.total_transitions_sampled} samples collected: " +
                f"mean return={np.mean([np.sum(traj['rewards']) for traj in self.replay_buffer.trajectories])} " +
                f"std return={np.std([np.sum(traj['rewards']) for traj in self.replay_buffer.trajectories])}"
            )

        warmup_transitions_sampled = self.total_transitions_sampled

        # warming up return models
        dataloader = create_dataloader(
            trajectories=self.replay_buffer.trajectories,
            num_iters=self.variant["num_updates_per_online_iter"],
            batch_size=self.variant["batch_size"],
            max_len=self.variant["K"],
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            state_mean=self.state_mean,
            state_std=self.state_std,
            reward_scale=self.reward_scale,
            action_range=self.action_range,
            max_future_len=self.variant["future_K"],
            weighted_by="return"
        )
        warmup_optimizer = torch.optim.Adam(
            self.model.predict_return_prior.parameters()
        )

        print("Warming up return predictor...")
        for _ in range(self.variant["return_warmup_iters"]):
            for _, trajs in enumerate(dataloader):
                train_outputs = trainer.return_warmup_step_stochastic(
                    loss_fn=trainer.return_warmup_loss_fn,
                    trajs=trajs,
                    optimizer=warmup_optimizer
                )

        if self.variant["fix_prior"]:
            for name, param in self.model.predict_prior.named_parameters():
                param.requires_grad = False
                print(f"fixing {name}...")

        if self.variant["fix_allbutreturn"]:
            for name, param in self.model.named_parameters():
                if "predict_return_prior" not in name:
                    param.requires_grad = False
                    print(f"fixing {name}...")

        # online finetuning
        while self.online_iter < self.variant["max_online_iters"] and self.total_transitions_sampled - warmup_transitions_sampled < 210_000:
            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                pretrain=False
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                max_future_len=self.variant["future_K"],
                weighted_by="return"
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=trainer.finetune_loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            if not self.variant["disable_log"]:
                self.logger.log_metrics(
                    outputs,
                    iter_num=self.pretrain_iter + self.online_iter,
                    total_transitions_sampled=self.total_transitions_sampled - warmup_transitions_sampled,
                    writer=writer,
                )

                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                )

            self.online_iter += 1

    def eval_only(self, eval_envs):
        print("\n\n\n*** Eval Only ***")

        if self.variant["record_video"]:
            from pyvirtualdisplay import Display

            virtual_display = Display(visible=0, size=(1400, 900))
            virtual_display.start()

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                record_video=self.variant["record_video"],
                pretrain=self.variant["eval_pretrained"]
            )
        ]

        eval_outputs, eval_reward = self.evaluate(eval_fns)
        print(self.variant["env"], eval_reward, eval_outputs["evaluation/return_std_gm"])

    def run(self):
        utils.set_seed_everywhere(args.seed)

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        target_goal = get_target_goal(env_name)
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(
                    i, env_name=env_name, target_goal=target_goal,
                )
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        if self.variant["eval_only"]:
            eval_envs = DummyVecEnv([
                get_env_builder(
                    i, env_name=env_name, target_goal=target_goal,
                )
                for i in range(self.variant["num_eval_episodes"])
            ])
            eval_envs.metadata["render_modes"].append("rgb_array")  # this fixs missing metadata of d4rl

            assert self.variant["model_path_prefix"] is not None, "Must provide a model path to evaluate"
            self.eval_only(eval_envs)

            eval_envs.close()
            return

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(
                        i + 100, env_name=env_name, target_goal=target_goal,
                    )
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_log", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="res")
    parser.add_argument("--exp_name", type=str, default="default")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    parser.add_argument("--future_K", type=int, default=20)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)
    parser.add_argument("--kl_div_weight", type=float, default=1)
    parser.add_argument("--mask_future", action="store_true")

    # shared evaluation options
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_pretrained", action="store_true")
    parser.add_argument("--model_path_prefix", type=str, default=None)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--model_name", type=str, default=None)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--online_warmup_samples", type=int, default=10000)
    parser.add_argument("--return_warmup_iters", type=int, default=5)
    parser.add_argument("--num_future_samples", type=int, default=256)
    parser.add_argument("--sample_topk", type=int, default=1)
    parser.add_argument("--fix_prior", action="store_true")
    parser.add_argument("--fix_allbutreturn", action="store_true")

    args = parser.parse_args()
    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment.run()
