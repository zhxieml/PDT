import numpy as np
import torch

MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    vec_env,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    use_mean=False,
    record_video=False,
    pretrain=True,
):
    def eval_episodes_fn(model):
        returns, lengths, _ = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            record_video=record_video,
            pretrain=pretrain,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    use_mean=False,
    record_video=False,
    pretrain=True,
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    if record_video:
        # from gym.wrappers.monitoring.video_recorder import VideoRecorder
        from stable_baselines3.common.vec_env import VecVideoRecorder

        vec_env = VecVideoRecorder(vec_env, video_folder="recording", record_video_trigger=lambda x: x == 0, video_length=max_ep_len,
                       name_prefix="random-agent")

        vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    futures = torch.zeros(0, device=device, dtype=torch.float32)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        futures = torch.cat(
            [
                futures,
                torch.zeros((num_envs, model.z_dim), device=device).reshape(
                    num_envs, -1, model.z_dim
                ),
            ],
            dim=1,
        )
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        assert futures is not None

        prediction_outputs = model.get_predictions(   # add context-length padding here
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            futures=futures.to(dtype=torch.float32),
            num_envs=num_envs,
            pretrain=pretrain
        )

        if model.stochastic_policy:
            if use_mean:
                action = prediction_outputs["action_dist"].mean.reshape(num_envs, -1, act_dim)[:, -1]
            else:
                # the return action is a SquashNormal distribution
                action = prediction_outputs["action_dist"].sample().reshape(num_envs, -1, act_dim)[:, -1]
        else:
            action = prediction_outputs["action_dist"]
        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        # replace the placeholder once the action is taken and the reward is revealed
        if prediction_outputs["future"] is not None:
            futures[:, -1] = prediction_outputs["future"]
        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )
