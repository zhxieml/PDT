import torch
import numpy as np
import random


MAX_EPISODE_LEN = 1000


class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
        max_future_len
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale
        self.max_future_len = max_future_len

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood. We clamp them to be within
        # the user defined action range.
        self.action_range = action_range

    def __call__(self, traj, end_idx=None):
        if end_idx is None:
            end_idx = random.randint(1, traj["rewards"].shape[0])
            start_idx = max(0, end_idx - self.max_len)

        # get sequences from dataset
        ss = traj["observations"][start_idx : end_idx].reshape(-1, self.state_dim)
        aa = traj["actions"][start_idx : end_idx].reshape(-1, self.act_dim)
        rr = traj["rewards"][start_idx : end_idx].reshape(-1, 1)
        if "terminals" in traj:
            dd = traj["terminals"][start_idx : end_idx]  # .reshape(-1)
        else:
            dd = traj["dones"][start_idx : end_idx]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(start_idx, end_idx)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        rtg = discount_cumsum(traj["rewards"][start_idx:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        ss = (ss - self.state_mean) / self.state_std

        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])

        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)

        fss = traj["observations"][end_idx : end_idx + self.max_future_len].reshape(-1, self.state_dim)
        faa = traj["actions"][end_idx : end_idx + self.max_future_len].reshape(-1, self.act_dim)
        frr = traj["rewards"][end_idx : end_idx + self.max_future_len].reshape(-1, 1)
        if "terminals" in traj:
            fdd = traj["terminals"][end_idx : end_idx + self.max_future_len]  # .reshape(-1)
        else:
            fdd = traj["dones"][end_idx : end_idx + self.max_future_len]  # .reshape(-1)

        # get the total length of a trajectory
        ftlen = fss.shape[0]

        ftimesteps = np.arange(end_idx, end_idx + ftlen)  # .reshape(-1)
        fordering = np.arange(ftlen)
        if ftlen:
            fordering[ftimesteps >= MAX_EPISODE_LEN] = -1
            fordering[fordering == -1] = fordering.max()
        ftimesteps[ftimesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        frtg = discount_cumsum(traj["rewards"][end_idx:], gamma=1.0)[: ftlen + 1].reshape(
            -1, 1
        )
        if frtg.shape[0] <= ftlen:
            frtg = np.concatenate([frtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        fact_len = faa.shape[0]
        if ftlen != fact_len:
            raise ValueError

        fss = np.concatenate([np.zeros((self.max_future_len - ftlen, self.state_dim)), fss])
        fss = (fss - self.state_mean) / self.state_std
        faa = np.concatenate([np.zeros((self.max_future_len - ftlen, self.act_dim)), faa])
        frr = np.concatenate([np.zeros((self.max_future_len - ftlen, 1)), frr])
        fdd = np.concatenate([np.ones((self.max_future_len - ftlen)) * 2, fdd])
        frtg = (
            np.concatenate([np.zeros((self.max_future_len - ftlen, 1)), frtg])
            * self.reward_scale
        )
        ftimesteps = np.concatenate([np.zeros((self.max_future_len - ftlen)), ftimesteps])
        fordering = np.concatenate([np.zeros((self.max_future_len - ftlen)), fordering])
        fpadding_mask = np.concatenate([np.zeros(self.max_future_len - ftlen), np.ones(ftlen)])

        fss = torch.from_numpy(fss).to(dtype=torch.float32)
        faa = torch.from_numpy(faa).to(dtype=torch.float32).clamp(*self.action_range)
        frr = torch.from_numpy(frr).to(dtype=torch.float32)
        fdd = torch.from_numpy(fdd).to(dtype=torch.long)
        frtg = torch.from_numpy(frtg).to(dtype=torch.float32)
        ftimesteps = torch.from_numpy(ftimesteps).to(dtype=torch.long)
        fordering = torch.from_numpy(fordering).to(dtype=torch.long)
        fpadding_mask = torch.from_numpy(fpadding_mask)

        return (
            ss, aa, rr, dd, rtg, timesteps, ordering, padding_mask,
            fss, faa, frr, fdd, frtg, ftimesteps, fordering, fpadding_mask
        )

def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    weighted_by,
    max_future_len,
    num_workers=4,
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size, weighted_by=weighted_by)

    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
        max_future_len=max_future_len
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
    )


def discount_cumsum(x, gamma):
    if x.size == 0:
        return np.array([[] for _ in range(x.shape[0])])
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size, weighted_by="return"):
    if weighted_by == "length":
        traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
        p_sample = traj_lens / np.sum(traj_lens)
    elif weighted_by == "return":
        traj_returns = np.array([np.sum(traj["rewards"]) for traj in trajectories])
        traj_returns -= np.min(traj_returns)
        traj_returns += 1e-6
        p_sample = traj_returns / np.sum(traj_returns)
    else:
        raise NotImplementedError

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds
