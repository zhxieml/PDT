"""
This file was adapted from the following:
* https://github.com/facebookresearch/online-dt/blob/main/decision_transformer/models/decision_transformer.py
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

The first one is licensed under the CC BY-NC, whereas the remaining two are licensed under the MIT License.
"""

import torch
import torch.nn as nn

import transformers

import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd

from src.models.model import TrajectoryModel
from src.models.trajectory_gpt2 import GPT2Model


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)

class Gaussian:
    """ Represents a gaussian distribution """
    # TODO: implement a dict conversion function
    def __init__(self, mu, log_sigma=None):
        """

        :param mu:
        :param log_sigma: If none, mu is divided into two chunks, mu and log_sigma
        """
        if log_sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb; pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, -1)

        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) if isinstance(log_sigma, torch.Tensor) else \
                            np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None

    @staticmethod
    def ten2ar(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif np.isscalar(tensor):
            return tensor
        elif hasattr(tensor, 'to_numpy'):
            return tensor.to_numpy()
        else:
            import pdb; pdb.set_trace()
            raise ValueError('input to ten2ar cannot be converted to numpy array')

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
               / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussian distribution."""
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    @staticmethod
    def stack(*argv, dim):
        return Gaussian._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return Gaussian._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussian(mu, log_sigma)

    def average(self, dists):
        """Fits single Gaussian to a list of Gaussians."""
        mu_avg = torch.stack([d.mu for d in dists]).sum(0) / len(dists)
        sigma_avg = torch.stack([d.mu ** 2 + d.sigma ** 2 for d in dists]).sum(0) - mu_avg**2
        return type(self)(mu_avg, torch.log(sigma_avg))

    def chunk(self, *args, **kwargs):
        return [type(self)(chunk) for chunk in torch.chunk(self.tensor(), *args, **kwargs)]

    def view(self, shape):
        self.mu = self.mu.view(shape)
        self.log_sigma = self.log_sigma.view(shape)
        self._sigma = self.sigma.view(shape)
        return self

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self.log_sigma[item])

    def tensor(self):
        return torch.cat([self.mu, self.log_sigma], dim=-1)

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(self.mu.detach(), self.log_sigma.detach())

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return type(self)(self.ten2ar(self.mu), self.ten2ar(self.log_sigma))


class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

    @staticmethod
    def stack(*argv, dim):
        return MultivariateGaussian(Gaussian.stack(*argv, dim=dim).tensor())

    @staticmethod
    def cat(*argv, dim):
        return MultivariateGaussian(Gaussian.cat(*argv, dim=dim).tensor())


class DiagGaussian(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super().__init__()

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_std = nn.Linear(hidden_dim, z_dim)

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, x):
        mu = self.mu(x)
        log_std = self.log_std(x)

        return MultivariateGaussian(mu, log_std)

class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        z_dim=16,
        kl_div_weight=1,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        num_future_samples=256,
        sample_topk=1,
        mask_future=False,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.num_future_samples = num_future_samples
        self.sample_topk = sample_topk
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_future = torch.nn.Linear(z_dim, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # for anti-causal transformer
        self.future_transformer = FutureTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            z_dim=z_dim,
            max_length=max_length,
            max_ep_len=max_ep_len,
            **kwargs
        )

        self.kl_div_weight = kl_div_weight
        self.predict_prior = DiagGaussian(hidden_size, z_dim)
        self.predict_return_prior = DiagGaussian(hidden_size * 2, 1)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range
        self.mask_future = mask_future

        if stochastic_policy:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        timesteps,
        ordering,
        padding_mask=None,
        fstates=None,
        factions=None,
        ftimesteps=None,
        fordering=None,
        fpadding_mask=None,
        futures=None,
        pretrain=True,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        if pretrain:
            predicted_return = None
            z_prior = self.predict_prior(F.relu(state_embeddings))
            z_fixed_prior = Gaussian(torch.zeros_like(z_prior.mu), torch.zeros_like(z_prior.log_sigma))

            if futures is None:  # training
                # auto-encoded latent embeddings
                z_posterior = self.future_transformer(states=fstates, actions=factions, timesteps=ftimesteps, attention_mask=fpadding_mask)
                z_random = z_posterior.sample()
                z_random *= padding_mask.unsqueeze(-1).repeat(1, 1, self.z_dim)
                z_embeddings = self.embed_future(z_random)
            else:  # evaluation
                z_posterior = None
                z_random = z_prior.sample()
                futures[:, -1] = z_random[:, -1]
                z_embeddings = self.embed_future(futures)
        else:
            z_prior, z_posterior, z_fixed_prior = None, None, None

            if futures is None:  # training
                z_prior = self.predict_prior(F.relu(state_embeddings))
                z_fixed_prior = Gaussian(torch.zeros_like(z_prior.mu), torch.zeros_like(z_prior.log_sigma))
                z_posterior = self.future_transformer(states=fstates, actions=factions, timesteps=ftimesteps, attention_mask=fpadding_mask)
                z_random = z_posterior.sample()
                z_random *= padding_mask.unsqueeze(-1).repeat(1, 1, self.z_dim)
                z_embeddings = self.embed_future(z_random)

                predicted_return = self.predict_return_prior(
                    F.relu(
                        torch.cat(
                            (
                                state_embeddings,
                                z_embeddings
                            ),
                            dim=-1
                        )
                    )
                )
            else:  # evaluation
                z_prior = self.predict_prior(F.relu(state_embeddings[:, -1]))
                z_prior_repeated = MultivariateGaussian.cat(*([z_prior] * self.num_future_samples), dim=0)
                new_future_repeated = z_prior_repeated.sample()
                new_z_embedding_repeated = self.embed_future(new_future_repeated)
                predicted_return_repeated = self.predict_return_prior(
                    F.relu(
                        torch.cat(
                            (
                                state_embeddings[:, -1].repeat(self.num_future_samples, 1),
                                new_z_embedding_repeated
                            ),
                            dim=-1
                        )
                    )
                ).mu
                predicted_return_repeated = predicted_return_repeated.view(self.num_future_samples, batch_size, -1)
                new_future_repeated = new_future_repeated.view(self.num_future_samples, batch_size, -1)
                selected_idxs = predicted_return_repeated.topk(k=self.sample_topk, dim=0).indices[-1]  # (batch_size, 1)
                selected_future = torch.gather(new_future_repeated, 0, selected_idxs.unsqueeze(0).repeat(1, 1, self.z_dim)).squeeze(0)  # (batch_size, z_dim)
                predicted_return = torch.gather(predicted_return_repeated, 0, selected_idxs.unsqueeze(0)).squeeze(0)  # (batch_size, 1)
                futures[:, -1] = selected_future
                z_embeddings = self.embed_future(futures)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        if self.mask_future:
            z_embeddings = torch.zeros_like(state_embeddings)

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        z_embeddings = z_embeddings + order_embeddings

        # this makes the sequence look like (s_1, z_1, a_1, s_2, z_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (state_embeddings, z_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        if self.mask_future:
            future_mask = torch.zeros_like(padding_mask)
        else:
            future_mask = padding_mask

        stacked_padding_mask = (
            torch.stack((padding_mask, future_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # states (0), futures (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        action_preds = self.predict_action(x[:, 1])

        return {
            "action_preds": action_preds,
            "z_prior": z_prior,
            "z_posterior": z_posterior,
            "z_fixed_prior": z_fixed_prior,
            "futures": futures,
            "predicted_return": predicted_return
        }

    def get_predictions(
        self, states, actions, timesteps, futures, pretrain=True, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        futures = futures.reshape(num_envs, -1, self.z_dim)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            futures = futures[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            futures = torch.cat(
                [
                    torch.zeros(
                        (
                            futures.shape[0],
                            self.max_length - futures.shape[1],
                            self.z_dim,
                        ),
                        device=futures.device,
                    ),
                    futures,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        outputs = self.forward(
            states,
            actions,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            futures=futures,
            pretrain=pretrain,
            **kwargs
        )

        future_res = outputs["futures"][:, -1] if outputs["futures"] is not None else None

        if self.stochastic_policy:
            action_res = outputs["action_preds"]
        else:
            action_res = self.clamp_action(outputs["action_preds"][:, -1])

        return {
            "action_dist": action_res,
            "future": future_res,
        }

    def clamp_action(self, action):
        return action.clamp(*self.action_range)

class FutureTransformer(TrajectoryModel):
    """
    Using second transformer as anti-causal aggregator
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            z_dim,
            ordering=0,
            max_length=None,
            max_ep_len=4096,
            **kwargs
    ):
        super().__init__(state_dim, act_dim=act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)
        self.ordering = ordering

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_z = DiagGaussian(hidden_size, z_dim)

    def forward(self, states, actions, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        if self.ordering:
            time_embeddings = self.embed_timestep(timesteps)
        else:
            time_embeddings = 0.0

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (s_0, a_0, s_1, a_1, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to
        # predicting states (1)
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        z_preds = self.predict_z(x[:,1])

        return z_preds
