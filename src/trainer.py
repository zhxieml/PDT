import numpy as np
import torch
import time
from tqdm import tqdm
from collections import defaultdict


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        pretrain=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.pretrain = pretrain
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        logged_info = defaultdict(list)
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss_outputs = self.train_step_stochastic(loss_fn, trajs)
            for k, v in loss_outputs.items():
                logged_info[k].append(v)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(logged_info["loss"])
        logs["training/train_loss_std"] = np.std(logged_info["loss"])
        for k, v in logged_info.items():
            if k == "loss" or len(v) == 0: continue
            logs[f"training/{k}"] = v[-1]
        if self.model.stochastic_policy:
            logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
            fstates,
            factions,
            frewards,
            fdones,
            frtg,
            ftimesteps,
            fordering,
            fpadding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)
        fstates = fstates.to(self.device)
        factions = factions.to(self.device)
        fdones = fdones.to(self.device)
        frtg = frtg.to(self.device)
        ftimesteps = ftimesteps.to(self.device)
        fordering = fordering.to(self.device)
        fpadding_mask = fpadding_mask.to(self.device)

        action_target = torch.clone(actions)
        rtg_target = torch.clone(rtg)

        outputs = self.model.forward(
            states,
            actions,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            fstates=fstates,
            factions=factions,
            ftimesteps=ftimesteps,
            fordering=fordering,
            fpadding_mask=fpadding_mask,
            pretrain=self.pretrain,
        )

        loss_outputs = loss_fn(
            outputs,
            action_target,
            rtg_target[:, :-1],
            padding_mask,
            self.model.temperature().detach() if self.model.stochastic_policy else 0.0,  # no gradient taken here
            self.model.kl_div_weight,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if self.log_temperature_optimizer is not None:
            self.log_temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss = (
                self.model.temperature() * (loss_outputs["entropy"] - self.model.target_entropy).detach()
            )
            if temperature_loss.requires_grad:
                temperature_loss.backward()
            self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v.detach().cpu().item() for k, v in loss_outputs.items()}

    def return_warmup_step_stochastic(self, loss_fn, trajs, optimizer):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
            fstates,
            factions,
            frewards,
            fdones,
            frtg,
            ftimesteps,
            fordering,
            fpadding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)
        fstates = fstates.to(self.device)
        factions = factions.to(self.device)
        fdones = fdones.to(self.device)
        frtg = frtg.to(self.device)
        ftimesteps = ftimesteps.to(self.device)
        fordering = fordering.to(self.device)
        fpadding_mask = fpadding_mask.to(self.device)

        action_target = torch.clone(actions)
        rtg_target = torch.clone(rtg)

        outputs = self.model.forward(
            states,
            actions,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            fstates=fstates,
            factions=factions,
            ftimesteps=ftimesteps,
            fordering=fordering,
            fpadding_mask=fpadding_mask,
            pretrain=self.pretrain,
        )

        loss_outputs = loss_fn(
            outputs,
            action_target,
            rtg_target[:, :-1],
            padding_mask,
            self.model.temperature().detach() if self.model.stochastic_policy else 0.0,  # no gradient taken here
            self.model.kl_div_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_outputs["loss"].backward()
        optimizer.step()

        return {k: v.detach().cpu().item() for k, v in loss_outputs.items()}

    def pretrain_loss_fn(
        self,
        outputs,
        a,
        rtg,
        attention_mask,
        entropy_reg,
        z_reg,
    ):
        if hasattr(outputs["action_preds"], "log_likelihood"):
            log_likelihood = outputs["action_preds"].log_likelihood(a)[attention_mask > 0].mean()
            entropy = outputs["action_preds"].entropy().mean()
        else:
            log_likelihood = -torch.mean((outputs["action_preds"] - a)[attention_mask > 0] ** 2)
            entropy = torch.zeros(1, device=self.device)

        return_pred_loss = torch.zeros(1, device=self.device)
        reg_kl = outputs["z_posterior"].kl_divergence(outputs["z_fixed_prior"])[attention_mask > 0].mean()
        prior_kl = outputs["z_posterior"].detach().kl_divergence(outputs["z_prior"])[attention_mask > 0].mean()
        loss = -(log_likelihood + entropy_reg * entropy) + z_reg * reg_kl + prior_kl

        return {
            "loss": loss,
            "nll": -log_likelihood,
            "entropy": entropy,
            "reg_kl": reg_kl,
            "prior_kl": prior_kl,
            "return_pred_loss": return_pred_loss
        }

    def finetune_loss_fn(
        self,
        outputs,
        a,
        rtg,
        attention_mask,
        entropy_reg,
        z_reg,
    ):
        if hasattr(outputs["action_preds"], "log_likelihood"):
            log_likelihood = outputs["action_preds"].log_likelihood(a)[attention_mask > 0].mean()
            entropy = outputs["action_preds"].entropy().mean()
        else:
            log_likelihood = -torch.mean((outputs["action_preds"] - a)[attention_mask > 0] ** 2)
            entropy = torch.zeros(1, device=self.device)

        reg_kl = outputs["z_posterior"].kl_divergence(outputs["z_fixed_prior"])[attention_mask > 0].mean()
        prior_kl = outputs["z_posterior"].detach().kl_divergence(outputs["z_prior"])[attention_mask > 0].mean()
        return_pred_loss = outputs["predicted_return"].nll(rtg)[attention_mask > 0].mean()
        loss = -(log_likelihood + entropy_reg * entropy) + z_reg * reg_kl + prior_kl + return_pred_loss

        return {
            "loss": loss,
            "nll": -log_likelihood,
            "entropy": entropy,
            "reg_kl": reg_kl,
            "prior_kl": prior_kl,
            "return_pred_loss": return_pred_loss
        }

    def return_warmup_loss_fn(
        self,
        outputs,
        a,
        rtg,
        attention_mask,
        entropy_reg,
        z_reg,
        pretrain=True
    ):
        return_pred_loss = outputs["predicted_return"].nll(rtg)[attention_mask > 0].mean()
        loss = return_pred_loss

        return {
            "loss": loss,
            "entropy": torch.zeros(1, device=self.device),
            "return_pred_loss": return_pred_loss
        }
