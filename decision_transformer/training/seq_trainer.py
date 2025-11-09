import numpy as np
import torch
import torch.nn.functional as F
import time

import copy

from torch import nn
import kornia.losses

from decision_transformer.training.trainer import Trainer
from utils import flatten_prompt

from info_nce import InfoNCE, info_nce


class SequenceTrainer(Trainer):
    def pure_train_iteration_mix(self, num_steps, no_prompt=False):

        train_losses = []
        action_losses = []
        reg_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss, action_loss, reg_loss = self.train_step_mix(no_prompt=no_prompt)
            train_losses.append(train_loss)
            action_losses.append(action_loss)
            reg_losses.append(reg_loss)
            if self.scheduler is not None:
                self.scheduler.step()


        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/action_loss_mean'] = np.mean(action_losses)
        logs['training/action_loss_std'] = np.std(action_losses)
        logs['training/reg_loss_mean'] = np.mean(reg_losses)
        logs['training/reg_loss_std'] = np.std(reg_losses)
        logs['total_data_sample_processed'] = self.total_data_sample_processed

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs

    def train_step_mix(self, no_prompt=False):
        prompt, batch, label = self.get_prompt_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        num_samples = states.shape[0]

        if no_prompt:
            state_preds, action_preds, reward_preds, logits = self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds, logits = self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, prompt=prompt
            )
        self.step += 1
        self.total_data_sample_processed += num_samples
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]


        action_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        loss = action_loss


        if self.args["infoNCE"]:
            # logits_norm = F.normalize(logits, p=2, dim=1)
            # sim_matrix = torch.matmul(logits_norm, logits_norm.T)/self.args["temperature"]
            # label=torch.tensor(label).to(self.device)
            # label = label.view(-1, 1)
            # mask = torch.eq(label, label.T).float()
            # mask = mask.fill_diagonal_(0)
            # exp_sim_matrix = torch.exp(sim_matrix)
            # exp_sim_matrix_sum = torch.sum(exp_sim_matrix, dim=1, keepdim=True)
            # pos_sim = torch.sum(exp_sim_matrix * mask, dim=1)
            # reg_loss = -torch.log(pos_sim / (exp_sim_matrix_sum - torch.exp(sim_matrix.diag()/self.args["temperature"])))
            # loss += self.args["infoNCE_lambda"]*torch.mean(reg_loss)
            info_loss = InfoNCE()
            task_nums = int(logits.shape[0]/self.batch_size)
            query = []
            key = []
            for i in range(task_nums):
                query.append(logits[i*self.batch_size])
                key.append(logits[i*self.batch_size+1])
            query = torch.stack(query)
            key = torch.stack(key)
            reg_loss = info_loss(query, key)
            loss += self.args["infoNCE_lambda"] * reg_loss

        elif self.args["classifier"]:
            label=torch.tensor(label).to(self.device)
            # Try on focal loss
            # focal_loss_fn = kornia.losses.FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
            # classifier_loss = focal_loss_fn(logits, label)
            classifier_criterion = nn.CrossEntropyLoss()
            reg_loss = classifier_criterion(logits, label)
            loss += self.args["classifier_lambda"] * reg_loss


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item(), action_loss.detach().cpu().item(), reg_loss.detach().cpu().item()

    def finetune_eval_iteration_multienv(self, get_prompt, get_batch, test_prompt_trajectories_list,
                                         test_trajectories_list,
                                         eval_episodes, env_name_list, info,
                                         variant, env_list, iter_num=0, print_logs=False,
                                         no_prompt=False, group='test-finetune',
                                         finetune_opt=False):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()
        self.current_model_dict = copy.deepcopy(self.model.state_dict())

        eval_start = time.time()
        if finetune_opt:
            fintune_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=variant['finetune_lr'],
                weight_decay=1e-4,
            )
        else:
            fintune_optimizer = None
        for env_id, env_name in enumerate(env_name_list):
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in
                             info[env_name]['env_targets']]
            self.get_prompt = get_prompt(test_prompt_trajectories_list[env_id], info[env_name], variant)
            self.get_batch = get_batch(test_trajectories_list[env_id], info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(), batch_size=1)  # one prompt for the whole batch now
            else:
                self.prompt = None

            self.model.train()
            # finetune the model on the data for this task
            finetune_losses = []
            for _ in range(variant['finetune_steps']):
                finetune_loss = self.train_step(
                    batch_size_overwrite=variant['finetune_batch_size'],
                    optimizer=fintune_optimizer)
                finetune_losses.append(finetune_loss)

            self.model.eval()
            # need to sample eval_fn and prompt together
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

            self.model.load_state_dict(self.current_model_dict)

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self, batch_size_overwrite=None, optimizer=None):
        if batch_size_overwrite is not None:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size_overwrite)
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        # print('state shape after batch', states.shape)
        # print('self.batch_size', self.batch_size)
        # print('enter train step')
        # input()
        state_preds, action_preds, reward_preds, logits = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, prompt=self.prompt
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        action_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss = action_loss
        if optimizer is None:
            self.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        if optimizer is None:
            self.optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()

    def eval_iteration_multienv(self, get_prompt, prompt_trajectories_list, eval_episodes, env_name_list, info,
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            # need to sample eval_fn and prompt together (no best-of-N)
            self.eval_fns = [
                eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name)
                for tar in info[env_name]['env_targets']
            ]
            print(len(prompt_trajectories_list[env_id]))
            print(info[env_name])
            self.get_prompt = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(), batch_size=1)
            else:
                self.prompt = None
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v


            logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def save_model(self, env_name, postfix, folder):
        model_name = '/prompt_model_' + postfix + '.pth'
        torch.save(self.model.state_dict(), folder + model_name)  # model save
        print('model saved to ', folder + model_name)  # , lm_loss.detach().cpu().item()
