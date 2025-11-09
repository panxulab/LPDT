import pickle
import random
import json
import metaworld
import numpy as np
import torch
from torch import optim
from itertools import chain
import timm.optim.optim_factory as optim_factory
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
from prompt_dt.prompt_evaluate_episodes import prompt_evaluate_episode_rtg
from collections import OrderedDict

from src.envs import PointEnv, HopperRandParamsEnv, WalkerRandParamsWrappedEnv, ReachEnv, PickPlaceEnv



def get_optimizer(args, model):
    if args["model_type"] == "llama":
        param_groups = optim_factory.param_groups_weight_decay(model, args["weight_decay"])
        optimizer = optim.AdamW(param_groups, lr=args["learning_rate"], betas=(0.9, 0.95))
        print(optimizer)
    
    elif args["pretrained_lm"] or args["model_type"] == "llama_original":
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" not in str(type(module)).lower())
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args["learning_rate"], weight_decay=args["weight_decay"], eps=1e-6)
    else:
        for name, param in model.named_parameters():
            print(f"{name}: {param.requires_grad}")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )  
    return optimizer



def gen_env(env_name, dataset_mode, config_save_path):
    if 'cheetah_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/HalfCheetahDir-v0/{dataset_mode}/task_goals.pkl', 'rb') as f:
            task_goals = pickle.load(f)
        if task_idx % 2 == 0:
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal=False)
        elif task_idx % 2 == 1:
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal=False)
        with open(f'./datasets/HalfCheetahDir-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 200
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [1500.]
        else:
            env_targets = [1000.]
        scale = 500.
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/HalfCheetahVel-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        task_goal =task_info[f'task {task_idx}']['goal']
        tasks.append({'velocity':task_goal})
        env = HalfCheetahVelEnv(tasks, include_goal=False)
        max_ep_len = 200
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [0.]
        else:
            env_targets = [0.]
        scale = 500.
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/AntDir-v0/{dataset_mode}/task_goals.pkl', 'rb') as f:
            task_goals = pickle.load(f)
        tasks.append(task_goals[task_idx])
        env = AntDirEnv(tasks, len(tasks), include_goal=False)
        with open(f'./datasets/AntDir-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 200
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [1500.]
        else:
            env_targets = [800.]
        scale = 500.
    elif 'point_robot' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        task_goals = np.load(f'./datasets/PointRobot-v0/{dataset_mode}/task_goals.npy')
        env = PointEnv(max_episode_steps=20, num_tasks=1)
        env.load_all_tasks([task_goals[task_idx]])
        with open(f'./datasets/PointRobot-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 20
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [0.]
        else:
            env_targets = [0.]
        scale = 10.
    elif 'hopper_param' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/HopperRandParams-v0/{dataset_mode}/task_goals.pkl', 'rb') as f:
            task_goals = pickle.load(f)
        tasks.append(task_goals[task_idx])
        env = HopperRandParamsEnv(tasks=tasks)
        with open(f'./datasets/HopperRandParams-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 200
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [1000.]
        else:
            env_targets = [800.]
        scale = 500.
    elif 'walker_param' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/WalkerRandParams-v0/{dataset_mode}/task_goals.pkl', 'rb') as f:
            task_goals = pickle.load(f)
        tasks.append(task_goals[task_idx])
        env = WalkerRandParamsWrappedEnv(tasks=tasks)
        with open(f'./datasets/WalkerRandParams-v0/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 200
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [1500.]
        else:
            env_targets = [1000.]
        scale = 500.
    elif 'reach' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        with open(f'./datasets/Reach/{dataset_mode}/task_goals.pkl', 'rb') as f:
            task_goals = pickle.load(f)
        tasks.append(task_goals[task_idx])
        env = ReachEnv(tasks=tasks)
        with open(f'./datasets/Reach/{dataset_mode}/task_info.json', 'rb') as f:
            task_info = json.load(f)
        max_ep_len = 500
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [5000.]
        else:
            env_targets = [3000.]
        scale = 500.
    elif 'pick_place' in env_name:
        task_idx = int(env_name.split('-')[-1])
        tasks = []
        tasks.append(task_idx)
        env = PickPlaceEnv(task_list=tasks)
        max_ep_len = 500
        if dataset_mode == 'expert' or dataset_mode == 'medium-expert':
            env_targets = [5000.]
        else:
            env_targets = [3000.]
        scale = 500.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale


def get_env_list(env_name_list, dataset_mode, config_save_path, device):
    info = {}  # store all the attributes for each env
    env_list = []

    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, dataset_mode=dataset_mode, config_save_path=config_save_path)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0]
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list

""" prompts """


def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:, :-1, :]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1))
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask


def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes * sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        # s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        # for i in range(int(num_episodes * sample_size)):
        #     if variant["stochastic_prompt"]:
        #         traj = prompt_trajectories[int(batch_inds[i])]  # random select traj
        #     else:
        #         traj = prompt_trajectories[int(sorted_inds[-i])]  # select the best traj with highest rewards
        #         # traj = prompt_trajectories[i]
        #     si = max(0, traj['rewards'].shape[0] - max_len - 1)  # select the last traj with length max_len
                
        #     # get sequences from dataset
        #     s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        #     a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        #     r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        #     if 'terminals' in traj:
        #         d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        #     else:
        #         d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        #     timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        #     timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
        #     rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        #     if rtg[-1].shape[1] <= s[-1].shape[1]:
        #         rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        #     # padding and state + reward normalization
        #     tlen = s[-1].shape[1]
        #     # if tlen !=args.K:
        #     #     print('tlen not equal to k')
        #     s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        #     if not variant['no_state_normalize']:
        #         s[-1] = (s[-1] - state_mean) / state_std
        #     a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        #     r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        #     d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        #     rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        #     timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        #     mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        # s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        # a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes * sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])]  # random select traj
            else:
                traj = prompt_trajectories[int(sorted_inds[-i])]  # select the best traj with highest rewards
                # traj = prompt_trajectories[i]
            # Try different segment method
            if variant["segment"] == "start":
                indices = np.arange(max_len)  # 从轨迹开头取 max_len 个连续点
            elif variant["segment"] == "end":
                indices = np.arange(traj['rewards'].shape[0] - max_len, traj['rewards'].shape[0])  # 从轨迹末端取 max_len 个连续点
            elif variant["segment"] == "random":
                start_idx = np.random.randint(0, traj['rewards'].shape[0] - max_len + 1)  # 随机选取起始点
                indices = np.arange(start_idx, start_idx + max_len)  # 构造从起始点开始的连续索引
            else:
                raise ValueError("Invalid segment option. Choose from 'start', 'end', or 'random'.")
                
            # get sequences from dataset
            s.append(traj['observations'][indices].reshape(1, -1, state_dim))
            a.append(traj['actions'][indices].reshape(1, -1, act_dim))
            r.append(traj['rewards'][indices].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][indices].reshape(1, -1))
            else:
                d.append(traj['dones'][indices].reshape(1, -1))
            timesteps.append(indices.reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][indices], gamma=1.).reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # Padding and normalization steps (remain the same as before)
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask
    return fn


def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list):
    per_env_batch_size = variant['batch_size']

    def fn(batch_size=per_env_batch_size):
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        label = []
        for env_id, env_name in enumerate(train_env_name_list):
            if prompt_trajectories_list:
                get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            else:
                get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant)
            prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size)
            p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
            p_s_list.append(p_s)
            p_a_list.append(p_a)
            p_r_list.append(p_r)
            p_d_list.append(p_d)
            p_rtg_list.append(p_rtg)
            p_timesteps_list.append(p_timesteps)
            p_mask_list.append(p_mask)

            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)
            for i in range(batch_size):
                label.append(env_id)

        p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask
        return prompt, batch, label

    return fn


""" batches """


def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_finetune(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['prompt_length']  # use the same amount of data for funetuning

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len - 1)  # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    return fn


""" data processing """


def process_total_data_mean(trajectories, mode):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std


def process_dataset(trajectories, mode, env_name, dataset, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_data(env_name_list, data_save_path, dataset, prompt_mode, args):
    trajectories_list = []
    prompt_trajectories_list = []
    print(f"load dataset with ratio: {args.ratio}")
    for env_name in env_name_list:
        idx = env_name.split('-')[-1]
        env = env_name.split('-')[0]
        trajectories = []
        start_ind = 0
        if env == 'ant_dir':
            dataset_path = data_save_path + f'/AntDir-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/AntDir-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'cheetah_dir':
            dataset_path = data_save_path + f'/HalfCheetahDir-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/HalfCheetahDir-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'cheetah_vel':
            dataset_path = data_save_path + f'/HalfCheetahVel-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/HalfCheetahVel-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'point_robot':
            dataset_path = data_save_path + f'/PointRobot-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/PointRobot-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'hopper_param':
            dataset_path = data_save_path + f'/HopperRandParams-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/HopperRandParams-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'walker_param':
            dataset_path = data_save_path + f'/WalkerRandParams-v0/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/WalkerRandParams-v0/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'reach':
            dataset_path = data_save_path + f'/Reach/{dataset}/dataset_task_{idx}_{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/Reach/{dataset}/dataset_task_prompt{idx}.pkl'
        elif env == 'pick_place':
            dataset_path = data_save_path + f'/ML1-pick-place-v2/{dataset}/ML1-pick-place-v2-{idx}-expert-{args.ratio}.pkl'
            prompt_dataset_path = data_save_path + f'/ML1-pick-place-v2/{dataset}/ML1-pick-place-v2-{idx}-prompt-expert.pkl'
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        if env=='cheetah_dir':
            if dataset =='medium':
                data['observations']=data['states']
                data['next_observations']=data['next_states']
                data['terminals']=data['dones']

        if env == 'point_robot':
            for ind in range(len(data['rewards'])):
                if (ind+1) % 20 ==0:
                    traj = OrderedDict()
                    for key, value in data.items():
                        traj[key] = value[start_ind : ind+1]
                    trajectories.append(traj)
                    start_ind = ind + 1
                if len(trajectories)==100:
                    break
        elif env == 'reach':
            for ind in range(len(data['rewards'])):
                if (ind+1) % 500 ==0:
                    traj = OrderedDict()
                    for key, value in data.items():
                        traj[key] = value[start_ind : ind+1]
                    trajectories.append(traj)
                    start_ind = ind + 1
                if len(trajectories)==100:
                    break
        elif env == 'pick_place':
            print('pick_place')
            trajectories = data
        else:
            for ind in range(len(data['rewards'])):
                if (ind+1) % 200 ==0:
                    traj = OrderedDict()
                    for key, value in data.items():
                        traj[key] = value[start_ind : ind+1]
                    trajectories.append(traj)
                    start_ind = ind + 1
                if len(trajectories)==100:
                    break

        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        
        trajectories_list.append(trajectories)
        prompt_trajectories_list.append(prompt_trajectories)
    return trajectories_list, prompt_trajectories_list


def process_info(env_name_list, trajectories_list, info, mode, dataset, pct_traj, variant):
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], dataset=dataset, pct_traj=pct_traj)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std
        if variant['average_state_mean']:
            info[env_name]['state_mean'] = variant['total_state_mean']
            info[env_name]['state_std'] = variant['total_state_std']
    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


""" evaluation """


def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, prompt=None):
        returns = []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']
                )
            returns.append(ret)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
        }

    return fn

