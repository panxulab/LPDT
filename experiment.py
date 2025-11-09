import itertools
import json
from collections import namedtuple
import re

import gym
import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import sys
import os
import loralib as lora
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from utils import get_optimizer, process_info, process_total_data_mean, load_data, get_env_list, \
    get_prompt_batch, get_prompt, get_batch, get_batch_finetune, eval_episodes



def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def experiment(
    exp_prefix,
    variant,
):

    # set seed
    np.random.seed(variant["seed"])
    random.seed(variant["seed"])
    # set seed for torch
    torch.manual_seed(variant["seed"])
    torch.cuda.manual_seed(variant["seed"])
    torch.cuda.manual_seed_all(variant["seed"])
    # set seed for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)
    model_type = variant["model_type"]

    if variant["classifier"]:
        print("classifier with lambda="+str(variant["classifier_lambda"]))
    elif variant["infoNCE"]:
        print("infoNCE with lambda="+str(variant["infoNCE_lambda"]))

    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'datasets')
    data_save_path = os.path.join(cur_dir, 'datasets')
    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)

    config_path_dict = {
        'cheetah_vel': "HalfCheetahVel-v0/cheetah_vel_50.json",
        'cheetah_dir': "HalfCheetahDir-v0/cheetah_dir_4.json",
        'ant_dir': "AntDir-v0/ant_dir_50.json",
        'hopper_param': "HopperRandParams-v0/hopper_param_50.json",
        'walker_param': "WalkerRandParams-v0/walker_param_50.json",
        'point_robot': "PointRobot-v0/point_robot_50.json",
        'reach': "Reach/reach_20.json",
        'pick_place': "ML1-pick-place-v2/pick_place_20.json"
    }

    # load task config
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


    train_env_name_list, test_env_name_list = [], []
    for task_ind in task_config.train_tasks:
        train_env_name_list.append(args.env + '-' + str(task_ind))
    for task_ind in task_config.test_tasks:
        test_env_name_list.append(args.env + '-' + str(task_ind))

    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']
    # training envs
    info, env_list = get_env_list(train_env_name_list, dataset_mode, config_save_path, device)

    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, test_dataset_mode, config_save_path, device)

    print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')



    # load dataset
    trajectories_list, prompt_trajectories_list = load_data(train_env_name_list, data_save_path, dataset_mode,
                                                                   train_prompt_mode, args)
    # load testing dataset
    test_trajectories_list, test_prompt_trajectories_list = load_data(test_env_name_list, data_save_path,
                                                                             test_dataset_mode, test_prompt_mode, args)

    # change to total train trajecotry
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        print(len(total_traj_list))
        total_state_mean, total_state_std = process_total_data_mean(total_traj_list, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant)


    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj,
                             variant)

    ######
    # construct dt model and trainer
    ######

    exp_prefix = exp_prefix + '-' + args.env
    num_env = len(train_env_name_list)
    ratio = variant['ratio']
    if variant['classifier']:
        group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}-classifier-{ratio}'
    elif variant['infoNCE']:
        group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}-infoNCE-{ratio}'
    elif variant['reconstruction']:
        group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}-reconstruction-{ratio}'
    
    from datetime import datetime
    current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_prefix = f'{current_time_str}-{group_name}'
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]



    if model_type == "dt":
        model = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=1000,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
            mlp_embedding=variant["mlp_embedding"],
            num_class=variant["num_class"],
            classifier=variant["classifier"],
            infoNCE=variant["infoNCE"]
        )
            
        if variant["adapt_mode"]:
            if variant["lora"] == False:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                print("adapt lora.")
                lora.mark_only_lora_as_trainable(model, bias='lora_only', depreciate_lora_rank=True)
                # lora.mark_only_lora_as_trainable(model, bias='all')
                # NOTE: Don't put this part below other adaptation part.
            if variant["adapt_wte"]:
                print("adapt wte.")
                for param in model.transformer.wte.parameters():
                    param.requires_grad = True
            if variant["adapt_wpe"]:
                print("adapt wpe.")
                for param in model.transformer.wpe.parameters():
                    param.requires_grad = True
            if variant["adapt_embed"]:
                print("adapt embeddings.")
                # adapt the embeddings in DecisionTransformer
                for name, param in model.named_parameters():
                    if ("embed" in name or "predict" in name):
                        param.requires_grad = True
            if variant["adapt_ln"]:
              print("adapt layer norms.")
              # adapt the LayerNorm in the transformer's blocks
              for block in model.transformer.h:
                  for param in block.ln_1.parameters():
                      param.requires_grad = True
                  for param in block.ln_2.parameters():
                      param.requires_grad = True
              # adapt the final LayerNorm in the transformer
              for param in model.transformer.ln_f.parameters():
                  param.requires_grad = True
            if variant["adapt_attn"]:
                print("adapt attention.")
                for block in model.transformer.h:
                # adapt the attention weights and biases
                    for param in block.attn.parameters():
                        param.requires_grad = True
            if variant["adapt_ff"]:
                print("adapt feed-forward.")
                for block in model.transformer.h:
                    # adapt the feed_forward weights and biases
                    for param in block.mlp.parameters():
                        param.requires_grad = True
            if variant["only_adapt_last_two_blocks"]:
                print("for transformer, only adapt the last two blocks.")
                for block in model.transformer.h[0:-2]:
                    for param in block.parameters():
                        param.requires_grad = False
            if variant["adapt_last_two_blocks"]:
                print("for transformer, adapt the last two blocks.")
                for block in model.transformer.h[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else:
            print("fintune all.")
            
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError


    # print model parameters (overall and transformer-specific)
    total_param_size = sum(p.numel() for p in model.parameters())
    trainable_param_size_overall = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (overall): {trainable_param_size_overall} / Total params: {total_param_size}")

    trainable_param_size = 0
    frozen_param_size = 0
    for name, param in model.named_parameters():
        if "transformer" not in name:
            continue
        if param.requires_grad:
            trainable_param_size += param.numel()
        else:
            frozen_param_size += param.numel()
    if (trainable_param_size + frozen_param_size) > 0:
        print(f"Transformer trainable params: {trainable_param_size}")
        print(f"Transformer frozen params: {frozen_param_size}")
        print(f"Transformer trainable ratio: {trainable_param_size/(trainable_param_size + frozen_param_size):.4f}")
    
    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    model = model.to(device=device)
    
    visualize = variant["visualize"]

    if "dt" in model_type:
        env_name = train_env_name_list[0]
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch(trajectories_list[0], info[env_name], variant),
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=None,
            get_prompt=get_prompt(prompt_trajectories_list[0], info[env_name], variant),
            get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list),
            device = device,
            num_class=variant["num_class"]
        )
    elif model_type == "bc":
        env_name = train_env_name_list[0]
        trainer = ActTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch(trajectories_list[0], info[env_name], variant),
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=None,
        )
    if not variant['evaluation']:
        if log_to_wandb:
            wandb_api_key = os.environ.get('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)
            wandb.init(
                name=exp_prefix,
                group=group_name,
                # NOTE: fill in the name of your own wandb project
                project='LPDT_v2_dataset_compute',
                config=variant,
            )
        save_path += wandb.run.name
        save_path += str(random.randint(1,100000))
        os.mkdir(save_path)

        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        # accumulate timing
        total_training_time_s = 0.0
        total_inference_time_s = 0.0
        num_eval_calls = 0

        for iter in range(variant['max_iters']):
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'],
                no_prompt=args.no_prompt
            )

            # accumulate training time from trainer logs
            if 'time/training' in outputs:
                total_training_time_s += outputs['time/training']

            # start evaluation
            if iter % args.test_eval_interval == 0:
                # evaluate test
                if not args.finetune:
                    test_eval_logs = trainer.eval_iteration_multienv(
                        get_prompt, test_prompt_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1,
                        print_logs=True, no_prompt=args.no_prompt, group='test')
                    # accumulate inference time
                    if 'time/evaluation' in test_eval_logs:
                        total_inference_time_s += test_eval_logs['time/evaluation']
                        num_eval_calls += 1
                    outputs.update(test_eval_logs)
                else:
                    test_eval_logs = trainer.finetune_eval_iteration_multienv(
                        get_prompt, get_batch_finetune, test_prompt_trajectories_list, test_trajectories_list,
                        eval_episodes, test_env_name_list, test_info,
                        variant, test_env_list, iter_num=iter + 1,
                        print_logs=True, no_prompt=args.no_prompt,
                        group='finetune-test', finetune_opt=variant['finetune_opt'])
                    if 'time/evaluation' in test_eval_logs:
                        total_inference_time_s += test_eval_logs['time/evaluation']
                        num_eval_calls += 1
                    outputs.update(test_eval_logs)
            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env,
                    # postfix=model_post_fix + '_iter_' + str(iter),
                    postfix=str(iter),
                    folder=save_path)

            outputs.update({"global_step": iter})  # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)

        # print timing summary
        avg_eval_time = (total_inference_time_s / num_eval_calls) if num_eval_calls > 0 else 0.0
        print(f"Total training time (s): {total_training_time_s:.3f}")
        print(f"Total inference/eval time (s): {total_inference_time_s:.3f}, runs: {num_eval_calls}, avg per run (s): {avg_eval_time:.3f}")

    else:
        ####
        # start evaluating
        ####

        saved_model_path = os.path.join(save_path, variant['load_path'])
        # if 'checkpoint' in variant and variant['checkpoint'] != "prompt_model_.*":
        #     saved_model_path = os.path.join(saved_model_dir, variant['checkpoint'])
        #     print(f"Using provided checkpoint: {variant['checkpoint']}")
        # else:
        #     pattern = re.compile(r'prompt_model_(\d+)')
        #     model_files = [f for f in os.listdir(saved_model_dir) if pattern.match(f)]
            
        #     assert model_files, "No model files found matching the pattern."
            
        #     latest_model_file = max(model_files, key=lambda f: int(pattern.match(f).group(1)))
        #     saved_model_path = os.path.join(saved_model_dir, latest_model_file)
        #     print(f"Using latest model file: {latest_model_file}")

        assert os.path.exists(saved_model_path), "Model file not found or unable to load."
        model.load_state_dict(torch.load(saved_model_path))
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1].split('.')[0])
        total_inference_time_s = 0.0
        for i in range(variant['num_eval']):
            eval_logs = trainer.eval_iteration_multienv(
                get_prompt, test_prompt_trajectories_list,
                eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num,
                print_logs=True, no_prompt=args.no_prompt, group='eval')
            print(eval_logs)
            if 'time/evaluation' in eval_logs:
                total_inference_time_s += eval_logs['time/evaluation']

        avg_eval_time = total_inference_time_s / max(1, variant['num_eval'])
        print(f"Total inference/eval time (s): {total_inference_time_s:.3f}, runs: {variant['num_eval']}, avg per run (s): {avg_eval_time:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--dataset_mode", type=str, default="expert")
    parser.add_argument("--test_dataset_mode", type=str, default="expert")
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning
    # data sampling
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--data_suffix", type=str, default="mydataset")
    
    parser.add_argument("--segment", type=str, default="random")
    # training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)
    parser.add_argument("--visualize", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--description", type=str, default="")
    
    # evaluation
    parser.add_argument("--load_path", type=str, required=True, default=".*")
    parser.add_argument("--checkpoint", type=str, default="prompt_model_.*")
    
    # architecture, don't need to care about in our method
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    # learning hyperparameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--num_eval", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=5001)
    parser.add_argument("--num_steps_per_iter", type=int, default=10)
    # implementations
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--mlp_embedding", action="store_true", default=False)
    # adaptations
    parser.add_argument("--adapt_mode", action="store_true", default=False)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--only_adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_ln", action="store_true", default=False)
    parser.add_argument("--adapt_attn", action="store_true", default=False)
    parser.add_argument("--adapt_ff", action="store_true", default=False)
    parser.add_argument("--adapt_embed", action="store_true", default=False)
    parser.add_argument("--adapt_wte", action="store_true", default=False)
    parser.add_argument("--adapt_wpe", action="store_true", default=False)
    # lm co-training

    parser.add_argument('--train_eval_interval', type=int, default=500)
    parser.add_argument('--test_eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=500)

    parser.add_argument('--average_state_mean', action='store_true', default=True)
    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--no_state_normalize', action='store_true', default=False)


    parser.add_argument("--ratio", action="store", type=float, default=1.0)

    parser.add_argument("--classifier", action="store_true", default=False)
    parser.add_argument("--classifier_lambda", type=float, default=0.1)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--infoNCE", action="store_true", default=False)
    parser.add_argument("--infoNCE_lambda", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1)
    args = parser.parse_args()
    experiment("d4rl-experiment", variant=vars(args))
