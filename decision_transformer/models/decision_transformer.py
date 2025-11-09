import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os

# use ../../decision_transformer as decision_transformer when run as main
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2LMHeadModel
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Config_LoRA

from decision_transformer.models.utils import ResidualBlock, MLPBlock


class DecisionTransformer(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    @property
    def transformer(self):
        return self.transformer_model.transformer

    def __init__(
            self,
            args,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            num_class=2,
            classifier=False,
            infoNCE=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size

        if args["pretrained_lm"] is not None:
            print("Loading from pretrained " + args["pretrained_lm"] + " model")
            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained(args["pretrained_lm"])
                self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                    args["pretrained_lm"],
                    config=config
                )
            else:
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                config.resid_pdrop = args["dropout"]
                self.transformer_model = GPT2LMHeadModel.from_pretrained(
                    args["pretrained_lm"],
                    config=config,
                )
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        else:

            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained("gpt2")
                self.transformer_model = GPT2LMHeadModel_LoRA(config)
            else:
                config = transformers.GPT2Config(
                    n_embd=hidden_size,
                    **kwargs
                )
                # config = transformers.GPT2Config.from_pretrained("gpt2")
                # config.resid_pdrop = args["dropout"]
                # NOTE: If you comment two lines above, then we adopt non-pretrained 3-layer DT; otherwise we use the same config as the pretrained gpt2 model, but with random weights
                self.transformer_model = GPT2LMHeadModel(config)
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        if max_ep_len > config.n_positions and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k: (k + step)] = self.transformer.wpe.weight[
                                                       orig_k: min(max_ep_len - k + orig_k, current_max_pos)
                                                       ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        if args["mlp_embedding"]:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)

            self.prompt_embed_return = ResidualBlock(1, hidden_size)
            self.prompt_embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.prompt_embed_action = ResidualBlock(self.act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

            self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
            self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["mlp_embedding"]:
            if args["share_input_output_proj"]: raise ValueError(
                "With MLP in embeddings, you cannot share the projections")
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = MLPBlock(self.hidden_size, self.act_dim, self.hidden_size)
            self.predict_return = torch.nn.Linear(hidden_size, 1)
        else:
            if args["share_input_output_proj"]:
                self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
                self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
                self.predict_action = lambda x: F.tanh(
                    F.linear(x, self.embed_action.weight.t())
                )
            else:
                self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
                self.predict_action = nn.Sequential(
                    *(
                            [nn.Linear(hidden_size, self.act_dim)]
                            + ([nn.Tanh()] if action_tanh else [])
                    )
                )
                self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.classifier = classifier
        self.infoNCE = infoNCE

        if self.infoNCE:
            self.prompt_encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))


        if self.classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_class))

        self.past_key_values = None
        print(self)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, past_key_values=None, prompt=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        logits = None

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:, :-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)
            if self.classifier:
                #classifier logits
                logits = self.classifier_head(torch.mean(prompt_stacked_inputs, dim=1))

            if self.infoNCE:
                logits = self.prompt_encoder(torch.mean(prompt_stacked_inputs, dim=1))

            # stacked_inputs add prompted sequence
            if prompt_stacked_inputs.shape[1] == 3 * seq_length:  # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat(
                    (prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else:  # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]
        self.past_key_values = transformer_outputs["past_key_values"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state

        return state_preds, action_preds, return_preds, logits

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, past_key_values=None, prompt=None, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([
                    torch.zeros(self.max_length - states.shape[1]),
                torch.ones(states.shape[1]), ])
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat([
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
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
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
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
                states,  actions, None, returns_to_go, timesteps, attention_mask=attention_mask, prompt=prompt)

        return action_preds[0, -1]
