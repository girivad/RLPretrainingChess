"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
from torch import autograd
import torch.nn as nn
from torch.nn import functional as F
from torch import logsumexp
from torch.distributions import Categorical
from utils import smooth
from contextlib import nullcontext

SPACE_TOKEN = 0
EOS_TOKEN = 15

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.block_size = config.block_size
        self.k_cache = None
        self.v_cache = None

    def create_kv_cache(self, bsz, device):
        self.k_cache = torch.zeros(
            (
                bsz,
                self.block_size,
                self.n_embd
            ),
            device = device
        )

        self.v_cache = torch.zeros(
            (
                bsz,
                self.block_size,
                self.n_embd
            ), device = device
        )

    def forward(self, x, start_pos = 0, kv_cache = False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # print(f"Projected Tokens {start_pos}:{start_pos + T}:", q.size(), k.size(), v.size())
        
        if kv_cache and self.k_cache is not None:
            self.k_cache[:B, start_pos : start_pos + T] = k
            self.v_cache[:B, start_pos : start_pos + T] = v
            k = self.k_cache[:B, : start_pos + T]
            v = self.v_cache[:B, : start_pos + T]
        elif kv_cache:
            raise Exception("KV Cache invoked before created.")

        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal= (start_pos == 0 or not kv_cache))
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, Tq, Tk)
            att = att.masked_fill(self.bias[:,:,start_pos : start_pos + T,: start_pos + T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, -1, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def create_kv_cache(self, bsz, device):
        self.attn.create_kv_cache(bsz, device)

    def forward(self, x, start_pos = 0, kv_cache = False):
        x = x + self.attn(self.ln_1(x), start_pos = start_pos, kv_cache = kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

# Reusable Loss Terms
def ce_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

def kd_loss(st_logits, se_logits):
    b, t, _ = st_logits.size()
    st_probs = smooth(F.softmax(st_logits, dim = -1)) 
    se_probs = F.softmax(se_logits, dim = -1)
    kld = torch.sum(
        torch.multiply(
            torch.log(st_probs), 
            se_probs
        )
    )
    se_entropy = torch.mean(Categorical(probs = smooth(se_probs)).entropy())
    return -1 * kld / (b * t), se_entropy

def z_loss(logits):
    return 1e-4 * (logsumexp(logits, dim = -1) ** 2).sum()

# Auxiliary Pretraining Losses for Seer Forcing:
def aux_seer_loss(st_logits, se_logits, idx, base_loss, loss_tensor, lamda = 0.5):
    loss = base_loss

    # KL-Divergence between Student and Seer
    kld, se_entropy = kd_loss(st_logits[:, :-1], se_logits[:, 1:].detach().clone())
    loss_tensor[2] = kld
    loss_tensor[3] = se_entropy
    loss = lamda * loss + (1 - lamda) * kld 
    assert not torch.any(torch.isnan(loss))

    # Cross-Entropy for Seer
    # The seer only predicts tokens within context window, no right-shifting needed.
    se_ce = ce_loss(se_logits, idx)
    loss_tensor[4] = se_ce
    loss = loss + se_ce
    assert not torch.any(torch.isnan(loss))

    return loss

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 29 # PGN character vocabulary without move numbers
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_dim: int = 3072
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_batch_size: int = 100

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # Student Model:
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        num_params = self.get_num_params()
        if num_params < 1e9:
            print("number of parameters in GPT: %.2fM" % (num_params/1e6,))
        else:
            print("number of parameters in GPT: %.2fB" % (num_params/1e9,))

    def create_kv_cache(self, bsz, device):
        for blk in self.transformer.h:
            blk.create_kv_cache(bsz, device)

    def get_num_params(self, non_embedding=True, train = True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() + self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def latent_forward(self, x, start_layer = 0, end_layer = None, start_pos = 0, kv_cache = False):
        if end_layer is None:
            end_layer = self.config.n_layer + 1
        device = x.device
        t = x.size(1)
        assert t + start_pos <= self.config.block_size, f"Cannot forward sequence of length {start_pos}/{t}, block size is only {self.config.block_size}"
        pos = torch.arange(start_pos, t + start_pos, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        if start_layer == 0:
            tok_emb = self.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        for layer_idx, block in enumerate(self.transformer.h):
            if layer_idx + 1 < start_layer or layer_idx >= end_layer:
                continue
            x = block(x, start_pos = start_pos, kv_cache = kv_cache)

        return x

    def forward(self, x, start_layer = 0, end_layer = None, targets = None, evaluate = False, batch_inf = False, start_pos = 0, kv_cache = False):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor to the model.
            start_layer (int, optional): The starting layer for the forward pass. Defaults to 0.
            end_layer (int, optional): The ending layer for the forward pass. If None, defaults to the total number of layers + 1.
            targets (torch.Tensor, optional): Target tensor for loss computation. If None, inference mode is assumed. Defaults to None.
            evaluate (bool, optional): If True, returns logits and loss for evaluation. Defaults to False.
            batch_inf (bool, optional): If True, enables batch inference mode. Defaults to False.
            start_pos (int, optional): Starting position for the forward pass. Defaults to 0.
            kv_cache (bool, optional): If True, enables key-value caching for inference. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, None], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - If `end_layer` is less than or equal to the number of layers, returns the latent representation `x`.
                - If in inference mode and `batch_inf` is False, returns the logits for the last token and None.
                - If in inference mode and `batch_inf` is True, returns the logits for all tokens and None.
                - If `evaluate` is True, returns the logits and the language modeling loss.
                - Otherwise, returns the logits, the total loss (includes aux losses), and a tensor containing individual loss components.

        Raises:
            Exception: If an error occurs during the unembedding process, an exception is raised with details about the error.
        """
        inference = targets is None

        if end_layer is None:
            end_layer = self.config.n_layer + 1

        loss_tensor = torch.zeros((2, ))

        # forward the GPT model itself
        x = self.latent_forward(x, start_layer = start_layer, end_layer = end_layer, start_pos = start_pos, kv_cache = kv_cache and inference and not batch_inf)

        if end_layer <= self.config.n_layer: # Latent: 0th layer is embedding, nth layer is latent, n + 1-th layer is decoded result.
            return x

        x = self.transformer.ln_f(x)

        if inference and not batch_inf:
            try:
                return self.lm_head(x[:, [-1], :]), None
            except Exception as e:
                raise Exception(f"Error Unembedding: {e}. x: {x.size()}. Start Pos: {start_pos}")
        elif inference:
            return self.lm_head(x), None

        # if we are given some desired targets also calculate the loss
        st_logits = self.lm_head(x) # Tokens 2:ctx_len + 1
        loss = ce_loss(st_logits, targets)
        loss_tensor[0] = loss
        assert not torch.any(torch.isnan(loss))

        if evaluate:
            return st_logits, loss

        st_z = z_loss(st_logits)
        loss_tensor[1] = st_z
        loss = loss + st_z
        assert not torch.any(torch.isnan(loss))

        return st_logits, loss, loss_tensor

    def compute_gradient(self, X, Y, gradient_accumulation_steps, ctx = nullcontext(), scaler = None, ddp_model = None):
        model = ddp_model
        if ddp_model is None:
            # Not using DDP
            model = self

        with ctx:
            _, loss, micro_loss_tensor = model(X, targets = Y)
            loss = loss / gradient_accumulation_steps

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

        return loss, micro_loss_tensor

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        blks = self.transformer.h
        for block in blks:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate_token(self, idx, temperature = 1.0, top_k = None, start_pos = 0, kv_cache = False):
        assert len(idx.size()) == 2, (idx.size(), idx, start_pos)
        assert idx.size(1) > 0, (idx.size(), idx, start_pos)
        logits, _ = self(idx, start_pos = start_pos, kv_cache = kv_cache)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample next tokens
        next_tokens = torch.multinomial(probs, num_samples = 1)

        return next_tokens
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_next = self.generate_token(idx_cond, temperature = temperature, top_k = top_k)            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_moves(self, games, device, max_move_size = 5, overwrite_spaces = True, temperature = 1.0, top_k = None, space_token = SPACE_TOKEN, eos_token = EOS_TOKEN, start_pos = 0, kv_cache = False, completed_msk = None):
        """
        Take a list of tokenized games. Autoregressively predict the next move with up to max_move_size tokens for each game, and output as token ids.
        """

        max_prompt_len = max((len(game) for game in games)) # Maximum number of tokens in a single game (also acts as the first token idx to fill in)
        max_token = max_prompt_len + max_move_size # Final Token to predict is max_move_size shifted from the max_prompt_len
        games_tensor = torch.tensor(
            [
                list(game) + [-1] * max_move_size + [eos_token] * (max_token - len(game) - max_move_size) for game in games
            ],
            device = device
        ) # Each game is represented as a sequence of game tokens, a limited move-sized slot and then padded with end-of-game tokens.

        mv_msk = games_tensor == -1 # The move mask identifies which slots should be retrieved from the final game state as the current move. Even completed games are allocated a slot, though the slot there must be filled in with padding tokens.

        if completed_msk is not None:
            games_tensor = torch.where(
                ~completed_msk.view(-1, 1),
                games_tensor,
                torch.full((1, max_token), eos_token, device = device)
            ) # For completed games, replace with sequences of only padding tokens. 

        sp_msk = games_tensor == space_token if space_token is not None else torch.full_like(games_tensor, False)

        if completed_msk is not None:
            assert not torch.any(torch.any(sp_msk, dim = 1) & completed_msk)
            assert torch.all(torch.all(games_tensor == eos_token, dim = 1) | ~completed_msk)

        min_prompt_len = None
        for game_idx in range(games_tensor.size(0)):
            mpl = -1
            for tok in range(games_tensor.size(1) - 1, -1, -1):
                if space_token is not None and sp_msk[game_idx, tok]:
                    mpl = tok
                    break
                if space_token is None and mv_msk[game_idx, tok]:
                    mpl = tok
            if mpl < 1:
                continue
            
            if min_prompt_len is None:
                min_prompt_len = mpl
            else:
                min_prompt_len = min(min_prompt_len, mpl)

        min_prompt_len = min_prompt_len or 1

        # if device == "cuda:0":
        #     print("mPL:", min_prompt_len, "MPL:", max_prompt_len, "MT:", max_token)

        temp_start_pos = start_pos
        final_start_pos = None

        for token in range(min_prompt_len, max_token):        
            if not torch.any(mv_msk[:, token]) and not (overwrite_spaces and torch.any(sp_msk[:, token])):
                continue
            # if device == "cuda:0":
            #     print(f"Generating Token {token} from context {temp_start_pos}:{token} ({games_tensor[:, :temp_start_pos]} + {games_tensor[:, temp_start_pos : token]})")
            # if device == "cuda:0":
            #     print("TSP:", temp_start_pos, "Token:", token)
            assert temp_start_pos != token, (device, temp_start_pos, games_tensor[0, :temp_start_pos])
            assert not torch.any(games_tensor[:, temp_start_pos : token] < 0), (device, games_tensor[:, temp_start_pos : token], temp_start_pos, token, min_prompt_len, max_token)
            next_tokens = self.generate_token(games_tensor[:, temp_start_pos:token], temperature = temperature, top_k = top_k, start_pos = temp_start_pos, kv_cache = kv_cache).view(-1)
            # if device == "cuda:0":
            #     print("Prediction:", next_tokens)
            # Store next tokens if it is writing into an allotted move slot (within the max_move_size)
            # Can only overwrite a space if terminating the game (i.e. resigning).
            write_msk = mv_msk[:, token] | (overwrite_spaces & sp_msk[:, token] & next_tokens == eos_token)
            games_tensor[:, token] = torch.where(
                write_msk, 
                next_tokens, 
                games_tensor[:, token]
            )
            mv_msk[:, token] = mv_msk[:, token] | (overwrite_spaces & sp_msk[:, token] & next_tokens == eos_token)
            
            if kv_cache:
                temp_start_pos = token
                # if device == "cuda:0":
                #     print(f"Moving up temp_start_pos: {temp_start_pos}")

            if final_start_pos is None and completed_msk is not None and torch.any(~completed_msk & mv_msk[:, token]):
                # The start_pos for the next iteration will be the first token of the first move made for an incomplete game.
                # if device == "cuda:0":
                #     print("Setting next start position:", token)
                final_start_pos = token

        # if device == "cuda:0":
        #     print("Total Tokens:", games_tensor)

        return [
            [idx_tensor.item() for idx_tensor in games_tensor[s, mv_msk[s]]]
            for s in range(games_tensor.size(0))
        ], final_start_pos
    
@dataclass
class SFGPTConfig(GPTConfig):
    n_slayer: int = 0
    lamda: float = 0.5

class SFGPT(GPT):
    def __init__(self, config: SFGPTConfig):
        super().__init__(config)
        assert config.n_slayer > 0
        self.seer = nn.ModuleDict(
            dict(
                h = nn.ModuleList([Block(config) for _ in range(config.n_slayer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
                ln_p1 = LayerNorm(config.n_embd, bias = config.bias),
                mha_p = SelfAttention(config),
                w_p = nn.Linear(config.n_embd, config.n_embd),
                ln_p2 = LayerNorm(config.n_embd, bias = config.bias),
                ln_f1 = LayerNorm(config.n_embd, bias = config.bias),
                mha_f = SelfAttention(config),
                w_f = nn.Linear(config.n_embd, config.n_embd),
                ln_f2 = LayerNorm(config.n_embd, bias = config.bias),
                fusion = MLP(config)
            )
        )

        # init all weights (particularly the seer, here)
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        num_params = self.get_num_params()
        if num_params < 1e9:
            print("number of parameters in SF-GPT: %.2fM" % (num_params/1e6,))
        else:
            print("number of parameters in SF-GPT: %.2fB" % (num_params/1e9,))
    
    def get_num_params(self, non_embedding = True, train = True):
        # Use super method to calculate the number of parameters in the whole model
        num_params = super().get_num_params(non_embedding = non_embedding, train = train)

        # If not training, Seer Parameters should not be treated as part of the model
        if not train:
            num_params -= sum(param.numel() for param in self.seer.parameters())

        return num_params

    def seer_forward(self, emb, device):
        b, _, e = emb.size()
        xl2r = emb[:, :-1]
        xr2l = torch.flip(emb[:, 1:], dims = (1, ))

        for blk in self.seer.h:
            xl2r = blk(xl2r)
            xr2l = blk(xr2l)

        xl2r = xl2r + self.seer.mha_p(self.seer.ln_p1(xl2r))
        xr2l = xr2l + self.seer.mha_f(self.seer.ln_f1(xr2l))

        x = self.seer.w_f(
            torch.concat(
                [ 
                  torch.flip(self.seer.ln_f2(xr2l), (1, )), # B x (S - 1) x d_m
                  torch.zeros((b, 1, e), device = device)
                ],
                dim = 1
              )
          ) + self.seer.w_p(
              torch.concat(
                [
                    torch.zeros((b, 1, e), device = device), 
                    self.seer.ln_p2(xl2r)  # B x (S - 1) x d_m
                ],
                dim = 1                  
              )
          )

        x = self.seer.fusion(x)
        return self.seer.ln_f(x)
    
    def forward(self, input, start_layer = 0, end_layer = None, targets = None, evaluate = False, batch_inf = False, start_pos = 0, kv_cache = False):
        """
        Forward pass for the model.

        Args:
            input (torch.Tensor): Input tensor to the model.
            start_layer (int, optional): The starting layer for the forward pass. Defaults to 0.
            end_layer (int, optional): The ending layer for the forward pass. If None, defaults to the total number of layers + 1.
            targets (torch.Tensor, optional): Target tensor for loss computation. If None, inference mode is assumed. Defaults to None.
            evaluate (bool, optional): If True, returns logits and loss for evaluation. Defaults to False.
            batch_inf (bool, optional): If True, enables batch inference mode. Defaults to False.
            start_pos (int, optional): The starting position for positional encoding. Defaults to 0.
            kv_cache (bool, optional): If True, enables key-value caching for faster inference. Defaults to False.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
                - If 'end_layer' is less than or equal to the number of layers, returns the latent representation 'x'.
                - If inference and not batch_inf: Returns logits for the last token and None.
                - If inference: Returns logits for all tokens and None.
                - Otherwise: Returns logits, loss, and loss tensor.
        Notes:
            - The forward pass is divided into multiple stages:
                1. Embedding computation using `latent_forward` with start_layer=0 and end_layer=0.
                2. Forwarding through the transformer layers from start_layer=1 to end_layer.
                3. If end_layer exceeds the number of layers, applies layer normalization and computes logits.
            - Loss computation includes:
                1. Cross-entropy loss for the standard logits.
                2. Auxiliary loss using Seer logits.
                3. Z-loss regularization for both standard and Seer logits.
        """
                
        inference = targets is None

        if end_layer is None:
            end_layer = self.config.n_layer + 1

        loss_tensor = torch.zeros((6, ))

        assert start_layer == 0 # To keep things simple, requires inputs to be token ids.

        # forward the GPT model itself
        emb = self.latent_forward(input, start_layer = 0, end_layer = 0, start_pos = start_pos, kv_cache = False) # Only calculates embedding
        x = self.latent_forward(emb, start_layer = 1, end_layer = end_layer, start_pos = start_pos, kv_cache = kv_cache and inference and not batch_inf)

        if end_layer <= self.config.n_layer:
            return x

        x = self.transformer.ln_f(x)
        assert len(x.size()) == 3, (x.size())

        if inference and not batch_inf:
            try:
                return self.lm_head(x[:, [-1], :]), None
            except Exception as e:
                raise Exception(f"Error Unembedding: {e}. x_in: ({x.size()}), x: {x.size()}. Start Pos: {start_pos}")
        elif inference:
            return self.lm_head(x), None

        # if we are given some desired targets also calculate the loss
        st_logits = self.lm_head(x) # Tokens 2:ctx_len + 1
        loss = ce_loss(st_logits, targets)
        loss_tensor[0] = loss
        assert not torch.any(torch.isnan(loss))

        if evaluate:
            return st_logits, loss

        h = self.seer_forward(emb, device = x.device)
        se_logits = self.lm_head(h) # 1 : ctx_len
        loss = aux_seer_loss(st_logits, se_logits, input, loss, loss_tensor, lamda = self.config.lamda)
        assert not torch.any(torch.isnan(loss))

        st_z = z_loss(st_logits)
        loss_tensor[1] = st_z
        loss = loss + st_z
        assert not torch.any(torch.isnan(loss))

        se_z = z_loss(se_logits)
        loss_tensor[5] = se_z
        loss = loss + se_z
        assert not torch.any(torch.isnan(loss))

        return st_logits, loss, loss_tensor

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        blks = self.transformer.h + self.seer.h
        for block in blks:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer + cfg.n_slayer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

@dataclass
class MTPGPTConfig(GPTConfig):
    k: int = 5
    discount_rate: float = 0.99

class MTPGPT(GPT):
    def __init__(self, config: MTPGPTConfig):
        super().__init__(config)
        assert config.k > 0
        self.heads = nn.ModuleList(
            [
                Block(config) for _ in range(config.k)
            ]
        )

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.2 / math.sqrt(2 * config.n_layer))
        
        num_params = self.get_num_params()
        if num_params < 1e9:
            print(f"number of parameters in {self.config.k}-MTP-GPT: %.2fM" % (num_params/1e6,))
        else:
            print(f"number of parameters in {self.config.k}-MTP-GPT: %.2fB" % (num_params/1e9,))        

    def get_num_params(self, non_embedding=True, train=True):
        num_params = super().get_num_params(non_embedding, train)

        if not train:
            num_params -= sum(param.numel() for param in self.heads.parameters())
        
        return num_params
    
    def forward(self, x, start_layer = 0, end_layer = None, targets = None, evaluate = False, batch_inf = False, start_pos = 0, kv_cache = False, k = 1):
        """
            Forward pass for MTP-GPT.

            Arguments:
                x: Input tensor of shape (batch_size, sequence_length) if using start_layer = 0 else (batch_size, sequence_length, embedding_dim) [processes intermediate latents].
                start_layer: The starting layer index for the forward pass.
                end_layer: The ending layer index for the forward pass. If None, defaults to the total number of layers + 1.
                targets: Target tensor for loss computation of shape (batch_size, sequence_length)
                evaluate: Boolean flag indicating evaluation mode.
                batch_inf: Boolean flag for batch inference mode (that is, generate predictions for all tokens in the sequence in parallel - used during GRPO).
                start_pos: The starting position index for positional embeddings.
                kv_cache: Boolean flag indicating whether to use key-value caching for attention.
                k: The head index to use for the final decoding layer.
            
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
                - If 'end_layer' is less than or equal to the number of layers, returns the latent representation 'x'.
                - If in inference mode and not batch_inf, returns logits for the last token and None.
                - If in inference mode and batch_inf, returns logits for all tokens and None.
                - Otherwise, returns logits, loss, and loss tensor.
        """
        inference = targets is None
        loss_tensor = torch.zeros((2, ))

        if end_layer is None:
            end_layer = self.config.n_layer + 1

        # forward the GPT model itself
        x = self.latent_forward(x, start_layer = start_layer, end_layer = end_layer, start_pos = start_pos, kv_cache = kv_cache and inference and not batch_inf)

        if end_layer <= self.config.n_layer:
            return x

        x = self.heads[k](x)
        x = self.transformer.ln_f(x)

        if inference and not batch_inf:
            try:
                return self.lm_head(x[:, [-1], :]), None
            except Exception as e:
                raise Exception(f"Error Unembedding: {e}. x_in: ({x.size()}), x: {x.size()}. Start Pos: {start_pos}")
        elif inference:
            return self.lm_head(x), None

        # if we are given some desired targets also calculate the loss
        st_logits = self.lm_head(x) # Tokens 2:ctx_len + 1
        loss = ce_loss(st_logits, targets)
        loss_tensor[0] = loss
        assert not torch.any(torch.isnan(loss))

        if evaluate:
            return st_logits, loss

        st_z = z_loss(st_logits)
        loss_tensor[1] = st_z
        loss = loss + st_z
        assert not torch.any(torch.isnan(loss))

        return st_logits, loss, loss_tensor

    def compute_gradient(self, X, Y, gradient_accumulation_steps, ctx=nullcontext(), scaler=None, ddp_model = None):
        model = ddp_model or self
        loss_tensors = []
        total_loss = 0

        with ctx:
            trunk_latent = model(X, end_layer = self.config.n_layer)
            # trunk_latent.requires_grad_(True)
            trunk_latent_k = trunk_latent

        if hasattr(model, "require_backward_grad_sync"):
            grad_sync = model.require_backward_grad_sync
            model.require_backward_grad_sync = False

        later_params = [p for ps in [self.transformer.ln_f.parameters(), self.lm_head.parameters()] for p in ps]

        loss = 0

        for k in range(self.config.k):
            if k == self.config.k - 1 and hasattr(model, "require_backward_grad_sync"):
                model.require_backward_grad_sync = grad_sync
            with ctx:
                _, k_loss, micro_loss_tensor = model(trunk_latent_k, targets = Y, start_layer = self.config.n_layer, end_layer = self.config.n_layer + 1, k = k)
                k_loss = k_loss / gradient_accumulation_steps

                # Discount Future Losses
                k_loss = k_loss * (self.config.discount_rate) ** k
                micro_loss_tensor = micro_loss_tensor * (self.config.discount_rate) ** k

                # Loss Logging
                loss_tensors.append(micro_loss_tensor)
                total_loss = total_loss + k_loss.clone().detach()
                loss = loss + k_loss

            # backward pass, with gradient scaling if training in fp16 
            # autograd.backward(scaler.scale(loss)) #, inputs = (trunk_latent, *([p for p in self.heads[k].parameters()] + later_params)), retain_graph = (k < self.config.k - 1), create_graph = False)

            trunk_latent_k = trunk_latent_k[:, :-1]
            Y = Y[:, 1:]

        scaler.scale(loss).backward()
        # trunk_latent.backward(trunk_latent.grad)
        # trunk_latent.grad.zero_()

        return total_loss, torch.cat(loss_tensors)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        blks = self.transformer.h + self.heads
        for block in blks:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer + 1, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

model_configs = {"gpt": GPTConfig, "sf-gpt": SFGPTConfig, "mtp-gpt": MTPGPTConfig}
models = {"gpt": GPT, "sf-gpt": SFGPT, "mtp-gpt": MTPGPT}

def name_losses(loss_tensor, architecture):
    loss_names = ["st_ce", "st_z"]
    if architecture == "sf-gpt":
        loss_names += [
            "kld", "se_e", "se_ce", "se_z"
        ]
    elif architecture == "mtp-gpt":
        K = (loss_tensor.size(0) // 2)
        loss_names = sum(
            [
                [f"st_ce_{k}", f"st_z_{k}"] for k in range(K)
            ], 
            []
        )
    
    return {
        loss_names[idx]: loss_tensor[idx].item() for idx in range(loss_tensor.size(0))
    }

@dataclass
class ProbeHeadConfig:
    n_embd: int
    n_classes: int
    task: str = "classification" # Options: classification, regression
    architecture: str = "linear" # Options: linear, transformer+linear

class ProbeHead(nn.Module):
    def __init__(self, config: ProbeHeadConfig, base_config: GPTConfig):
        super().__init__()
        self.config = config

        head_layers = []

        for layer in config.architecture.split("+")[:-1]:
            if layer == "transformer":
                head_layers.append(Block(base_config))
            elif layer == "linear":
                head_layers.append(nn.Linear(config.n_embd, config.n_embd))
        
        if config.architecture.split("+")[-1] != "linear":
            raise NotImplementedError("Only linear output heads are implemented for ProbeHead")
        elif config.architecture.split("+")[-1] == "linear":
            if config.task == "classification":
                head_layers.append(nn.Linear(config.n_embd, config.n_classes))
                head_layers.append(nn.Softmax(dim = -1))
                self.loss_fn = nn.CrossEntropyLoss()
            elif config.task == "regression":
                head_layers.append(nn.Linear(config.n_embd, 1))
                head_layers.append(nn.Tanh())
                self.loss_fn = nn.MSELoss()

        assert len(head_layers) > 0, "ProbeHead must have at least one layer"

        self.head = nn.Sequential(*head_layers)
    
    def forward(self, x, targets = None, evaluate = False):
        """
            Forward Pass for the Probe Head.
            Args:
                x (torch.Tensor): Input tensor to the probe head (B, S, D).
                targets (torch.Tensor, optional): Target tensor for loss computation. If None, inference mode is assumed. Defaults to None.
                evaluate (bool, optional): If True, returns logits and loss for evaluation. Defaults to False.
            Returns:
                Union[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                - If in inference mode: Returns logits and None.
                - If 'evaluate' is True: Returns logits and loss.
                - Otherwise: Returns logits and loss.
        """
        # Probe Probabilities (B, S, C)
        probs = self.head(x)
        valid_token_filter = ~torch.isnan(targets) if targets is not None else None

        filtered_probs = probs[valid_token_filter]
        filtered_targets = targets[valid_token_filter]

        if self.config.task == "classification":
            filtered_targets = filtered_targets.long()

        loss = self.loss_fn(
            filtered_probs,
            filtered_targets
        ) if targets is not None else None

        return probs, loss

class ProbeWrapper(nn.Module):
    def __init__(self, base_model: GPT, probe_head_config: ProbeHeadConfig):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False
        
        self.white_turn_probes = nn.ModuleList(
            [
                ProbeHead(probe_head_config, base_model.config)
                for _ in range(base_model.config.n_layer)
            ]
        )
        self.black_turn_probes = nn.ModuleList(
            [
                ProbeHead(probe_head_config, base_model.config)
                for _ in range(base_model.config.n_layer)   
            ]
        )

    def forward(self, x, targets = None, evaluate = False):
        """
        Forward Pass for the Probe Wrapper.

        Args:
            x (torch.Tensor): Input tensor to the model.
            targets (torch.Tensor, optional): Target tensor for loss computation. If None, inference mode is assumed. Defaults to None.
            evaluate (bool, optional): If True, returns logits and base loss for evaluation. Defaults to False.

        Returns:
            Union[List[torch.Tensor], Tuple[List[torch.Tensor], Optional[torch.Tensor]]]:
            - If in inference mode: A list of probe logits for each layer and None.
            - If 'evaluate' is True: A list of probe logits for each layer, the total loss and the per-layer loss components.
            - Otherwise: A list of probe logits for each layer, the total loss and the per-layer loss components.
        """
        inference = targets is None
        probe_outputs = []
        total_loss = 0
        loss_components = []

        for layer_idx in range(self.base_model.config.n_layer):
            # Latent representation at Layer layer_idx (B, S, D)
            latent = self.base_model.latent_forward(x, start_layer = layer_idx, end_layer = layer_idx + 1)
            x = latent

            # Probe Probabilities of White Turns at Layer layer_idx (B, S, C)
            white_probe_probs, white_probe_loss = self.white_turn_probes[layer_idx](
                latent[:, ::2, :], 
                targets = targets[:, ::2],
                evaluate = evaluate
            ) # Probe Probabilities: (B, S/2, C)

            black_probe_probs, black_probe_loss = self.black_turn_probes[layer_idx](
                latent[:, 1::2, :], 
                targets = targets[:, 1::2],
                evaluate = evaluate
            ) # Probe Probabilities: (B, S/2, C)
            
            probe_probs = torch.stack(
                (white_probe_probs, black_probe_probs),
                dim = 3
            ).transpose(-1, -2).flatten(start_dim = 1, end_dim = 2) # Final Probe Logits: (B, S, C)

            probe_outputs.append(probe_probs) 

            # If not inference, accumulate loss
            if inference:
                continue
            
            total_loss += white_probe_loss + black_probe_loss
            loss_components.append(white_probe_loss.item() + black_probe_loss.item())

        if inference:
            return probe_outputs, None

        return probe_outputs, total_loss, torch.tensor(loss_components)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            [
                {
                'params': [p for p in self.parameters()],
                'weight_decay': weight_decay
                }
            ], 
            lr=learning_rate, 
            betas=betas, 
            **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def reset_grad(self):
        for p in self.parameters():
            if p.grad is not None and p.requires_grad:
                p.grad.zero_()

def name_probe_losses(micro_loss_tensor):
    loss_names = []
    for layer_idx in range(micro_loss_tensor.size(0)):
        loss_names.append(f"layer_{layer_idx}_probe_loss")
    
    return {
        loss_names[idx]: micro_loss_tensor[idx].item() for idx in range(micro_loss_tensor.size(0))
    }