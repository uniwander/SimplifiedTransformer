import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN
import math


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        """
        Root Mean Square Layer Normalization, from https://github.com/bzhangGo/rmsnorm
        :param d: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * self.d ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed


class LeakyReLU(nn.Module):
    # LeakyReLU nonlinearity.
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)
    
class SimplifiedTransformerBlock(nn.Module):
    """
    A customisable GPT2Block that implements Pre-LN, parallel, and skipless blocks.
    """
    def __init__(self, hidden_size, ff_mult=4, 
                 max_position_embeddings=128,
                 n_layer=12,
                 num_attention_heads=12,
                 scale_attn_weights=False,
                 scale_attn_by_inverse_layer_idx=False,
                 first_layer_value_resid_gain=1,
                 value_resid_gain=0,
                 value_skip_gain=1,
                 val_init_type="id",
                 trainable_value_gains=True,
                 trainable_proj_gains=True,
                 last_layer_proj_resid_gain=None,
                 proj_resid_gain=0,
                 proj_skip_gain=1,
                 proj_init_type="id",
                 query_init_std=0,
                 key_init_std=None,
                 val_proj_init_std=None,
                 attn_mat_resid_gain=1,
                 trainable_attn_mat_gains=True,
                 attn_mat_skip_gain=1,
                 centre_attn=True,
                 centre_attn_gain=1,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 is_cross_attention=False,
                 parallel_layers=True, 
                 norm_type="none", 
                 layer_norm_epsilon=1e-08, 
                 norm_position='pre', 
                 attn_block_resid_gain=1,
                 trainable_attn_block_gains=True, 
                 attn_block_skip_gain=0,  
                 mlp_block_resid_gain=0.1, 
                 trainable_mlp_block_gains=True, 
                 mlp_block_skip_gain=0, 
                 activation_function="leaky_relu",
                 lrelu_neg_slope=0, 
                 mlp_proj_init_std=None,
                 decoder_only=False,
                 layer_idx=None):
        super().__init__()
        self.parallel_layers = parallel_layers
        self.layer_idx = layer_idx

        if norm_type == "ln":
            self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        elif norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
            self.ln_2 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        elif norm_type == "none":
            # always use LN in first layer to normalise input activation norms.
            self.ln_1 = (
                nn.Identity()
                if layer_idx > 0
                else nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            )
            self.ln_2 = nn.Identity()
        else:
            raise NotImplementedError

        self.norm_position = norm_position

        self.attn = SimplifiedAttention(
                 max_position_embeddings=max_position_embeddings,
                 n_layer=n_layer,
                 hidden_size=hidden_size,
                 num_attention_heads=num_attention_heads,
                 scale_attn_weights=scale_attn_weights,
                 scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
                 first_layer_value_resid_gain=first_layer_value_resid_gain,
                 value_resid_gain=value_resid_gain,
                 value_skip_gain=value_skip_gain,
                 val_init_type=val_init_type,
                 trainable_value_gains=trainable_value_gains,
                 trainable_proj_gains=trainable_proj_gains,
                 last_layer_proj_resid_gain=last_layer_proj_resid_gain,
                 proj_resid_gain=proj_resid_gain,
                 proj_skip_gain=proj_skip_gain,
                 proj_init_type=proj_init_type,
                 query_init_std=query_init_std,
                 key_init_std=key_init_std,
                 val_proj_init_std=val_proj_init_std,
                 attn_mat_resid_gain=attn_mat_resid_gain,
                 trainable_attn_mat_gains=trainable_attn_mat_gains,
                 attn_mat_skip_gain=attn_mat_skip_gain,
                 centre_attn=centre_attn,
                 centre_attn_gain=centre_attn_gain,
                 attn_pdrop=attn_pdrop,
                 resid_pdrop=resid_pdrop,
                 is_cross_attention=is_cross_attention,
                 decoder_only=decoder_only,
                 layer_idx=layer_idx)

        self.mlp = SimplifiedMLP(ff_mult, hidden_size, activation_function,
                 lrelu_neg_slope, resid_pdrop, mlp_proj_init_std)

        self.attn_block_resid_gain = nn.Parameter(
            torch.Tensor([attn_block_resid_gain]),
            requires_grad=trainable_attn_block_gains,
        )
        self.attn_block_skip_gain = nn.Parameter(
            torch.Tensor([attn_block_skip_gain]),
            requires_grad=trainable_attn_block_gains
            and not self.parallel_layers,
        )

        self.mlp_block_resid_gain = nn.Parameter(
            torch.Tensor([mlp_block_resid_gain]),
            requires_grad=trainable_mlp_block_gains,
        )
        self.mlp_block_skip_gain = nn.Parameter(
            torch.Tensor([mlp_block_skip_gain]),
            requires_grad=trainable_mlp_block_gains and not self.parallel_layers,
        )

        self.add_attn_skip = attn_block_skip_gain != 0
        self.add_mlp_skip = mlp_block_skip_gain != 0

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        val_len: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        if self.norm_position == "post":
            hidden_states = self.ln_1(hidden_states)
        skip_branch = hidden_states
        if self.norm_position == "pre":
            hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            val_len=val_len,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        if self.parallel_layers:
            # Parallel block of GPT-J
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = (
                self.mlp_block_resid_gain * feed_forward_hidden_states
                + self.attn_block_resid_gain * attn_output
            )
            if self.add_mlp_skip:
                hidden_states += self.mlp_block_skip_gain * skip_branch

        else:
            # Attention residual connection
            hidden_states = self.attn_block_resid_gain * attn_output
            if self.add_attn_skip:
                hidden_states += self.attn_block_skip_gain * skip_branch

            if self.norm_position == "post":
                hidden_states = self.ln_2(hidden_states)

            skip_branch = hidden_states

            if self.norm_position == "pre":
                hidden_states = self.ln_2(hidden_states)

            feed_forward_hidden_states = self.mlp(hidden_states)

            # MLP residual connection
            hidden_states = self.mlp_block_resid_gain * feed_forward_hidden_states
            if self.add_mlp_skip:
                hidden_states += self.mlp_block_skip_gain * skip_branch

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs[0]


class SimplifiedAttention(nn.Module):
    """
    A customisable Attn sub-block that can implement Shaped Attention, and identity value/projection weights.
    """
    def __init__(self, max_position_embeddings=128,
                 n_layer=12,
                 hidden_size=768,
                 num_attention_heads=12,
                 scale_attn_weights=False,
                 scale_attn_by_inverse_layer_idx=False,
                 first_layer_value_resid_gain=1,
                 value_resid_gain=0,
                 value_skip_gain=1,
                 val_init_type="id",
                 trainable_value_gains=True,
                 trainable_proj_gains=True,
                 last_layer_proj_resid_gain=None,
                 proj_resid_gain=0,
                 proj_skip_gain=1,
                 proj_init_type="id",
                 query_init_std=0,
                 key_init_std=None,
                 val_proj_init_std=None,
                 attn_mat_resid_gain=1,
                 trainable_attn_mat_gains=True,
                 attn_mat_skip_gain=1,
                 centre_attn=True,
                 centre_attn_gain=1,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 decoder_only=False,
                 is_cross_attention=False, layer_idx=None):
        super().__init__()
        assert is_cross_attention == False
        max_positions = max_position_embeddings

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = scale_attn_weights

        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.decoder_only = decoder_only

        self.qk_attn = SimplifiedConv1D(
            2 * self.embed_dim,
            self.embed_dim,
        )

        if first_layer_value_resid_gain is not None and layer_idx == 0:
            value_resid_gain = first_layer_value_resid_gain
        else:
            value_resid_gain = value_resid_gain
        if (
            value_skip_gain != 1
            or value_resid_gain != 0
            or val_init_type != "id"
        ):
            self.v_attn = SimplifiedConv1D(
                self.embed_dim,
                self.embed_dim,
                resid_gain=value_resid_gain,
                skip_gain=value_skip_gain,
                trainable_gains=trainable_value_gains,
                init_type=val_init_type,
                bias=False,
            )
        else:
            self.v_attn = nn.Identity()

        if (
            last_layer_proj_resid_gain is not None
            and layer_idx == n_layer - 1
        ):
            proj_resid_gain = last_layer_proj_resid_gain
        else:
            proj_resid_gain = proj_resid_gain
        if (
            proj_skip_gain != 1
            or proj_resid_gain != 0
            or proj_init_type != "id"
        ):
            self.c_proj = SimplifiedConv1D(
                self.embed_dim,
                self.embed_dim,
                resid_gain=proj_resid_gain,
                skip_gain=proj_skip_gain,
                trainable_gains=trainable_proj_gains,
                init_type=proj_init_type,
                bias=False,
            )
        else:
            self.c_proj = nn.Identity()

        self.split_size = self.embed_dim
        query_weight, key_weight = self.qk_attn.weight.data.split(
            self.split_size, dim=1
        )

        if query_init_std is not None:
            query_weight.normal_(mean=0.0, std=query_init_std)

        if key_init_std is not None:
            key_weight.normal_(mean=0.0, std=key_init_std)

        if val_proj_init_std is not None:
            self.v_attn.weight.data.normal_(mean=0.0, std=val_proj_init_std)
            self.c_proj.weight.data.normal_(mean=0.0, std=val_proj_init_std)

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.pruned_heads = set()

        self.attn_mat_resid_gain = nn.Parameter(
            attn_mat_resid_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains,
        )
        self.attn_mat_skip_gain = nn.Parameter(
            attn_mat_skip_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains,
        )

        self.centre_attn = centre_attn
        # Centered attention, from https://arxiv.org/abs/2306.17759
        uniform_causal_attn_mat = torch.ones(
            (max_positions, max_positions), dtype=torch.float32
        ) / torch.arange(1, max_positions + 1).view(-1, 1)
        self.register_buffer(
            "uniform_causal_attn_mat",
            torch.tril(
                uniform_causal_attn_mat,
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.centre_attn_gain = nn.Parameter(
            centre_attn_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=trainable_attn_mat_gains
            and centre_attn_gain != 0,
        )
        self.register_buffer(
            "diag",
            torch.eye(max_positions).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        
        if self.decoder_only:
            """
            create a mask matrix with upper diagonal values all to be negative infinity so that
            one token only attends to itself and previous tokens
            """
            attn_weights_size = torch.Size([query.size(2), query.size(2)])
            attention_mask  = torch.zeros(attn_weights_size)
            attention_mask[:] = float('-inf')
            attention_mask = torch.triu(attention_mask, diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).repeat(query.size(1), 1, 1)
            attention_mask = attention_mask.unsqueeze(0).repeat(query.size(0), 1, 1, 1)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        new_attn_weights = self.attn_mat_resid_gain * attn_weights.type(value.dtype)

        if self.centre_attn:
            post_sm_bias_matrix = (
                self.attn_mat_skip_gain * self.diag[:, :, :key_length, :key_length]
            ) - self.centre_attn_gain * (
                self.uniform_causal_attn_mat[
                    :, :, key_length - query_length : key_length, :key_length
                ]
            )
            new_attn_weights = new_attn_weights + post_sm_bias_matrix

        new_attn_weights = self.attn_dropout(new_attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            new_attn_weights = new_attn_weights * head_mask

        attn_output = torch.matmul(new_attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        val_len: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert encoder_hidden_states is None
        (query, key) = self.qk_attn(hidden_states).split(self.split_size, dim=2)
        value = self.v_attn(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if type(val_len)  == torch.Tensor:
            """
            if the sequences in a minibatch have different valid length, then create an attention
            mask so that any tokens after the valid length for a sequence will not be in attention
            """
            attn_weights_size = torch.Size([query.size(0), query.size(1), query.size(2), query.size(2)])
            attention_mask  = torch.zeros(attn_weights_size)
            maxlen = hidden_states.size(1)
            mask = torch.arange(maxlen)[None, :] < val_len[:, None]
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            mask = mask.unsqueeze(2).repeat(1, 1, query.size(2), 1)
            attention_mask[~mask] = float('-inf')
        else:
            attention_mask = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        proj_output = self.c_proj(attn_output)
        proj_output = self.resid_dropout(proj_output)

        outputs = (proj_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class SimplifiedMLP(nn.Module):
    def __init__(self, ff_mult, hidden_size=256, activation_function='leaky_relu',
                 lrelu_neg_slope=0, resid_pdrop=0.1, mlp_proj_init_std=None):
        super().__init__()
        embed_dim = hidden_size
        intermediate_size = hidden_size*ff_mult

        self.c_fc = SimplifiedConv1D(intermediate_size, embed_dim, bias=False)
        self.c_proj = SimplifiedConv1D(embed_dim, intermediate_size, bias=False)

        if activation_function != "leaky_relu":
            self.act = ACT2FN[activation_function]
        else:
            self.act = LeakyReLU(negative_slope=lrelu_neg_slope)

        self.dropout = nn.Dropout(resid_pdrop)

        if mlp_proj_init_std is not None:
            nn.init.normal_(self.c_proj.weight, std=mlp_proj_init_std)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SimplifiedConv1D(nn.Module):
    """
    (Linear) 1D-convolutional layer that can be reparameterised into skip (see Eq. 6 of paper).

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
        resid_gain (float): Residual weight.
        skip_gain (float): Skip weight, if None then defaults to standard Linear layer.
        trainable_gains (bool): Whether or not gains are trainable.
        init_type (one of ["orth", "id", "normal"]): Type of weight initialisation.
        bias (bool): Whether or not to use bias parameters.
    """

    def __init__(
        self,
        nf,
        nx,
        resid_gain=None,
        skip_gain=None,
        trainable_gains=False,
        init_type="normal",
        bias=True,
    ):
        super().__init__()
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)
        
        if skip_gain is None:
            # Standard linear layer
            self.weight = nn.Parameter(torch.empty(nx, nf))
            if init_type == "orth":
                nn.init.orthogonal_(self.weight)
            elif init_type == "id":
                self.weight.data = torch.eye(nx)
            elif init_type == "normal":
                nn.init.normal_(self.weight, std=0.02)
            else:
                raise NotImplementedError
            self.skip = False

        elif skip_gain is not None:
            # Reparameterised linear layer
            assert nx == nf
            self.resid_gain = nn.Parameter(
                torch.Tensor([resid_gain]), requires_grad=trainable_gains
            )
            self.skip_gain = nn.Parameter(
                torch.Tensor([skip_gain]),
                requires_grad=trainable_gains,
            )

            self.weight = nn.Parameter(torch.zeros(nx, nx))
            if init_type == "orth":
                self.id = nn.init.orthogonal_(torch.empty(nx, nx)).cuda()
            elif init_type == "id":
                self.id = torch.eye(nx).cuda()
            elif init_type == "normal":
                self.id = nn.init.normal_(
                    torch.empty(nx, nx), std=1 / math.sqrt(nx)
                ).cuda()
            else:
                raise NotImplementedError
            self.skip = True
            self.init_type = init_type

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.skip:
            self.id = self.id.to(self.skip_gain.device)
            if self.resid_gain == 0 and self.init_type == "id":
                x = torch.add(self.bias, x * self.skip_gain)
            else:
                x = torch.addmm(
                    self.bias,
                    x.view(-1, x.size(-1)),
                    self.resid_gain * self.weight + self.skip_gain * self.id,
                )
        else:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)

        return x
