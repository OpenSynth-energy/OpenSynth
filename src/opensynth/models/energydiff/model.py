# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
#
# The EnergyDiff model is made available via Nan Lin and Pedro P. Vergara
# from the Delft University of Technology. Nan Lin and Pedro P. Vergara are
# funded via the ALIGN4Energy Project (with project number NWA.1389.20.251) of
# the research programme NWA ORC 2020 which is (partly) financed by the Dutch
# Research Council (NWO), The Netherland.
"""
Name: nn modules for diffusion
Author: Nan Lin (sentient-codebot)
Date: Nov 2024

"""
import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

# model components


class SinusoidalPosEmb(nn.Module):
    """for position t, dimension i of a d-dim vector, the embedding is
    1/(10000**(i/(d/2-1)))

    dim must be even.

    Args:
        dim (int): dimension of the input

    Forward:
        pos: (batch, sequence)

    Return:
        emb: (batch, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dimension must be even")
        self.dim = dim

    def forward(self, pos: Tensor) -> Tensor:
        """
        Args:
            pos: (batch, sequence)
        """
        device = pos.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device)
            * -math.log(10000)
            / (half_dim - 1)
        )  # shape: (dim/2,)
        emb = pos.unsqueeze(-1) * emb.unsqueeze(
            0
        )  # shape: (batch, 1) * (1, dim/2) -> (batch, dim/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch, dim)
        return emb


class RMSNorm(nn.Module):
    """do normalization over channel dimension \
        and multiply with a learnable parameter g"""

    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=2) * self.g * math.sqrt(x.shape[2])


class SelfAttention(nn.Module):
    r"""(batch, sequence, dim) -> (batch, sequence, dim)"""

    def __init__(self, dim: int, dim_head: int, num_head: int = 4):
        super().__init__()
        self.dim_inout = dim
        self.scale = dim_head**-0.5
        self.num_head = num_head
        self.hidden_dim = dim_head * num_head
        self.to_qkv = nn.Linear(self.dim_inout, self.hidden_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim_inout), RMSNorm(self.dim_inout)
        )
        self.rms_norm_q = nn.Sequential(
            Rearrange("b num_head l d -> b l (num_head d)"),
            RMSNorm(self.hidden_dim),  # in dim 2
            Rearrange(
                "b l (num_head d) -> b num_head l d", num_head=self.num_head
            ),
        )
        self.rms_norm_k = nn.Sequential(
            Rearrange("b num_head l d -> b l (num_head d)"),
            RMSNorm(self.hidden_dim),  # in dim 1
            Rearrange(
                "b l (num_head d) -> b num_head l d", num_head=self.num_head
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(
            3, dim=2
        )  # 3 * (batch, sequence, hidden_dim)
        q, k, v = map(
            lambda t: rearrange(
                t, "b l (num_head d) -> b num_head l d", num_head=self.num_head
            ),
            qkv,
        )  # shape: 3 * (batch, num_head, sequence, dim_head)

        # such normalization: one independent attention for each head
        q = self.rms_norm_q(q)
        k = self.rms_norm_k(k)
        q = q * self.scale  # scale

        sim = einsum(q, k, "b h l dq, b h n dq -> b h l n")
        sim = sim.softmax(dim=-1)  # over keys sequence
        out = einsum(
            sim, v, "b h l n, b h n dv -> b h dv l"
        )  # shape: (batch, num_head, sequence, dim_head)
        out = rearrange(
            out, "b h dv l -> b l (h dv)"
        )  # shape: (batch, sequence, hidden_dim)
        return self.to_out(out)  # shape: (batch, sequence, dim)


class DecoderBlock(nn.Module):
    """GPT2-style decoder block

    input: (batch, sequence, dim)
    condition: (batch, #sequence, dim)
    output: (batch, sequence, dim)
    """

    def __init__(
        self,
        dim_base: int,  # model dimension
        num_attn_head: int = 4,
        dim_head: None | int = None,  # default dim_base // num_head
        dim_feedforward: int = 2048,
        dropout: float = 0.1,  # at ff layer
        conditioning: bool = False,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.dim_head = dim_head if dim_head else dim_base // num_attn_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning

        self.ln_1 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )
        self.attn = SelfAttention(
            dim_base,
            dim_head=dim_base // self.num_attn_head,
            num_head=self.num_attn_head,
        )
        self.ln_2 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )

        # no cross attention compared with the GPT2
        self.mlp = nn.Sequential(
            nn.Linear(dim_base, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_base),
            nn.Dropout(self.dropout),
        )

        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )

    def forward(
        self,
        x: Tensor,
        c: None | Tensor = None,
    ) -> Tensor:
        if c is not None and self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(
                c
            )  # shape: (batch, 1, channel * 6)
            (
                scale_attn,
                shift_attn,
                gate_attn,
                scale_mlp,
                shift_mlp,
                gate_mlp,
            ) = cond_scale_shift_gate.chunk(6, dim=2)
        else:
            (
                scale_attn,
                shift_attn,
                gate_attn,
                scale_mlp,
                shift_mlp,
                gate_mlp,
            ) = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

        # attention
        x = x + gate_attn * self.attn(
            self.ln_1(x) * (1.0 + scale_attn) + shift_attn
        )

        # mlp
        x = x + gate_mlp * self.mlp(
            self.ln_2(x) * (1.0 + scale_mlp) + shift_mlp
        )

        return x


class DecoderTransformer(nn.Module):
    """GPT2-style decoder-only transformer

    input: (batch, sequence, dim)
    condition: (batch, #sequence, dim)
    output: (batch, sequence, dim)
    """

    def __init__(
        self,
        dim_base: int,
        num_attn_head: int = 4,
        num_layer=6,
        dim_feedforward=2048,  # need to be big for transformer
        dropout=0.1,  # at ff layer
        conditioning: bool = False,  # for conditioning on t (diffusion)
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.num_layer = num_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning

        # layers
        self.decoders = nn.ModuleList([])
        for idx in range(self.num_layer):
            self.decoders.append(
                DecoderBlock(
                    dim_base=dim_base,
                    num_attn_head=num_attn_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    conditioning=conditioning,
                )  # shape: (batch, sequence, dim) -> (batch, sequence, dim)
            )

    def forward(
        self,
        x: Tensor,
        c: None | Tensor = None,
    ) -> Tensor:
        for decoder in self.decoders:
            x = decoder(x, c)
        return x  # shape: (batch, sequence, dim)


class InitProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class DenoisingTransformer(nn.Module):
    """wraps DecoderTransformer with additional layers for denoising
    input:
        - x: (batch, sequence, dim) # TODO maybe adapt?
        - time: (batch, )

    return:
        - x: (batch, sequence, dim)

    """

    def __init__(
        self,
        dim_base: int,
        dim_in: int,  # input dimension (batch, ..., dim_in)
        num_attn_head: int = 4,
        num_decoder_layer: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        learn_variance: bool = False,  # learnable variance
        disable_init_proj: bool = False,  # disable initial projection
    ):
        super().__init__()
        self.dim_base = dim_base
        self.dim_in = dim_in
        self.dim_out = dim_in * (2 if learn_variance else 1)
        self.num_attn_head = num_attn_head
        self.num_decoder_layer = num_decoder_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learn_variance = learn_variance

        # layers
        time_pos_emb = SinusoidalPosEmb(self.dim_base)
        self.time_mlp = nn.Sequential(
            time_pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base * 3),
        )  # scale, shift, input_to_decoder_blocks

        _transformer_pos_emb = SinusoidalPosEmb(self.dim_base)
        self.transformer_pos_emb = nn.Sequential(
            _transformer_pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base),
        )

        # initial projection
        # self.init_proj = nn.Sequential(
        #     Rearrange('batch seq dim_in -> batch dim_in seq'),
        #     nn.Conv1d(self.dim_in, self.dim_base, kernel_size=5, padding=2),
        #     Rearrange('batch dim_base seq -> batch seq dim_base')
        # ) # (batch, sequence, dim_in) -> (batch, sequence, dim_base)
        if not disable_init_proj:
            self.init_proj = InitProjection(
                self.dim_in, self.dim_base, kernel_size=5, padding=2
            )
        else:
            self.init_proj = nn.Linear(self.dim_in, self.dim_base)

        # decoder
        self.transformer = DecoderTransformer(
            dim_base=dim_base,
            num_attn_head=num_attn_head,
            num_layer=num_decoder_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conditioning=True,  # on timestep t
        )

        # conditioning
        self.final_ln = nn.LayerNorm(
            self.dim_base, elementwise_affine=False, eps=1e-6
        )
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim_base, self.dim_base * 2),
        )

        # final linear
        self.final_linear = nn.Linear(self.dim_base * 2, self.dim_out)

        # intialize weights
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if getattr(module, "weight", None) is not None and not isinstance(
                module, nn.LayerNorm
            ):
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # better not zero out the weights of final output layer
        # the final linear: (dim_base * 2) -> (dim_out)
        nn.init.constant_(self.final_linear.bias, 0)

        for decoder in self.transformer.decoders:
            nn.init.constant_(decoder.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(decoder.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        time: Tensor,
    ) -> Tensor:
        # encoder time
        _encoded_t = self.time_mlp(time)  # shape: (batch, dim_base * 3)
        _encoded_t = rearrange(_encoded_t, "batch dim -> batch 1 dim")
        scale, shift, encoded_t = _encoded_t.chunk(
            3, dim=2
        )  # shape: 3 * (batch, 1, dim_base)

        # init projection
        x = self.init_proj(x)  # shape: (batch, sequence, dim_base)
        x_copy = x.clone()
        x = x * (1 + scale) + shift  # shape: (batch, sequence, dim_base)
        x = F.silu(x)  # shape: (batch, sequence, dim_base)

        # transformer
        pos_seq = torch.arange(
            x.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        pos_emb_seq = self.transformer_pos_emb(
            pos_seq
        )  # shape: (sequence, dim_base)
        pos_emb_seq = rearrange(
            pos_emb_seq, "seq dim -> 1 seq dim"
        )  # shape: (1, sequence, dim_base)
        x = x + pos_emb_seq

        x = self.transformer(
            x, encoded_t
        )  # shape: (batch, sequence, dim_base)

        # final adaln modulation
        scale_final, shift_final = self.final_adaLN_modulation(
            encoded_t
        ).chunk(
            2, dim=2
        )  # shape: 2 * (batch, 1, dim_base)
        x = self.final_ln(x) * (1 + scale_final) + shift_final

        # final linear
        x = torch.concat(
            (x, x_copy), dim=2
        )  # shape: (batch, sequence, dim_base * 2)
        x = self.final_linear(x)

        return x  # shape: (batch, sequence, dim_out)
