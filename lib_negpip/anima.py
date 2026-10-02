# https://github.com/david419kr/sd-webui-negpip/blob/main/scripts/negpip.py

from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from backend.diffusion_engine.anima import Anima as AnimaEngine
    from backend.nn.anima import Anima
    from backend.text_processing.anima_engine import AnimaTextProcessingEngine
    from modules.prompt_parser import SdConditioning

import torch
import torch.nn.functional as F
from einops import rearrange

from backend.nn.anima import SelfCrossAttention
from backend.sampling import condition, sampling_function
from modules import shared


def patch_anima_negpip(cls: "NegPiP", *, unpatch=False):
    if unpatch != cls._patched[1]:
        return

    cls._patched[1] = not cls._patched[1]

    model: "AnimaEngine" = shared.sd_model
    dit: "Anima" = model.forge_objects.unet.model.diffusion_model
    _hook_get_learned_conditioning(model, unpatch)
    _hook_dit_forward(dit, unpatch)
    _hook_forwards(unpatch)
    _hook_compile_conditions(unpatch)


# ================================================================================ #


def _hook_get_learned_conditioning(model: "AnimaEngine", remove: bool):
    if remove:
        if hasattr(model, "orig_forward"):
            model.get_learned_conditioning = model.orig_forward
            del model.orig_forward
        return

    model.orig_forward = model.get_learned_conditioning

    # Forge Neo 2.29.2 renamed the engine (AnimaTextProcessingEngine -> Qwen06Engine)
    engine: "AnimaTextProcessingEngine" = getattr(
        model, "text_processing_engine_anima", None
    ) or model.text_processing_engine_qwen

    @torch.inference_mode()
    @wraps(model.orig_forward)
    def negpip_learned_conditioning(prompt: "SdConditioning"):
        conds = model.orig_forward(prompt)
        assert isinstance(conds, list)
        assert len(prompt) == len(conds)

        crossattn = []
        negpip_mask = []
        _count = 0

        for line, cond in zip(prompt, conds):
            assert isinstance(cond, torch.Tensor)

            cond_data = cond.reshape(-1, cond.shape[-1])
            assert cond_data.ndim == 2

            mask = _build_negpip_mask(
                engine,
                line,
                cond_data.shape[0],
                cond_data.device,
                cond_data.dtype,
            )

            _count += int((mask < 0).sum())

            crossattn.append(cond_data * mask.unsqueeze(-1).to(cond_data))
            negpip_mask.append(mask.unsqueeze(-1).to(cond_data))

        if _count > 0:
            key = "Negative" if prompt.is_negative_prompt else "Positive"
            print(f"NegPiP Enable ({key}: {_count})")

        return {
            "crossattn": torch.stack(crossattn, dim=0),
            "c_negpip_mask": torch.stack(negpip_mask, dim=0),
        }

    model.get_learned_conditioning = negpip_learned_conditioning


def _build_negpip_mask(
    text_processing_engine: "AnimaTextProcessingEngine",
    line: str,
    token_length: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
):
    multipliers = _t5_multipliers(text_processing_engine, line)

    if len(multipliers) == 0:
        return torch.ones(token_length, device=device, dtype=dtype)

    weights = torch.tensor(multipliers, device=device, dtype=dtype)
    ones = torch.ones_like(weights)
    mask = torch.where(weights < 0, -ones, ones)

    if mask.shape[0] < token_length:
        mask = F.pad(mask, (0, token_length - mask.shape[0]), value=1.0)
    elif mask.shape[0] > token_length:
        mask = mask[:token_length]

    return mask


def _t5_multipliers(text_processing_engine, line: str) -> list[float]:
    """the per-token weights the engine multiplies into the T5-aligned conditioning"""

    if hasattr(text_processing_engine, "tokenize_line"):
        multipliers = []
        for chunk in text_processing_engine.tokenize_line(line):
            multipliers.extend(getattr(chunk, "t5_multipliers", []))
        return multipliers

    # Forge Neo 2.29.2+: mirror Qwen06Engine.__call__
    emphasis_name = text_processing_engine.emphasis.name
    if emphasis_name == "Ignore":
        return []

    chunk = text_processing_engine.t5_tokenizer.tokenize_with_weights(
        line, disable_weights=emphasis_name == "None"
    )
    return [weight for _, weight in chunk[0]]


def _hook_dit_forward(dit: "Anima", remove: bool):
    if remove:
        if hasattr(dit, "orig_forward"):
            if getattr(dit.forward, "_negpip", False):
                dit.forward = dit.orig_forward
            del dit.orig_forward
        return

    dit.orig_forward = dit.forward

    @torch.inference_mode()
    @wraps(dit.orig_forward)
    def negpip_forward(
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})

        negpip_mask = kwargs.get("c_negpip_mask", None)
        if negpip_mask is None:
            negpip_mask = torch.ones(
                context.shape[0],
                context.shape[1],
                1,
                device=context.device,
                dtype=context.dtype,
            )

        transformer_options["negpip_mask"] = negpip_mask
        kwargs["transformer_options"] = transformer_options

        return dit.orig_forward(x, timesteps, context, padding_mask, **kwargs)

    negpip_forward._negpip = True
    dit.forward = negpip_forward


def _hook_forwards(remove: bool):
    if remove:
        if hasattr(SelfCrossAttention, "negpip_orig_forward"):
            if getattr(SelfCrossAttention.forward, "_negpip", False):
                SelfCrossAttention.forward = SelfCrossAttention.negpip_orig_forward
            del SelfCrossAttention.negpip_orig_forward
        return

    SelfCrossAttention.negpip_orig_forward = SelfCrossAttention.forward

    @torch.inference_mode()
    @wraps(SelfCrossAttention.negpip_orig_forward)
    def negpip_forward(
        self: SelfCrossAttention,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ):
        if self.is_SelfAttn:
            return self.negpip_orig_forward(x, context, rope_emb, transformer_options)

        negpip_mask: torch.Tensor = transformer_options.get("negpip_mask", None)

        q = self.q_proj(x)
        context_k = x if context is None else context
        context_v = context_k
        if negpip_mask is not None:
            assert negpip_mask.ndim == context_v.ndim
            if (batch := (x.size(0) // negpip_mask.size(0))) > 1:
                negpip_mask = negpip_mask.repeat(batch, 1, 1)
            context_v = context_v * negpip_mask.to(context_v)

        k = self.k_proj(context_k)
        v = self.v_proj(context_v)

        q, k, v = map(
            lambda t: rearrange(
                t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim
            ),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_SelfAttn and rope_emb is not None:
            q = self.apply_rotary_pos_emb(q, rope_emb)
            k = self.apply_rotary_pos_emb(k, rope_emb)

        return self.compute_attention(q, k, v, transformer_options=transformer_options)

    negpip_forward._negpip = True
    SelfCrossAttention.forward = negpip_forward


def _hook_compile_conditions(remove: bool):
    if remove:
        if hasattr(condition, "orig_forward"):
            condition.compile_conditions = condition.orig_forward
            sampling_function.compile_conditions = condition.orig_forward
            del condition.orig_forward
        return

    condition.orig_forward = condition.compile_conditions

    @wraps(condition.orig_forward)
    def compile_conditions(cond):
        if cond is None:
            return None

        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            cross_attn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(cross_attn)}
            if "c_negpip_mask" in cond:
                model_conds["c_negpip_mask"] = condition.Condition(
                    cond["c_negpip_mask"]
                )
            return [dict(cross_attn=cross_attn, model_conds=model_conds)]

        return condition.orig_forward(cond)

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions
