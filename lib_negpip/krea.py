# https://github.com/blue-pen5805/ComfyUI-krea2-negpip

from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from backend.diffusion_engine.krea import Krea2 as Krea2Engine
    from backend.nn.krea import Attention, SingleStreamDiT
    from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine
    from modules.prompt_parser import SdConditioning

import torch
import torch.nn.functional as F
from einops import rearrange

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_function
from backend.nn.flux import apply_rope
from backend.text_processing import emphasis, parsing
from lib_negpip.anima import _hook_compile_conditions
from modules import shared


_PATCHED_MODEL = None
_PATCHED_DIT = None


def patch_krea2_negpip(cls: "NegPiP", *, unpatch=False):
    global _PATCHED_MODEL, _PATCHED_DIT

    if unpatch != cls._patched[2]:
        return

    if unpatch:
        if _PATCHED_MODEL is not None:
            _hook_get_learned_conditioning(_PATCHED_MODEL, True)
        if _PATCHED_DIT is not None:
            _hook_dit_forward(_PATCHED_DIT, True)
            _hook_attn_forwards(_PATCHED_DIT, True)
        _hook_compile_conditions(True)

        _PATCHED_MODEL = None
        _PATCHED_DIT = None
        cls._patched[2] = False
        return

    model: "Krea2Engine" = shared.sd_model
    dit: "SingleStreamDiT" = model.forge_objects.unet.model.diffusion_model
    _hook_get_learned_conditioning(model, False)
    _hook_dit_forward(dit, False)
    _hook_attn_forwards(dit, False)
    _hook_compile_conditions(False)

    _PATCHED_MODEL = model
    _PATCHED_DIT = dit
    cls._patched[2] = True


# ================================================================================ #


def _hook_get_learned_conditioning(model: "Krea2Engine", remove: bool):
    if remove:
        if hasattr(model, "_negpip_orig_get_learned_conditioning"):
            if getattr(model.get_learned_conditioning, "_negpip", False):
                model.get_learned_conditioning = model._negpip_orig_get_learned_conditioning
            del model._negpip_orig_get_learned_conditioning
        return

    orig_get_learned_conditioning = model.get_learned_conditioning
    model._negpip_orig_get_learned_conditioning = orig_get_learned_conditioning

    engine: "Qwen3VLTextProcessingEngine" = model.text_processing_engine_qwen

    @torch.inference_mode()
    @wraps(orig_get_learned_conditioning)
    def negpip_learned_conditioning(prompt: "SdConditioning"):
        memory_management.load_model_gpu(model.forge_objects.clip.patcher)

        engine.emphasis = emphasis.get_current_option(shared.opts.emphasis)()
        if any(emphasis.uses_emphasis(x) for x in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = engine.emphasis.name

        crossattn = []
        negpip_mask = []
        _count = 0
        cache = {}

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(engine, line)
            cond, mask = cache[line]

            _count += int((mask < 0).sum())

            crossattn.append(cond)
            negpip_mask.append(mask)

        if _count > 0:
            key = "Negative" if prompt.is_negative_prompt else "Positive"
            print(f"NegPiP Enable ({key}: {_count})")

        return {
            "crossattn": crossattn,
            "c_negpip_mask": negpip_mask,
        }

    negpip_learned_conditioning._negpip = True
    model.get_learned_conditioning = negpip_learned_conditioning


def _tokenize_line_negpip(
    engine: "Qwen3VLTextProcessingEngine", line: str
) -> tuple[list, list[float]]:
    """
    tokenize like Qwen3VLTextProcessingEngine.tokenize_line, but apply the chat
    template once around the whole prompt instead of once per weighted segment,
    so that the weights only cover the user text
    """

    parsed = parsing.parse_prompt_attention(line, engine.emphasis.name)

    if all(weight == 1.0 for _, weight in parsed):
        chunk = engine.tokenize_line(line)[0]
        return chunk.tokens, chunk.multipliers

    prefix, suffix = engine.llama_template.split("{}")
    tokenized = engine.tokenizer([prefix, *(text for text, _ in parsed), suffix])["input_ids"]

    tokens = list(tokenized[0])
    multipliers = [1.0] * len(tokens)

    for segment, (_, weight) in zip(tokenized[1:-1], parsed):
        tokens.extend(segment)
        multipliers.extend([weight] * len(segment))

    tokens.extend(tokenized[-1])
    multipliers.extend([1.0] * len(tokenized[-1]))

    return tokens, multipliers


def _encode_line(
    engine: "Qwen3VLTextProcessingEngine", line: str
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, multipliers = _tokenize_line_negpip(engine, line)

    neutral = [1.0] * len(multipliers)
    magnitude_idx = [i for i, m in enumerate(multipliers) if abs(m) != 1.0]

    if magnitude_idx:
        # apply the weight magnitudes on the encoder output, by lerping between a
        # neutral (empty) encoding and the actual encoding; scaling the input
        # embeddings instead barely has any effect, as Qwen3-VL RMSNorms them away
        reference = [engine.id_pad] * len(tokens)
        z = engine.process_tokens([tokens, reference], [neutral, neutral])
        cond, ref = z[0:1], z[1:2]

        idx = torch.tensor(magnitude_idx, device=cond.device, dtype=torch.long)
        scale = torch.tensor(
            [abs(multipliers[i]) for i in magnitude_idx],
            device=cond.device,
            dtype=cond.dtype,
        ).reshape(1, 1, -1, 1)
        cond[:, :, idx, :] = torch.lerp(ref[:, :, idx, :], cond[:, :, idx, :], scale)
    else:
        cond = engine.process_tokens([tokens], [neutral])

    weights = torch.tensor(multipliers, dtype=torch.float32)
    ones = torch.ones_like(weights)
    mask = torch.where(weights < 0, -ones, ones)

    cond = engine.strip_template(cond, tokens)
    mask = engine.strip_template(mask.reshape(1, 1, -1, 1), tokens)

    if mask.shape[1] < cond.shape[1]:
        mask = F.pad(mask, (0, 0, 0, cond.shape[1] - mask.shape[1]), value=1.0)
    elif mask.shape[1] > cond.shape[1]:
        mask = mask[:, : cond.shape[1]]

    return cond, mask.to(device=cond.device, dtype=cond.dtype)


def _hook_dit_forward(dit: "SingleStreamDiT", remove: bool):
    if remove:
        if hasattr(dit, "_negpip_orig_forward"):
            if getattr(dit.forward, "_negpip", False):
                dit.forward = dit._negpip_orig_forward
            del dit._negpip_orig_forward
        return

    orig_forward = dit.forward
    dit._negpip_orig_forward = orig_forward

    @torch.inference_mode()
    @wraps(orig_forward)
    def negpip_forward(
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options: dict = {},
        **kwargs,
    ):
        negpip_mask: Optional[torch.Tensor] = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
            if negpip_mask.ndim == 4:
                negpip_mask = negpip_mask.squeeze(1)
            transformer_options = {**transformer_options, "negpip_mask": negpip_mask}

        return orig_forward(
            x,
            timesteps,
            context,
            attention_mask=attention_mask,
            transformer_options=transformer_options,
            **kwargs,
        )

    negpip_forward._negpip = True
    dit.forward = negpip_forward


def _hook_attn_forwards(dit: "SingleStreamDiT", remove: bool):
    for block in getattr(dit, "blocks", ()):
        _hook_attn_forward(block.attn, remove)


def _hook_attn_forward(module: "Attention", remove: bool):
    if remove:
        if hasattr(module, "_negpip_orig_forward"):
            if getattr(module.forward, "_negpip", False):
                module.forward = module._negpip_orig_forward
            del module._negpip_orig_forward
        return

    orig_forward = module.forward
    module._negpip_orig_forward = orig_forward

    @torch.inference_mode()
    @wraps(orig_forward)
    def negpip_forward(
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        transformer_options: dict = {},
    ):
        negpip_mask: torch.Tensor = transformer_options.get("negpip_mask", None)
        if negpip_mask is None:
            return orig_forward(x, freqs, mask, transformer_options)

        q, k, v, gate = module.wq(x), module.wk(x), module.wv(x), module.gate(x)

        m = negpip_mask.to(v)
        if (batch := (x.size(0) // m.size(0))) > 1:
            m = m.repeat(batch, 1, 1)
        txtlen = min(m.size(1), v.size(1))
        v[:, :txtlen] = v[:, :txtlen] * m[:, :txtlen]

        q = rearrange(q, "B L (H D) -> B H L D", H=module.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=module.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=module.kvheads)
        q, k = module.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if module.kvheads != module.heads:
            rep = module.heads // module.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = attention_function(q, k, v, module.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return module.wo(out * F.sigmoid(gate))

    negpip_forward._negpip = True
    module.forward = negpip_forward
