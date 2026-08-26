# https://github.com/blue-pen5805/ComfyUI-krea2-negpip

from contextlib import contextmanager
from contextvars import ContextVar
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
from backend.quant_ops import ck
from backend.text_processing import emphasis, parsing
from lib_negpip.anima import _hook_compile_conditions
from modules import shared

_V_SCALING: ContextVar[float] = ContextVar("negpip_v_scaling", default=0.0)


_PATCHED_MODEL = None
_PATCHED_DIT = None


@contextmanager
def v_scaling_scope(strength: float):
    token = _V_SCALING.set(strength)
    try:
        yield
    finally:
        _V_SCALING.reset(token)


def scope_v_scaling_method(obj, name: str, strength: float):
    current = getattr(obj, name, None)
    if current is None:
        return

    # functools.wraps copies __dict__, so a foreign wrapper built around our
    # scoped function inherits these attributes; only unwrap through them when
    # the self-reference proves the function really is our own wrapper.
    if getattr(current, "_negpip_scoped", None) is current:
        original = current._negpip_original
    else:
        original = current

    @wraps(original)
    def scoped(*args, **kwargs):
        try:
            with v_scaling_scope(strength):
                return original(*args, **kwargs)
        finally:
            if getattr(obj, name, None) is scoped:
                setattr(obj, name, original)

    scoped._negpip_original = original
    scoped._negpip_scoped = scoped
    setattr(obj, name, scoped)


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
        v_scaling = _V_SCALING.get()

        if not prompt.is_negative_prompt:
            references = [*getattr(model, "ref_latents", ())]
            if (ini_latent := getattr(model, "ini_latent", None)) is not None:
                references.insert(0, ini_latent)

            if getattr(shared.opts, "krea2_do_reference", False) and references:
                print("NegPiP Positive Disabled (Krea 2 Reference active)")
                return orig_get_learned_conditioning(prompt)

            # Mirror Forge Neo's no-reference path. In particular, consume a
            # pending img2img latent and prevent latents from a previous job
            # from leaking into the diffusion model.
            if hasattr(model, "ini_latent"):
                model.ini_latent = None
            dynamic_args.ref_latents.clear()

        engine.emphasis = emphasis.get_current_option(shared.opts.emphasis)()
        if any(emphasis.uses_emphasis(x) for x in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = engine.emphasis.name

        crossattn = []
        negpip_mask = []
        _count = 0
        cache = {}

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(engine, line, v_scaling)
            cond, mask = cache[line]

            _count += int((mask[..., -1] < 0).sum())

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
    engine: "Qwen3VLTextProcessingEngine", line: str, v_scaling: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, multipliers = _tokenize_line_negpip(engine, line)

    neutral = [1.0] * len(multipliers)
    encoder_fade = min(max(v_scaling, 0.0), 1.0)
    if encoder_fade >= 1.0:
        encoder_scales = [1.0] * len(multipliers)
    else:
        # Fade the ComfyUI-compatible encoder lerp out continuously as V-scaling
        # takes over, while retaining the exact endpoints at Strength 0 and 1.
        encoder_scales = [abs(m) + (1.0 - abs(m)) * encoder_fade for m in multipliers]
    magnitude_idx = [i for i, scale in enumerate(encoder_scales) if scale != 1.0]

    if magnitude_idx:
        # apply the weight magnitudes on the encoder output, by lerping between a
        # neutral (empty) encoding and the actual encoding; scaling the input
        # embeddings instead barely has any effect, as Qwen3-VL RMSNorms them away
        reference = [engine.id_pad] * len(tokens)
        z = engine.process_tokens([tokens, reference], [neutral, neutral])
        cond, ref = z[0:1], z[1:2]

        idx = torch.tensor(magnitude_idx, device=cond.device, dtype=torch.long)
        scale = torch.tensor(
            [encoder_scales[i] for i in magnitude_idx],
            device=cond.device,
            dtype=cond.dtype,
        ).reshape(1, 1, -1, 1)
        cond[:, :, idx, :] = torch.lerp(ref[:, :, idx, :], cond[:, :, idx, :], scale)
    else:
        cond = engine.process_tokens([tokens], [neutral])

    weights = torch.tensor(multipliers, dtype=torch.float32)
    ones = torch.ones_like(weights)
    sign_mask = torch.where(weights < 0, -ones, ones)
    if v_scaling > 0.0:
        # Strength is an exponent: 0 gives the sign-only mask, 1 gives the raw
        # weight, and values above 1 strengthen magnitude without crossing zero.
        # Floor |w| so a zero weight still fades continuously from the sign-only
        # mask at Strength 0 instead of snapping to 0 for any Strength above it.
        image_mask = sign_mask * weights.abs().clamp_min(1e-4).pow(v_scaling)
        mask = torch.stack((image_mask, sign_mask), dim=-1)
    else:
        mask = sign_mask.unsqueeze(-1)

    cond = engine.strip_template(cond, tokens)
    mask = engine.strip_template(mask.reshape(1, 1, mask.shape[0], mask.shape[1]), tokens)

    if mask.shape[1] < cond.shape[1]:
        mask = F.pad(mask, (0, 0, 0, cond.shape[1] - mask.shape[1]), value=1.0)
    elif mask.shape[1] > cond.shape[1]:
        mask = mask[:, : cond.shape[1]]

    cond, mask = _reshape_conditioning_for_dit(cond, mask, _PATCHED_DIT)

    return cond, mask.to(device=cond.device, dtype=cond.dtype)


def _reshape_conditioning_for_dit(
    cond: torch.Tensor,
    mask: torch.Tensor,
    dit: Optional["SingleStreamDiT"],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Forge Neo through 2.27 keeps conditioning flattened here and unpacks it
    # inside SingleStreamDiT.forward(). Newer versions expect the text engine
    # to return the tapped encoder layers as a separate dimension.
    if dit is None:
        raise RuntimeError("Krea 2 DiT is not initialized for NegPiP conditioning")

    if hasattr(dit, "_unpack_context"):
        return cond, mask

    if cond.ndim != 3:
        raise RuntimeError(
            f"Unexpected Krea 2 conditioning shape: expected 3 dimensions, got {tuple(cond.shape)}"
        )

    batch, sequence, fused = cond.shape
    layers = dit.txtlayers
    features = dit.txtdim
    expected = layers * features
    if fused != expected:
        raise RuntimeError(
            f"Unexpected Krea 2 conditioning width: expected {layers}x{features}={expected}, got {fused}"
        )

    cond = cond.reshape(batch * sequence, layers, features)
    mask = mask.reshape(batch * sequence, mask.shape[-1])
    return cond, mask


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
        split_queries = m.size(-1) > 1
        image_mask = m[..., :1]
        sign_mask = m[..., -1:]

        # Text queries always read sign-masked values. When image queries need
        # different magnitudes, reuse this V tensor after text attention instead
        # of retaining a second head-expanded copy.
        v[:, :txtlen] = v[:, :txtlen] * sign_mask[:, :txtlen]

        q = rearrange(q, "B L (H D) -> B H L D", H=module.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=module.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=module.kvheads)
        q, k = module.qknorm(q, k)
        if freqs is not None:
            q, k = ck.apply_rope(q, k, freqs)
        if module.kvheads != module.heads:
            rep = module.heads // module.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if not split_queries:
            out = attention_function(q, k, v, module.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        else:
            txt_mask = _slice_query_mask(mask, 0, txtlen, q.size(2))
            img_mask = _slice_query_mask(mask, txtlen, q.size(2), q.size(2))
            out_txt = attention_function(q[:, :, :txtlen], k, v, module.heads, mask=txt_mask, skip_reshape=True, transformer_options=transformer_options)

            image_ratio = (image_mask * sign_mask).unsqueeze(1)
            v[:, :, :txtlen] = v[:, :, :txtlen] * image_ratio[:, :, :txtlen]
            out_img = attention_function(q[:, :, txtlen:], k, v, module.heads, mask=img_mask, skip_reshape=True, transformer_options=transformer_options)
            out = torch.cat((out_txt, out_img), dim=1)
        return module.wo(out * F.sigmoid(gate))

    negpip_forward._negpip = True
    module.forward = negpip_forward


def _slice_query_mask(mask: Optional[torch.Tensor], start: int, end: int, query_len: int):
    if mask is not None and mask.ndim >= 3 and mask.shape[-2] == query_len:
        return mask[..., start:end, :]
    return mask
