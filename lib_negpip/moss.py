# MOSS-Image, through either of two backends:
#
# - the moss-image-forge-neo extension (https://github.com/eduardoabreu81/moss-image-forge-neo):
#   engine MossImageEngine, text engine MossTextEngine, DiT MossImageDiT
# - Forge Neo's own support (Haoming02/sd-webui-forge-classic#1564):
#   engine MOSSImage, text engine Qwen359BEngine, DiT backend.nn.moss_image.MOSSImage
#
# Both encode the prompt with Qwen3.5-9B through Forge Neo's ComfyUI-compatible
# SDClipModel and run joint [image, text] single-stream blocks, so the NegPiP
# hooks are the same; only the names and the block internals differ, which the
# _Backend classes below absorb. Everything is resolved from the loaded model.

import json
import sys
from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from modules.prompt_parser import SdConditioning

import torch
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from lib_negpip.anima import _hook_compile_conditions
from lib_negpip.krea import _V_SCALING
from lib_negpip.utils import has_negative
from modules import shared

_PATCHED_MODEL = None
_PATCHED_DIT = None
_BACKEND: Optional["_Backend"] = None

_IM_START = "<|im_start|>"


def is_moss_image(model) -> bool:
    return bool(getattr(model, "is_moss_image", False)) or type(model).__name__ in ("MossImageEngine", "MOSSImage")


# ================================================================================ #


def _module_of(obj):
    return sys.modules.get(type(obj).__module__)


class _Backend:
    """what the hooks need from a MOSS-Image implementation"""

    name = ""
    # the functions the block hook takes from the DiT's module
    transformer_api: tuple[str, ...] = ()
    block_api: tuple[str, ...] = ()

    def __init__(self, model, dit):
        self.model = model
        self.dit = dit
        self.engine = None
        self.template = ""
        self.max_length: Optional[int] = None

    @classmethod
    def matches(cls, model) -> bool:
        raise NotImplementedError

    def unsupported(self) -> Optional[str]:
        """a reason not to patch this implementation, or None"""
        engine = self.engine
        if engine is None:
            return f"{self.name}: the engine has no text engine"
        if "{}" not in self.template:
            return f"{self.name}: the text engine has no chat template"
        if not hasattr(getattr(engine, "text_encoder", None), "encode_token_weights"):
            return f"{self.name}: the text encoder has no encode_token_weights"
        if not hasattr(getattr(engine, "tokenizer", None), "tokenizer"):
            return f"{self.name}: the tokenizer wrapper has no tokenizer"

        layers = getattr(self.dit, "layers", None)
        if not layers:
            return f"{self.name}: the DiT has no layers"
        module = _module_of(self.dit)
        for name in self.transformer_api:
            if not hasattr(module, name):
                return f"{self.name}: the transformer module has no {name}"
        for name in self.block_api:
            if not hasattr(layers[0], name):
                return f"{self.name}: the block has no {name}"
        return None

    # ---- text ----

    @property
    def hf_tokenizer(self):
        return self.engine.tokenizer.tokenizer

    def restore_layout(self, text: str) -> str:
        return text

    def own_template(self, text: str) -> bool:
        """the user supplied the chat template, so none is added"""
        return False

    def token_ids(self, line: str) -> list[int]:
        """the tokens of the whole line, as the implementation feeds them"""
        raise NotImplementedError

    def encode(self, line: str) -> torch.Tensor:
        """the implementation's own encoding of the line: (T, 4096)"""
        raise NotImplementedError

    def encode_weighted(self, pairs: list[tuple[int, float]]) -> torch.Tensor:
        # the ComfyUI-compatible encoder applies the magnitudes on its output, by
        # lerping each scaled token between the empty (pad) encoding and its own
        return self.engine.text_encoder.encode_token_weights([pairs])[0][0]

    # ---- DiT ----

    def block_forward(self, block, args: tuple, kwargs: dict, negpip_mask: torch.Tensor) -> torch.Tensor:
        """Block.forward with the attention values of the text tokens scaled by the mask"""
        raise NotImplementedError


class _ExtensionBackend(_Backend):
    name = "extension"
    transformer_api = ("rotate", "head_norm", "situ", "kernels")
    block_api = ("norm1", "norm2", "q_norm", "k_norm", "proj", "heads", "fused_qkv")

    def __init__(self, model, dit):
        super().__init__(model, dit)
        self.engine = getattr(model, "text_engine_moss", None)
        self._module = _module_of(self.engine) if self.engine is not None else None
        self.template = getattr(self._module, "TEMPLATE", "")
        self._restore = getattr(self._module, "restore_layout_json", None)

    @property
    def max_length(self) -> Optional[int]:
        return getattr(self._module, "MAX_LENGTH", None)

    @max_length.setter
    def max_length(self, value):
        pass  # the extension's module constant is the source of truth

    @classmethod
    def matches(cls, model) -> bool:
        return hasattr(model, "text_engine_moss")

    def unsupported(self):
        if self.engine is not None and (not hasattr(self.engine, "token_ids") or not hasattr(self.engine, "encode")):
            return f"{self.name}: the text engine has no token_ids / encode"
        return super().unsupported()

    def restore_layout(self, text):
        return self._restore(text) if self._restore is not None else text

    def token_ids(self, line):
        return list(self.engine.token_ids(line))

    def encode(self, line):
        return self.engine.encode(line)

    def block_forward(self, block, args, kwargs, negpip_mask):
        x, rope, mod, dtype = args[:4]
        options = kwargs.get("transformer_options") or (args[4] if len(args) > 4 else None) or {}
        transformer = _module_of(block)

        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
        y = (block.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
        if block.fused_qkv:
            q, k, v = block.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)
        q = transformer.head_norm(block.q_norm, q, block.heads)
        k = transformer.head_norm(block.k_norm, k, block.heads)

        m, imglen, image_mask, sign_mask = _text_masks(negpip_mask, v)
        # Text queries always read sign-masked values. When image queries need
        # different magnitudes, reuse this V tensor after text attention instead
        # of retaining a second head-expanded copy. (q and k are already projected
        # into fresh tensors, so writing into a chunk() view of qkv is safe.)
        v[:, imglen:] = v[:, imglen:] * sign_mask

        heads = block.heads
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = transformer.rotate(q.reshape(shape), rope).transpose(1, 2).to(v.dtype)
        k = transformer.rotate(k.reshape(shape), rope).transpose(1, 2).to(v.dtype)
        v = v.reshape(shape).transpose(1, 2)

        if image_mask is None:
            out = transformer.kernels.attention(q, k, v, transformer_options=options)
        else:
            out_txt = transformer.kernels.attention(q[:, :, imglen:], k, v, transformer_options=options)
            v[:, :, imglen:] = v[:, :, imglen:] * (image_mask * sign_mask).unsqueeze(1)
            out_img = transformer.kernels.attention(q[:, :, :imglen], k, v, transformer_options=options)
            out = torch.cat((out_img, out_txt), dim=1)

        out = block.proj["o"](out.to(dtype))
        x = x + (out * gate_a).to(x.dtype)
        y = (block.norm2(x) * (1 + scale_f) + shift_f).to(dtype)
        hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + (block.proj["ffn_down"](transformer.situ(hidden, gate)) * gate_f).to(x.dtype)


class _NativeBackend(_Backend):
    name = "Forge Neo"
    transformer_api = ("attention", "modulated_rms_norm", "gated_residual", "linear_input_act", "attention_function", "weights_manual_cast", "main_stream_worker", "ck")
    block_api = ("norm1", "norm2", "q_norm", "k_norm", "proj", "num_heads", "fused_qkv")

    def __init__(self, model, dit):
        super().__init__(model, dit)
        self.engine = getattr(model, "text_processing_engine_qwen", None)
        self.template = getattr(self.engine, "llama_template", "")

    @classmethod
    def matches(cls, model) -> bool:
        return type(model).__name__ == "MOSSImage" and hasattr(model, "text_processing_engine_qwen")

    def own_template(self, text):
        return text.startswith(_IM_START)

    def token_ids(self, line):
        # Qwen359BEngine.__call__: strip, an empty prompt is one space, the
        # template unless the prompt brings its own
        text = line.strip() or " "
        if not self.own_template(text):
            text = self.template.format(text)
        return list(self.hf_tokenizer(text)["input_ids"])

    def encode(self, line):
        return self.engine([line])[0]

    def block_forward(self, block, args, kwargs, negpip_mask):
        x, mod, pe = args[:3]
        options = kwargs.get("transformer_options") or (args[3] if len(args) > 3 else None) or {}
        nn = _module_of(block)

        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = nn.modulated_rms_norm(x, block.norm1, scale_a, shift_a)
        if block.fused_qkv:
            q, k, v = block.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)

        m, imglen, image_mask, sign_mask = _text_masks(negpip_mask, v)
        v[:, imglen:] = v[:, imglen:] * sign_mask
        heads = block.num_heads

        if image_mask is None:
            out = nn.attention(q, k, v, block.q_norm, block.k_norm, pe, heads, options)
        else:
            # nn.attention() with the queries split: text queries read the
            # sign-masked values, image queries the magnitude-scaled ones
            B, N, _ = q.shape
            q = q.view(B, N, heads, -1)
            k = k.view(B, N, heads, -1)
            q_scale, _, q_stream = nn.weights_manual_cast(block.q_norm, q)
            k_scale, _, k_stream = nn.weights_manual_cast(block.k_norm, k)
            with nn.main_stream_worker(q_scale, None, q_stream), nn.main_stream_worker(k_scale, None, k_stream):
                q, k = nn.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, block.q_norm.eps)
            q, k = q.flatten(2), k.flatten(2)
            out_txt = nn.attention_function(q[:, imglen:], k, v, heads, transformer_options=options)
            v[:, imglen:] = v[:, imglen:] * (image_mask * sign_mask)
            out_img = nn.attention_function(q[:, :imglen], k, v, heads, transformer_options=options)
            out = torch.cat((out_img, out_txt), dim=1)

        x = nn.gated_residual(x, block.proj["o"](out), gate_a)
        y = nn.modulated_rms_norm(x, block.norm2, scale_f, shift_f)
        return nn.gated_residual(x, nn.linear_input_act(block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"), gate_f)


_BACKENDS = (_ExtensionBackend, _NativeBackend)


def _resolve(model, dit=None) -> Optional[_Backend]:
    for backend in _BACKENDS:
        if backend.matches(model):
            return backend(model, dit)
    return None


def _text_masks(negpip_mask: torch.Tensor, v: torch.Tensor):
    """
    the mask aligned with v (B, L, ...): repeated over the batch, cut to the text
    length; returns (mask, image token count, image-query mask or None, sign mask).
    The text tokens follow the image tokens in the MOSS-Image sequence.
    """
    m = negpip_mask.to(v)
    if (batch := (v.size(0) // m.size(0))) > 1:
        m = m.repeat(batch, 1, 1)
    txtlen = min(m.size(1), v.size(1))
    imglen = v.size(1) - txtlen
    image_mask = m[:, :txtlen, :1] if m.size(-1) > 1 else None
    sign_mask = m[:, :txtlen, -1:]
    return m, imglen, image_mask, sign_mask


# ================================================================================ #


def any_weighted(model, p) -> bool:
    """
    utils.any_negative / any_weighted, but looking only at the captions of a
    layout prompt (one JSON object, e.g. written by Regional Boxes): the brackets
    of its boxes are data, not emphasis
    """
    backend = _resolve(model)
    lines = [
        *p.prompts,
        *p.negative_prompts,
        *(getattr(p, "hr_prompts", None) or []),
        *(getattr(p, "hr_negative_prompts", None) or []),
    ]
    for line in lines:
        found = _layout(backend, line.strip()) if backend is not None else None
        if found is not None:
            layout, remainder = found
            texts = [remainder]
            for start, end in _caption_spans(layout):
                try:
                    texts.append(json.loads(f'"{layout[start:end]}"'))
                except ValueError:
                    pass
        else:
            texts = [line]
        if any(has_negative(t) or emphasis.uses_emphasis(t) for t in texts):
            return True
    return False


def patch_moss_negpip(cls: "NegPiP", *, unpatch=False) -> bool:
    global _PATCHED_MODEL, _PATCHED_DIT, _BACKEND

    if unpatch != cls._patched[3]:
        return True

    if unpatch:
        if _PATCHED_MODEL is not None:
            _hook_get_learned_conditioning(_PATCHED_MODEL, True)
        if _PATCHED_DIT is not None:
            _hook_dit_forward(_PATCHED_DIT, True)
            _hook_block_forwards(_PATCHED_DIT, True)
        _hook_compile_conditions(True)

        _PATCHED_MODEL = None
        _PATCHED_DIT = None
        _BACKEND = None
        cls._patched[3] = False
        return True

    model = shared.sd_model
    dit = model.forge_objects.unet.model.diffusion_model

    backend = _resolve(model, dit)
    if backend is None:
        print("NegPiP Disabled (MOSS-Image: unknown implementation)")
        return False
    problem = backend.unsupported()
    if problem is not None:
        print(f"NegPiP Disabled (MOSS-Image: {problem})")
        return False

    _BACKEND = backend
    _hook_get_learned_conditioning(model, False)
    _hook_dit_forward(dit, False)
    _hook_block_forwards(dit, False)
    _hook_compile_conditions(False)

    _PATCHED_MODEL = model
    _PATCHED_DIT = dit
    cls._patched[3] = True
    return True


# ================================================================================ #


def _hook_get_learned_conditioning(model, remove: bool):
    if remove:
        if hasattr(model, "_negpip_orig_get_learned_conditioning"):
            if getattr(model.get_learned_conditioning, "_negpip", False):
                model.get_learned_conditioning = model._negpip_orig_get_learned_conditioning
            del model._negpip_orig_get_learned_conditioning
        return

    orig_get_learned_conditioning = model.get_learned_conditioning
    model._negpip_orig_get_learned_conditioning = orig_get_learned_conditioning

    backend = _BACKEND

    @torch.inference_mode()
    @wraps(orig_get_learned_conditioning)
    def negpip_learned_conditioning(prompt: "SdConditioning"):
        memory_management.load_model_gpu(model.forge_objects.clip.patcher)
        v_scaling = _V_SCALING.get()
        emphasis_name = emphasis.get_current_option(shared.opts.emphasis).name

        crossattn = []
        negpip_mask = []
        _count = 0
        cache = {}

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(backend, line, emphasis_name, v_scaling)
            cond, mask = cache[line]

            _count += int((mask[..., -1] < 0).sum())

            crossattn.append(cond)
            negpip_mask.append(mask)

        # both implementations hard-wire EmphasisNone (and record "None" while
        # encoding); the weights follow the user setting here, so record that
        if any(emphasis.uses_emphasis(x) for x in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = emphasis_name

        if _count > 0:
            key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
            print(f"NegPiP Enable ({key}: {_count})")

        return {
            "crossattn": crossattn,
            "c_negpip_mask": negpip_mask,
        }

    negpip_learned_conditioning._negpip = True
    model.get_learned_conditioning = negpip_learned_conditioning


def _layout(backend: _Backend, text: str) -> Optional[tuple[str, str]]:
    """
    (layout, remainder) of a MOSS-Image layout prompt: the leading JSON object
    (with the braces restored as the extension does) and whatever follows it,
    e.g. the Extra text Forge Neo's JSON Builder appends; None when the line
    does not start with a layout
    """
    text = backend.restore_layout(text)
    if not text.startswith("{"):
        return None
    try:
        layout, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    if not isinstance(layout, dict):
        return None
    return text[:end], text[end:]


def _is_layout(backend: _Backend, text: str) -> bool:
    return _layout(backend, text) is not None


_CAPTION_KEY = "caption"


def _json_strings(text: str) -> list[tuple[int, int]]:
    """(start, end) of every JSON string token in text, quotes included"""
    spans = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
        else:
            i += 1
    return spans


def _caption_spans(text: str) -> list[tuple[int, int]]:
    """
    (start, end) of the content of every string that is the value of a
    "caption" key: the scene caption and the item captions carry the prompt
    words, everything else in a layout (text to render, meta, boxes) is data
    """
    strings = _json_strings(text)
    spans = []
    for k in range(len(strings) - 1):
        key_start, key_end = strings[k]
        if text[key_start:key_end] != f'"{_CAPTION_KEY}"':
            continue
        between = text[key_end : strings[k + 1][0]]
        if between.strip() != ":":
            continue
        value_start, value_end = strings[k + 1]
        spans.append((value_start + 1, value_end - 1))
    return spans


def _escape(segment: str) -> str:
    return json.dumps(segment, ensure_ascii=False)[1:-1]


def _parse_attention(text: str, emphasis_name: str) -> list[tuple[str, float]]:
    parsed = parsing.parse_prompt_attention(text, emphasis_name)
    if emphasis_name == "Ignore":
        parsed = [(t, 1.0) for t, _ in parsed]
    # MOSS-Image has no chunking; the parser marks BREAK with -1 and eats the
    # whitespace around it, restore the plain word the implementation would see
    return [(" BREAK ", 1.0) if (t == "BREAK" and w == -1.0) else (t, w) for t, w in parsed]


def _parse_layout(layout: str, emphasis_name: str) -> list[tuple[str, float]]:
    """
    the layout as weighted segments: the structure verbatim, the captions parsed
    for prompt weights (and re-escaped where a weight applies)
    """
    parsed = []
    pos = 0
    for start, end in _caption_spans(layout):
        parsed.append((layout[pos:start], 1.0))
        raw = layout[start:end]
        try:
            caption = json.loads(f'"{raw}"')
        except ValueError:
            parsed.append((raw, 1.0))
            pos = end
            continue
        segments = _parse_attention(caption, emphasis_name)
        if all(w == 1.0 for _, w in segments):
            parsed.append((raw, 1.0))
        else:
            parsed.extend((_escape(t), w) for t, w in segments)
        pos = end
    parsed.append((layout[pos:], 1.0))
    return parsed


def _parse_line(backend: _Backend, line: str, emphasis_name: str) -> list[tuple[str, float]]:
    found = _layout(backend, line.strip())
    if found is None:
        return _parse_attention(line, emphasis_name)
    layout, remainder = found
    parsed = _parse_layout(layout, emphasis_name)
    if remainder.strip():
        # plain prompt text after the layout: weights apply as usual
        parsed.extend(_parse_attention(remainder, emphasis_name))
    return parsed


def _tokenize_line_negpip(backend: _Backend, line: str, emphasis_name: str) -> tuple[list[int], list[float]]:
    """
    tokenize like the implementation does (chat template once around the whole
    prompt), with the prompt weights attached to the tokens of their segments
    """
    parsed = _parse_line(backend, line, emphasis_name)

    if all(w == 1.0 for _, w in parsed):
        tokens = backend.token_ids(line)
        return tokens, [1.0] * len(tokens)

    texts = [t for t, _ in parsed]
    weights = [w for _, w in parsed]
    # the whitespace around the whole prompt is dropped
    texts[0] = texts[0].lstrip()
    texts[-1] = texts[-1].rstrip()
    # Qwen's BPE merges a leading space into the word ("Ġred"); move the
    # whitespace before a segment into it so that the segment tokenizes the
    # same way as inside the whole prompt and the weight covers the whole token
    for i in range(len(texts) - 1):
        stripped = texts[i].rstrip()
        texts[i + 1] = texts[i][len(stripped):] + texts[i + 1]
        texts[i] = stripped

    if backend.own_template("".join(texts)):
        prefix = suffix = ""
    else:
        prefix, suffix = backend.template.split("{}")
    hf_tokenizer = backend.hf_tokenizer

    tokens = list(hf_tokenizer.encode(prefix, add_special_tokens=False)) if prefix else []
    multipliers = [1.0] * len(tokens)

    for segment, weight in zip(texts, weights):
        if not segment:
            continue
        ids = hf_tokenizer.encode(segment, add_special_tokens=False)
        tokens.extend(ids)
        multipliers.extend([weight] * len(ids))

    if suffix:
        ids = hf_tokenizer.encode(suffix, add_special_tokens=False)
        tokens.extend(ids)
        multipliers.extend([1.0] * len(ids))

    if backend.max_length is not None:
        tokens, multipliers = tokens[: backend.max_length], multipliers[: backend.max_length]
    return tokens, multipliers


def _weight_masks(multipliers: list[float], v_scaling: float = 0.0) -> tuple[list[float], torch.Tensor]:
    """
    split the per-token prompt weights between the text encoder and the attention
    values, exactly as krea._encode_line does; returns (encoder_scales, mask) where
    mask is [tokens, 1] (sign only) or [tokens, 2] (image-query mask, sign mask)
    """
    encoder_fade = min(max(v_scaling, 0.0), 1.0)
    if encoder_fade >= 1.0:
        encoder_scales = [1.0] * len(multipliers)
    else:
        # Fade the ComfyUI-compatible encoder lerp out continuously as V-scaling
        # takes over, while retaining the exact endpoints at Strength 0 and 1.
        encoder_scales = [abs(m) + (1.0 - abs(m)) * encoder_fade for m in multipliers]

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

    return encoder_scales, mask


def _encode_line(backend: _Backend, line: str, emphasis_name: str, v_scaling: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, multipliers = _tokenize_line_negpip(backend, line, emphasis_name)
    encoder_scales, mask = _weight_masks(multipliers, v_scaling)

    if all(w == 1.0 for w in multipliers):
        # identical to the implementation's own encoding
        cond = backend.encode(line)
    else:
        cond = backend.encode_weighted(list(zip(tokens, encoder_scales)))

    # (T, 4096) without a batch dimension, as both implementations return it
    if cond.ndim != 2:
        raise RuntimeError(f"Unexpected MOSS-Image conditioning shape: expected 2 dimensions, got {tuple(cond.shape)}")

    if mask.shape[0] < cond.shape[0]:
        mask = F.pad(mask, (0, 0, 0, cond.shape[0] - mask.shape[0]), value=1.0)
    elif mask.shape[0] > cond.shape[0]:
        mask = mask[: cond.shape[0]]

    return cond, mask.to(device=cond.device, dtype=cond.dtype)


# ================================================================================ #


def _hook_dit_forward(dit, remove: bool):
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
    def negpip_forward(x, timestep, context, transformer_options=None, **kwargs):
        negpip_mask: Optional[torch.Tensor] = kwargs.pop("c_negpip_mask", None)

        if negpip_mask is not None:
            while negpip_mask.ndim > 3:
                negpip_mask = negpip_mask.squeeze(1)
            transformer_options = {**(transformer_options or {}), "negpip_mask": negpip_mask}

        if transformer_options is None:
            return orig_forward(x, timestep, context, **kwargs)
        return orig_forward(x, timestep, context, transformer_options=transformer_options, **kwargs)

    negpip_forward._negpip = True
    dit.forward = negpip_forward


def _hook_block_forwards(dit, remove: bool):
    for block in getattr(dit, "layers", ()):
        _hook_block_forward(block, remove)


def _hook_block_forward(block, remove: bool):
    if remove:
        if hasattr(block, "_negpip_orig_forward"):
            if getattr(block.forward, "_negpip", False):
                block.forward = block._negpip_orig_forward
            del block._negpip_orig_forward
        return

    orig_forward = block.forward
    block._negpip_orig_forward = orig_forward
    backend = _BACKEND

    @torch.inference_mode()
    @wraps(orig_forward)
    def negpip_forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if options is None:
            # the last positional argument of both Block.forward signatures
            options = next((a for a in reversed(args) if isinstance(a, dict)), None)
        negpip_mask: Optional[torch.Tensor] = (options or {}).get("negpip_mask", None)
        if negpip_mask is None:
            return orig_forward(*args, **kwargs)
        return backend.block_forward(block, args, kwargs, negpip_mask)

    negpip_forward._negpip = True
    block.forward = negpip_forward
