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
from backend.quant_ops import ck, QuantizedTensor
from backend.text_processing import emphasis, parsing
from lib_negpip.anima import _hook_compile_conditions
from modules import shared



_V_SCALING: ContextVar[float] = ContextVar("negpip_v_scaling", default=0.0)


_PATCHED_MODEL = None
_PATCHED_DIT = None

_KREA2_W4A8_OP = None
_KREA2_W4A4_OP = None


def _get_w4a8_op():
    """Register or return the opaque custom op for AsymW4A8Int8 linear execution under TorchDynamo."""
    global _KREA2_W4A8_OP
    if _KREA2_W4A8_OP is not None:
        return _KREA2_W4A8_OP

    try:
        from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

        @torch.library.custom_op("krea2::w4a8_linear", mutates_args=())
        def w4a8_linear(
            x: torch.Tensor,
            qdata: torch.Tensor,
            s_rel: torch.Tensor,
            s_channel: torch.Tensor,
            codebook: torch.Tensor | None,
            correction: torch.Tensor | None,
            bias: torch.Tensor | None,
            group_size: int,
            convrot_groupsize: int,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            return w4a8_int8_linear(
                x,
                qdata,
                s_rel,
                s_channel,
                codebook=codebook,
                correction=correction,
                bias=bias,
                group_size=group_size,
                convrot_groupsize=convrot_groupsize,
                out_dtype=out_dtype,
            )

        @w4a8_linear.register_fake
        def _(x, qdata, s_rel, s_channel, codebook, correction, bias, group_size, convrot_groupsize, out_dtype):
            return x.new_empty((*x.shape[:-1], qdata.shape[0]), dtype=out_dtype)

        _KREA2_W4A8_OP = w4a8_linear
    except Exception as exc:
        _KREA2_W4A8_OP = None
    return _KREA2_W4A8_OP


def _get_w4a4_op():
    """Register or return the opaque custom op for ConvRotW4A4 linear execution under TorchDynamo."""
    global _KREA2_W4A4_OP
    if _KREA2_W4A4_OP is not None:
        return _KREA2_W4A4_OP

    try:
        from comfy_kitchen.registry import registry as ck_registry

        @torch.library.custom_op("krea2::w4a4_linear", mutates_args=())
        def w4a4_linear(
            x: torch.Tensor,
            qweight: torch.Tensor,
            wscales: torch.Tensor,
            bias: torch.Tensor | None,
            convrot_groupsize: int,
            quant_group_size: int,
            linear_dtype: str,
        ) -> torch.Tensor:
            impl = ck_registry.get_implementation("convrot_w4a4_linear", kwargs={
                "x": x, "qweight": qweight, "wscales": wscales, "bias": bias,
                "convrot_groupsize": convrot_groupsize,
                "quant_group_size": quant_group_size, "linear_dtype": linear_dtype,
            })
            return impl(
                x, qweight, wscales, bias=bias,
                convrot_groupsize=convrot_groupsize,
                quant_group_size=quant_group_size,
                linear_dtype=linear_dtype,
            )

        @w4a4_linear.register_fake
        def _(x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype):
            return x.new_empty(x.shape[:-1] + (qweight.shape[0],))

        _KREA2_W4A4_OP = w4a4_linear
    except Exception as exc:
        _KREA2_W4A4_OP = None
    return _KREA2_W4A4_OP


_SAGE_CUSTOM_OP = None
_ORIG_ATTN_SAGE = None
_ORIG_ATTN_FUNCTION = None
_ORIG_KREA_ATTN_FUNCTION = None
_ORIG_SAGEATTN_CALLABLE = None


def _get_sage_op():
    """Register or return the opaque custom op for SageAttention under TorchDynamo."""
    global _SAGE_CUSTOM_OP
    if _SAGE_CUSTOM_OP is not None:
        return _SAGE_CUSTOM_OP

    try:
        import sageattention

        @torch.library.custom_op("sage::sageattn", mutates_args=())
        def sage_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, tensor_layout: str) -> torch.Tensor:
            return sageattention.sageattn(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)

        @sage_op.register_fake
        def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, tensor_layout: str) -> torch.Tensor:
            return q.new_empty(q.shape)

        _SAGE_CUSTOM_OP = sage_op
    except Exception:
        _SAGE_CUSTOM_OP = None
    return _SAGE_CUSTOM_OP


def _install_sage_custom_op():
    """Enable zero-graph-break compilation for SageAttention layers."""
    global _ORIG_ATTN_SAGE, _ORIG_ATTN_FUNCTION, _ORIG_KREA_ATTN_FUNCTION, _ORIG_SAGEATTN_CALLABLE
    op = _get_sage_op()
    if op is None:
        return

    try:
        import sageattention
        import backend.attention as attn
        import backend.nn.krea as krea_nn

        if _ORIG_SAGEATTN_CALLABLE is None:
            _ORIG_SAGEATTN_CALLABLE = sageattention.sageattn
            orig_fn = _ORIG_SAGEATTN_CALLABLE

            def smart_sageattn(q, k, v, attn_mask=None, is_causal=False, tensor_layout="HND"):
                if torch.compiler.is_compiling() and attn_mask is None:
                    return op(q, k, v, is_causal, tensor_layout)
                return orig_fn(q, k, v, attn_mask=attn_mask, is_causal=is_causal, tensor_layout=tensor_layout)

            sageattention.sageattn = smart_sageattn

        unwrapped = getattr(attn.attention_sage, "_torchdynamo_orig_callable", None)
        if unwrapped is not None:
            if _ORIG_ATTN_SAGE is None:
                _ORIG_ATTN_SAGE = attn.attention_sage
                _ORIG_ATTN_FUNCTION = attn.attention_function
                _ORIG_KREA_ATTN_FUNCTION = getattr(krea_nn, "attention_function", None)

            attn.attention_sage = unwrapped
            attn.attention_function = unwrapped
            krea_nn.attention_function = unwrapped
            global attention_function
            attention_function = unwrapped
    except Exception as e:
        print(f"[NegPiP-A] SageAttention custom op setup notice: {e}")


def _uninstall_sage_custom_op():
    """Restore original SageAttention wrappers when unpatching."""
    global _ORIG_ATTN_SAGE, _ORIG_ATTN_FUNCTION, _ORIG_KREA_ATTN_FUNCTION, _ORIG_SAGEATTN_CALLABLE
    try:
        import sageattention
        import backend.attention as attn
        import backend.nn.krea as krea_nn

        if _ORIG_SAGEATTN_CALLABLE is not None:
            sageattention.sageattn = _ORIG_SAGEATTN_CALLABLE
            _ORIG_SAGEATTN_CALLABLE = None

        if _ORIG_ATTN_SAGE is not None:
            attn.attention_sage = _ORIG_ATTN_SAGE
            attn.attention_function = _ORIG_ATTN_FUNCTION
            if _ORIG_KREA_ATTN_FUNCTION is not None:
                krea_nn.attention_function = _ORIG_KREA_ATTN_FUNCTION
            global attention_function
            attention_function = _ORIG_ATTN_FUNCTION
            _ORIG_ATTN_SAGE = None
            _ORIG_ATTN_FUNCTION = None
            _ORIG_KREA_ATTN_FUNCTION = None
    except Exception:
        pass


def _install_krea2_custom_op(module: torch.nn.Module) -> bool:
    """Route quantized Linear layers through opaque custom ops when TorchDynamo is compiling."""
    if getattr(module, "_krea2_op_installed", False):
        return True
    weight = getattr(module, "weight", None)
    if not isinstance(weight, QuantizedTensor):
        return False

    layout_cls = getattr(weight, "_layout_cls", None)
    if layout_cls == "AsymW4A8Int8Layout":
        op = _get_w4a8_op()
        if op is None:
            return False
        params = weight._params
        if getattr(params, "transposed", False):
            return False

        from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

        original = module.forward

        def forward(x, *args, **kwargs):
            w = module.weight
            if (args or kwargs or x.ndim < 2 or x.requires_grad
                    or getattr(module, "weight_function", None)
                    or getattr(module, "bias_function", None)
                    or getattr(module, "forge_force_cast_weights", False)
                    or getattr(module, "_full_precision_mm", False)
                    or w._qdata.device != x.device):
                return original(x, *args, **kwargs)

            if torch.compiler.is_compiling():
                qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(w)
                p = w._params
                bias = module.bias
                if bias is not None and bias.dtype != x.dtype:
                    bias = bias.to(dtype=x.dtype)
                return op(
                    x, qdata, s_rel, s_channel, codebook, correction, bias,
                    p.group_size, p.convrot_groupsize, x.dtype,
                )
            return original(x, *args, **kwargs)

        module.forward = forward
        module._krea2_orig_forward = original
        module._krea2_op_installed = True
        return True

    elif layout_cls == "TensorCoreConvRotW4A4Layout":
        op = _get_w4a4_op()
        if op is None:
            return False
        params = weight._params
        if getattr(params, "transposed", False):
            return False

        from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

        original = module.forward
        groupsize = int(params.convrot_groupsize)
        quant_group_size = int(params.quant_group_size)
        linear_dtype = str(params.linear_dtype)

        def forward(x, *args, **kwargs):
            w = module.weight
            if (args or kwargs or x.ndim < 2 or x.requires_grad
                    or getattr(module, "weight_function", None)
                    or getattr(module, "bias_function", None)
                    or getattr(module, "forge_force_cast_weights", False)
                    or getattr(module, "_full_precision_mm", False)
                    or w._qdata.device != x.device):
                return original(x, *args, **kwargs)

            if torch.compiler.is_compiling():
                qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(w)
                bias = module.bias
                if bias is not None and bias.dtype != x.dtype:
                    bias = bias.to(dtype=x.dtype)
                return op(x, qweight, wscales, bias, groupsize, quant_group_size, linear_dtype)
            return original(x, *args, **kwargs)

        module.forward = forward
        module._krea2_orig_forward = original
        module._krea2_op_installed = True
        return True

    return False


def _install_krea2_custom_ops(dit: torch.nn.Module):
    count = 0
    for m in dit.modules():
        if _install_krea2_custom_op(m):
            count += 1
    if count > 0:
        print(f"[NegPiP-A] Installed TorchDynamo custom ops on {count} quantized linear layers")


def _uninstall_krea2_custom_ops(dit: torch.nn.Module):
    for m in dit.modules():
        if getattr(m, "_krea2_op_installed", False):
            if hasattr(m, "_krea2_orig_forward"):
                m.forward = m._krea2_orig_forward
                del m._krea2_orig_forward
            m._krea2_op_installed = False


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


_ORIG_KMODEL_APPLY_MODEL = None
_ORIG_INSTANCE_APPLY_MODEL = {}
_ORIG_KMODEL_SETATTR = None


def _mark_tensor_dynamic(t):
    """Mark sequence length dimension dynamic so TorchInductor and TorchDynamo do not recompile on prompt text changes."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    # In Krea 2 TextFusionTransformer, context is (batch, seq, layers, dim) or (batch, layers, seq, dim)
    if t.ndim == 4:
        try:
            torch._dynamo.mark_dynamic(t, 1)
        except Exception:
            pass
        try:
            torch._dynamo.mark_dynamic(t, 2)
        except Exception:
            pass
    # In standard conditioning, context is (batch, seq, dim) -> seq is dim 1
    elif t.ndim == 3:
        try:
            torch._dynamo.mark_dynamic(t, 1)
        except Exception:
            pass
    # In flattened masks/tokens, (seq, dim) -> seq is dim 0
    elif t.ndim == 2:
        try:
            torch._dynamo.mark_dynamic(t, 0)
        except Exception:
            pass


class _NegPiPDynamicCompiledWrapper(torch.nn.Module):
    def __init__(self, orig_compiled):
        super().__init__()
        object.__setattr__(self, '_orig_compiled', orig_compiled)
        object.__setattr__(self, '_negpip_dynamic_wrapped', True)

    def _apply_dynamic_guards(self, *d_args, **d_kwargs):
        for arg in d_args:
            _mark_tensor_dynamic(arg)
        for k, v in d_kwargs.items():
            if isinstance(v, torch.Tensor):
                _mark_tensor_dynamic(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        _mark_tensor_dynamic(sub_v)

    def forward(self, *d_args, **d_kwargs):
        self._apply_dynamic_guards(*d_args, **d_kwargs)
        return self._orig_compiled(*d_args, **d_kwargs)

    def __call__(self, *d_args, **d_kwargs):
        return self.forward(*d_args, **d_kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_compiled, name)


def _wrap_compiled_module(kmodel):
    compiled = getattr(kmodel, "_forge_compiled_model", None)
    if compiled is not None and not getattr(compiled, "_negpip_dynamic_wrapped", False):
        wrapper = _NegPiPDynamicCompiledWrapper(compiled)
        setattr(kmodel, "_forge_compiled_model", wrapper)


try:
    import torch._inductor.config as inductor_config
    inductor_config.search_autotune_cache = True
except Exception:
    pass

def _hook_kmodel_apply_model(kmodel=None, remove: bool = False):
    """Wrap KModel.apply_model and intercept _forge_compiled_model to dynamically mark variable text conditioning shapes before TorchDynamo compiles."""
    global _ORIG_KMODEL_APPLY_MODEL, _ORIG_INSTANCE_APPLY_MODEL, _ORIG_KMODEL_SETATTR
    try:
        from backend.modules.k_model import KModel
    except Exception:
        KModel = None

    if remove:
        if kmodel is not None and id(kmodel) in _ORIG_INSTANCE_APPLY_MODEL:
            orig = _ORIG_INSTANCE_APPLY_MODEL.pop(id(kmodel))
            if getattr(getattr(kmodel, "apply_model", None), "_negpip", False):
                kmodel.apply_model = orig
            compiled = getattr(kmodel, "_forge_compiled_model", None)
            if compiled is not None and getattr(compiled, "_negpip_dynamic_wrapped", False):
                setattr(kmodel, "_forge_compiled_model", getattr(compiled, "_orig_compiled", compiled))
        if KModel is not None and _ORIG_KMODEL_APPLY_MODEL is not None:
            if getattr(KModel.apply_model, "_negpip", False):
                KModel.apply_model = _ORIG_KMODEL_APPLY_MODEL
            _ORIG_KMODEL_APPLY_MODEL = None
        if KModel is not None and _ORIG_KMODEL_SETATTR is not None:
            KModel.__setattr__ = _ORIG_KMODEL_SETATTR
            _ORIG_KMODEL_SETATTR = None
        return

    # Install class-level __setattr__ hook on KModel so _forge_compiled_model is wrapped immediately upon creation
    if KModel is not None and _ORIG_KMODEL_SETATTR is None:
        _ORIG_KMODEL_SETATTR = KModel.__setattr__
        orig_setattr = _ORIG_KMODEL_SETATTR

        def negpip_kmodel_setattr(self, name, value):
            if name == "_forge_compiled_model" and value is not None and not getattr(value, "_negpip_dynamic_wrapped", False):
                value = _NegPiPDynamicCompiledWrapper(value)
            orig_setattr(self, name, value)

        KModel.__setattr__ = negpip_kmodel_setattr

    # 1. Wrap kmodel instance if available
    if kmodel is not None:
        _wrap_compiled_module(kmodel)
        if not getattr(getattr(kmodel, "apply_model", None), "_negpip", False):
            orig_inst_apply = kmodel.apply_model
            _ORIG_INSTANCE_APPLY_MODEL[id(kmodel)] = orig_inst_apply

            @wraps(orig_inst_apply)
            def negpip_inst_apply_model(*args, **kwargs):
                _wrap_compiled_module(kmodel)
                res = orig_inst_apply(*args, **kwargs)
                _wrap_compiled_module(kmodel)
                return res

            negpip_inst_apply_model._negpip = True
            kmodel.apply_model = negpip_inst_apply_model

    # 2. Wrap KModel class as fallback
    if KModel is not None and not getattr(getattr(KModel, "apply_model", None), "_negpip", False):
        orig_apply_model = KModel.apply_model
        _ORIG_KMODEL_APPLY_MODEL = orig_apply_model

        @wraps(orig_apply_model)
        def negpip_kmodel_apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            _wrap_compiled_module(self)
            return orig_apply_model(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)

        negpip_kmodel_apply_model._negpip = True
        KModel.apply_model = negpip_kmodel_apply_model


def patch_krea2_negpip(cls: "NegPiP", *, unpatch=False):
    global _PATCHED_MODEL, _PATCHED_DIT

    if unpatch != cls._patched[2]:
        return

    model: "Krea2Engine" = getattr(shared, "sd_model", None)
    if model is None:
        return

    unet = getattr(getattr(model, "forge_objects", None), "unet", None)
    kmodel = getattr(unet, "model", None)
    dit: "SingleStreamDiT" = getattr(kmodel, "diffusion_model", None) if kmodel is not None else None

    if unpatch:
        if _PATCHED_MODEL is not None:
            _hook_get_learned_conditioning(_PATCHED_MODEL, True)
        if _PATCHED_DIT is not None:
            _hook_dit_forward(_PATCHED_DIT, True)
            _hook_attn_forwards(_PATCHED_DIT, True)
            _uninstall_krea2_custom_ops(_PATCHED_DIT)
            _uninstall_sage_custom_op()
        _hook_compile_conditions(True)
        _hook_kmodel_apply_model(kmodel, True)

        _PATCHED_MODEL = None
        _PATCHED_DIT = None
        cls._patched[2] = False
        return

    if dit is None:
        return

    _hook_get_learned_conditioning(model, False)
    _install_sage_custom_op()
    _install_krea2_custom_ops(dit)
    _hook_dit_forward(dit, False)
    _hook_attn_forwards(dit, False)
    _hook_compile_conditions(False)
    _hook_kmodel_apply_model(kmodel, False)

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

        # Forge Neo 2.29.2+ hard-wires Krea 2 to EmphasisNone (read-only
        # property), so follow the user setting here instead of the engine's
        emphasis_name = emphasis.get_current_option(shared.opts.emphasis).name
        if _is_legacy_engine(engine):
            engine.emphasis = emphasis.get_current_option(shared.opts.emphasis)()
        if any(emphasis.uses_emphasis(x) for x in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = emphasis_name

        crossattn = []
        negpip_mask = []
        _count = 0
        cache = {}

        for line in prompt:
            if line not in cache:
                cache[line] = _encode_line(engine, line, emphasis_name, v_scaling)
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


_ID_PAD = 151643
_ID_IM_START = 151644


def _is_legacy_engine(engine) -> bool:
    # Forge Neo through 2.29.1 ships Qwen3VLTextProcessingEngine; 2.29.2 replaced
    # it with the ComfyUI-based Qwen3VL4BEngine, which has no process_tokens
    return hasattr(engine, "process_tokens")


def _hf_tokenizer(engine):
    return engine.tokenizer if _is_legacy_engine(engine) else engine.tokenizer.tokenizer


def _encode_tokens(engine, batch_tokens: list[list[int]]) -> torch.Tensor:
    """encode without weights; returns [batch, layers, sequence, features]"""
    if _is_legacy_engine(engine):
        return engine.process_tokens(batch_tokens, [[1.0] * len(t) for t in batch_tokens])
    return engine.text_encoder(batch_tokens)[0]


def _strip_template(out: torch.Tensor, tokens: list[int]) -> torch.Tensor:
    """
    port of the template stripping in Forge Neo's Krea 2 text engine;
    [batch, layers, sequence, features] -> [batch, sequence, layers * features]
    """
    template_end = 0
    count_im_start = 0

    for i, token in enumerate(tokens):
        if token == _ID_IM_START and count_im_start < 2:
            template_end = i
            count_im_start += 1

    if out.shape[2] > (template_end + 3):
        if tokens[template_end + 1] == 872 and tokens[template_end + 2] == 198:
            template_end += 3

    out = out[:, :, template_end:]

    b, n, seq, h = out.shape
    return out.permute(0, 2, 1, 3).reshape(b, seq, n * h)


def _tokenize_line_negpip(
    engine: "Qwen3VLTextProcessingEngine", line: str, emphasis_name: str
) -> tuple[list, list[float]]:
    """
    tokenize like Qwen3VLTextProcessingEngine.tokenize_line, but apply the chat
    template once around the whole prompt instead of once per weighted segment,
    so that the weights only cover the user text
    """

    parsed = parsing.parse_prompt_attention(line, emphasis_name)
    if emphasis_name == "Ignore":
        parsed = [(text, 1.0) for text, _ in parsed]

    if all(weight == 1.0 for _, weight in parsed):
        if _is_legacy_engine(engine):
            chunk = engine.tokenize_line(line)[0]
            return chunk.tokens, chunk.multipliers

        text = "".join(text for text, _ in parsed).strip()
        tokens = _hf_tokenizer(engine)(engine.llama_template.format(text))["input_ids"]
        return tokens, [1.0] * len(tokens)

    prefix, suffix = engine.llama_template.split("{}")
    tokenized = _hf_tokenizer(engine)([prefix, *(text for text, _ in parsed), suffix])["input_ids"]

    tokens = list(tokenized[0])
    multipliers = [1.0] * len(tokens)

    for segment, (_, weight) in zip(tokenized[1:-1], parsed):
        tokens.extend(segment)
        multipliers.extend([weight] * len(segment))

    tokens.extend(tokenized[-1])
    multipliers.extend([1.0] * len(tokenized[-1]))

    return tokens, multipliers


def _encode_line(
    engine: "Qwen3VLTextProcessingEngine",
    line: str,
    emphasis_name: str,
    v_scaling: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, multipliers = _tokenize_line_negpip(engine, line, emphasis_name)

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
        reference = [_ID_PAD] * len(tokens)
        z = _encode_tokens(engine, [tokens, reference])
        cond, ref = z[0:1], z[1:2]

        idx = torch.tensor(magnitude_idx, device=cond.device, dtype=torch.long)
        scale = torch.tensor(
            [encoder_scales[i] for i in magnitude_idx],
            device=cond.device,
            dtype=cond.dtype,
        ).reshape(1, 1, -1, 1)
        cond[:, :, idx, :] = torch.lerp(ref[:, :, idx, :], cond[:, :, idx, :], scale)
    else:
        cond = _encode_tokens(engine, [tokens])

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

    cond = _strip_template(cond, tokens)
    mask = _strip_template(mask.reshape(1, 1, mask.shape[0], mask.shape[1]), tokens)

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
        try:
            m = getattr(shared, "sd_model", None)
            dit = getattr(getattr(getattr(m, "forge_objects", None), "unet", None), "model", None).diffusion_model
        except Exception:
            pass

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

        # Text queries always read sign-masked values. Use functional slicing
        # to ensure compatibility with TorchDynamo and TorchInductor compilation.
        if txtlen < v.size(1):
            v = torch.cat((v[:, :txtlen] * sign_mask[:, :txtlen], v[:, txtlen:]), dim=1)
        else:
            v = v * sign_mask

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
            if txtlen < v.size(2):
                v_img = torch.cat((v[:, :, :txtlen] * image_ratio[:, :, :txtlen], v[:, :, txtlen:]), dim=2)
            else:
                v_img = v * image_ratio
            out_img = attention_function(q[:, :, txtlen:], k, v_img, module.heads, mask=img_mask, skip_reshape=True, transformer_options=transformer_options)
            out = torch.cat((out_txt, out_img), dim=1)
        return module.wo(out * F.sigmoid(gate))

    negpip_forward._negpip = True
    module.forward = negpip_forward


def _slice_query_mask(mask: Optional[torch.Tensor], start: int, end: int, query_len: int):
    if mask is not None and mask.ndim >= 3 and mask.shape[-2] == query_len:
        return mask[..., start:end, :]
    return mask
