import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import torch
from lib_negpip import IS_NEO, INCOMPATIBLE_EXTENSIONS
from lib_negpip.anima import patch_anima_negpip
from lib_negpip.krea import patch_krea2_negpip
from lib_negpip.sd import patch_sd_negpip
from lib_negpip.utils import NEG_PATTERN, any_negative, any_weighted, hr_dealer, reset_prompt_cache

from modules import scripts
from modules.prompt_parser import (
    SdConditioning,
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
)
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser


def _verify_ext(p: " StableDiffusionProcessing"):
    for ext in p.scripts.scripts:
        if ext.title() not in INCOMPATIBLE_EXTENSIONS:
            continue
        if p.script_args[ext.args_from] is True:
            return False

    return True


class NegPiP(scripts.Script):
    _patched: list[bool] = [False, False, False]

    def __init__(self):
        self.active: bool = False

        self.is_xl: bool
        self.is_anima: bool
        self.is_krea2: bool
        self.is_hr: bool

        self.tokenizer: torch.nn.Module

        self.has_hr_p: bool
        self.has_hr_n: bool
        self.rev: bool
        self.batch_size: int

        self.conds: list[torch.Tensor]
        self.c_len: int
        self.c_tokens: list[int]
        self.conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

        self.unconds: list[torch.Tensor]
        self.uc_len: int
        self.uc_tokens: list[int]
        self.unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

        on_cfg_denoiser(self.denoiser_callback)

    def reset(self):
        self.active = False

        self.is_xl = False
        self.is_anima = False
        self.is_krea2 = False
        self.is_hr = False

        self.tokenizer = None

        self.conds = None
        self.c_tokens = None
        self.conds_all = None
        self.hr_conds_all = None

        self.unconds = None
        self.uc_tokens = None
        self.unconds_all = None
        self.hr_unconds_all = None

        patch_sd_negpip(None, NegPiP, unpatch=True)
        patch_anima_negpip(NegPiP, unpatch=True)
        patch_krea2_negpip(NegPiP, unpatch=True)

    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def process_batch(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.reset()

        if IS_NEO and not p.sd_model.is_webui_legacy_model():
            model_type = type(p.sd_model).__name__
            self.is_anima = model_type == "Anima"
            self.is_krea2 = model_type == "Krea2"

            if self.is_anima:
                active = any_negative(p)
            elif self.is_krea2:
                # the weight magnitudes only apply correctly while patched,
                # so also activate on positive weights
                active = any_negative(p) or any_weighted(p)
            else:
                return

            if not active:
                return

            if not _verify_ext(p):
                print("NegPiP Disabled")
                return

            if self.is_anima:
                patch_anima_negpip(NegPiP)
            else:
                patch_krea2_negpip(NegPiP)

            reset_prompt_cache(p)
            p.extra_generation_params["NegPiP"] = True
            self.active = True
            return

        if not any_negative(p):
            return

        if not _verify_ext(p):
            print("NegPiP Disabled")
            return

        self.is_xl = p.sd_model.is_sdxl
        self.batch_size = p.batch_size
        self.has_hr_p, self.has_hr_n = hr_dealer(p)
        self.rev = p.sampler_name not in ("DDIM", "PLMS", "UniPC")

        if IS_NEO:
            self.tokenizer = (
                p.sd_model.text_processing_engine_l.tokenize_line
                if self.is_xl
                else p.sd_model.text_processing_engine.tokenize_line
            )
        else:
            self.tokenizer = (
                p.sd_model.conditioner.embedders[0].tokenize_line
                if self.is_xl
                else p.sd_model.cond_stage_model.tokenize_line
            )

        nip = self._getScheduledNegPip(p.prompts, p.steps)
        pin = self._getScheduledNegPip(p.negative_prompts, p.steps)

        self.conds_all = self._calc_conds(p, nip)
        self.unconds_all = self._calc_conds(p, pin)

        hr_steps: int = getattr(p, "hr_second_pass_steps", 0) or p.steps

        if self.has_hr_p:
            hr_nip = self._getScheduledNegPip(p.hr_prompts, hr_steps)
            self.hr_conds_all = self._calc_conds(p, hr_nip)

        if self.has_hr_n:
            hr_pin = self._getScheduledNegPip(p.hr_negative_prompts, hr_steps)
            self.hr_unconds_all = self._calc_conds(p, hr_pin)

        def calcChunks(a: int, b: int) -> int:
            return a // b if a % b == 0 else a // b + 1

        self.c_len = calcChunks(self.tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(self.tokenizer(p.negative_prompts[0])[1], 75)

        patch_sd_negpip(self, NegPiP)
        reset_prompt_cache(p)
        p.extra_generation_params["NegPiP"] = True
        self.active = True

        if len(self.conds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Positive: {self.conds_all[0][0][1][0][1]})")
        if len(self.unconds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Negative: {self.unconds_all[0][0][1][0][1]})")

    def postprocess(self, *args, **kwargs):
        self.reset()

    def before_hr(self, *args, **kwargs):
        self.is_hr = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if (not self.active) or self.is_anima or self.is_krea2:
            return

        conds_list = []
        tokens_list = []

        if self.is_hr and self.has_hr_p:
            conds = self.hr_conds_all
        else:
            conds = self.conds_all

        if conds is not None:
            for step, regions in conds[0]:
                if step >= params.sampling_step + 2:
                    for conds, tokens in regions:
                        conds_list.append(conds)
                        tokens_list.append(tokens)
                    break
            self.conds = conds_list
            self.c_tokens = tokens_list

        unconds_list = []
        uc_tokens_list = []

        if self.is_hr and self.has_hr_n:
            unconds = self.hr_unconds_all
        else:
            unconds = self.unconds_all

        if unconds is not None:
            for step, regions in unconds[0]:
                if step >= params.sampling_step + 2:
                    for unconds, uc_tokens in regions:
                        unconds_list.append(unconds)
                        uc_tokens_list.append(uc_tokens)
                        break
            self.unconds = unconds_list
            self.uc_tokens = uc_tokens_list

    @staticmethod
    def _getScheduledNegPip(
        prompts: list[str], steps: list[int]
    ) -> list[list[tuple[int, list[tuple[str, float]]]]]:
        """extract the prompts with negative weights"""

        output = []

        scheduled = get_learned_conditioning_prompt_schedules(prompts, steps)
        for i, batch_schedule in enumerate(scheduled):
            stepout = []

            for step, prompt in batch_schedule:
                neg_matches: list[str] = re.findall(NEG_PATTERN, prompt)
                neg_targets = []

                for minusmatch in neg_matches:
                    prompts[i] = prompts[i].replace(minusmatch, "")
                    neg_targets.append(minusmatch.strip("(").strip(")"))

                neg_targets: list[tuple[str, str]] = [x.split(":") for x in neg_targets]
                text_weights: list[tuple[str, float]] = []

                for text, weight in neg_targets:
                    if text.strip() in ("BREAK", "AND", "ADDCOL", "ADDROW"):
                        continue
                    if (weight := float(weight)) < 0.0:
                        text_weights.append((text, weight))

                stepout.append((step, text_weights))

            output.append(stepout)

        return output

    def _cond_dealer(
        self, p: "StableDiffusionProcessing", target: tuple[str, float]
    ) -> tuple[torch.Tensor, int]:
        conds = []

        input = SdConditioning(
            [f"({target[0]}:{-target[1]})"],
            width=p.width,
            height=p.height,
        )

        cond = get_learned_conditioning(p.sd_model, input, p.steps)

        _, token_len = self.tokenizer(target[0])

        conds.append(
            cond[0][0].cond[1 : token_len + 2, :]
            if not self.is_xl
            else cond[0][0].cond["crossattn"][1 : token_len + 2, :]
        )

        conds = torch.cat(conds, 0).unsqueeze(0)
        return conds.repeat(self.batch_size, 1, 1), conds.shape[1]

    def _calc_conds(
        self,
        p: "StableDiffusionProcessing",
        targetlist: list[list[tuple[int, list[tuple[str, float]]]]],
    ) -> list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]:
        outconds = []
        for batch in targetlist:
            stepconds = []
            for step, regions in batch:
                regionconds = []
                for targets in regions:
                    conds, c_tokens = self._cond_dealer(p, targets)
                    regionconds.append((conds, c_tokens))
                stepconds.append((step, regionconds))
            outconds.append(stepconds)
        return outconds
