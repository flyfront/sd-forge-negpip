# Experimental Krea 2 NegPiP for Forge Neo

A personal, AI-assisted fork that experiments with NegPiP and prompt weighting for Krea 2 (Krea2) in Forge Neo. It may help when prompt weights such as `(word:1.5)` or `(word:-1.0)` have little or no visible effect.

This fork adds experimental Krea 2 support to the [original sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip). It also retains NegPiP support for **SD1, SDXL, and Anima**.

> [!NOTE]
> This is not an official Krea 2 implementation. It was developed largely with AI assistance for my own use and is shared in case it helps someone with the same problem. If the [upstream extension](https://github.com/Haoming02/sd-forge-negpip) adds Krea 2 support, prefer that.

## Features

- NegPiP-style negative weights for Krea 2 prompts
- Negative prompt weights inside the positive prompt, even at CFG 1
- Non-unit positive weights, although their effect on Krea 2 is usually subtle
- NegPiP behavior for SD1, SDXL, and Anima inherited from the original extension

## Installation

In Forge Neo, open **Extensions → Install from URL** and enter:

```text
https://github.com/flyfront/sd-forge-negpip
```

Set **Branch name** to `neo-krea2`, install the extension, and restart the web UI.

| Branch | Krea 2 behavior | Intended use |
| --- | --- | --- |
| `neo-krea2` (this branch) | NegPiP; positive weights are usually subtle | Original NegPiP-style suppression |
| `neo-krea2-emphasis` (default) | NegPiP plus optional V-Scaling | More noticeable positive and negative prompt weighting |

## How to Use

- add `(foo:-1.0)` in the `positive prompt` to **remove** a concept
- add `(bar:-1.0)` in the `negative prompt` to **enforce** a concept

## Krea 2 Prompt Weighting

Forge Neo's built-in prompt weighting has practically no effect on Krea 2. This experimental fork adds NegPiP-style weighting: negative weights such as `(word:-1.0)` can suppress concepts, just like NegPiP on the other models. Positive weights such as `(word:1.5)` are also applied, but they may be much less visible than negative weights because Krea 2 tends to normalize away simple magnitude changes.

- The extension activates whenever any non-unit weight is present in the prompts, not just negative weights; the activation is recorded as `NegPiP: True` in the infotext
- Krea 2 has no chunking and `BREAK` is not treated specially, same as without this extension; note that the prompt parser internally marks `BREAK` with a weight of `-1`, so a bare `BREAK` behaves like `(BREAK:-1.0)` while the extension is active
- When Krea 2 Reference is active with a reference image, the positive prompt uses Forge Neo's image-aware conditioning path and NegPiP weighting is disabled for that positive prompt. Negative-prompt NegPiP remains available. The console reports `NegPiP Positive Disabled (Krea 2 Reference active)` in this case

> [!NOTE]
> The Krea 2 support is based on [ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip). Following that implementation, weighted prompts are tokenized with the chat template applied once around the entire prompt, rather than once per weighted segment like the built-in emphasis does. Weight magnitudes are applied on the text encoder output by interpolating each weighted token between a neutral (empty) encoding and its actual encoding. Since Krea 2 uses normalization heavily, positive weights mostly affect the remaining directional difference and can look subtle, while negative weights additionally flip the token direction in attention values and tend to be much more noticeable. Weighted prompts render differently from extension-off.

## Examples

> [!NOTE]
> These example images and prompts are inherited from the upstream extension and were not generated with Krea 2. They demonstrate the original NegPiP behavior; Krea 2 results, especially with positive weights, may differ.

<table>
    <tr>
        <th>Base</th>
        <th><code>(aqua hair:-1.0)</code><br>in <b>Positive</b> Prompt</th>
        <th><code>(aqua hair:1.5)</code><br>in <b>Negative</b> Prompt</th>
    </tr>
    <tr>
        <td><img src="./img/off.webp" width=256></td>
        <td><img src="./img/negpip.webp" width=256></td>
        <td><img src="./img/neg.webp" width=256></td>
    </tr>
</table>

- **Full Prompts**

```
masterpiece, best quality, high quality, 1girl, solo, hatsune miku, vocaloid, casual, looking at viewer, smile, simple background, white background,
anime screenshot, anime coloring, screencap, flat color, masterpiece, best quality, very aesthetic, absurdres, aesthetic, detailed, beautiful color, amazing quality, highres, safe
Negative prompt: (signature), worst quality, bad quality, low quality, text, name, watermark, (hdr, cinematic, high contrast), logo, username, bad anatomy, bad proportions, extra limbs, extra digit, extra legs, extra legs and arms, disfigured, missing arms, too many fingers, fused fingers, missing fingers, unclear eyes, censored
```
