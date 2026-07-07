# SD Forge Negative Prompt in Prompt
This is an Extension for Forge [Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic) / [Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which implements **NegPip**, allowing you to give words a negative emphasis inside the positive prompt field to suppress a concept, or vice versa.

> [!IMPORTANT]
> Only **SD1** and **SDXL** are supported<br>
> 🔥 **New:** Now supports **Anima** & **Krea 2**

## How to Use

- add `(foo:-1.0)` in the `positive prompt` to **remove** a concept
- add `(bar:-1.0)` in the `negative prompt` to **enforce** a concept

## Krea 2

Due to how it is implemented, the built-in prompt weighting has practically no effect on Krea 2. With this extension, the weights work as intended — not only the negative ones, but also the positive ones: `(word:2.0)` now actually strengthens the concept.

- The extension activates whenever any non-unit weight is present in the prompts, not just negative weights; the activation is recorded as `NegPiP: True` in the infotext
- Krea 2 has no chunking and `BREAK` is not treated specially, same as without this extension; note that the prompt parser internally marks `BREAK` with a weight of `-1`, so a bare `BREAK` behaves like `(BREAK:-1.0)` while the extension is active

> [!NOTE]
> The Krea 2 support is based on [ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip). Following that implementation, weighted prompts are tokenized with the chat template applied once around the entire prompt, rather than once per weighted segment like the built-in emphasis does; the weight magnitudes are then applied on the text encoder output, by interpolating each weighted token between a neutral (empty) encoding and its actual encoding. Weighted prompts therefore render differently from extension-off.

## Examples

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
