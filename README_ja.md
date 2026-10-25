# NegPiP / Prompt Weighting for Forge Neo (MOSS-Image, Krea 2, Anima, SD1, SDXL)

[English README](README.md)

[Haoming02 の sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) を Forge Neo 向けに、個人用として AI の支援を受けて fork したものです。MOSS-Image と Krea 2(Krea2)に NegPiP とプロンプトの重み付けを追加しています。`(word:1.5)` や `(word:-1.0)` のような重みがほとんど、あるいは全く見た目に効かない場合に役立つかもしれません。SD1、SDXL、Anima の NegPiP 対応も引き継いでいます。詳しくは [MOSS-Image のプロンプト重み付け](#moss-image-のプロンプト重み付け)と [Krea 2 のプロンプト重み付け](#krea-2-のプロンプト重み付け)を参照してください。

> [!NOTE]
> 公式の実装ではありません。自分用に、主に AI の支援で開発したものを、同じ問題を抱えている人の役に立つかもしれないので公開しています。Haoming02 の sd-forge-negpip は 2026-09-30 にアーカイブされたため、Forge Neo 版はこの fork で続いています。大元の [sd-webui-negpip](https://github.com/hako-mikan/sd-webui-negpip) も Forge Neo で MOSS-Image(2026-10-25 以降、Forge Neo 組み込みの MOSS-Image 対応のみ)と Krea 2(2026-09-04 以降)に対応しています。負の重みだけで足りるならそちらを使ってください。この fork はそれに加えて `(word:1.5)` のような正の重みも適用します。両者は方式も重みの尺度も異なるため、同じプロンプトを入れても同じ絵にはなりません。[大元の sd-webui-negpip との違い](#大元の-sd-webui-negpip-との違い)を参照してください。

## 機能

- MOSS-Image と Krea 2 で、ポジティブプロンプトの中で負の重みを使える。CFG 1 でも有効
- MOSS-Image の NegPiP 重み付け。MOSS-Image は本来プロンプトの重みを完全に無視する。Forge Neo 自身の MOSS-Image 対応と moss-image-forge-neo 拡張の両方に対応
- 任意の「V-Scaling」。重み付きトークンの attention value(V)をスケールし、MOSS-Image と Krea 2 の正負の重みをより目に見える形で効かせる
- SD1、SDXL、Anima の NegPiP 動作は元の拡張から継承
- Forge Neo の txt2img と img2img に対応

## インストール

Forge Neo の「Extensions → Install from URL」に次の URL を入力します。

```text
https://github.com/flyfront/sd-forge-negpip
```

ブランチ名の指定は不要です。`neo` が既定ブランチで、MOSS-Image を含む対応モデルすべてをカバーしています。インストール後に Web UI を再起動してください。

以前に `neo-krea2-emphasis` ブランチでインストールした場合も、今のところはそのまま動きます。そのブランチは `neo` のミラーとして維持され、同じコミットを受け取っています。ただしミラーは暫定的なもので、いずれ更新を止める予定なので、都合のよいときに `neo` へ切り替えておくことを推奨します。`extensions/sd-forge-negpip` の中で `git switch neo` を実行してください。`neo-moss`(MOSS-Image 対応を開発したブランチ)と、それより古い `neo-krea2`(V-Scaling なしの NegPiP)はすでに凍結しています。更新を受け取り続けるには同じ方法で `neo` に切り替えてください。

V-Scaling のアコーディオンのラベルは「NegPiP V-Scaling」です(2026-10-22 より前は「NegPiP V-Scaling (Krea 2)」でした)。適用されるモデルは Strength スライダーの説明文に列挙しています。このラベルは `ui-config.json` のキーにもなっているので、旧ラベルで保存した既定値は設定し直す必要があります。

## 使い方

- ポジティブプロンプトに `(foo:-1.0)` を書くと、その概念を取り除く
- ネガティブプロンプトに `(bar:-1.0)` を書くと、その概念を強制する
- MOSS-Image と Krea 2 では、「NegPiP V-Scaling」を開いて `(word:1.5)` のような正の重みを使うと、より強く強調できる

## MOSS-Image のプロンプト重み付け

MOSS-Image を Forge Neo で使う方法は 2 つあります。Forge Neo 自身の対応([Haoming02/sd-webui-forge-classic#1564](https://github.com/Haoming02/sd-webui-forge-classic/pull/1564)、2026-10-21 マージ)と、それより前からある [moss-image-forge-neo](https://github.com/eduardoabreu81/moss-image-forge-neo) 拡張です。どちらもプロンプトをそのまま Qwen3.5-9B テキストエンコーダに渡すため、`(word:1.5)` や `(word:-1.0)` のような重みはパースすらされず、括弧と数字がテキストのままエンコーダに届きます。この拡張はどちらの実装がモデルを読み込んだかを検出し、そのテキストエンジンと DiT を次の節の Krea 2 と同じ方法でフックします。設定は何も要りません。

- 重み付きプロンプトは Forge Neo のプロンプトパーサで解釈され(`Emphasis` 設定が適用されます)、MOSS-Image のチャットテンプレートをプロンプト全体に 1 回だけ適用してトークナイズします。重み付き区間は、素のプロンプトと同じトークンになるようにトークナイズされます
- 重みの大きさは Forge Neo の ComfyUI 互換テキストエンコーダ経路(両実装が使っているもの)を通して適用し、各重み付きトークンを空のエンコーディングとそのトークン自身のエンコーディングの間で補間します。つまり ComfyUI 標準のプロンプト重み付けです
- 負の重みは、DiT の 42 個の joint block すべてでそのトークンの attention value(V)の符号を反転します。テキストのみの精錬ブロックは触りません。任意の「V-Scaling」は下の Krea 2 の節で説明するとおりに働き、画像トークンは `sign(weight) × abs(weight) ** Strength` を読み、テキストトークンは符号だけを読みます
- 実効的な重みのないプロンプトは、各実装が単独で行うのと全く同じにエンコードされます
- レイアウトプロンプト(JSON オブジェクト。[Regional Boxes](https://github.com/eduardoabreu81/sd-regional-boxes-neo) や Forge Neo の JSON Builder が書き出すもの)では、重みはキャプションの中(シーンの `caption` と各アイテムの `caption`)だけで読み取ります。描画する文字列(`text`)、`meta`、ボックスはデータとしてそのまま渡すので、ボックスの角括弧が強調と誤認されることはありません。レイアウトの後に続く素のプロンプト(JSON Builder の `Extra` 欄)は通常どおり重み付けされます。重みのないレイアウトはそのまま渡され、拡張は有効になりません
- MOSS-Image では `BREAK` は普通の単語です。NegPiP が有効なあいだも普通の単語のままです
- ネガティブプロンプト側のブロックスキップはそのまま動きます。拡張側のもの(`[MOSS-Image] Blocks skipped in the negative-prompt pass`)も、Forge Neo の Skip Layer Guidance も
- 少ステップの蒸留 LoRA でも動きます(lodestones の 8-step DMD LoRA と akhaliq の 4-step LoRA、CFG 1 で確認)。ポジティブプロンプトの `(word:-1.0)` はそのまま概念を取り除きます。これらの LoRA は CFG 1 で動くためネガティブプロンプトが評価されず、ネガティブプロンプト内の重みはそこでは効きません

両方の経路を MOSS-Image SFT モデル(`moss_image_sft_int8_convrot`)で確認しました。拡張経路は Forge Neo 2.30 と moss-image-forge-neo 0.1.1(0.1.2 はオフラインで確認)、Forge Neo 経路は 2026-10-22 時点の #1564 ブランチ(Skip Layer Guidance オン)で、結果は一致しました。どちらの実装も認識できない場合や内部構造が変わった場合は、MOSS-Image では NegPiP はオフのままになり、コンソールに `NegPiP Disabled (MOSS-Image: ...)` と表示されます。モデル自体はそのまま動きます。

## Krea 2 のプロンプト重み付け

Forge Neo 組み込みのプロンプト重み付けは Krea 2 では意図どおりに働きません。入力埋め込みをスケールする方式で、それはテキストエンコーダの正規化でほぼ打ち消されるため、`(word:1.5)` と書いてもその語は強くなりません。重み付きプロンプトの絵は目に見えて変わることがありますが、意図した方向には変わりません。この実験的な fork は NegPiP 式の重み付けを追加します。`(word:-1.0)` のような負の重みは、他のモデルでの NegPiP と同じように概念を抑え込めます。`(word:1.5)` のような正の重みも適用されますが、Krea 2 は単純な大きさの変化を正規化で消してしまう傾向があるため、負の重みよりずっと目立ちにくいことがあります。

さらに、任意でより強いモードがあります。txt2img / img2img ページの「NegPiP V-Scaling」アコーディオンを有効にすると、プロンプトの重みが重み付きトークンの attention value(V)、つまり各トークンがモデル内部でどれだけ寄与するかを直接スケールし、正負どちらの重みもより目に見えるようになります。「Strength」スライダーは通常の挙動(`0`)からプロンプトの重みそのまま(`1`)までを動き、`2` までの値は符号を保つ累乗カーブで大きさを強めるので、重みが強調から抑制へ、あるいはその逆に反転することはありません。絵がかなり変わるので、必要なときだけ使ってください。オンのあいだは強さが infotext に `NegPiP V-Scaling` として記録されます。アコーディオンがオフのときは、上で説明した素の NegPiP 式の重み付けとして動作します。

- この拡張は、負の重みだけでなく 1 以外の重みがプロンプトに含まれていれば有効になります。有効になったことは infotext に `NegPiP: True` として記録されます
- Krea 2 にはチャンク分けがなく、`BREAK` は特別扱いされません。これは拡張なしの場合と同じです。なお、プロンプトパーサは内部で `BREAK` に重み `-1` を付けるため、拡張が有効なあいだは素の `BREAK` が `(BREAK:-1.0)` と同じ振る舞いになります
- 参照画像ありで Krea 2 Reference が有効なときは、ポジティブプロンプトは Forge Neo の画像を考慮した条件付け経路を使い、そのポジティブプロンプトでは NegPiP の重み付けが無効になります。ネガティブプロンプト側の NegPiP は使えます。この場合コンソールに `NegPiP Positive Disabled (Krea 2 Reference active)` と表示されます

> [!NOTE]
> Krea 2 対応は [ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) を基にしています。その実装に従い、重み付きプロンプトはチャットテンプレートをプロンプト全体に 1 回だけ適用してトークナイズします(組み込みの emphasis のように重み付き区間ごとに適用するのではありません)。重みの大きさはテキストエンコーダの出力に対して、各重み付きトークンを中立(空)のエンコーディングと実際のエンコーディングの間で補間する形で適用します。Krea 2 は正規化を多用するため、正の重みは残った方向成分の差にしか影響せず控えめに見えることがあります。負の重みはさらに attention value 内でトークンの向きを反転させるため、ずっと目立つ傾向があります。V-Scaling を `0` から `1` に上げていくと、エンコーダ側の補間は連続的にフェードアウトし(大きく外挿した重みはトークン埋め込みを本来の意味から逸らしかねません)、代わりに attention value(V)が `sign(weight) × abs(weight) ** Strength` で大きさを受け持ちます。V は attention の中で Krea 2 の正規化が触らない唯一のテンソルなので、トークンは意味を保ったまま寄与量だけが変わり、重みに正負両側で階調が付きます。たとえば `(word:-2.0)` は `(word:-1.0)` より強く抑制します。スケール後の値を読むのは画像トークンだけで、テキストトークンは元の符号マスクを読み続けるため、大きな重みでもテキストストリームは壊れません(cross-attention モデルでの NegPiP の働き方を踏襲しています)。重み付きプロンプトの絵は、拡張オフのときとは異なる仕上がりになります。

## 大元の sd-webui-negpip との違い

hako-mikan の大元の [sd-webui-negpip](https://github.com/hako-mikan/sd-webui-negpip) は Forge Neo で Z-Image、Anima、Krea 2、MOSS-Image(Forge Neo 組み込み対応のみ)に対応しています。この fork は Haoming02 の移植版から派生しており、そちらが新しいアーキテクチャに対して別の方式を採ったため、MOSS-Image と Krea 2 では素の `(word:-1.0)` の場合でも両者の挙動が異なります。両方を同時にインストールしないでください。比較するときは Extensions タブで片方を無効にしてください。

- 方式: 大元は `(word:-1.0)` をプロンプトから取り除き、その語を単独でエンコードし、そのトークンを conditioning に追加して attention value を反転します。この fork は語をプロンプトに残したまま、その語自身のトークン位置の attention value を反転します(Haoming02 版が Anima 向けに導入した位置マスク方式)。その結果、こちらではプロンプトの残りの部分はそのままエンコードされ、構図は重みなしの絵に近く保たれます。大元はトークン列そのものが変わるので、構図もそれに合わせて動きます。一方で、大元の語は周囲のテキストから一切見えませんが、こちらではテキストエンコーダが文脈の中でその語を読むため、概念の一部が後続の語に残ることがあります。`(red:-1.0) apple` のような属性の否定は大元の方がきれいに抜けるかもしれません
- 重みの尺度: 2026-09 以降、大元はこれらのモデルで attention value に重みをそのまま掛けるので、`-2` は `-1` より強く、`-3` あたりで絵全体が変わります(大元の README より)。この fork では V-Scaling オフのとき、MOSS-Image と Krea 2 では重みの符号だけが効きます。`-0.5`、`-1`、`-2` は同じ絵になり、`(word:1.5)` は上で説明した素の重み付けとして適用されます。階調が欲しいときは「NegPiP V-Scaling」を有効にしてください。画像トークンが `sign(weight) × abs(weight) ** Strength` を読み、テキストトークンは符号だけを読み続けるので、大きな重みでも大元より絵が崩れにくくなります。数値は両者で互換ではありません
- 正の重み: 大元はこれらのモデルでは適用しません。`(word:1.5)` は拡張なしのときと同じようにテキストエンコーダに届きます。この fork は適用します(MOSS-Image と Krea 2 の節を参照)
- SD1 と SDXL は、この fork でも大元の方式と尺度のままです。そこでは大元と同じく `(word:-2.0)` は `(word:-1.0)` の 2 倍の強さになります
- 同じ点: MOSS-Image でちょうど `-1.0` のとき、両者は同じ操作を行います。joint block の attention value を反転し、テキストのみのブロックは触らないので、どちらでも語は取り除かれます。Krea 2 では `-1.0` の並列比較で同等の結果が得られました。MOSS-Image では両者の並列比較はまだ行っていません

## 作例

### MOSS-Image

`moss_image_sft_int8_convrot`、Forge Neo 2.30 で生成(Euler、MOSS スケジュール、20 steps、CFG 3.0、1024×1024、同一シード)。

<table>
    <tr>
        <th>ベース</th>
        <th><code>(a red apple:-1.0)</code><br>をポジティブプロンプトに</th>
    </tr>
    <tr>
        <td><img src="./img/moss-base.webp" width=256></td>
        <td><img src="./img/moss-negpip.webp" width=256></td>
    </tr>
</table>

```
a cute cat sitting on a sofa in a living room, a red apple on the table
```

<table>
    <tr>
        <th>この拡張を無効</th>
        <th>V-Scaling オフ</th>
        <th>V-Scaling オン(Strength 1.0)</th>
    </tr>
    <tr>
        <td><img src="./img/moss-weights-ignored.webp" width=256></td>
        <td><img src="./img/moss-vscaling-off.webp" width=256></td>
        <td><img src="./img/moss-vscaling-on.webp" width=256></td>
    </tr>
</table>

```
a cute cat sitting on a sofa in a living room, red theme, (blue theme:2.0)
```

この拡張を無効にすると、MOSS-Image 拡張は `(blue theme:2.0)` を素のテキストとして受け取り、部屋は完全に赤になります。NegPiP 式の重み付けでは青が目に見えて押し出され、V-Scaling オンでは青が勝ちます。

### Krea 2

`krea2_turbo_fp8_scaled`、Forge Neo 2.29 で生成(Euler、8 steps、CFG 1.0、1024×1024、同一シード)。

<table>
    <tr>
        <th>ベース</th>
        <th><code>(television:-1.0)</code><br>をポジティブプロンプトに</th>
    </tr>
    <tr>
        <td><img src="./img/krea2-base.webp" width=256></td>
        <td><img src="./img/krea2-negpip.webp" width=256></td>
    </tr>
</table>

```
a cute cat sitting on a sofa in a living room
```

<table>
    <tr>
        <th>V-Scaling オフ</th>
        <th>V-Scaling オン(Strength 1.0)</th>
    </tr>
    <tr>
        <td><img src="./img/krea2-vscaling-off.webp" width=256></td>
        <td><img src="./img/krea2-vscaling-on.webp" width=256></td>
    </tr>
</table>

```
a cute cat sitting on a sofa in a living room, red theme, (blue theme:2.0)
```

V-Scaling オフでは、正の重みは赤が支配する結果をほとんど変えません。V-Scaling オンでは綱引きに青が勝ち、上で説明したとおり全体の描写も変わります。

### SD1 / SDXL / Anima

> [!NOTE]
> これらの作例画像とプロンプトは上流の拡張から引き継いだもので、Krea 2 で生成したものではありません。元々の NegPiP の挙動を示しています。3 枚目はネガティブプロンプト内の重みを使っていますが、CFG 1 の Krea 2 turbo モデルでは Forge Neo がネガティブプロンプトを省略するため効果がありません。

<table>
    <tr>
        <th>ベース</th>
        <th><code>(aqua hair:-1.0)</code><br>をポジティブプロンプトに</th>
        <th><code>(aqua hair:1.5)</code><br>をネガティブプロンプトに</th>
    </tr>
    <tr>
        <td><img src="./img/off.webp" width=256></td>
        <td><img src="./img/negpip.webp" width=256></td>
        <td><img src="./img/neg.webp" width=256></td>
    </tr>
</table>

- プロンプト全文

```
masterpiece, best quality, high quality, 1girl, solo, hatsune miku, vocaloid, casual, looking at viewer, smile, simple background, white background,
anime screenshot, anime coloring, screencap, flat color, masterpiece, best quality, very aesthetic, absurdres, aesthetic, detailed, beautiful color, amazing quality, highres, safe
Negative prompt: (signature), worst quality, bad quality, low quality, text, name, watermark, (hdr, cinematic, high contrast), logo, username, bad anatomy, bad proportions, extra limbs, extra digit, extra legs, extra legs and arms, disfigured, missing arms, too many fingers, fused fingers, missing fingers, unclear eyes, censored
```
