# chashma

**A tiny GUI grounding model that runs on a laptop GPU.** Give it a screenshot and
an instruction like *"open the pricing page"* and it tells you where to click.

Computer-use agents spend most of their budget on one question, *where on this
screen is the thing I want?*, and today they answer it by sending every screenshot
to a frontier model. chashma is a ~256M-parameter vision-language model fine-tuned
for just that question, trained entirely on a 6 GB RTX 3060 laptop, on data that
labels itself.

```bash
chashma predict shot.png "search for a department" --save out.png
```

## How it works

```
 seed URLs ──► Playwright ──► screenshot + DOM boxes ──► set-of-marks ──► Claude / GPT
                 │  3 viewports, light/dark,         (free labels)     overlay     writes instructions
                 │  scrolled positions                                              (direct / functional / visual)
                 ▼
          stability + occlusion filters ──► dataset (split by domain) ──► SmolVLM fine-tune ──► eval + API
```

### 1. Data that labels itself

The browser already knows where every button is. The crawler renders real sites with
Playwright and runs [`extract.js`](src/chashma/data/extract.js) in the page to pull
every interactive element with its exact bounding box. The hard part is throwing away
the boxes that would be wrong labels:

| problem | fix |
| --- | --- |
| hidden / `display:none` / `aria-hidden` elements | computed-style walk up the tree |
| elements covered by cookie banners, modals, sticky headers | `elementFromPoint` probes at 5 points, must be on top at the center + majority |
| carousels and animations moving between extraction and screenshot | extract **before and after** the screenshot, keep only elements that didn't move |
| `<a><button>` nesting producing duplicate targets | IoU-based dedupe |
| full-page click handlers | drop anything bigger than half the viewport |

Each page is captured in a sample of variants (desktop / laptop / mobile, light / dark,
top / scrolled) so the model sees the same UI in different shapes.

### 2. Instructions from an LLM, one call per page

Boxes alone aren't enough; the model needs to learn how people *ask* for things.
[`annotate.py`](src/chashma/data/annotate.py) draws numbered marks on ~8
non-overlapping targets (round-robin over icons, inputs and text so icons aren't drowned
out) and asks a vision LLM for three instruction styles per mark:

- **direct**: "click Sign in"
- **functional**: "log into my account" (the goal, not the label)
- **visual**: "the magnifying glass at the top right" (essential for icons)

One call covers all marks on a page, so annotation costs about one image per
screenshot rather than one per element. Instructions that mention the overlay ("click
box 3") are filtered out, since there are no marks at inference time. Free template
instructions ("click Pricing") are mixed in, but only for labels that are unique on the
page.

### 3. A split that doesn't cheat

A site's header and nav repeat on every page, so a random split leaks test targets into
training. Splits are assigned **per domain** (hash-bucketed), and near-identical
captures (e.g. a "dark" variant of a site without dark mode) are dropped with a
perceptual hash.

### 4. Training on 6 GB

The model answers in a tiny fixed format, `<point>x,y</point>` on a 0..999 grid,
so outputs are a few tokens and resolution-independent. The loss is masked to the
answer tokens only. SmolVLM-256M fits a full fine-tune in bf16 with gradient
checkpointing; SmolVLM-500M trains with LoRA. Configs are in [`configs/`](configs).

### 5. Evaluation

Click accuracy = the predicted point falls inside the target box. The same harness
scores local checkpoints and API baselines (`anthropic:...`, `openai:...`) on:

- the held-out chashma test split (unseen domains), broken down by instruction style,
  icon vs text, viewport and theme
- [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) (web, plus
  mobile/desktop to measure transfer)

## Results

_Coming once the first full crawl + training run is done._

## Usage

```bash
uv sync --all-extras
uv run playwright install chromium
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY for openai:<model>

chashma crawl --seeds configs/seeds.txt          # screenshots + boxes -> data/captures
chashma annotate --model anthropic:claude-haiku-5-5
chashma build                                    # -> data/dataset/{train,val,test}.jsonl
chashma train --config configs/train_256m.yaml   # -> checkpoints/chashma-256m

chashma eval --model checkpoints/chashma-256m
chashma eval --model checkpoints/chashma-256m --dataset screenspot --platform web
chashma eval --model anthropic:claude-haiku-5-5 --limit 200   # baseline

chashma serve --model checkpoints/chashma-256m
curl -F image=@shot.png -F instruction="open settings" localhost:8000/v1/ground
```

Docker (GPU):

```bash
docker build -t chashma .
docker run --gpus all -p 8000:8000 -v $PWD/checkpoints/chashma-256m:/models/chashma chashma
```

## Layout

```
src/chashma/
  data/      extract.js, crawler, set-of-marks, annotation, templates, dataset build
  model/     answer format, local VLM + API grounders
  train/     SFT (full or LoRA)
  eval/      click-accuracy harness, ScreenSpot loader
  serve/     FastAPI app
```

## Roadmap

- [ ] first full crawl (Tranco top ~2k) + training run, results table
- [ ] crop/zoom augmentation for small targets
- [ ] verifier pass: a second model must click the right mark, or the instruction is dropped
- [ ] demo agent: Claude plans in text, chashma grounds each step
- [ ] ONNX / int8 export for CPU inference
- [ ] RL fine-tuning with a "click landed in box" reward
