# chashma

**A small GUI grounding model you can run on a laptop GPU.** Give it a screenshot and an
instruction like *"open the pricing page"* and it tells you where to click.

Computer-use agents spend most of their budget on one question, *where on this screen is
the thing I want?*, and usually answer it by sending every screenshot to a frontier
model. chashma is [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
fine-tuned for just that question on 60k examples from real websites that label
themselves. It runs locally in ~5 GB of VRAM and was trained on one free Kaggle T4.

```bash
chashma predict mailchimp.png "call the Mailchimp sales phone number" --save out.png
# click at (949, 75)  raw='(741, 93)'  1481ms      <- lands on "Sales: +1 (...)" in the header
```

## Results

Click accuracy = the predicted point lands inside the target element's box. The chashma
test split is 300 questions on websites (domains) never seen in training.

| model | chashma test | ScreenSpot web | p50 latency | runs |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B (base) | 78.0% | 81.9% | 613 ms | local, RTX 3060 |
| **chashma (Qwen3-VL-2B + LoRA)** | **84.7%** | 82.6% | 613 ms | local, RTX 3060 |
| Claude Haiku 5.5 | 90.7% | 93.1% | 1680 ms | API |

Where fine-tuning helped most (chashma test, base → fine-tuned):

| slice | n | base | chashma | Haiku |
| --- | --- | --- | --- | --- |
| mobile viewport | 60 | 46.7% | **66.7%** | 75.0% |
| direct ("click Sign in") | 29 | 75.9% | **96.6%** | 100% |
| dark theme | 106 | 73.6% | **83.0%** | 90.6% |
| icons | 43 | 72.1% | **76.7%** | 90.7% |
| functional ("log into my account") | 35 | 60.0% | 68.6% | 94.3% |

On unseen websites from the same kind of crawl, fine-tuning closes about half the gap
between the base model and a frontier API model, at the same speed. On ScreenSpot, a
different benchmark built from other sites with its own labeling, it holds level (+0.7,
within noise for n=436): better on dev tools (68.3% → 79.1%), worse on forums
(85.4% → 77.5%). The data makes the model better at the web it was trained on without
making it a better general pointer, and the weakest spot is clear: instructions that
describe a *goal* rather than the thing on screen need more reasoning than a 2B model
brings.

## How it works

```
 seed URLs ──► Playwright ──► screenshot + DOM boxes ──► templates + LLM instructions
                 │  3 viewports, light/dark,        (free labels)
                 │  scrolled positions
                 ▼
       stability + occlusion filters ──► dataset (split by domain) ──► LoRA on Qwen3-VL ──► eval / API / agent
```

### 1. Data that labels itself

The browser already knows where every button is. The crawler renders real sites with
Playwright and runs [`extract.js`](src/chashma/data/extract.js) in the page to pull every
interactive element with its exact bounding box. The hard part is throwing away boxes that
would be wrong labels:

| problem | fix |
| --- | --- |
| hidden / `display:none` / `aria-hidden` elements | computed-style walk up the tree |
| elements covered by cookie banners, modals, sticky headers | `elementFromPoint` probes at 5 points, must be on top at the center + majority |
| carousels and animations moving between extraction and screenshot | extract **before and after** the screenshot, keep only elements that didn't move |
| `<a><button>` nesting producing duplicate targets | IoU-based dedupe |
| full-page click handlers | drop anything bigger than half the viewport |

Seeds come from the [Tranco](https://tranco-list.eu) top sites
([`scripts/make_seeds.py`](scripts/make_seeds.py)), filtered for adult and infrastructure
domains, one site per brand. Each page is captured in a sample of variants (desktop /
laptop / mobile, light / dark, top / scrolled).

Current dataset: 3,442 screenshots from 1,373 domains, 67,884 questions (10k on icons).

### 2. Instructions: free templates plus an LLM

Boxes alone aren't enough; the model has to learn how people *ask* for things.

- **templates** ("click Pricing", "open the search field") are free and come straight from
  the element's text, aria-label or placeholder, only when that label is unique on the page.
- **LLM instructions** ([`annotate.py`](src/chashma/data/annotate.py)): numbered marks are
  drawn on ~8 non-overlapping targets and a vision LLM writes three styles per mark in one
  call per page: *direct* ("click Sign in"), *functional* ("log into my account") and
  *visual* ("the magnifying glass at the top right"). Instructions that mention the
  overlay ("click box 3") are dropped, since there are no marks at inference time.

### 3. A split that doesn't cheat

A site's header and nav repeat on every page, so a random split leaks test targets into
training. Splits are assigned **per domain**, and near-identical captures (e.g. a "dark"
variant of a site without dark mode) are dropped with a perceptual hash.

### 4. Training

- **Answer format**: Qwen's native `(x, y)` on a 0..1000 grid, so the model fine-tunes the
  pointing it already knows instead of learning a new syntax.
- **Packing**: up to 16 questions about one screenshot go in one multi-turn sequence, so the
  ~1000 image tokens are encoded once instead of 16 times. A group never asks about the same
  element twice (otherwise later turns copy earlier answers), and is capped at 800 text
  tokens (non-Latin pages tokenize long and used to run the T4 out of memory).
- **Loss** on the answer tokens only.
- **LoRA** r=16 on the language model's attention and MLP (17M trainable parameters).
- **Checkpoint selection by click accuracy**, measured by generating on held-out sites
  during training. Eval loss bottoms out early, while the model still guesses common
  positions without looking at the image.

Laptops thermal-throttle on multi-hour runs (the 3060 dropped to 210 MHz at 89 °C), so
[`scripts/kaggle_train.sh`](scripts/kaggle_train.sh) ships the dataset and a wheel to a free
Kaggle T4, waits for that exact dataset version, and pulls the checkpoint back. One epoch
takes ~5 h.

#### What didn't work

The first version fine-tuned SmolVLM-256M. Packing, more data and better checkpoint
selection all helped the loss, but click accuracy stayed at 5-7%: a model with no
localization pretraining learned to guess where buttons usually are, not to look. A
real-vs-blank-image loss check (0.831 vs 0.827) made that visible. Starting from a model
that already points (Qwen3-VL, 78% zero-shot) and teaching it this domain worked on the
first full run.

### 5. Evaluation

The same harness scores local checkpoints and API baselines (`anthropic:...`,
`openai:...`) on the held-out test split, broken down by instruction style, icon vs text,
viewport and theme, and on
[ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) web.

### 6. The agent: plan in words, click with chashma

[`agent/browser.py`](src/chashma/agent/browser.py) splits a browser agent in two. The
planner LLM sees a downscaled screenshot and only decides *what* to do next, in words
("type asyncio into the quick search field"). chashma decides *where*. The planner never
emits coordinates, so it works from cheap low-res images, and the precise part runs
locally. Every run saves per-step screenshots with the click drawn on them, a
`trajectory.json` and a GIF.

```bash
chashma agent "open the documentation page for the asyncio module" \
  --url https://docs.python.org/3/ --headed
```

`--grounder` also takes `http://localhost:8000` (a running `chashma serve`) or
`anthropic:<model>` to compare against a frontier model doing both jobs.

## Usage

```bash
uv sync --all-extras
uv run playwright install chromium
echo "ANTHROPIC_API_KEY=..." > .env    # only for LLM instructions, baselines and the agent planner

uv run python scripts/make_seeds.py --top 3000    # -> data/seeds_tranco.txt
chashma crawl --seeds data/seeds_tranco.txt      # screenshots + boxes -> data/captures
chashma annotate --model anthropic:claude-haiku-5-5   # optional, costs API calls
chashma build                                    # -> data/dataset/{train,val,test}.jsonl

chashma train                                    # configs/train_qwen3vl_2b.yaml
# or on a free Kaggle T4 (KAGGLE_TOKEN in .env):
scripts/kaggle_train.sh push && scripts/kaggle_train.sh watch

chashma eval --split test --limit 300
chashma eval --dataset screenspot --platform web
chashma eval --model anthropic:claude-haiku-5-5 --split test --limit 300   # baseline

chashma serve
curl -F image=@shot.png -F instruction="open settings" localhost:8000/v1/ground
```

The default model is `checkpoints/chashma-qwen3vl-2b`; every command takes `--model`.

### Docker (GPU)

```bash
chashma publish --repo you/chashma-qwen3vl-2b       # merged weights -> dist/hf/chashma-qwen3vl-2b
docker build -t chashma .
docker run --gpus all -p 8000:8000 \
  -v $PWD/dist/hf/chashma-qwen3vl-2b:/models/chashma:ro chashma
curl localhost:8000/health
```

`--gpus all` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Without it the same image runs on CPU (~9 GB RAM, ~20 s per click).

## Layout

```
src/chashma/
  data/      extract.js, crawler, set-of-marks, annotation, templates, dataset build
  model/     answer format, local VLM / API / HTTP grounders
  train/     SFT with image packing (full or LoRA)
  eval/      click-accuracy harness, ScreenSpot loader
  serve/     FastAPI app
  agent/     planner + grounder browser agent
scripts/     Tranco seeds, Kaggle training
```

## Roadmap

- [ ] more LLM-written functional instructions (the weakest slice)
- [ ] crop/zoom augmentation for small targets
- [ ] agent benchmark: task success and cost, chashma vs frontier grounding
- [ ] int8 / GGUF export for CPU inference
- [ ] RL fine-tuning with a "click landed in box" reward
