# loratrain

A small command line front end for kohya's `sd-scripts` that does the three things
the training scripts leave to you: check the dataset, build a config that matches
the model you are actually training, and launch it.

Three presets ship, each encoding one specific failure mode:

- `sdxl` — eyes come out wrong because the dataset was trained at 512, faces are
  small in frame, or the text encoders ran in fp16.
- `flux` - anatomy renders wrong or not at all because it occupies too few pixels
  in the dataset and the captions never name it.
- `krea` - the face looks untrained because a LoRA trained on `flux1-dev` was
  applied to Krea, whose weights are different enough to dilute the delta.

## Install

```
pip install -e .
```

Or without installing: `python -m loratrain ...` from the repository root.

No runtime dependencies. Pillow is used for reading image sizes and for
near-duplicate detection when it is present, and image headers are parsed directly
when it is not — install it with `pip install -e ".[images]"` if you want the
perceptual hash.

## Using models you already have in ComfyUI

`--comfy` points at the ComfyUI install and fills any model path you did not pass
from `<root>/models`, choosing the file that can actually train:

```
loratrain run ~/data/sdxl --preset sdxl --comfy ~/ComfyUI --trigger mytok
loratrain run ~/data/krea --preset krea --comfy ~/ComfyUI --trigger mytok
```

Explicit `--base` / `--clip-l` / `--t5xxl` / `--ae` always win over what it finds.

What works:

- `models/checkpoints/sd_xl_base_1.0.safetensors` — yes, that is the layout
  `sdxl_train_network.py` loads.
- `models/diffusion_models/flux1-dev.safetensors` — yes, when it is the transformer
  alone in bf16 (~23.8 GB).
- `models/text_encoders/clip_l.safetensors` and `t5xxl_fp16.safetensors` — yes,
  they are exactly the `--clip_l` / `--t5xxl` files.
- `models/vae/ae.safetensors`, `sdxl_vae.safetensors` — yes.
- `models/loras/*.safetensors` — only as an initialisation
  (`--set network_weights=...`), never as a training base.

What does not, and why:

- All-in-one Flux checkpoints, with the transformer, text encoders and VAE in one
  file. sd-scripts loads a transformer, so the file has to be split. `--comfy`
  spots this from the tensor names and says so instead of failing at launch.
- Pre-quantised fp8 Flux weights. `--fp8_base` quantises a bf16 file on the fly;
  handing it already-scaled fp8 weights throws away what it is meant to compute.
- GGUF models — not trainable at all.

```
loratrain doctor --comfy ~/ComfyUI
```

lists every model it found, marks each usable or not, and gives the reason for the
ones that are not.

## Where it finds kohya, and where it looks for accelerate

`loratrain` needs three things that can live in different places: the
`sd-scripts` folder, a Python that has torch and accelerate, and the `accelerate`
executable. The scripts are found in this order:

1. `--sd-scripts <path>` — either the `sd-scripts` folder or the `kohya_ss` root
   containing it
2. the `LORATRAIN_SD_SCRIPTS` environment variable
3. the current directory, then `./sd-scripts`
4. `~/sd-scripts`, `~/kohya_ss`, `~/kohya-ss`
5. `/opt/kohya_ss`, `/opt/kohya-ss`, `/opt/sd-scripts`, `/usr/local/sd-scripts`

A folder only counts if it actually holds `sdxl_train_network.py`,
`flux_train_network.py` or `train_network.py`, so it will not mistake a random
directory for an install. `loratrain doctor` prints what it found.

`accelerate` is looked for next to the `--python` interpreter first, and only then
on `PATH`. That is the venv case: `accelerate`, torch and sd-scripts' requirements
live in one environment, so `--python ~/kohya_ss/venv/bin/python` is enough even
when that venv is not activated and nothing of it is on `PATH`.

### Kubuntu

Ubuntu's Python is marked externally managed (PEP 668), so nothing installs into it
and the venv is not optional:

```
sudo apt install python3-venv python3-pip git
python3 -m venv ~/kohya_ss/venv
source ~/kohya_ss/venv/bin/activate
pip install -r ~/kohya_ss/sd-scripts/requirements.txt   # brings torch + accelerate
```

Then either activate that venv before running, or point at it explicitly:

```
loratrain doctor --sd-scripts ~/kohya_ss --python ~/kohya_ss/venv/bin/python
loratrain run ~/data/sdxl --preset sdxl --comfy ~/ComfyUI \
  --sd-scripts ~/kohya_ss --python ~/kohya_ss/venv/bin/python --trigger mytok
```

`--python` decides both which interpreter probes `sd-scripts --help` and which
environment `accelerate` is taken from, so the two cannot disagree.

## Commands

```
loratrain presets
loratrain doctor
loratrain check   <dataset> --preset sdxl --trigger mytok
loratrain caption <dataset> --trigger mytok --vlm-url http://localhost:11434 --vlm-model qwen2.5vl:7b --write
loratrain plan    <dataset> --preset sdxl --base ~/models/sd_xl_base_1.0.safetensors --trigger mytok
loratrain run     <dataset> --preset sdxl --base ~/models/sd_xl_base_1.0.safetensors --trigger mytok
loratrain watch   <out> --follow
loratrain serve   <out>
```

Flux and Krea need four model files:

```
loratrain run ~/data/krea --preset krea \
  --base ~/models/flux1-krea-dev.safetensors \
  --clip-l ~/models/clip_l.safetensors \
  --t5xxl ~/models/t5xxl_fp16.safetensors \
  --ae ~/models/ae.safetensors \
  --trigger mytok
```

`plan` writes `<out>/train.toml` and prints the exact command without running it.
`run` does the same and then starts training, streaming output to the terminal and
to `<out>/train.log`. `run --dry-run` is `plan` with the run's error handling.

`run` also reports on the run while it happens, every minute:

```
[watch] step 320/2000 (16%)  loss 0.128  -4.2%  2.13 it/s  eta 13m10s
```

`--report-every 0` silences it.

## Watching a run

`watch` reads the same numbers out of any log — including a run started in kohya's
GUI or another terminal:

```
loratrain watch ~/out/sdxl_mylora --follow        # live, from now
loratrain watch ~/out/sdxl_mylora --from-start    # live, including the past
loratrain watch ~/out/sdxl_mylora                 # a finished log, in full
```

It prints the step, loss, the loss change over the last 50 points as a percentage,
iterations per second, and the ETA. On a finished log it prints the whole curve:
first loss, current loss, change since the start, loss per 100 samples, the lr
decay, and warnings.

The percentage is the answer to "is it learning": `-4.2%` means the loss fell 4.2%
across the recent window. `+0.3%` means it did not move.

The warnings are the point of the exercise:

- `loss went to nan/inf` — diverged; stop, reload the last checkpoint, lower the lr.
- `loss is rising (+7.1%)` — overfitting or the lr is too high.
- `loss is flat (-0.2%)` — more steps are unlikely to help; look at the samples.
- `throughput fell from 2.10 to 0.80 it/s` — data loading or latent caching.

sd-scripts paints its bar with carriage returns and no newlines while it runs, so
the runner reads both and the log is stored raw; `watch` splits on both as well.
That is why reading one paint behind is normal for a fraction of a second.

## The web view, with the sample images

The terminal cannot show you the samples, so there is a page for that:

```
loratrain run ~/data/sdxl --preset sdxl --comfy ~/ComfyUI --serve --trigger mytok
```

prints `live view   http://127.0.0.1:8000/` before training starts, and the page
fills in as it goes. For a run already going — or one started in kohya's GUI —
point it at the output folder:

```
loratrain serve ~/out/sdxl_mylora          # opens a browser
loratrain serve ~/out/sdxl_mylora --no-browser --port 8010
```

It shows the step and percentage as a bar, the loss curve over the last 400
readings, speed, ETA, the current lr, any warnings from the progress monitor, and a
grid of the sample images sd-scripts writes to `<out>/sample/` — newest first,
labelled with the step parsed out of the filename, click to open full size.

Implementation notes worth knowing: it is one stdlib `http.server` on
`127.0.0.1`, no build step and no CDN, and the page polls `/api/state` every two
seconds. The log is read incrementally from the last offset, so polling a 200 MB
log costs nothing; a truncated log (a new run over the same folder) resets it.
Sample serving rejects any path that is not a plain filename.

`--serve` only stays up while the training process lives. If you want the page to
outlive it, run `loratrain serve` in another terminal.

## What each command actually does

**`check`** reads every image in the folder, computes the bucket sd-scripts will
use, and reports the two numbers that decide whether detail survives: which
buckets the dataset falls into, and how many images get upscaled into their bucket
because they are smaller than the minimum bucket resolution. It also reports
captions that are missing, captions that are reused, and how much of the dataset
actually contains the trigger word.

**`caption`** reads the folder, groups byte-identical and near-identical images,
reports anything under the resolution rule (1024px on the short side by default),
and only then sends the remaining images to a vision model — one at a time, or
`--workers` in parallel. The model is asked for JSON with `caption`, `attributes`
and `tags`. With `--write` the caption lands next to each image as `<stem>.txt`
with the trigger first, and the replies are cached by file digest in
`.loratrain-cache.json`, so a re-run costs nothing until an image changes.

For `--subject character` (the default) it also takes a vote on each attribute
across the dataset — hair colour, hair style, eye colour, expression, framing —
keeps the ones with 80% consensus, and reports the images that disagree. Those
images' captions get the deviant value spelled out (`hair_color: blonde`) instead
of being blended into the trigger, and the report says to consider dropping them:
naming the deviation is mitigation, not a fix.

Any OpenAI-compatible vision endpoint works:

```
loratrain caption ~/data/sdxl --trigger mytok \
  --vlm-url http://localhost:1234/v1 --vlm-model qwen2-vl-7b \
  --workers 4 --write
```

Without `--vlm-url` it runs the offline half only (duplicates and size), which is
free. `--prompt` / `--prompt-file` replace the built-in instruction; the default
describes only what is visible, so the wording for your domain is yours to supply.

### Local vision endpoints

Two shapes are supported and the right one is picked automatically. A bare host is
probed once: if `/api/tags` answers it is Ollama, otherwise it is treated as
OpenAI-compatible. A URL containing `/v1` is always OpenAI-compatible.

**Ollama** — point at the bare port. Its native `/api/chat` is used because it
takes raw base64 and `format: "json"` constrains the reply, which is what makes a
7B model return parseable JSON:

```
ollama pull qwen2.5vl:7b
loratrain caption ~/data/sdxl --trigger mytok \
  --vlm-url http://localhost:11434 --vlm-model qwen2.5vl:7b \
  --write
```

Ollama serializes requests by default, so `--workers` above 1 mostly queues;
raise `OLLAMA_NUM_PARALLEL` if you want real concurrency.

**vLLM** — OpenAI-compatible, so point at `/v1`:

```
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --limit-mm-per-prompt image=1
loratrain caption ~/data/sdxl --trigger mytok \
  --vlm-url http://localhost:8000/v1 --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --workers 8 --write
```

vLLM batches properly, so `--workers 8` is worth it there and 8x faster than 1.

Check either server before a run:

```
loratrain doctor --vlm-url http://localhost:11434 --vlm-model qwen2.5vl:7b
```

It reports which flavour the URL resolved to, the model names the server lists, and
warns when the name you passed is not among them — the error you would otherwise
hit on the first image. `--vlm-timeout` (default 300s) covers the first request,
which includes loading the model.

**`plan`** does the check, then builds the config. Two things worth knowing:

- It probes the installed `sd-scripts` once with `<script> --help` and drops or
  renames any key this version does not accept (`t5xxl_max_token_length` becomes
  `t5_max_token_length`, and so on). The dropped keys are printed; `--no-filter`
  disables this. If loratrain lives in a different virtualenv from sd-scripts,
  pass `--python /path/to/that/venv/bin/python` so the probe uses the right
  interpreter.
- The command runs with `cwd` set to the sd-scripts checkout, because
  `network_module = "networks.lora"` only resolves from there.

**`doctor`** reports what it found: the sd-scripts directory, which model scripts
exist, which flag names that version supports, whether `accelerate` is available
(and from where), and — with `--vlm-url` or `--comfy` — the vision endpoint and the
ComfyUI models.

## What it does not do

It does not train anything itself, does not download models, and has no model of
its own: `caption` uses whatever endpoint you point it at. It cannot tell you
whether a caption is accurate — it can show you what the model said, where the
dataset disagrees with itself, which images repeat, and which are too small, and
you judge.

## Tests

```
python -m pytest
```

The tests cover bucket maths, the image header parser, the generated TOML,
duplicate grouping, attribute voting, the VLM reply parser, caption assembly and
the progress-bar parser.
