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

No runtime dependencies. Pillow is used for reading image sizes when it is
present and is otherwise worked around by parsing image headers directly.

## Commands

```
loratrain presets
loratrain doctor
loratrain check   <dataset> --preset sdxl --trigger mytok
loratrain plan    <dataset> --preset sdxl --base ~/models/sd_xl_base_1.0.safetensors --trigger mytok
loratrain run     <dataset> --preset sdxl --base ~/models/sd_xl_base_1.0.safetensors --trigger mytok
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

## What each command actually does

**`check`** reads every image in the folder, computes the bucket sd-scripts will
use, and reports the two numbers that decide whether detail survives: which
buckets the dataset falls into, and how many images get upscaled into their bucket
because they are smaller than the minimum bucket resolution. It also reports
captions that are missing, captions that are reused, and how much of the dataset
actually contains the trigger word.

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
exist, which learning-rate/parameter names that version supports, and whether
`accelerate` is on PATH.

## What it does not do

It does not train anything itself, does not download models, and does not look at
image content. It cannot tell you whether a caption describes the image well, only
whether the trigger is consistently present. Caption quality is still a human job.

## Tests

```
python -m pytest
```

The tests cover bucket maths, the image header parser and the generated TOML.
