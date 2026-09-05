# Dada Dataset Creator

Local FastAPI app for creating, curating, captioning, exporting, and training LoRA datasets. It is designed around three workflows:

- **Anima LoRA**, compatible with `kohya-ss/sd-scripts` and `anima_train_network.py`.
- **Qwen Image Edit-2511 LoRA**, compatible with `kohya-ss/musubi-tuner`.
- **Krea 2 LoRA**, compatible with `kohya-ss/musubi-tuner` (`krea2_train_network.py`).
- **Ideogram4 LoRA**, compatible with `ostris/ai-toolkit`.
- **MiniMax H3 image LoRA**, using Ostris AI Toolkit with still images and `.txt` captions.

The app works fully locally for image uploads, manual caption editing, and dataset export. Codex image generation can also be used to create synthetic datasets from prompts and references. Codex, training, and Modal features are optional.

## Requirements

- Python 3.12+
- `uv`
- `git`
- For local training: `accelerate`, PyTorch/GPU, and the dependencies for the relevant trainer.
- For Codex features: an authenticated `codex` CLI and the optional `codex` dependency group.
- For remote Qwen training: a Modal account and the optional `modal` dependency group.

## Run

```bash
uv sync --group dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://127.0.0.1:8000 on this machine, or `http://<your-machine-ip>:8000` from another device on the same network.

## Basic Workflow

1. Create a dataset from the home page.
2. Choose the dataset type:
   - `Anima`: images with tag-style captions.
   - `Qwen Image Edit-2511`: control/target pairs with edit instructions.
   - `Krea 2`: images with captions (text-to-image, no control images).
   - `Ideogram4`: images with structured JSON captions and estimated bboxes.
3. Upload images or edit pairs.
4. Optionally use Codex image generation to create synthetic training images or Qwen edit pairs.
5. Edit captions manually, or use Codex to generate captions.
6. Download `dataset.toml` or `export.zip`.
7. Optionally launch training from the dataset page.

## Output Structure

Each dataset is written to `./datasets/<dataset_slug>/`.

Anima dataset:

```text
datasets/
  my_dataset/
    settings.json
    dataset.toml
    jobs_state.json
    images/
      000001.png
      000001.txt
      000001.meta.json
    references/
      ref_001.png
    raw/
    curator/
    outputs/
```

Qwen Image Edit-2511 dataset:

```text
datasets/
  my_qwen_dataset/
    settings.json
    dataset.toml
    images/
      000001.png
      000001.txt
      000001.meta.json
    controls/
      000001.png
    cache/
    references/
    raw/
    curator/
    outputs/
```

Ideogram4 dataset:

```text
datasets/
  my_ideogram_dataset/
    settings.json
    dataset.toml
    ideogram4_training_config.yaml
    images/
      000001.png
      000001.txt
      000001.meta.json
    references/
    raw/
    curator/
    outputs/
```

Krea 2 dataset:

```text
datasets/
  my_krea_dataset/
    settings.json
    dataset.toml
    krea2_training_config.json
    images/
      000001.png
      000001.txt
      000001.meta.json
    cache/
    references/
    raw/
    curator/
    outputs/
```

`dataset.toml` is regenerated when settings change and before training or export.

## Curator

Each dataset has a curator view at:

```text
/datasets/<slug>/curator
```

From there you can:

- Upload raw images.
- Upload visual references.
- Ask Codex to transform raw images using an instruction.
- Review generated candidates.
- Approve one or more candidates into the final dataset.

For Qwen datasets, approving a candidate copies the raw image as the `control` and the curated image as the `target`.

## Codex

Codex features run as background jobs and stream live logs in the dataset page.

Install the optional dependency group and authenticate the CLI:

```bash
uv sync --group dev --group codex
codex login
```

Available features:

- Generate synthetic images for Anima datasets from prompts and optional reference images.
- Generate synthetic control/target edit pairs for Qwen Image Edit-2511.
- Import recent images from `~/.codex/generated_images`.
- Caption a single image or a batch, including Ideogram4 JSON captions with estimated bboxes.
- Curate raw images using references.

This makes it possible to build fully synthetic datasets: describe the target concept, generate candidate images or edit pairs with Codex imagegen, import them into the dataset, then caption, curate, export, or train from the same UI.

Control how many Codex sessions can run in parallel:

```bash
CODEX_MAX_PARALLEL_JOBS=2 uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The integration uses your local `codex` CLI session and loads the SDK dynamically. It does not call the OpenAI Images API directly.

## Anima Training

From the dataset page, you can launch two job types:

- `setup_sd_scripts`: clones `kohya-ss/sd-scripts` into `vendor/sd-scripts`, or updates the existing checkout with a fast-forward pull.
- `train_anima_lora`: runs `accelerate launch anima_train_network.py` with the dataset's `dataset.toml`.

Logs appear in the jobs panel. Output defaults to:

```text
datasets/<dataset_slug>/outputs
```

To resume an interrupted Anima run, enable `save training state` before starting the original training. After interruption, start a new Anima training job with `Resume state path` pointing to the saved state directory from the output folder. The app passes this to `sd-scripts` as `--resume=<path>`.

The Anima form defaults to the current lower-VRAM path: latent caching, text encoder output caching, the 2D Qwen-Image VAE (`--qwen_image_vae_2d`), TF32, and cuDNN benchmark. `torch.compile` is exposed but off by default because it requires a compatible PyTorch/Triton environment; when enabled, the app uses `--compile_mode=default` and `--compile_cache_size_limit=32`.

The app does not automatically install PyTorch or GPU dependencies for `sd-scripts`. Install those dependencies in the same environment used to launch FastAPI.

Example for an RTX 50-series setup:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cd vendor/sd-scripts
uv pip install -r requirements.txt
cd ../..
uv run accelerate config default
```

## Qwen Image Edit-2511 Training

For Qwen Image Edit-2511, the app uses `kohya-ss/musubi-tuner`.

From the dataset page:

- `setup_musubi_tuner`: clones `kohya-ss/musubi-tuner` into `vendor/musubi-tuner` and installs it as an editable package. If the checkout already exists, it is fast-forward pulled first so newer scripts (such as the Krea 2 trainers) become available.
- `train_qwen_edit_lora`: runs latent caching, text encoder caching, and training.

Local training runs:

```text
python src/musubi_tuner/qwen_image_cache_latents.py
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py
accelerate launch src/musubi_tuner/qwen_image_train_network.py
```

Notes:

- Use bf16 checkpoints for `dit` and `text_encoder`.
- Do not use fp8 checkpoints as base files; use the `fp8_base`, `fp8_scaled`, and `fp8_vl` flags for VRAM savings.
- The default preset uses `blocks_to_swap=36`.

## Krea 2 Training

For Krea 2, the app uses `kohya-ss/musubi-tuner` (the same trainer as Qwen Image Edit-2511). Krea 2 is a single-stream MMDiT text-to-image model that uses **Qwen3-VL-4B-Instruct** as the text encoder and the **Qwen-Image VAE**. The recommended workflow is to **train on the RAW DiT** and run inference on the distilled **Turbo** DiT.

From the dataset page:

- `setup_musubi-tuner`: clones `kohya-ss/musubi-tuner` into `vendor/musubi-tuner` (or fast-forward pulls the existing checkout so the Krea 2 scripts are available) and installs it as an editable package.
- `train_krea2_lora`: runs latent caching, text encoder output caching, and training.

Local training runs:

```text
python src/musubi_tuner/krea2_cache_latents.py
python src/musubi_tuner/krea2_cache_text_encoder_outputs.py
accelerate launch src/musubi_tuner/krea2_train_network.py
```

Notes:

- Use bf16 checkpoints for `dit` (RAW) and `text_encoder`. Do not pass fp8 model files; use the `fp8 base` / `fp8 scaled` toggles for DiT VRAM savings. Krea 2 requires `--fp8_base` and `--fp8_scaled` together, so enabling either toggle enables both.
- Default LoRA targets all Linear layers in the DiT with rank/alpha 32 (`networks.lora_krea2`), matching the model authors' recommended default.
- The default timestep sampling is `shift` with `discrete_flow_shift 2.5` (the K2 inference time-shift at 1024×1024). For varying-resolution training, switch to `krea2_shift`, which reproduces K2's resolution-aware schedule per sample.
- `blocks_to_swap` maximum is 26 (28 blocks − 2). The app defaults Krea 2 local training to `fp8_base`, `fp8_scaled`, `blocks_to_swap=26`, H2D-only block swap, and `block_swap_ring_size=1` for 16 GB class GPUs; reduce or clear block swap only when you have enough VRAM.
- `Turbo DiT for samples` is optional: when set, sample images during training are generated on the Turbo model with the trained LoRA applied on top. `--turbo_dit` cannot be combined with `--blocks_to_swap`.
- Model files: RAW DiT `raw.safetensors` from `krea/Krea-2-Raw`, Turbo DiT `turbo.safetensors` from `krea/Krea-2-Turbo`, VAE from `Comfy-Org/Qwen-Image-Edit_ComfyUI`, and the text encoder `qwen3vl_4b_bf16.safetensors` from `Comfy-Org/Qwen3-VL`.
- The Modal backend runs `modal run app/modal_krea2.py` and defaults to volume `dada-krea2` with GPU `RTX-PRO-6000`. Keep bf16 checkpoints for RAW DiT and text encoder; the Modal path still uses Musubi's `--fp8_base` / `--fp8_scaled` flags for memory savings.

## Ideogram4 Training

For Ideogram4 and MiniMax H3, the app uses `ostris/ai-toolkit`.

MiniMax H3 also supports **Training backend → Modal**, with `RTX-PRO-6000` as the default GPU, matching Krea 2 ([Modal GPU identifiers](https://modal.com/docs/guide/gpu)). Install the Modal dependency group (`uv sync --group modal`) and authenticate with `uv run modal setup` before starting a cloud job. Local AI Toolkit setup is unnecessary for Modal.

The Modal worker (`app/modal_minimax_h3.py`) installs a pinned AI Toolkit revision, uses a persistent `dada-minimax-h3` volume, and requests 128 GiB of host RAM for model loading/offloading. The timeout defaults to 86400 seconds. GPU, volume, timeout, and remote output directory can be changed in the form. Remote output directories must be under `/data/outputs/`.

Each cloud run uploads a fresh snapshot of images and `.txt` captions. A local Comfy-layout model directory or local adapter file is uploaded automatically; repository weights download remotely and remain cached in the volume. Existing `/data/...` model paths refer to that volume. After successful training, LoRA checkpoints download to the configured local output directory. If downloading fails, the job retains its Modal output location. Dataset snapshots, models, caches, and checkpoints remain in the volume until removed. Cancelling stops the attached Modal CLI, as in the Krea 2 integration; inspect the Modal dashboard if an interrupted remote run remains active.

The Modal integration is covered by simulated execution/upload tests; a real cloud image build and GPU training run have not been verified.

To train MiniMax H3 from images:

1. Create a **MiniMax H3 (images)** dataset, upload PNG/JPEG/WebP images, and edit or generate their `.txt` captions.
2. Click **Setup AI Toolkit**. An existing checkout must already include MiniMax H3 support; otherwise update it with `git -C vendor/ai-toolkit pull --ff-only`, then rerun setup to install its requirements.
3. Use `Comfy-Org/MiniMax-H3` or a local folder with the same component layout. Missing weights download at training time. The default partition is `fl2va_pruned`; quantized weights are loaded in their existing format.
4. Start with resolution `512`, rank `16`, and batch size `1`. Resolution entries must be multiples of 32. Dataset settings supply batch size and repeats. Layer offloading defaults to 95%; actual GPU/RAM requirements depend on the model and configuration.
5. Click **Start MiniMax H3 image training** and follow the job logs. Jobs can be cancelled from the panel.

The generated `minimax_h3_training_config.yaml` uses JSON syntax (valid YAML), with `arch: minimax_h3`, `num_frames: 1`, and audio disabled. Latents and text embeddings are cached; automatic preview sampling is disabled. An optional assistant training adapter can be supplied. Outputs are stored under `outputs/<output_name>/` by default. Training images teach appearance/style; this workflow supplies no motion or audio examples.

The configuration follows the upstream [MiniMax H3 implementation](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py) and [dataset configuration](https://github.com/ostris/ai-toolkit/blob/main/toolkit/config_modules.py). Automated tests verify configuration and job execution with a simulated training process; they do not verify GPU training or model quality.

From the dataset page:

- `setup_ai_toolkit`: clones `ostris/ai-toolkit` into `vendor/ai-toolkit` and installs `requirements.txt`.
- `train_ideogram4_lora`: writes `ideogram4_training_config.yaml` and runs `python run.py <config>`.

Ideogram4 captions are saved as JSON in each image's `.txt` file. Codex captioning produces `high_level_description`, `style_description`, and `compositional_deconstruction.elements` with estimated `[x1, y1, x2, y2]` bboxes normalized to the dataset resolution.

## Qwen Training on Modal

Install the optional dependency group:

```bash
uv sync --group dev --group modal
modal setup
```

When you choose the `modal` backend, the app runs:

```bash
modal run app/modal_qwen_edit.py
```

The script uploads the dataset to a Modal volume, builds a CUDA image with `musubi-tuner`, runs cache + training, and writes outputs to the configured volume.

Relevant controls:

- `modal_volume_name`: Modal volume, defaults to `dada-qwen-edit`.
- `modal_gpu`: Modal GPU, defaults to `L40S`.
- `modal_timeout`: job timeout, defaults to `86400`.
- `modal_bake_models`: includes local model files inside the Modal image.

## Jobs

Jobs are stored inside the dataset and recovered after app restart if they were still queued.

Main types:

- Codex: generation, captioning, recent import, and curation.
- Training: trainer setup and training.

Only training jobs can be cancelled from the UI.

Control training parallelism:

```bash
TRAINING_MAX_PARALLEL_JOBS=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Export

From the dataset page, you can download:

- `dataset.toml`
- `<dataset_slug>.zip`

You can also open them directly:

```text
/datasets/<slug>/dataset.toml
/datasets/<slug>/export.zip
```

## Tests

```bash
uv run pytest
```

## Acknowledgments

This project builds on and integrates with:

- [FastAPI](https://fastapi.tiangolo.com/) for the local web app.
- [uv](https://docs.astral.sh/uv/) for Python dependency and environment management.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) for Anima LoRA training.
- [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) for Qwen Image Edit-2511 and Krea 2 LoRA training.
- [Modal](https://modal.com/) for optional remote Qwen training.
- Codex for optional local generation, captioning, and curation workflows.

## License

This project is licensed under the Apache License 2.0, matching the primary license used by the training scripts it integrates with. See [LICENSE](LICENSE).

Third-party projects keep their own licenses. In particular, `kohya-ss/sd-scripts` is Apache License 2.0, and `kohya-ss/musubi-tuner` is Apache License 2.0 for its main codebase, with some model-specific subdirectories following their upstream licenses.
