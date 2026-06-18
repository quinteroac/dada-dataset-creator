# Dada Dataset Creator

Local FastAPI app for creating, curating, captioning, exporting, and training LoRA datasets. It is designed around two workflows:

- **Anima LoRA**, compatible with `kohya-ss/sd-scripts` and `anima_train_network.py`.
- **Qwen Image Edit-2511 LoRA**, compatible with `kohya-ss/musubi-tuner`.

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
- Caption a single image or a batch.
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

- `setup_musubi_tuner`: clones `kohya-ss/musubi-tuner` into `vendor/musubi-tuner` and installs it as an editable package.
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
- [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) for Qwen Image Edit-2511 LoRA training.
- [Modal](https://modal.com/) for optional remote Qwen training.
- Codex for optional local generation, captioning, and curation workflows.

## License

This project is licensed under the Apache License 2.0, matching the primary license used by the training scripts it integrates with. See [LICENSE](LICENSE).

Third-party projects keep their own licenses. In particular, `kohya-ss/sd-scripts` is Apache License 2.0, and `kohya-ss/musubi-tuner` is Apache License 2.0 for its main codebase, with some model-specific subdirectories following their upstream licenses.
