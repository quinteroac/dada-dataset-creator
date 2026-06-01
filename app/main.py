from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.codex_service import CodexClientService, CodexUnavailableError
from app.job_runner import CodexJobRunner
from app.jobs import JobStore
from app.storage import DatasetNotFoundError, DatasetStore
from app.training_runner import TrainingJobRunner
from app.training_service import TrainingService
from app.models import DATASET_TYPE_QWEN_IMAGE_EDIT_2511


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.job_store = JobStore(Path("datasets"))
    fastapi_app.state.job_store.recover_interrupted_jobs()
    fastapi_app.state.runner = CodexJobRunner(DatasetStore(Path("datasets")), fastapi_app.state.job_store)
    fastapi_app.state.training_runner = TrainingJobRunner(DatasetStore(Path("datasets")), fastapi_app.state.job_store)
    await fastapi_app.state.runner.start()
    await fastapi_app.state.training_runner.start()
    try:
        yield
    finally:
        await fastapi_app.state.runner.stop()
        await fastapi_app.state.training_runner.stop()


app = FastAPI(title="Dada Dataset Creator", lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/media/datasets", StaticFiles(directory="datasets", check_dir=False), name="dataset_media")


def get_store() -> DatasetStore:
    return DatasetStore(Path("datasets"))


def get_job_store() -> JobStore:
    if hasattr(app.state, "job_store"):
        return app.state.job_store
    return JobStore(Path("datasets"))


def get_runner() -> CodexJobRunner:
    if not hasattr(app.state, "runner"):
        app.state.job_store = JobStore(Path("datasets"))
        app.state.runner = CodexJobRunner(get_store(), app.state.job_store)
    return app.state.runner


def get_training_runner() -> TrainingJobRunner:
    if not hasattr(app.state, "training_runner"):
        app.state.job_store = JobStore(Path("datasets"))
        app.state.training_runner = TrainingJobRunner(get_store(), app.state.job_store)
    return app.state.training_runner


def get_training_service() -> TrainingService:
    return TrainingService()


def get_codex() -> CodexClientService:
    return CodexClientService()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, store: DatasetStore = Depends(get_store)) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "datasets": store.list_datasets(),
        },
    )


@app.post("/datasets")
async def create_dataset(
    name: str = Form(...),
    dataset_type: str = Form(...),
    trigger_token: str = Form(...),
    resolution_width: int = Form(1024),
    resolution_height: int = Form(1024),
    batch_size: int = Form(1),
    num_repeats: int = Form(10),
    min_bucket_reso: int = Form(512),
    max_bucket_reso: int = Form(1536),
    bucket_reso_steps: int = Form(16),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    settings = store.create_dataset(
        name=name,
        dataset_type=dataset_type,
        trigger_token=trigger_token,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        batch_size=batch_size,
        num_repeats=num_repeats,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
    )
    return RedirectResponse(f"/datasets/{settings.slug}", status_code=303)


@app.get("/datasets/{slug}", response_class=HTMLResponse)
async def dataset_detail(
    slug: str,
    request: Request,
    message: str = "",
    error: str = "",
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
) -> HTMLResponse:
    try:
        settings = store.load_settings(slug)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    return templates.TemplateResponse(
        request,
        "dataset.html",
        {
            "settings": settings,
            "images": store.list_images(slug),
            "references": store.list_references(slug),
            "jobs": [job.to_json() for job in job_store.list_jobs(slug)],
            "training_ready": get_training_service().sd_scripts_ready(),
            "musubi_ready": get_training_service().musubi_tuner_ready(),
            "qwen_dataset_type": DATASET_TYPE_QWEN_IMAGE_EDIT_2511,
            "message": message,
            "error": error,
        },
    )


@app.get("/datasets/{slug}/curator", response_class=HTMLResponse)
async def dataset_curator(
    slug: str,
    request: Request,
    message: str = "",
    error: str = "",
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
) -> HTMLResponse:
    try:
        settings = store.load_settings(slug)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    return templates.TemplateResponse(
        request,
        "curator.html",
        {
            "settings": settings,
            "raw_images": store.list_raw_images(slug),
            "references": store.list_references(slug),
            "candidates": store.list_curator_candidates(slug),
            "jobs": [job.to_json() for job in job_store.list_jobs(slug)],
            "qwen_dataset_type": DATASET_TYPE_QWEN_IMAGE_EDIT_2511,
            "message": message,
            "error": error,
        },
    )


@app.post("/datasets/{slug}/curator/raw")
async def upload_curator_raw(
    slug: str,
    files: list[UploadFile] = File(...),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    saved = await store.save_raw_images(slug, files)
    return RedirectResponse(
        f"/datasets/{slug}/curator?message=Uploaded {len(saved)} raw images",
        status_code=303,
    )


@app.post("/datasets/{slug}/curator/references")
async def upload_curator_references(
    slug: str,
    files: list[UploadFile] = File(...),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    saved = await store.save_reference_images(slug, files)
    return RedirectResponse(
        f"/datasets/{slug}/curator?message=Uploaded {len(saved)} reference images",
        status_code=303,
    )


@app.post("/datasets/{slug}/curator/process")
async def process_curator_images(
    slug: str,
    instruction: str = Form(...),
    selection_mode: str = Form("selected"),
    reference_mode: str = Form("selected"),
    raw_names: list[str] | None = Form(None),
    reference_names: list[str] | None = Form(None),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    try:
        store.load_settings(slug)
        selected_raw = [path.name for path in store.list_raw_images(slug)] if selection_mode == "all" else list(raw_names or [])
        if reference_mode == "all":
            selected_references = [path.name for path in store.list_references(slug)]
        elif reference_mode == "none":
            selected_references = []
        else:
            selected_references = list(reference_names or [])
        if not selected_raw:
            raise ValueError("Select at least one raw image")
        job = job_store.create_job(
            slug,
            "curate_raw_images",
            {
                "raw_names": selected_raw,
                "reference_names": selected_references,
                "instruction": instruction,
                "selection_mode": selection_mode,
            },
        )
        runner.enqueue(job)
        return RedirectResponse(
            f"/datasets/{slug}/curator?message=Curator job queued",
            status_code=303,
        )
    except (CodexUnavailableError, FileNotFoundError, DatasetNotFoundError, ValueError) as exc:
        return RedirectResponse(f"/datasets/{slug}/curator?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/curator/candidates/{candidate_id}/approve")
async def approve_curator_candidate(
    slug: str,
    candidate_id: str,
    instruction: str = Form(""),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    try:
        store.approve_curator_candidate(slug, candidate_id, instruction)
        return RedirectResponse(
            f"/datasets/{slug}/curator?message=Candidate added to dataset",
            status_code=303,
        )
    except (DatasetNotFoundError, FileNotFoundError, ValueError) as exc:
        return RedirectResponse(f"/datasets/{slug}/curator?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/curator/candidates/approve-selected")
async def approve_selected_curator_candidates(
    slug: str,
    candidate_ids: list[str] | None = Form(None),
    instruction: str = Form(""),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    try:
        selected = list(candidate_ids or [])
        if not selected:
            raise ValueError("Select at least one candidate")
        approved = store.approve_curator_candidates(slug, selected, instruction)
        return RedirectResponse(
            f"/datasets/{slug}/curator?message=Added {len(approved)} candidates to dataset",
            status_code=303,
        )
    except (DatasetNotFoundError, FileNotFoundError, ValueError) as exc:
        return RedirectResponse(f"/datasets/{slug}/curator?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/settings")
async def update_settings(
    slug: str,
    trigger_token: str = Form(...),
    resolution_width: int = Form(...),
    resolution_height: int = Form(...),
    batch_size: int = Form(...),
    num_repeats: int = Form(...),
    min_bucket_reso: int = Form(...),
    max_bucket_reso: int = Form(...),
    bucket_reso_steps: int = Form(...),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    store.update_settings(
        slug,
        trigger_token=trigger_token,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        batch_size=batch_size,
        num_repeats=num_repeats,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
    )
    return RedirectResponse(f"/datasets/{slug}?message=Settings updated", status_code=303)


@app.post("/datasets/{slug}/upload")
async def upload_images(
    slug: str,
    files: list[UploadFile] = File(...),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    saved = await store.save_uploaded_images(slug, files)
    return RedirectResponse(
        f"/datasets/{slug}?message=Uploaded {len(saved)} images",
        status_code=303,
    )


@app.post("/datasets/{slug}/upload-edit-pairs")
async def upload_edit_pairs(
    slug: str,
    controls: list[UploadFile] = File(...),
    targets: list[UploadFile] = File(...),
    instructions: str = Form(""),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    saved = await store.save_uploaded_edit_pairs(slug, controls, targets, instructions)
    return RedirectResponse(
        f"/datasets/{slug}?message=Uploaded {len(saved)} edit pairs",
        status_code=303,
    )


@app.post("/datasets/{slug}/references")
async def upload_references(
    slug: str,
    files: list[UploadFile] = File(...),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    saved = await store.save_reference_images(slug, files)
    return RedirectResponse(
        f"/datasets/{slug}?message=Uploaded {len(saved)} references",
        status_code=303,
    )


@app.post("/datasets/{slug}/generate")
async def generate_images(
    slug: str,
    prompt: str = Form(...),
    count: int = Form(1),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    try:
        store.load_settings(slug)
        job = job_store.create_job(slug, "generate_images", {"prompt": prompt, "count": count})
        runner.enqueue(job)
        return RedirectResponse(
            f"/datasets/{slug}?message=Codex generation job queued",
            status_code=303,
        )
    except (CodexUnavailableError, FileNotFoundError, DatasetNotFoundError) as exc:
        return RedirectResponse(f"/datasets/{slug}?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/generate-edit-pairs")
async def generate_edit_pairs(
    slug: str,
    prompt: str = Form(""),
    count: int = Form(1),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    try:
        settings = store.load_settings(slug)
        if settings.dataset_type != DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
            raise ValueError("This dataset is not Qwen Image Edit-2511")
        prompt = prompt.strip() or settings.trigger_token or "Create varied Qwen Image Edit-2511 training pairs."
        job = job_store.create_job(slug, "generate_edit_pairs", {"prompt": prompt, "count": count})
        runner.enqueue(job)
        return RedirectResponse(
            f"/datasets/{slug}?message=Codex edit pair generation job queued",
            status_code=303,
        )
    except (CodexUnavailableError, FileNotFoundError, DatasetNotFoundError, ValueError) as exc:
        return RedirectResponse(f"/datasets/{slug}?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/import-recent-codex")
async def import_recent_codex_images(
    slug: str,
    minutes: int = Form(60),
    limit: int = Form(12),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    store.load_settings(slug)
    job = job_store.create_job(slug, "import_recent", {"minutes": minutes, "limit": limit})
    runner.enqueue(job)
    return RedirectResponse(
        f"/datasets/{slug}?message=Codex import job queued",
        status_code=303,
    )


@app.post("/datasets/{slug}/images/{stem}/caption")
async def update_caption(
    slug: str,
    stem: str,
    caption: str = Form(...),
    description: str = Form(""),
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    store.update_caption(slug, stem, caption, description)
    return RedirectResponse(f"/datasets/{slug}?message=Caption saved", status_code=303)


@app.post("/datasets/{slug}/images/{stem}/label")
async def label_image(
    slug: str,
    stem: str,
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    try:
        store.load_settings(slug)
        if stem not in {record.stem for record in store.list_images(slug)}:
            raise FileNotFoundError(stem)
        job = job_store.create_job(slug, "label_image", {"stem": stem})
        runner.enqueue(job)
        return RedirectResponse(f"/datasets/{slug}?message=Codex label job queued", status_code=303)
    except (CodexUnavailableError, StopIteration, FileNotFoundError) as exc:
        return RedirectResponse(f"/datasets/{slug}?error={str(exc)}", status_code=303)


@app.post("/datasets/{slug}/label-batch")
async def label_batch(
    slug: str,
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    runner: CodexJobRunner = Depends(get_runner),
) -> RedirectResponse:
    settings = store.load_settings(slug)
    stems = [record.stem for record in store.list_images(slug) if record.caption.strip() == settings.trigger_token]
    job = job_store.create_job(slug, "label_batch", {"stems": stems})
    runner.enqueue(job)
    return RedirectResponse(f"/datasets/{slug}?message=Codex batch label job queued", status_code=303)


@app.post("/datasets/{slug}/setup-sd-scripts")
async def setup_sd_scripts(
    slug: str,
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    training_runner: TrainingJobRunner = Depends(get_training_runner),
) -> RedirectResponse:
    store.load_settings(slug)
    job = job_store.create_job(slug, "setup_sd_scripts", {})
    training_runner.enqueue(job)
    return RedirectResponse(f"/datasets/{slug}?message=sd-scripts setup job queued", status_code=303)


@app.post("/datasets/{slug}/setup-musubi-tuner")
async def setup_musubi_tuner(
    slug: str,
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    training_runner: TrainingJobRunner = Depends(get_training_runner),
) -> RedirectResponse:
    store.load_settings(slug)
    job = job_store.create_job(slug, "setup_musubi_tuner", {})
    training_runner.enqueue(job)
    return RedirectResponse(f"/datasets/{slug}?message=musubi-tuner setup job queued", status_code=303)


@app.post("/datasets/{slug}/train")
async def train_anima_lora(
    slug: str,
    pretrained_model_name_or_path: str = Form(...),
    qwen3: str = Form(...),
    vae: str = Form(...),
    llm_adapter_path: str = Form(""),
    t5_tokenizer_path: str = Form(""),
    output_name: str = Form(""),
    output_dir: str = Form(""),
    network_dim: int = Form(8),
    learning_rate: str = Form("1e-4"),
    optimizer_type: str = Form("AdamW8bit"),
    lr_scheduler: str = Form("constant"),
    timestep_sampling: str = Form("sigmoid"),
    discrete_flow_shift: str = Form("1.0"),
    max_train_epochs: int = Form(10),
    save_every_n_epochs: int = Form(1),
    mixed_precision: str = Form("bf16"),
    vae_chunk_size: int = Form(64),
    gradient_checkpointing: bool = Form(False),
    cache_latents: bool = Form(False),
    cache_text_encoder_outputs: bool = Form(False),
    vae_disable_cache: bool = Form(False),
    train_text_encoder: bool = Form(False),
    extra_args: str = Form(""),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    training_runner: TrainingJobRunner = Depends(get_training_runner),
) -> RedirectResponse:
    store.load_settings(slug)
    payload = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "qwen3": qwen3,
        "vae": vae,
        "llm_adapter_path": llm_adapter_path,
        "t5_tokenizer_path": t5_tokenizer_path,
        "output_name": output_name or f"{slug}_anima_lora",
        "output_dir": output_dir or str((Path("datasets") / slug / "outputs").resolve()),
        "network_dim": network_dim,
        "learning_rate": learning_rate,
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": discrete_flow_shift,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "mixed_precision": mixed_precision,
        "vae_chunk_size": vae_chunk_size,
        "gradient_checkpointing": gradient_checkpointing,
        "cache_latents": cache_latents,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "vae_disable_cache": vae_disable_cache,
        "train_text_encoder": train_text_encoder,
        "extra_args": extra_args,
    }
    job = job_store.create_job(slug, "train_anima_lora", payload)
    training_runner.enqueue(job)
    return RedirectResponse(f"/datasets/{slug}?message=Anima training job queued", status_code=303)


@app.post("/datasets/{slug}/train-qwen-edit")
async def train_qwen_edit_lora(
    slug: str,
    dit: str = Form(...),
    vae: str = Form(...),
    text_encoder: str = Form(...),
    output_name: str = Form(""),
    output_dir: str = Form(""),
    training_backend: str = Form("local"),
    network_dim: int = Form(8),
    learning_rate: str = Form("5e-5"),
    optimizer_type: str = Form("adamw8bit"),
    max_train_epochs: int = Form(16),
    save_every_n_epochs: int = Form(1),
    vram_preset: str = Form("fast_16gb"),
    blocks_to_swap: str = Form("36"),
    use_pinned_memory_for_block_swap: bool = Form(False),
    gradient_checkpointing: bool = Form(False),
    fp8_base: bool = Form(False),
    fp8_scaled: bool = Form(False),
    fp8_vl: bool = Form(False),
    modal_volume_name: str = Form("dada-qwen-edit"),
    modal_gpu: str = Form("L40S"),
    modal_timeout: int = Form(86400),
    modal_output_dir: str = Form(""),
    modal_bake_models: bool = Form(False),
    extra_args: str = Form(""),
    store: DatasetStore = Depends(get_store),
    job_store: JobStore = Depends(get_job_store),
    training_runner: TrainingJobRunner = Depends(get_training_runner),
) -> RedirectResponse:
    settings = store.load_settings(slug)
    if settings.dataset_type != DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
        return RedirectResponse(f"/datasets/{slug}?error=This dataset is not Qwen Image Edit-2511", status_code=303)
    payload = {
        "dit": dit,
        "vae": vae,
        "text_encoder": text_encoder,
        "output_name": output_name or f"{slug}_qwen_edit_lora",
        "output_dir": output_dir or str((Path("datasets") / slug / "outputs").resolve()),
        "training_backend": training_backend,
        "network_dim": network_dim,
        "learning_rate": learning_rate,
        "optimizer_type": optimizer_type,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "vram_preset": vram_preset,
        "blocks_to_swap": blocks_to_swap,
        "use_pinned_memory_for_block_swap": use_pinned_memory_for_block_swap,
        "gradient_checkpointing": gradient_checkpointing,
        "fp8_base": fp8_base,
        "fp8_scaled": fp8_scaled,
        "fp8_vl": fp8_vl,
        "modal_volume_name": modal_volume_name,
        "modal_gpu": modal_gpu,
        "modal_timeout": modal_timeout,
        "modal_output_dir": modal_output_dir or f"/data/outputs/{slug}",
        "modal_bake_models": modal_bake_models,
        "resolution_width": settings.resolution_width,
        "resolution_height": settings.resolution_height,
        "batch_size": settings.batch_size,
        "num_repeats": settings.num_repeats,
        "extra_args": extra_args,
    }
    job = job_store.create_job(slug, "train_qwen_edit_lora", payload)
    training_runner.enqueue(job)
    return RedirectResponse(f"/datasets/{slug}?message=Qwen Edit training job queued", status_code=303)


@app.get("/datasets/{slug}/jobs")
async def list_jobs(
    slug: str,
    job_store: JobStore = Depends(get_job_store),
) -> JSONResponse:
    jobs = [job.to_json() for job in job_store.list_jobs(slug)]
    return JSONResponse({"jobs": jobs})


@app.get("/datasets/{slug}/jobs/{job_id}")
async def get_job(
    slug: str,
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> JSONResponse:
    try:
        job = job_store.get_job(slug, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JSONResponse({"job": job.to_json(), "log": job_store.read_log(slug, job_id)})


@app.post("/datasets/{slug}/jobs/{job_id}/cancel")
async def cancel_job(
    slug: str,
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    training_runner: TrainingJobRunner = Depends(get_training_runner),
) -> RedirectResponse:
    job = job_store.get_job(slug, job_id)
    if job.type not in {"train_anima_lora", "train_qwen_edit_lora", "setup_musubi_tuner", "setup_sd_scripts"}:
        return RedirectResponse(f"/datasets/{slug}?error=Only training jobs can be cancelled", status_code=303)
    await training_runner.cancel(slug, job_id)
    return RedirectResponse(f"/datasets/{slug}?message=Training job cancelled", status_code=303)


@app.post("/datasets/{slug}/images/{stem}/delete")
async def delete_image(
    slug: str,
    stem: str,
    store: DatasetStore = Depends(get_store),
) -> RedirectResponse:
    store.delete_image(slug, stem)
    return RedirectResponse(f"/datasets/{slug}?message=Image deleted", status_code=303)


@app.get("/datasets/{slug}/dataset.toml")
async def download_toml(slug: str, store: DatasetStore = Depends(get_store)) -> FileResponse:
    settings = store.load_settings(slug)
    toml_path = store.write_dataset_toml(settings)
    return FileResponse(toml_path, filename="dataset.toml")


@app.get("/datasets/{slug}/export.zip")
async def download_zip(slug: str, store: DatasetStore = Depends(get_store)) -> FileResponse:
    settings = store.load_settings(slug)
    store.write_dataset_toml(settings)
    dataset_dir = store.dataset_dir(slug)
    with TemporaryDirectory() as temp_dir:
        archive_base = Path(temp_dir) / slug
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=dataset_dir)
        persistent = dataset_dir / f"{slug}.zip"
        shutil.copy2(archive_path, persistent)
    return FileResponse(persistent, filename=f"{slug}.zip")
