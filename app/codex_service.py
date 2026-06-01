from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class CodexLabel:
    tags: list[str]
    description: str


@dataclass(slots=True)
class CodexEditPair:
    control_path: Path
    target_path: Path
    instruction: str


class CodexUnavailableError(RuntimeError):
    pass


class CodexClientService:
    """Small boundary around the experimental Codex SDK.

    The app remains useful without Codex: uploads, manual captions, and TOML export
    do not depend on this class succeeding.
    """

    def __init__(self, generated_images_dir: Path | None = None) -> None:
        self.generated_images_dir = generated_images_dir or Path.home() / ".codex" / "generated_images"

    async def generate_images(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
    ) -> list[Path]:
        sdk = self._load_sdk()
        before = self._snapshot_generated_images()
        request = self._generation_prompt(prompt, count, reference_paths, dataset_slug)
        await self._run_codex_request(sdk, request, reference_paths)
        return self._wait_for_new_images(before, count)

    async def generate_images_stream(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[Path]:
        sdk = self._load_sdk()
        before = self._snapshot_generated_images()
        request = self._generation_prompt(prompt, count, reference_paths, dataset_slug)
        await self._run_codex_request_stream(sdk, request, reference_paths, on_event)
        return self._wait_for_new_images(before, count)

    async def generate_edit_pairs_stream(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[CodexEditPair]:
        sdk = self._load_sdk()
        before = self._snapshot_generated_images()
        request = self._edit_pair_generation_prompt(prompt, count, reference_paths, dataset_slug)
        raw = await self._run_codex_request_stream(sdk, request, reference_paths, on_event)
        generated = self._wait_for_new_images(before, count * 2, timeout_seconds=45)
        return self._parse_edit_pairs(raw, generated, prompt)

    async def curate_raw_images_stream(
        self,
        instruction: str,
        raw_paths: list[Path],
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[Path]:
        sdk = self._load_sdk()
        before = self._snapshot_generated_images()
        request = self._curator_prompt(instruction, raw_paths, reference_paths, dataset_slug)
        await self._run_codex_request_stream(sdk, request, raw_paths + reference_paths, on_event)
        return self._wait_for_new_images(before, len(raw_paths), timeout_seconds=60)

    async def label_image(
        self,
        image_path: Path,
        trigger_token: str,
    ) -> CodexLabel:
        sdk = self._load_sdk()
        request = (
            "Etiqueta esta imagen para entrenar un LoRA anime de Anima. "
            "Devuelve solo JSON valido con las claves tags y description. "
            f"El primer tag debe ser exactamente {trigger_token!r}. "
            "tags debe ser una lista corta de tags estilo Danbooru/Booru: sujeto, estilo, ropa, pose, fondo, camara. "
            "description debe ser una frase natural en ingles."
        )
        raw = await self._run_codex_request(sdk, request, [image_path])
        return self._parse_label(raw, trigger_token)

    async def label_image_stream(
        self,
        image_path: Path,
        trigger_token: str,
        on_event: Callable[[str], None],
    ) -> CodexLabel:
        sdk = self._load_sdk()
        request = (
            "Etiqueta esta imagen para entrenar un LoRA anime de Anima. "
            "Devuelve solo JSON valido con las claves tags y description. "
            f"El primer tag debe ser exactamente {trigger_token!r}. "
            "tags debe ser una lista corta de tags estilo Danbooru/Booru: sujeto, estilo, ropa, pose, fondo, camara. "
            "description debe ser una frase natural en ingles."
        )
        raw = await self._run_codex_request_stream(sdk, request, [image_path], on_event)
        return self._parse_label(raw, trigger_token)

    async def label_edit_pair_stream(
        self,
        control_path: Path,
        target_path: Path,
        on_event: Callable[[str], None],
    ) -> str:
        sdk = self._load_sdk()
        request = (
            "Write one concise English image-edit instruction for training Qwen-Image-Edit-2511. "
            "The first image is the control/start image. The second image is the edited target image. "
            "Return only the instruction text, with no JSON and no commentary."
        )
        raw = await self._run_codex_request_stream(sdk, request, [control_path, target_path], on_event)
        return raw.strip().strip('"') or "Edit the control image to match the target image."

    def _load_sdk(self) -> Any:
        try:
            import codex_app_server as sdk  # type: ignore[import-not-found]
        except Exception as exc:
            raise CodexUnavailableError(
                "Codex SDK is not installed or importable. Install with: uv sync --group dev --group codex"
            ) from exc
        return sdk

    async def _run_codex_request(self, sdk: Any, prompt: str, image_paths: list[Path]) -> str:
        return await asyncio.to_thread(self._run_codex_request_sync, sdk, prompt, image_paths)

    async def _run_codex_request_stream(
        self,
        sdk: Any,
        prompt: str,
        image_paths: list[Path],
        on_event: Callable[[str], None],
    ) -> str:
        return await asyncio.to_thread(
            self._run_codex_request_stream_sync,
            sdk,
            prompt,
            image_paths,
            on_event,
        )

    def _run_codex_request_sync(self, sdk: Any, prompt: str, image_paths: list[Path]) -> str:
        codex_cls = getattr(sdk, "Codex", None)
        local_image_cls = getattr(sdk, "LocalImageInput", None)
        text_cls = getattr(sdk, "TextInput", None)
        config_cls = getattr(sdk, "AppServerConfig", None)
        if codex_cls is None or local_image_cls is None or text_cls is None:
            raise CodexUnavailableError("The installed Codex SDK does not expose Codex/TextInput/LocalImageInput.")

        input_items: list[Any] = [text_cls(prompt)]
        input_items.extend(local_image_cls(path=str(path)) for path in image_paths)

        codex_bin = shutil.which("codex")
        config = None
        if config_cls is not None and codex_bin:
            config = config_cls(codex_bin=codex_bin)

        try:
            with codex_cls(config=config) as codex:
                thread = codex.thread_start()
                result = thread.run(input_items)
        except Exception as exc:
            raise CodexUnavailableError(f"Codex SDK request failed: {exc}") from exc
        return self._stringify_result(result)

    def _run_codex_request_stream_sync(
        self,
        sdk: Any,
        prompt: str,
        image_paths: list[Path],
        on_event: Callable[[str], None],
    ) -> str:
        codex_cls = getattr(sdk, "Codex", None)
        local_image_cls = getattr(sdk, "LocalImageInput", None)
        text_cls = getattr(sdk, "TextInput", None)
        config_cls = getattr(sdk, "AppServerConfig", None)
        if codex_cls is None or local_image_cls is None or text_cls is None:
            raise CodexUnavailableError("The installed Codex SDK does not expose Codex/TextInput/LocalImageInput.")

        input_items: list[Any] = [text_cls(prompt)]
        input_items.extend(local_image_cls(path=str(path)) for path in image_paths)

        codex_bin = shutil.which("codex")
        config = config_cls(codex_bin=codex_bin) if config_cls is not None and codex_bin else None
        accumulated_agent_text: list[str] = []
        final_text = ""
        try:
            with codex_cls(config=config) as codex:
                thread = codex.thread_start()
                turn = thread.turn(input_items)
                on_event("[codex] invoked")
                for event in turn.stream():
                    line = self._line_from_stream_event(event)
                    if line:
                        on_event(line)
                    if event.method == "item/agentMessage/delta":
                        delta = getattr(event.payload, "delta", "")
                        accumulated_agent_text.append(str(delta))
                    if event.method == "turn/completed":
                        final_text = self._final_text_from_completed_event(event) or "".join(accumulated_agent_text)
                return final_text or "".join(accumulated_agent_text)
        except Exception as exc:
            raise CodexUnavailableError(f"Codex SDK request failed: {exc}") from exc

    def _generation_prompt(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
    ) -> str:
        refs = "\n".join(f"- {path}" for path in reference_paths) or "- none"
        return (
            f"Generate {count} anime training images for an Anima LoRA dataset named {dataset_slug}. "
            "Use image generation if available. Save the images as PNG files in the local Codex generated_images folder. "
            "Keep composition varied and suitable for training, not promotional text or collages.\n\n"
            f"Prompt base:\n{prompt}\n\nReference images:\n{refs}"
        )

    def _edit_pair_generation_prompt(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
    ) -> str:
        refs = "\n".join(f"- {path}" for path in reference_paths) or "- none"
        return (
            f"Generate {count} synthetic Qwen Image Edit-2511 training pairs for dataset {dataset_slug}. "
            "This is an image-edit dataset, so each target MUST be produced by editing its exact control image. "
            "Do not create the target as a separate text-to-image generation from the same description. "
            "Workflow for every pair: first create and save a control/start PNG; then use that saved control PNG "
            "as the image input/reference for an image edit operation; save the edited result as the target PNG. "
            "Only the requested edit may change. Preserve identity, subject count, pose, camera angle, composition, "
            "geometry, background layout, and all unrelated details from the control image. "
            "If exact image editing from the control is not available, return [] instead of inventing approximate pairs. "
            "Save files in the local Codex generated_images folder using names like "
            f"{dataset_slug}_001_control.png and {dataset_slug}_001_target.png. "
            "Return JSON only, as an array of objects with keys control_path, target_path, instruction, "
            "and target_created_by_editing_control. target_created_by_editing_control must be true only when the "
            "target was created by editing the exact control image file. instruction must be a concise English edit "
            "instruction suitable as the training caption.\n\n"
            f"Prompt base:\n{prompt}\n\nReference images:\n{refs}"
        )

    def _curator_prompt(
        self,
        instruction: str,
        raw_paths: list[Path],
        reference_paths: list[Path],
        dataset_slug: str,
    ) -> str:
        raw_refs = "\n".join(f"- {path}" for path in raw_paths) or "- none"
        style_refs = "\n".join(f"- {path}" for path in reference_paths) or "- none"
        return (
            f"Curate {len(raw_paths)} raw images for dataset {dataset_slug}. "
            "The raw images are the source images to transform. The reference images are only visual guidance "
            "for style, colors, lighting, composition, photographic look, texture, or finish. Do not treat "
            "reference images as target subjects unless the user explicitly asks for that. "
            "Create one curated output image for each raw image, preserving the main subject identity and useful "
            "training composition while applying the requested transformation. Save the curated outputs as PNG "
            "files in the local Codex generated_images folder. Do not modify files inside the dataset folder.\n\n"
            f"User instruction:\n{instruction}\n\n"
            f"Raw images to transform:\n{raw_refs}\n\n"
            f"Reference images for style/color/look only:\n{style_refs}"
        )

    def _parse_label(self, raw: str, trigger_token: str) -> CodexLabel:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise CodexUnavailableError("Codex did not return JSON labels.")
            data = json.loads(raw[start : end + 1])
        tags = [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()]
        if not tags or tags[0] != trigger_token:
            tags = [tag for tag in tags if tag != trigger_token]
            tags.insert(0, trigger_token)
        return CodexLabel(tags=tags, description=str(data.get("description", "")))

    def _parse_edit_pairs(self, raw: str, generated: list[Path], fallback_instruction: str) -> list[CodexEditPair]:
        path_by_name = {path.name: path for path in generated}
        pairs: list[CodexEditPair] = []
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            data = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else json.loads(raw)
        except json.JSONDecodeError:
            data = []

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                control = self._resolve_generated_path(str(item.get("control_path", "")), path_by_name)
                target = self._resolve_generated_path(str(item.get("target_path", "")), path_by_name)
                instruction = str(item.get("instruction", "")).strip() or fallback_instruction
                created_by_editing_control = item.get("target_created_by_editing_control") is True
                if control is not None and target is not None and created_by_editing_control:
                    pairs.append(CodexEditPair(control, target, instruction))

        return pairs

    def _resolve_generated_path(self, value: str, path_by_name: dict[str, Path]) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return path
        return path_by_name.get(path.name)

    def _line_from_stream_event(self, event: Any) -> str:
        method = str(getattr(event, "method", "event"))
        payload = getattr(event, "payload", None)
        delta = getattr(payload, "delta", None)
        if method == "item/agentMessage/delta" and delta:
            return str(delta)
        if method == "item/commandExecution/outputDelta" and delta:
            return f"[command] {delta}"
        if method == "item/fileChange/outputDelta" and delta:
            return f"[file] {delta}"
        if method == "turn/completed":
            turn = getattr(payload, "turn", None)
            status = getattr(turn, "status", "")
            error = getattr(turn, "error", None)
            message = getattr(error, "message", "") if error else ""
            return f"[codex] completed {status} {message}".strip()
        if method == "turn/started":
            return "[codex] turn started"
        return ""

    def _final_text_from_completed_event(self, event: Any) -> str:
        turn = getattr(getattr(event, "payload", None), "turn", None)
        items = getattr(turn, "items", []) or []
        messages = []
        for item_wrapper in items:
            item = getattr(item_wrapper, "root", item_wrapper)
            if getattr(item, "type", None) == "agentMessage":
                text = getattr(item, "text", "")
                if text:
                    messages.append(str(text))
        return messages[-1] if messages else ""

    def _snapshot_generated_images(self) -> set[Path]:
        if not self.generated_images_dir.exists():
            return set()
        return {
            path
            for path in self.generated_images_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }

    def recent_generated_images(self, minutes: int = 30) -> list[Path]:
        cutoff = time.time() - (minutes * 60)
        return sorted(
            [
                path
                for path in self._snapshot_generated_images()
                if path.stat().st_mtime >= cutoff
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def write_event(self, dataset_slug: str, event: str, detail: str = "") -> None:
        log_path = self.generated_images_dir.parent / "codex_events.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            log_path.read_text(encoding="utf-8") if log_path.exists() else "",
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as file:
            timestamp = datetime.now(timezone.utc).isoformat()
            file.write(f"{timestamp}\t{dataset_slug}\t{event}\t{detail}\n")

    def _wait_for_new_images(self, before: set[Path], count: int, timeout_seconds: int = 30) -> list[Path]:
        deadline = time.time() + timeout_seconds
        newest: list[Path] = []
        while time.time() < deadline:
            after = self._snapshot_generated_images()
            newest = sorted(after - before, key=lambda path: path.stat().st_mtime)
            if len(newest) >= count:
                return newest[:count]
            time.sleep(1)
        return newest[:count]

    def _stringify_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "output_text"):
            return str(result.output_text)
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)
