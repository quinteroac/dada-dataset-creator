import asyncio
import json
from types import SimpleNamespace

from app.codex_service import CodexEditPair
from app.codex_service import CodexClientService


def test_parse_label_inserts_trigger_first() -> None:
    service = CodexClientService()
    label = service._parse_label('{"tags":["1girl","blue hair"],"description":"A portrait."}', "tok")

    assert label.tags == ["tok", "1girl", "blue hair"]
    assert label.description == "A portrait."


def test_parse_label_extracts_json_from_text() -> None:
    service = CodexClientService()
    label = service._parse_label('Here: {"tags":["tok","solo"],"description":"Solo."}', "tok")

    assert label.tags == ["tok", "solo"]


def test_stream_event_mapping_for_agent_delta() -> None:
    service = CodexClientService()
    event = SimpleNamespace(
        method="item/agentMessage/delta",
        payload=SimpleNamespace(delta="hello"),
    )

    assert service._line_from_stream_event(event) == "hello"


def test_stream_event_mapping_for_completion() -> None:
    service = CodexClientService()
    event = SimpleNamespace(
        method="turn/completed",
        payload=SimpleNamespace(turn=SimpleNamespace(status="completed", error=None)),
    )

    assert "completed" in service._line_from_stream_event(event)


def test_generate_images_can_be_mocked(tmp_path) -> None:
    class FakeCodex(CodexClientService):
        async def generate_images(self, prompt, count, reference_paths, dataset_slug):
            image = tmp_path / "generated.png"
            image.write_bytes(b"fake")
            return [image]

    result = asyncio.run(FakeCodex().generate_images("prompt", 1, [], "slug"))

    assert result[0].name == "generated.png"


def test_codex_request_stream_uses_structured_text_item_when_no_images() -> None:
    captured = {}

    class TextInput:
        def __init__(self, text):
            self.text = text

    class LocalImageInput:
        def __init__(self, path):
            self.path = path

    class Turn:
        def stream(self):
            return []

    class Thread:
        def turn(self, input_items):
            captured["input_items"] = input_items
            return Turn()

    class Codex:
        def __init__(self, config=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def thread_start(self):
            return Thread()

    sdk = SimpleNamespace(Codex=Codex, TextInput=TextInput, LocalImageInput=LocalImageInput, AppServerConfig=None)
    service = CodexClientService()

    service._run_codex_request_stream_sync(sdk, "hello", [], lambda event: None)

    assert isinstance(captured["input_items"], list)
    assert isinstance(captured["input_items"][0], TextInput)
    assert captured["input_items"][0].text == "hello"


def test_edit_pair_prompt_requires_exact_control_edit() -> None:
    service = CodexClientService()

    prompt = service._edit_pair_generation_prompt("make the palette blue", 1, [], "slug")

    assert "target MUST be produced by editing its exact control image" in prompt
    assert "Do not create the target as a separate text-to-image generation" in prompt
    assert "target_created_by_editing_control" in prompt
    assert "return [] instead of inventing approximate pairs" in prompt


def test_parse_edit_pairs_rejects_unconfirmed_pairs(tmp_path) -> None:
    service = CodexClientService()
    control = tmp_path / "control.png"
    target = tmp_path / "target.png"
    control.write_bytes(b"control")
    target.write_bytes(b"target")

    raw = json.dumps([{"control_path": str(control), "target_path": str(target), "instruction": "make it blue"}])

    assert service._parse_edit_pairs(raw, [control, target], "fallback") == []


def test_parse_edit_pairs_accepts_confirmed_control_edit(tmp_path) -> None:
    service = CodexClientService()
    control = tmp_path / "control.png"
    target = tmp_path / "target.png"
    control.write_bytes(b"control")
    target.write_bytes(b"target")

    raw = json.dumps(
        [
            {
                "control_path": str(control),
                "target_path": str(target),
                "instruction": "make it blue",
                "target_created_by_editing_control": True,
            }
        ]
    )

    pairs = service._parse_edit_pairs(raw, [control, target], "fallback")

    assert pairs == [CodexEditPair(control, target, "make it blue")]
