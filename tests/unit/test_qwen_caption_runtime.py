from types import SimpleNamespace

from tasks.caption_vqa.qwen25vl_coco_caption import _is_qwen25vl_7b


def _args(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(model_name=model_name)


def _model(name_or_path: str) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(_name_or_path=name_or_path))


def test_qwen25vl_7b_detection_uses_args_model_name() -> None:
    assert _is_qwen25vl_7b(_args("model_checkpoint/Qwen2.5-VL-7B-Instruct"), _model(""))
    assert not _is_qwen25vl_7b(_args("model_checkpoint/Qwen2.5-VL-3B-Instruct"), _model(""))


def test_qwen25vl_7b_detection_falls_back_to_config_name() -> None:
    assert _is_qwen25vl_7b(_args("custom/path"), _model("Qwen2.5-VL-7B-Instruct"))
    assert not _is_qwen25vl_7b(_args("custom/path"), _model("Qwen2.5-VL-3B-Instruct"))
