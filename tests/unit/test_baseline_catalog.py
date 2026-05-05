from attribution_research.baselines import baseline_names, baseline_spec, group_baselines
from attribution_research.registry import task_algorithm_choices


def test_catalog_runnable_baselines_match_current_task_choices():
    assert baseline_names(task="classification", runnable_only=True) == task_algorithm_choices(
        "classification"
    )
    assert baseline_names(task="detection", runnable_only=True) == task_algorithm_choices(
        "detection"
    )
    assert baseline_names(task="caption_vqa", runnable_only=True) == task_algorithm_choices(
        "caption_vqa"
    )


def test_catalog_resolves_upstream_aliases_to_canonical_names():
    assert baseline_spec("Grad-ECLIP").name == "grad_eclip"
    assert baseline_spec("IGOS++").name == "igos_pp"
    assert baseline_spec("SSGrad-CAM++").name == "ssgrad_cam_pp"
    assert baseline_spec("LLaVA-CAM").name == "llavacam"


def test_catalog_grouping_keeps_native_and_catalog_distinctions():
    grouped = group_baselines(group_by="task")
    classification = {spec.name: spec.support for spec in grouped["classification"]}
    detection = {spec.name: spec.support for spec in grouped["detection"]}
    caption = {spec.name: spec.support for spec in grouped["caption_vqa"]}

    assert classification["grad_eclip"] == "native"
    assert classification["ig2"] == "native"
    assert classification["igos_pp"] == "native"

    assert detection["gradcam"] == "native"
    assert detection["odam"] == "native"
    assert detection["ssgrad_cam_pp"] == "native"

    assert caption["igos_pp"] == "native"
    assert caption["llavacam"] == "native"
    assert caption["tam"] == "catalog"
    assert {spec.support for specs in grouped.values() for spec in specs} == {"native", "catalog"}


def test_task_filtered_task_grouping_only_emits_requested_task():
    grouped = group_baselines(group_by="task", task="classification")
    assert tuple(grouped.keys()) == ("classification",)
