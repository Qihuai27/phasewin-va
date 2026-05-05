from attribution_research.registry import (
    algorithm_family,
    task_algorithm_choices,
    task_supports_algorithm,
    validate_task_algorithm,
)


def test_algorithm_families_match_research_structure():
    assert algorithm_family("gradient") == "gradient"
    assert algorithm_family("ig2") == "gradient"
    assert algorithm_family("igos_pp") == "gradient"
    assert algorithm_family("gradcam") == "gradient"
    assert algorithm_family("odam") == "gradient"
    assert algorithm_family("ssgrad_cam_pp") == "gradient"
    assert algorithm_family("greedy") == "search"
    assert algorithm_family("phasewin") == "search"
    assert algorithm_family("drise") == "search"
    assert algorithm_family("dhsic") == "search"


def test_task_algorithm_choices_are_explicit():
    assert task_algorithm_choices("classification") == (
        "greedy",
        "phasewin",
        "drise",
        "dhsic",
        "gradient",
        "grad_eclip",
        "ig2",
        "igos_pp",
    )
    assert task_algorithm_choices("detection") == (
        "greedy",
        "phasewin",
        "drise",
        "gradient",
        "gradcam",
        "odam",
        "ssgrad_cam_pp",
    )
    assert task_algorithm_choices("caption_vqa") == (
        "greedy",
        "phasewin",
        "drise",
        "gradient",
        "llavacam",
        "igos_pp",
    )


def test_task_support_matrix_rejects_invalid_pairs():
    assert task_supports_algorithm("classification", "igos_pp")
    assert task_supports_algorithm("detection", "odam")
    assert not task_supports_algorithm("detection", "dhsic")
    assert not task_supports_algorithm("caption_vqa", "dhsic")


def test_validate_task_algorithm_returns_normalized_name():
    assert validate_task_algorithm("caption-vqa", "PhaseWin") == "phasewin"
