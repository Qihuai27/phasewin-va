from types import SimpleNamespace

from attribution_research.runtime import build_run_tag


def _args(**kwargs):
    base = dict(
        algorithm="greedy",
        segmenter="superpixel",
        superpixel_algorithm="slico",
        division_number=64,
        patch_size=None,
        grid_rows=None,
        grid_cols=None,
        lambda1=1.0,
        lambda2=1.0,
        window_size=None,
        phasewin_window_frac=0.3,
        drise_n_masks=1000,
        dhsic_batch_size=32,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_build_run_tag_for_search_methods():
    assert build_run_tag(_args(algorithm="greedy")) == "greedy-slico-division-64-1.0-1.0"
    assert build_run_tag(_args(algorithm="phasewin")) == "phasewin-slico-division-64-1.0-1.0-window-pct-30"
    assert (
        build_run_tag(_args(algorithm="phasewin", window_size=12))
        == "phasewin-slico-division-64-1.0-1.0-window-12"
    )
    assert build_run_tag(_args(algorithm="drise")) == "drise-slico-division-64-1.0-1.0-nmasks-1000"


def test_build_run_tag_for_map_methods():
    assert build_run_tag(_args(algorithm="gradient")) == "gradient-slico-division-64-1.0-1.0"
    assert build_run_tag(_args(algorithm="dhsic")) == "dhsic-slico-division-64-batch-32-tf-cpu"


def test_build_run_tag_for_patch_segmenter():
    assert build_run_tag(_args(segmenter="patch", patch_size=16)) == "greedy-patch-size-16-1.0-1.0"
