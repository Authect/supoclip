import subprocess

from src.media.reframing import (
    MAX_CROP_KEYFRAMES,
    _refine_scene_cut,
    build_crop_trajectory,
    build_smooth_pan_expression,
)

WIDTH, CROP_W = 1920, 606
SAMPLE_FPS = 3.0


def two_shot_track(cut_sample: int = 30, second_center: float | None = 1400.0):
    """A face at x=400 cutting to a face at x=1400 (or to no face)."""
    track = []
    for i in range(60):
        center = 400.0 if i < cut_sample else second_center
        track.append((i / SAMPLE_FPS, center, 0.01))
    return track


def x_at(keys, t):
    """Evaluate the piecewise-linear crop position the ffmpeg expression renders."""
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def test_crop_switches_on_scene_cut_instead_of_panning_through_it():
    cut = 9.93  # refined edit time, between the samples at 9.67s and 10.0s
    keys = build_crop_trajectory(two_shot_track(), WIDTH, CROP_W, scene_cuts=[cut])

    old_x, new_x = 400 - CROP_W // 2, 1400 - CROP_W // 2
    assert all(abs(x - old_x) <= 2 for t, x in keys if t < cut)
    assert all(abs(x - new_x) <= 2 for t, x in keys if t >= cut)
    assert abs(x_at(keys, cut - 0.02) - old_x) <= 2
    assert abs(x_at(keys, cut + 0.02) - new_x) <= 2


def test_without_scene_cuts_the_crop_eases_across_the_change():
    keys = build_crop_trajectory(two_shot_track(), WIDTH, CROP_W)

    old_x, new_x = 400 - CROP_W // 2, 1400 - CROP_W // 2
    assert any(old_x + 50 < x < new_x - 50 for _, x in keys)


def test_shot_without_a_face_keeps_the_previous_framing():
    keys = build_crop_trajectory(
        two_shot_track(second_center=None), WIDTH, CROP_W, scene_cuts=[10.0]
    )

    assert {x for _, x in keys} == {400 - CROP_W // 2}


def test_refine_scene_cut_finds_the_edit_frame(tmp_path):
    video = tmp_path / "cut.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=1.4",
            "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30:d=1.6",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )

    # The 3 fps analysis reports the cut at the first sample after it (1.667s).
    refined = _refine_scene_cut(video, 5 / SAMPLE_FPS, 1 / SAMPLE_FPS)

    assert abs(refined - 1.4) <= 1 / 30 + 1e-6


def test_rapid_cutting_stays_within_ffmpeg_expression_limits():
    # A cut every second between two framings: more edits than one expression holds.
    track = [
        (i / SAMPLE_FPS, 400.0 if int(i / SAMPLE_FPS) % 2 == 0 else 1400.0, 0.01)
        for i in range(135)
    ]
    cuts = [second - 0.05 for second in range(1, 45)]

    keys = build_crop_trajectory(track, WIDTH, CROP_W, scene_cuts=cuts)

    assert len(keys) <= MAX_CROP_KEYFRAMES
    assert any(
        t1 - t0 < 0.01 and abs(x1 - x0) > 900
        for (t0, x0), (t1, x1) in zip(keys, keys[1:])
    )
    expression = build_smooth_pan_expression(keys)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=0.2",
            "-vf", f"crop={CROP_W}:1080:x='{expression}':y=0",
            "-f", "null", "-",
        ],
        check=True,
    )
