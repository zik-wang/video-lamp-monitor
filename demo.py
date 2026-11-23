"""Generate a synthetic recording, run the monitor, and check the results."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from openpyxl import load_workbook

from core import extract
from monitor import analyze
from pipeline import save


FPS = 20
SIZE = (480, 320)
INVENTORY = [
    {"id": "L01", "roi": [115, 130, 145, 170], "center": [130, 150]},
    {"id": "L02", "roi": [325, 160, 355, 200], "center": [340, 180]},
]
LIGHT_COLORS = {
    "off": (24, 26, 28),
    "red": (35, 45, 248),
    "green": (45, 242, 30),
    "yellow": (25, 224, 252),
}


def lamp_states(seconds):
    first = "off"
    second = "off"
    if 2 <= seconds < 5:
        first = "red"
    elif 5 <= seconds < 8:
        first = "green"
    if 1 <= seconds < 4:
        second = "yellow"
    elif 5 <= seconds < 5.6 or 6.2 <= seconds < 6.8:
        second = "green"
    return first, second


def make_video(path):
    width, height = SIZE
    rng = np.random.default_rng(81)
    background = np.full((height, width, 3), 40, np.uint8)
    for x, y, gray in rng.integers([6, 70, 65], [width - 6, height - 6, 225], size=(450, 3)):
        cv2.circle(background, (int(x), int(y)), 3, (int(gray),) * 3, -1)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic MJPG video.")
    try:
        for frame_index in range(12 * FPS):
            seconds = frame_index / FPS
            frame = background.copy()
            for lamp, state in zip(INVENTORY, lamp_states(seconds)):
                x1, y1, x2, y2 = lamp["roi"]
                cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (90, 94, 98), -1)
                cv2.ellipse(frame, tuple(lamp["center"]), (11, 15), 0, 0, 360, LIGHT_COLORS[state], -1)
            if 7 <= seconds < 9:
                transform = np.float32([[1, 0, 12], [0, 1, -8]])
                frame = cv2.warpAffine(frame, transform, SIZE)
            if 9 <= seconds < 10:
                frame[:] = 0
            writer.write(frame)
    finally:
        writer.release()


def create_profile(video, profile):
    profile.mkdir()
    save(profile / "inventory.json", {"lamps": INVENTORY})
    capture = cv2.VideoCapture(str(video))
    centers, guards, labels, exemplars = [], [], [], []
    overrides = {}
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    samples = [
        [(.5, "off"), (2.5, "red"), (5.4, "green")],
        [(.5, "off"), (2.5, "yellow"), (5.4, "green")],
    ]
    try:
        ok, reference = capture.read()
        if not ok:
            raise RuntimeError("Synthetic video could not be decoded.")
        cv2.imencode(".png", reference)[1].tofile(str(profile / "reference.png"))
        for index, lamp in enumerate(INVENTORY):
            lamp_centers, lamp_guards = [], []
            for cluster, (seconds, color) in enumerate(samples[index]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * FPS))
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode calibration frame at {seconds}s.")
                core, guard, _ = extract(frame, identity, [lamp])
                lamp_centers.append(core[0].reshape(-1) / 255)
                lamp_guards.append(guard[0])
                key = f"{lamp['id']}_{cluster}"
                labels.append({
                    "key": key, "lit": "off" if color == "off" else "on",
                    "confidence": "high", "visible": True,
                })
                exemplars.append({"key": key, "crop": core[0].tolist()})
                overrides[key] = color
            centers.append(lamp_centers)
            guards.append(lamp_guards)
    finally:
        capture.release()
    np.savez_compressed(
        profile / "appearance-model.npz",
        centers=np.asarray(centers, np.float32), radii=np.full((2, 3), .025),
        guards=np.asarray(guards, np.uint8),
    )
    save(profile / "appearance-labels.json", labels)
    save(profile / "appearance-labels-recheck.json", [])
    save(profile / "exemplars.json", exemplars)
    save(profile / "profile.json", {
        "color_overrides": overrides, "override_basis": "由合成视频的已知灯色校准",
    })


def verify(report, workbook_path):
    assert report["lamps"] == 2 and report["frames"] == 240 and report["model_calls"] == 0
    for row in report["summary"]:
        assert abs(row["on_seconds"] + row["off_seconds"] + row["unknown_seconds"] - 12) < 1e-8
        assert abs(row["unknown_seconds"] - 1) < .15
    color_spans = {(row["lamp"], row["state"]) for row in report["color_intervals"]}
    assert {("L01", "red"), ("L01", "green"), ("L02", "yellow"), ("L02", "green")} <= color_spans
    assert any(row["lamp"] == "L02" and row["type"] == "blink" for row in report["intervals"])
    assert any(row["lamp"] == "L02" and "黄色" in row["condition"] for row in report["alerts"])
    book = load_workbook(workbook_path, read_only=True)
    try:
        assert book.sheetnames == ["灯光时段", "颜色变化", "持续提醒", "灯位汇总"]
        assert len(list(book["灯位汇总"].values)) == 3
    finally:
        book.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    video, profile = output / "synthetic.avi", output / "profile"
    make_video(video)
    create_profile(video, profile)
    results = output / "results"
    settings = argparse.Namespace(
        video=video, out=results, profile=profile, endpoint=None, model=None,
        position_rule=False, alert_seconds=2.0, video_start="",
    )
    report = analyze(settings)
    verify(report, results / "report.xlsx")
    print(f"PASS · 2 lamps, 240 frames, color and blink intervals, unknown visibility, XLSX · {results}")


if __name__ == "__main__":
    main()
