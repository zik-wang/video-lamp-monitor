"""Analyze a recording with a reviewed lamp profile or an explicit vision endpoint."""

import argparse
import json
import math
import shutil
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from colors import build_color_map
from core import AppearanceModel
from pipeline import annotated, calibrate, locate, save, scan, summarize
from vision import ModelClient


REPORT_TABLES = {
    "灯光时段": ("intervals", {
        "lamp": "灯位", "start": "开始秒", "end": "结束秒", "kind": "状态",
        "color": "颜色", "duration": "持续秒", "on_seconds": "实亮秒",
        "off_seconds": "实暗秒", "end_reason": "区间结束依据",
    }),
    "颜色变化": ("color_intervals", {
        "lamp": "灯位", "start": "开始秒", "end": "结束秒", "label": "颜色或状态",
    }),
    "持续提醒": ("alerts", {
        "lamp": "灯位", "condition": "规则", "start": "开始秒",
        "fired_at": "达到门槛秒", "end": "结束秒", "evidence": "依据",
    }),
    "灯位汇总": ("summary", {
        "lamp": "灯位", "on_seconds": "亮灯秒", "off_seconds": "熄灭秒",
        "unknown_seconds": "无法判断秒", "blink_intervals": "闪烁段数", "duration": "总时长秒",
    }),
}


def export_workbook(report, path):
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, (key, columns) in REPORT_TABLES.items():
        sheet = workbook.create_sheet(title)
        sheet.append(list(columns.values()))
        for row in report[key]:
            values = [row.get(field) for field in columns]
            sheet.append(values)
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
                elif isinstance(cell.value, float):
                    cell.number_format = "0.000"
        for cell in sheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="263F57")
        for index in range(1, len(columns) + 1):
            sheet.column_dimensions[get_column_letter(index)].width = 24
        sheet.freeze_panes = "B2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def read_reference(video):
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not ok or not math.isfinite(fps) or fps <= 0 or frames < 1:
            raise ValueError("无法解码视频或读取帧率和帧数。")
        return frame
    finally:
        capture.release()


def validate_inventory(inventory, frame):
    height, width = frame.shape[:2]
    identifiers = set()
    if not inventory:
        raise ValueError("没有可分析的灯位。")
    for lamp in inventory:
        name = lamp["id"]
        if not name or name in identifiers:
            raise ValueError("灯位编号必须非空且唯一。")
        identifiers.add(name)
        x1, y1, x2, y2 = lamp["roi"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{name} 的灯框超出参考图。")


def analyze(args, notify=lambda *message: None):
    video = args.video.resolve(strict=True)
    output = args.out.resolve()
    if (not math.isfinite(args.alert_seconds) or args.alert_seconds <= 0):
        raise ValueError("提醒门槛须为正数。")
    if args.video_start:
        datetime.fromisoformat(args.video_start)
    if args.profile is None and (not args.endpoint or not args.model):
        raise ValueError("请提供校准目录，或显式指定模型地址和模型名。")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须为空。")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 512 * 1024 * 1024:
        raise RuntimeError("输出盘剩余空间不足 512 MiB。")

    cancel = threading.Event()
    first_frame = read_reference(video)
    model_client = None
    if args.profile is None:
        profile = output / "profile"
        profile.mkdir()
        reference = first_frame
        cv2.imencode(".png", reference)[1].tofile(str(profile / "reference.png"))
        model_client = ModelClient(args.endpoint, args.model, output, cancel)
        inventory = locate(reference, model_client, notify, cancel)
        save(profile / "inventory.json", {"lamps": inventory})
    else:
        profile = args.profile.resolve(strict=True)
        reference_file = profile / "reference.png"
        if not reference_file.exists():
            reference_file = profile / "reference.jpg"
        reference = cv2.imdecode(np.fromfile(reference_file, np.uint8), cv2.IMREAD_COLOR)
        if reference is None or reference.shape != first_frame.shape:
            raise ValueError("参考图无法读取，或与视频分辨率不同。")
        inventory = json.loads((profile / "inventory.json").read_text(encoding="utf-8-sig"))["lamps"]

    validate_inventory(inventory, reference)
    annotated(reference, inventory, output / "lamp-inventory.jpg")
    data = scan(video, reference, inventory, cancel, notify)
    if model_client is not None:
        calibrate(data, inventory, profile, model_client, notify, cancel, video, reference)

    model = AppearanceModel(profile)
    colors, evidence = build_color_map(model, profile, args.position_rule)
    save(output / "color-calibration.json", evidence)
    settings = {"alert_seconds": args.alert_seconds, "video_start": args.video_start}
    report = summarize(data, model, colors, settings, cancel)
    report["video_name"] = video.name
    report["model_calls"] = model_client.calls if model_client is not None else 0
    report["registration_failures"] = data["registration_failures"]
    save(output / "report.json", report)
    export_workbook(report, output / "report.xlsx")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--endpoint", help="显式启用可接收画面裁块的视觉模型地址")
    parser.add_argument("--model", help="支持图像输入的模型 ID")
    parser.add_argument("--position-rule", action="store_true", help="启用已确认的下层绿、上层红黄布局")
    parser.add_argument("--alert-seconds", type=float, default=60)
    parser.add_argument("--video-start", default="", help="实际录像起点，ISO 8601，可选")
    args = parser.parse_args()
    report = analyze(args, lambda stage, message, _: print(f"{stage} · {message}", flush=True))
    print(f"{report['lamps']} 灯位，{report['frames']} 帧，结果目录 {args.out}")


if __name__ == "__main__":
    main()
