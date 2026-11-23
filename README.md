# 多灯视频监测 · Video Lamp Monitor

从录制视频中跟踪多盏工业信号灯，输出每盏灯的亮灭、颜色变化、常亮和闪烁时段。结果包括灯位标注、可复核的 JSON，以及按灯位汇总的 Excel。

## Qwen 27B 与 OpenCV 如何分工

原型使用 Qwen 27B 做两类视觉判断：检查画面分块中的灯位候选，以及给每盏灯少量代表性外观样本标注亮、暗和不可判定。定位输出须通过坐标和去重约束。看到高光、反射或彩色外壳时，先核对发光区域，保留无法确定的状态。

OpenCV 负责参考画面配准和逐帧测量。程序把帧级判定转换成稳定的亮灭段，识别闪烁，计算持续时间，并保留无法配准的区间。颜色由校准记录单独确认。颜色证据不足时保留待确认状态，黄色提醒只接受已确认的黄色。

配准和时段计算可以脱离模型复跑。有已审核的校准目录时，全程使用本地视频和校准文件。

## 复现场景

需要 Python 3.10+。在希望存放项目、依赖和结果的磁盘上运行。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B demo.py --out runs/demo
```

演示生成 12 秒、240 帧的合成视频，包含两盏灯、画面平移和一秒完全不可见区间，不需要模型、密钥或现场素材。程序会自动检查灯色、闪烁、黄色持续提醒、未知时段和 Excel 报告。

| 灯位 | 亮灯秒 | 熄灭秒 | 无法判断秒 | 闪烁段 |
| --- | ---: | ---: | ---: | ---: |
| L01 | 6.00 | 5.00 | 1.00 | 0 |
| L02 | 4.20 | 6.80 | 1.00 | 1 |

输出位于 `runs/demo/results/`。下面用生成的校准目录，直接调用视频监测入口：

```powershell
.\.venv\Scripts\python.exe -B monitor.py --video runs/demo/synthetic.avi --profile runs/demo/profile --out runs/replay --alert-seconds 2
```

## 分析新视频

有对应机位的校准时使用 `--profile`。校准目录包含参考图、灯位框、外观样本和颜色映射。首次在新机位运行时，可显式指定兼容 OpenAI Chat Completions 接口的视觉模型根地址和模型 ID：

```powershell
.\.venv\Scripts\python.exe -B monitor.py --video input.mp4 --out runs/new-camera --endpoint http://127.0.0.1:8000 --model <YOUR_VISION_MODEL_ID>
```

该模式会把局部画面发送到指定服务，保留模型响应供复核。检查 `lamp-inventory.jpg` 中的编号和框选结果，并用 `color-calibration.json` 核对颜色依据。需要补充已确认颜色时，可在校准目录的 `profile.json` 中填写 `color_overrides`，形如 `{"L01_1": "red", "L01_2": "green"}`，其中后缀对应外观样本编号。

本仓库是离线录像原型。当前按约一秒间隔更新配准，保留逐帧外观用于时段分析。新机位、持续遮挡、快速转动及不同灯具应分别验收。公开验证覆盖上述合成回放，模型辅助校准需要使用者提供兼容的视觉服务。
