# INC1000 + Brewster 快速反馈扫描

脚本：`brewster_feedback_scan.py`。这是独立脚本，不改变原采集 GUI，也不执行 NUC。

## 现场流程

1. 使用原来的 `Measurement_QT_GUI.py` 界面瞄准太阳、居中，点击完成校准，保存 pan offset，然后关闭界面以释放串口和相机。
2. 运行本脚本。默认从 **zenith 70°（仰角 20°）** 往太阳方向升高，每层 2°。
3. INC1000 控制仰角 → 停稳 → 四个偏振角采集 → 根据 roll 旋转 Q/U → 拟合 U/I 与图像水平视角 → 小幅 pan 试探 → 自动对中 → 下一层。
4. 每次 pan 后重新检查仰角。Q/I 变号记录候选区间，但继续扫描到太阳仰角；拟合无效或反馈不收敛则停止并保存原因。

## 启动

在仓库根目录，用原有相机采集环境运行。软件依赖沿用原项目：numpy、pyserial、zaber-motion、VmbPy；图形曲线需要 matplotlib，太阳校准界面需要原 Qt 环境。

先无硬件验证：

```powershell
py Instrument_Operation/brewster_feedback_scan.py --simulate --plot
```

打开原太阳校准界面，完成校准并关闭后自动继续扫描：

```powershell
py Instrument_Operation/brewster_feedback_scan.py --calibrate-sun --roll-sign 1 --plot --output D:/Data/Brewster
```

已有此次安装的太阳校准结果时直接扫描：

```powershell
py Instrument_Operation/brewster_feedback_scan.py --roll-sign 1 --plot --output D:/Data/Brewster
```

X 符号另一种选择：把 `--roll-sign 1` 改为 `--roll-sign -1`。程序不会自动把“拟合更好”当作符号已校准；应通过已知相机旋转或独立图像水平基准确认。

默认端口：INC1000 COM1/115200，Moog COM7，Zaber COM6，可用 `--inc-port`、`--inc-baud`、`--moog-port`、`--zaber-port` 修改。

## 姿态和太阳瞄准

- `altitude = -INC1000_Y + tilt_offset`，`zenith = 90 - altitude`。
- `roll = roll_sign * INC1000_X + roll_offset`。
- `--tilt-offset` 是光轴仰角安装修正；不直接套用原 GUI 的 Moog tilt offset。
- `--roll-offset` 包含固定的偏振参考轴偏差，单位度。
- 这是按当前安装约定实现的快速模型，没有做倾角仪两个轴到欧拉角的完整转换。
- 使用原 GUI 保存的 `~/.ultrasip_auto_scan_settings.json` 中的 pan_offset、latitude、longitude；也可用 `--settings` 指定文件或 `--pan-offset`、`--latitude`、`--longitude` 覆盖。搬动底座后应重新太阳瞄准。
- 初始 pan 保留原约定：太阳的 Moog/SunCalc 方位角减 pan_offset。后续通过图像追线，不把 Moog pan 当作精确绝对方位角。
- 太阳在起点仰角以下时直接停止。终点太阳仰角在每层重新计算，默认到太阳；`--sun-margin 2` 可在太阳下方 2°结束。
- 原太阳校准 GUI 的关闭行为包含回零，脚本沿用该界面的行为。本扫描结束或中断时停止并释放连接，不额外回零。

## 快速计算约定

偏振片角度顺序为 0/45/90/135°，每组固定曝光：

```
I = (I0 + I45 + I90 + I135) / 2
Q = I0 - I90
U = I45 - I135
Q' = Q*cos(2*roll) + U*sin(2*roll)
U' = -Q*sin(2*roll) + U*cos(2*roll)
```

默认在中心宽 800 像素、高 40 像素的水平带里，平均每列有效像素的 U'/I，拟合 `u = a*degree + b`，零点为 `-b/a`。图像角度正向为列号增大方向；默认比例 0.002°/pixel 沿用现有采集界面，可用 `--deg-per-pixel` 修改。`--center-x/--center-y` 指定光轴像素，默认几何中心。

pan 方向和比例通过 0.2°试探的实际反馈测定，不直接把图像视角当成 pan。默认每次执行预测修正的 70%，单次不超过 1°。步进分辨率按现有 Moog 0.1°命令处理。U 零点距中心不超过 0.08°视为对中，实际仰角允许偏差 0.15°；可调节对应 tolerance 参数。

曝光默认 1000 µs，可用 `--exposure-us` 修改，本版不自动曝光。遇到饱和、太暗、斜率不足、零点不在 ROI、拟合残差过大、姿态漂移或运动不收敛时停止。接近太阳时可能因饱和提前停止；检查日志后降低曝光重测。

`root_se_deg` 仅是列平均线性拟合的形式标准误差，没有包括 NUC、相关噪声、光轴安装误差或 roll 符号误差。候选区间不是已标定的中性点结果。本版不自动细扫候选区间，可减小 `--zenith-step` 后重扫。

## 输出

每次运行创建一个独立时间目录：

- `config.json`：使用的参数和太阳方位偏移。
- `acquisition_XXXX/raw_000.npy` 等：四张完整原始图，无压缩保存以减少计算。
- `partial.json`：已完成帧的时间及采集前后 INC1000 读数，即使中断也保留。
- `result.json`、`u_fit.npz`：姿态、线性拟合结果和曲线。
- `measurements.jsonl`：所有采集，包括 pan 试探。
- `aligned.jsonl`：每层成功对中后的结果。
- `candidates.json`：有 Q 变号时生成候选天顶角区间。
- `status.json`：完成、中断或失败及其原因。

这是独立的快速验证格式，并非原 H5 分析程序的直接输入。后续可对保存的原图离线做 NUC 和精确定位。2848×2848 的 uint16 四帧约 65 MB/组，采集速度也会受磁盘写入速度影响。

按 Ctrl+C 停止。不自动修改 INC1000 零点、波特率、地址或 Flash 设置。

## 无硬件测试

```powershell
py -m unittest discover -s Instrument_Operation/tests -p "test_brewster_feedback_scan.py" -v
```
