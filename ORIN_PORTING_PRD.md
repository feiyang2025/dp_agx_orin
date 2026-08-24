# DragonPilot AGX Orin 移植 PRD（TW-T906G）

> 版本：v1.0 | 日期：2026-08-24
> 源项目：https://github.com/Guiimartinho/dragonpilot-adapt-jetson （DragonPilot 0.10.3 → Jetson AGX Xavier 适配版）
> 目标平台：图为 TW-T906G 边缘计算平台（内嵌 NVIDIA Jetson AGX Orin 32GB）
> 文档性质：移植需求与实施指南，按阶段逐项执行

---

## 1. 背景与目标

### 1.1 项目现状
本项目是 DragonPilot（openpilot 社区 fork，0.10.3 基线）向 NVIDIA Jetson 平台的移植版本，
原适配目标为 **Jetson AGX Xavier**（JetPack 5.x / CUDA 11.4 / Python 3.11 / sm_72）。

核心特性：
- 推理后端：tinygrad CUDA graphs（FP16），vision/policy/dmonitoring 三模型
- 图像预处理：CUDA 零拷贝（`Tensor.from_blob`），无 GPU→CPU→GPU 回环
- 相机：MIPI CSI 优先，自动回退 USB webcam（OpenCV + PyAV 转 NV12）
- UI：raylib，逻辑分辨率 2160×1080，实测 Xavier 上 SCALE=0.889 → 窗口 1920×960 居中于 1080p 屏

### 1.2 移植目标
| 项 | 说明 |
|---|---|
| 硬件 | 图为 TW-T906G（AGX Orin 32GB） |
| 系统 | JetPack 6.2（Ubuntu 22.04 / CUDA 12.x）→ 后期验证 JetPack 7 |
| 相机一期 | USB×2：远焦(road) + 近焦(wide)，**不做 DMS** |
| 相机二期 | GMSL IMX390 ×N（板载 8 路 GMSL2 接口，厂商提供设备树+驱动+BSP） |
| 功能保留 | 车速自动切换远/近焦显示、模型双流输入、UI、panda 控制 |

### 1.3 目标硬件规格（TW-T906G 手册摘录）
| 项目 | 规格 | 与 Xavier 差异 |
|---|---|---|
| SoC 模块 | Jetson AGX Orin 32GB | Xavier → Orin |
| CPU | 8 核 Cortex-A78AE @2.2GHz | Xavier 8核 A76 @2.2GHz（同核数！mtune 需改） |
| GPU | 1792 核 Ampere + 56 Tensor Core (sm_87) | Xavier 512 核 Volta (sm_72) |
| AI 算力 | 200 TOPS | 32 TOPS |
| DLA | 2× NVDLA v2.0 @1.4GHz | Xavier 无独立 NVDLA v2 |
| 内存 | 32GB LPDDR5 256-bit | 同容量，带宽更高 |
| 系统 | 出厂 Ubuntu 20.04（=JetPack 5.x），需刷至 JP 6.2 | — |
| 相机接口 | 板载 8× GMSL2（解串器在板上）+ 4× USB3.0 Type-A | — |
| 显示 | HDMI 2.0 @4K60 仅输出 | — |
| 供电 | 9-36V JAE 12pin，典型 45W / 最大 75W | — |
| 其他 | 2×CAN、3×RS232、2×GPIO(1.8V/3.3V)、4G/5G/WiFi 选配 | — |

### 1.4 不需要改动的部分（已确认与板卡解耦）
- 平台检测机制：`/JETSON` 标记文件 + `jarch64` arch（`SConstruct:35-36`，`system/hardware/__init__.py:10`）
- 进程管理：`system/manager/process_config.py`（JETSON 分支逻辑不变）
- VisionIPC / cereal / msgq / panda：架构无关
- UI 渲染层：raylib，仅依赖 OpenGL
- CUDA 零拷贝桥：`msgq_repo/msgq/visionipc/visionbuf_jetson.cc` 的 `cudaHostRegister`
  方案为纯 CUDA API，架构无关

---

## 2. 一期任务：平台适配（Xavier → Orin）

### 2.1 构建系统 SConstruct

**文件**：`SConstruct`

| 行号 | 现状 | 改为 | 原因 |
|---|---|---|---|
| 120 | `-march=armv8.2-a+fp16+dotprod -mtune=cortex-a76` | `-march=armv8.2-a+fp16+dotprod -mtune=cortex-a78ae` | T906G CPU 为 A78AE；A76 tune 参数在 A78AE 上次优 |
| 110-119 | CPPPATH/LIBPATH 含 `/usr/local/cuda/include`、`/usr/local/cuda/lib64` | 不变 | JP6 下 CUDA 安装路径相同（CUDA 12.x） |

验收：
```bash
scons -j8 && echo OK   # 全量编译零错误
```

### 2.2 硬件抽象层 hardware.py

**文件**：`system/hardware/jetson/hardware.py`

#### 2.2.1 功耗计 INA3221（行 83-116, 217-245）
- Xavier：I2C 总线 `1-0040`（ina3221 / ina3221x）
- Orin：通常位于 `3-0040` 或 `c-0040`，驱动名多为 `ina3221x`
- **改法**：不要硬编码路径，改为扫描 `/sys/bus/i2c/drivers/ina3221*/` 下所有 hwmon 并累加；
  或启动时探测一次缓存路径。T906G 实际路径以板上实测为准：
```bash
ls /sys/bus/i2c/drivers/ina3221*/*/hwmon/*/ 2>/dev/null
cat /sys/bus/i2c/devices/*/name | grep -i ina
```

#### 2.2.2 电源模式 set_power_save（行 138-161）
- `nvpmodel -m` 档位号在 Orin 上含义不同：Orin MAXN 仍为 `-m 0`，但低功耗档编号需查：
```bash
sudo nvpmodel -q --verbose   # 列出所有可用 mode
```
- **CPU 禁用循环 `range(4,8)` 不用改**（Orin 32GB 也是 8 核）
- `jetson_clocks` 命令不变

#### 2.2.3 频率 watchdog（行 199-211）
- 行 208：阈值 `if freq < 2_000_000:` → 改为 `< 2_100_000`（Orin 最大 2.2GHz，
  若锁频后被降超过 100MHz 视为 throttle）。更稳做法：读取 `scaling_max_freq` 动态比较：
```python
max_f = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq").read())
if freq < max_f * 0.9: ...
```

#### 2.2.4 GPU 占用（行 163-168）
- `/sys/devices/gpu.0/load` 在 Orin 上仍存在（格式同为 /10 百分比），暂不改；
  上机后验证一次即可。

#### 2.2.5 温度区 get_thermal_config（行 124-130）
| 区 | Xavier | Orin (待实测) |
|---|---|---|
| cpu | CPU-therm | CPU-therm ✅ |
| gpu | GPU-therm | GPU-therm ✅ |
| memory | Tdiode_tegra | CV0-therm / TdiodeTEGRA（上机确认） |
| pmic | PMIC-Die | 可能不存在，可置空 |

实测命令：
```bash
for z in /sys/class/thermal/thermal_zone*; do echo "$z: $(cat $z/type)"; done
```

#### 2.2.6 芯片序列号 get_serial（行 37-45）
- `/sys/module/fuse_burn/...` 和 `tegra_fuse` 路径在 Orin 上可能不同，
  失败时回退值已是 `"jetson-unknown"`，不阻塞，可选优化。

### 2.3 环境校验脚本

**文件**：`scripts/jetson_validate_env.sh`

改动点：
1. CUDA 版本断言：11.4 → 允许 ≥12.0（JP6）/ 预留 13.x（JP7）判断分支
2. Python 版本断言：JP6 自带 python3.10（uv 管理的 venv 用 3.11/3.12 亦可）——
   按 `pyproject.toml` 要求检查
3. JetPack 版本探测：解析 `/etc/nv_tegra_release` 的 L4T 版本号
   （JP6.2 = L4T 36.4.x；JP5 = 35.x）
4. 新增检查项：`nvidia-smi`（JP6 有）、DLA 设备节点、GMSL V4L2 设备（二期用）

### 2.4 编码器（可选优化，不阻塞）

**文件**：`system/loggerd/encoderd.cc`（行 9 `__JETSON__` 分支）

现状若为软编/CUDA 编码：Orin 有 NVENC 硬编码器（H.265 2×4K60），后续可通过
V4L2 NVENC 接口或 ffmpeg h264_nvmpi 替换。一期先保持现状跑通。

### 2.5 模型缓存重建

tinygrad 模型 pkl 缓存含 sm_72 编译产物，**必须删除重编**：
```bash
rm -rf ~/.cache/tinygrad selfdrive/modeld/models/*_tinygrad.pkl
# 首次运行 modeld 时自动重新 JIT 编译（sm_87）
```

---

## 3. 一期任务：USB 双相机系统

### 3.1 结论先行
- openpilot 原厂 C3/C4 为 远焦(road)+近焦(wide) 双摄；**模型同时消费两个流**
  （`selfdrive/modeld/modeld.py:358-366`，wide 为辅助输入）
- **UI 按车速自动切换远近焦**：`selfdrive/ui/onroad/augmented_road_view.py:30-31`
  - 车速 < 10 m/s (22mph) → 显示近焦 WIDE_CAM
  - 车速 > 15 m/s (34mph) → 显示远焦 ROAD_CAM
  - 10~15 m/s 过渡区间
- 无 wide 流时全链路自动降级为单 far-cam 模式（`selfdrived.py:427`）——
  即**最少 1 个相机就能跑**

### 3.2 配置方式（零代码改动）
现有代码原生支持（`tools/webcam/camerad.py:13-25`）：

```bash
export ROAD_CAM=0      # 远焦 → /dev/video0
export WIDE_CAM=1      # 近焦 → /dev/video1
# 不设 DRIVER_CAM —— 无 DMS 需求
```

链路自动选择：`system/camerad/jetson_camerad.py:114-125` 在无 CSI 相机时
自动委托 `tools.webcam.camerad`，无需干预。

建议补充（小改动）：`process_config.py:91` 中 dmonitoringmodeld 在 JETSON 上默认启用，
因无 driver 相机会空转占内存。可在无 DRIVER_CAM 时禁用：
```
enabled=(WEBCAM or not PC or JETSON) and os.getenv("DRIVER_CAM") is not None
```
（非必须——空转无害，只是省 ~200MB 内存和进程槽位）

### 3.3 采集管线现状（CPU 路径）
```
USB UVC (请求 1280×720@25fps)
 → cv.VideoCapture 采集            tools/webcam/camera.py:16-20
 → cv.flip(frame, -1) 180°翻转     camera.py:36        [CPU]
 → PyAV BGR→NV12 转换             camera.py:27-28,37  [CPU]
 → VisionIPC 共享内存（每流20 buffer）
 → modeld：CUDA 零拷贝预处理 + tinygrad 推理   [GPU]
 → UI：VisionIPC 读流，按车速切流              [GPU]
```

### 3.4 双相机同步问题分析
- 现状：每相机独立线程 + 独立 frame_id 计数器 + 各自 Ratekeeper(20fps)
  （`tools/webcam/camerad.py:54-59`），**无硬件同步、无时间戳对齐**
- 可接受原因：modeld 对两流各自取最新帧独立推理；comma 原厂相机才是硬件同步的，
  几十 ms 错位对模型输出影响很小
- **真正风险是 USB 带宽而非同步**：
  两个 UVC 相机未压缩 YUYV@720p25 ≈ 55MB/s/个，同一控制器可能带宽不足掉帧

对策（按优先级）：
1. 两个相机分别插不同 USB 控制器（T906G 4 个 USB3.0 口分属根集线器，实测确认）：
   ```bash
   lsusb -t          # 查看拓扑分布
   v4l2-ctl -d /dev/video0 --list-formats-ext   # 确认支持 MJPEG
   ```
2. 启用 MJPEG 输出（需改 `tools/webcam/camera.py`，约 5 行）：
   ```python
   self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
   # read_frames() 中改为 imdecode → bgr2nv12
   ```
3. 监控实际帧率（验收项 3.6）

### 3.5 二期优化项：GPU 化转换（不阻塞一期）

| 方案 | 做法 | 收益 | 成本 |
|---|---|---|---|
| A. VIC 硬件转换（推荐） | GStreamer 管线：`v4l2src ! image/jpeg ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM)` appsink | MJPEG硬解(NVDEC)+BGR→NV12(VIC)，每相机 CPU 从~1核降至~0.1核 | 重写 camera.py 采集段；NVMM→CUDA 映射需 EGL/NvBufSurface 桥接 |
| B. CUDA 核 | cv2.cuda 上传 BGR → 自写 flip+cvtColor NV12 核（~50行） | 省转换计算，不省传输 | 需自编带 CUDA 的 OpenCV |
| C. 转换挪入 modeld | camera.py 发原始帧，CUDA bridge 内转换 | 最彻底 | 侵入 modeld 核心 + UI 读流也要改 |

**决策**：一期用现有 CPU 路径跑通并实测占用（预计单相机 <25% 单核，Orin 8核完全够）；
实测吃紧再上方案 A。

---

## 4. 二期任务：GMSL 相机接入（IMX390）

### 4.1 前提条件（厂商交付物清单）
向图为科技索取（已确认厂商提供）：
- [ ] JP 6.2 版 BSP / 整包刷机镜像（含板载 GMSL 解串器设备树）
- [ ] IMX390 传感器内核驱动（ko 或内置）
- [ ] 解串器驱动（确认芯片型号，大概率 MAX9296/MAX96712 系列）
- [ ] Argus ISP override 文件（IMX390 为 RAW 传感器，必须过 ISP）
- [ ] GMSL 相机测试工具/说明（厂商一般附 v4l2 测试脚本）

### 4.2 数据通路（与 USB 相机的本质区别）
```
IMX390 (RAW Bayer) → GMSL2串行 → 板载解串器 → VI/CSI → Jetson ISP(Argus)
→ ISP 输出 NV12（ISP 已完成去马赛克/AE/AWB/翻转，无需软件转换！）
→ V4L2 /dev/video* 或 Argus API 取流
→ VisionIPC → modeld/UI（下游零改动）
```
注意：RAW 型号走 ISP 路径，**不能**复用 USB 相机的"VIC 简单转换"方案；
但 ISP 直接吐 NV12 反而使转换层最简单。

### 4.3 集成步骤
1. 用厂商 BSP 刷机，验证 `dmesg | grep -iE "imx390|max929"` 无报错
2. `v4l2-ctl --list-devices` 确认 GMSL 相机枚举为 /dev/video*
3. 单帧抓图验证 ISP 输出：
   ```bash
   nvgstcapture-1.0 --sensor-id=N    # 或 gst-launch nvarguscamerasrc sensor-id=N
   ```
4. 在 `CameraSource` 抽象下新增 `GmslBackend`（见 4.4），
   取流方式优先 nvarguscamerasrc(GStreamer) → 映射进 VisionIPC
5. 校准：`common/transformations/camera.py` 中为新相机添加内参
   （DEVICE_CAMERAS 注册 imx390 条目：焦距/畸变/安装位置）

### 4.4 CameraSource 统一抽象设计（预留）
```
CameraSource (抽象基类)
 ├── open() -> bool
 ├── read_frame() -> (nv12_bytes, ts_ns)
 ├── close()
 └── properties: width/height/fps/stream_type
      ↑
      ├── UvcBackend   ← 一期（OpenCV/GStreamer，支持 MJPEG 开关）
      └── GmslBackend  ← 二期（nvarguscamerasrc，sensor_id 配置）
```
重构范围：`tools/webcam/camerad.py` 的 CAMERAS 表改为实例化 backend；
`jetson_camerad.py` 的 CSI/webcam 自动探测逻辑并入统一入口。

### 4.5 GMSL 多路扩展说明
T906G 板载 8 路 GMSL2。openpilot 模型最多消费 road+wide 两路视觉流；
多余通道可用于环视/记录等自定义功能，不在本 PRD 范围。

---

## 5. JetPack 升级路径

### 5.1 现状与目标
出厂：Ubuntu 20.04 = JetPack 5.x（L4T R35）。
一期目标：**JetPack 6.2**（Ubuntu 22.04 / L4T 36.4 / CUDA 12.x）。
远期：JetPack 7（CUDA 13）列为风险验证项。

### 5.2 刷机要点
- **必须使用图为提供的 JP6 BSP/镜像**，不可直接刷 NVIDIA 官方 SDK Manager 镜像——
  否则板载 GMSL 解串器设备树缺失，8 路 GMSL 全部失效
- 刷机方式：T906G 进入 RECOVER 模式（按住 REC → 按 RES 2秒 → 松开 RES → 松开 REC，
  主机 `lsusb` 见 "NVIDIA Corp" 即成功），SDK Manager/flash.sh 烧写
- 刷机后创建平台标记：
  ```bash
  sudo touch /JETSON && sudo touch /AGNOS_VERSION_CHECK_SKIP 2>/dev/null
  ```
  注意：`launch_chffrplus.sh` 的 `agnos_init` 会读 `/VERSION` 做 AGNOS 校验，
  Orin 上应跳过（`/AGNOS` 不存在时该分支自然跳过 ✅，确认即可）

### 5.3 JP6 对本项目的具体影响
| 组件 | JP5 (Xavier) | JP6.2 (Orin) | 动作 |
|---|---|---|---|
| CUDA | 11.4 | 12.x | tinygrad/torch 重装（pip 依赖按 pyproject 重装即可） |
| sm 架构 | sm_72 | sm_87 | 清模型缓存重编（§2.5） |
| Python | 3.11 (venv) | 3.10 系统自带 | uv 创建 venv 时指定版本，按 pyproject.toml requires |
| nvpmodel | MAXN=0 | MAXN=0（档位表变） | §2.2.2 实测 |
| 编译器 | clang 默认 | clang 默认 | scons CC=clang 不变 |

### 5.4 JP7 风险项（二期后评估）
- tinygrad 对 CUDA 13 的兼容性（上游跟进情况未知）——升级前先在 PC CUDA13 环境
  验证 tinygrad 推理通过再动板子
- 厂商 BSP 是否发布 JP7 版（决定 GMSL 能否继续用）

---

## 6. 实施顺序与验收标准

### 阶段 P0：环境就绪（0.5 天）
- [ ] 厂商 JP6.2 BSP 刷机成功，HDMI 出画面
- [ ] `touch /JETSON`，uname -m = aarch64
- [ ] CUDA 12.x：`nvcc --version`；GPU：`nvidia-smi`
- [ ] 克隆本项目到 /data/openpilot，uv sync 依赖安装成功

### 阶段 P1：构建适配（0.5 天）
- [ ] 改 SConstruct（§2.1）、hardware.py（§2.2）、validate_env.sh（§2.3）
- [ ] `scons -j8` 全绿
- [ ] `./scripts/jetson_validate_env.sh` 通过

### 阶段 P2：单相机冒烟（1 天）
- [ ] USB 相机插 /dev/video0，`export ROAD_CAM=0`
- [ ] 清模型缓存，单独起 webcamerad：帧率达标（v4l2 实际能力 ≥20fps）
- [ ] 起 modeld：日志出现 "tinygrad CUDA graphs + CUDA zero-copy preprocessing"，
      median 推理时间 < Xavier 基线（13.66ms）或合理水平
- [ ] 起 UI：60FPS，模型车道线渲染正常

### 阶段 P3：双相机完整功能（1 天）
- [ ] 第二相机插另一 USB 总线口，`export WIDE_CAM=1`
- [ ] `lsusb -t` 确认分属不同控制器；两路均稳定 20fps（写帧率监控脚本观察 10 分钟无掉帧）
- [ ] replay/demo 模式下车速 <10m/s UI 显示近焦、>15m/s 切远焦（augmented_road_view 生效）
- [ ] manager 全进程绿色（除 dmonitoringmodeld 按 §3.2 处理）
- [ ] 温度/功耗上报数值合理（hardwared 日志核对 §2.2 实测路径）
- [ ] 30 分钟压力测试：GPU <40%、CPU <60%、无 thermal throttling（watchdog 未触发频繁 jetson_clocks）

### 阶段 P4：实车联调（视车况）
- [ ] panda CAN 连接、车辆识别（Fingerprint）
- [ ] 控制横向/纵向闭环正常
- [ ] loggerd 录制回放正常

### 阶段 P5：GMSL 二期（依赖厂商资料到位）
- [ ] 按 §4.3 步骤逐项验收
- [ ] IMX390 内参标定录入 DEVICE_CAMERAS
- [ ] GmslBackend 合入 CameraSource 抽象

---

## 附录 A：关键文件索引（本项目 Jetson 相关全部清单）

| 文件 | 作用 | 是否需改 |
|---|---|---|
| `SConstruct` | 平台检测(jarch64)+编译flags | ✅ §2.1 |
| `system/hardware/__init__.py` | /JETSON 检测+HARDWARE 选择 | ❌ |
| `system/hardware/jetson/hardware.py` | 温度/功耗/nvpmodel/watchdog | ✅ §2.2 |
| `system/hardware/jetson/fan_controller.py` | PWM 风扇控制 | ⚠️ 上机验证（T906G 导被动散热为主，风扇策略需实测） |
| `system/hardware/jetson/dfs.py` | 动态调频 | ⚠️ 验证 |
| `system/hardware/jetson/hugepages.py` | 大页内存(CUDA TLB) | ❌ |
| `system/hardware/hw.h` / `jetson/hardware.h` | C++ 侧硬件头 | ❌ |
| `system/hardwared.py` | 硬件守护进程(JETSON 分支) | ⚠️ 验证 dfs 调用 |
| `system/manager/process_config.py` | 进程编排(WEBCAM/JETSON) | ⚠️ 可选 §3.2 |
| `system/camerad/jetson_camerad.py` | CSI/USB 自动切换入口 | ❌（自动回退USB） |
| `system/camerad/cameras/camera_jetson.py` | CSI V4L2 驱动 | ❌（二期 GMSL 参考） |
| `tools/webcam/camerad.py` | USB 相机守护(多摄) | ❌（配置即用） |
| `tools/webcam/camera.py` | OpenCV 采集+NV12转换 | ⚠️ 可选 MJPEG §3.4 |
| `selfdrive/modeld/modeld.py` | 视觉模型(DEV=CUDA) | ❌ |
| `selfdrive/modeld/dmonitoringmodeld.py` | DMS 模型 | ❌（不用 DMS） |
| `msgq_repo/msgq/visionipc/visionbuf_jetson.cc` | 共享内存 CUDA 注册 | ❌ |
| `system/loggerd/encoderd.cc` | 录制编码 | ⚠️ 可选 NVENC §2.4 |
| `selfdrive/ui/ui.py` | UI 核绑定(core 6) | ❌ |
| `selfdrive/ui/onroad/augmented_road_view.py` | 远近焦速度切换 | ❌ |
| `scripts/jetson_validate_env.sh` | 环境自检 | ✅ §2.3 |
| `scripts/jetson_replay.sh` / `jetson_stress_test.sh` | 调试工具 | ⚠️ 按需 |
| `JETSON_SETUP.md` / `JETSON_OPTIMIZATION.md` | 文档 | ⚠️ 移植后更新 |

## 附录 B：Xavier vs Orin 快查表

| 维度 | AGX Xavier (源) | TW-T906G / AGX Orin 32GB (目标) |
|---|---|---|
| arch flag | jarch64, cortex-a76 | jarch64, **cortex-a78ae** |
| GPU | Volta 512核 sm_72 | Ampere 1792核 **sm_87** |
| CUDA | 11.4 | **12.x** (JP6) |
| L4T | R35 | **R36.4** (JP6.2) |
| Ubuntu | 20.04 | **22.04** |
| CPU 核数 | 8 | 8（相同） |
| CPU 峰值 | 2.2GHz | 2.2GHz（watchdog 阈值微调） |
| 功耗计 | INA3221 bus1-0040 | INA3221x **bus3/c-0040（实测）** |
| DLA | — | 2× NVDLA v2.0（dmonitoring/TRT 可用，本项目不用） |
| GMSL | 无 | **板载 8 路 GMSL2** |
| 模型缓存 | sm_72 pkl | 必须清缓存重编 |

---
*本文档基于对源码库的静态审查 + TW-T906G 用户手册 V4.0 编写。
所有标注"实测/待验证"的项目须上机确认后回填本文档。*
