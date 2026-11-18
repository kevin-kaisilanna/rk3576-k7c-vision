# RK3576 / KICKPI K7C 视觉工程

KICKPI K7C（RK3576，Ubuntu 24.04）上跑的标准 CPython + OpenCV 钢珠识别与局域网图传，不是 MicroPython。算法由 K230 仓库的灰度双边缘识别移植而来。

## 这个工程做什么

- V4L2 采集 MJPG 640×480 @ 30 FPS，在裁出的 ROI 内做一维双边缘对比度检测，三帧确认 + 速度预测 + 丢点保持，把位置经 `/dev/ttyS8`（115200，帧尾 LF）发给主控。
- 局域网 MJPEG 图传 + 网页端 MP4 录像（MP4V），默认端口 `8081`。
- `deploy/` 提供 systemd 开机自启：自动建 NetworkManager 热点 `K7C-AP`（`10.42.0.1/24`），服务异常自动重启。
- 与 K230 的差异：采集用 `cv2.VideoCapture`（`media.sensor` → V4L2），处理用标准 `cv2` + `numpy`（`cv_lite` / `ulab` → CPython），串口帧尾 CRLF → LF，`/sdcard/main.py` → systemd。

## 部署

1. **接硬件**：USB 摄像头（需支持 MJPG，默认 `/dev/video73`）、UART `/dev/ttyS8` 接主控；确认 `ls /dev/video*`、`ls /dev/ttyS*` 存在。
2. **装依赖**：

   ```bash
   python3 -m venv /venv
   /venv/bin/pip install opencv-python numpy pyserial
   ```

3. **放代码**：把整个仓库放到 `/userdata/project/`（`deploy/k7c-ball.service` 里的路径是硬编码，换目录需同步改 unit 的 `WorkingDirectory` 与 `ExecStart`）。
4. **填配置**：`deploy/k7c_ball_autostart.py` 里的 AP 密码占位符改成自己的密码；`--uart-port`、`--camera` 与实机不符时改 `k7c-ball.service` 的 `ExecStart`。
5. **前台试跑**（先确认能出画面再配自启）：

   ```bash
   cd /userdata/project && source /venv/bin/activate
   python3 grayscale_ball_tracker_rk3576.py --camera 73 --width 640 --height 480 --fps 30 \
     --search-x-min 95 --search-x-max 505 --stream --port 8081 \
     --stream-fps 20 --stream-width 1024 --jpeg-quality 80 --log-every 30
   ```

6. **验证**：浏览器打开 `http://<板子IP>:8081/` 看 MJPEG 画面；串口日志每 30 帧打印一次位置；网页可录制 MP4。
7. **开机自启**（可选）：

   ```bash
   sudo python3 deploy/k7c_ball_autostart.py install     # 建 K7C-AP + 装 k7c-ball.service
   sudo python3 deploy/k7c_ball_autostart.py status      # 查看服务与网卡状态
   sudo systemctl start k7c-ball                        # install 不会立即启动，避免切断当前 SSH
   sudo python3 deploy/k7c_ball_autostart.py uninstall  # 移除服务（保留 AP 配置）
   ```

## 文件

| 文件 | 说明 |
| --- | --- |
| `grayscale_ball_tracker_rk3576.py` | 主程序：识别、串口输出、MJPEG 图传、录像 |
| `deploy/k7c_ball_autostart.py` | 自启安装器，支持 `install` / `ensure-ap` / `status` / `uninstall` |
| `deploy/k7c-ball.service` | systemd unit |

## 第三方内容（均未上传）

`opencv-python`（Apache-2.0）、`numpy`（BSD-3-Clause）、`pyserial` 用 pip 安装；V4L2 驱动、systemd、NetworkManager 与厂商系统镜像由板卡 / 发行版提供；录像 MP4、演示视频与竞赛官方文档未分发。
真实 Wi-Fi 名称 / 密码、AP 密码与共享密钥已脱敏为 `YOUR_WIFI_SSID`、`YOUR_WIFI_PASSWORD`、`YOUR_AP_PASSWORD`、`YOUR_SHARED_SECRET`，**运行前必须填回**。

## 项目系列

同一竞赛项目按**硬件环境**拆分的独立仓库，各平台固件 API 与运行栈互不相通，代码不能混用。

| 硬件环境 / 运行栈 | 仓库 |
| --- | --- |
| 嘉楠 K230 · CanMV K230（MicroPython 固件） | [canmv-k230-vision](https://github.com/kevin-kaisilanna/canmv-k230-vision) |
| 逐飞 OMV RT5 · OpenMV 兼容固件 | [omv-rt5-openmv-vision](https://github.com/kevin-kaisilanna/omv-rt5-openmv-vision) |
| 逐飞 OpenART Mini · 板级显示 / SPI | [openart-mini-display](https://github.com/kevin-kaisilanna/openart-mini-display) |
| KICKPI K7C · RK3576（Ubuntu + CPython/OpenCV） | **rk3576-k7c-vision** ← 本仓库 |
| 竞赛设计报告（跨平台文档，无代码） | [rolling-ball-design-report](https://github.com/kevin-kaisilanna/rolling-ball-design-report) |

> 所有仓库均只收录自研代码；第三方开源库、厂商 SDK、平台生成代码与预训练模型未上传，来源见各仓库 README。

## 许可证

仅收录自研代码，尚未指定开源许可证（默认保留所有权利）；第三方组件权利归各自权利人。
