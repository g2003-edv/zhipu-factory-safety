"""厂区智能安防监控：YOLOv8 本地行人检测 + 智谱 GLM-4V 云端复核。

设计要点：
- 本地 YOLO 逐帧检测行人，云端多模态模型只在触发告警时做二次复核；
- 同一摄像头有冷却窗口（默认 60s）与连续帧去重，避免按帧调用付费接口；
- 云端不可用（网络 / 鉴权 / 返回异常）时自动降级为仅本地 YOLO 告警，
  主循环不中断，并在终端打印明确警告。

所有配置均从环境变量读取（也支持 .env 文件），详见 .env.example 与 README。
"""

from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from zhipuai import ZhipuAI

try:  # .env 为可选配置项，未安装 python-dotenv 时直接读系统环境变量
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - 依赖缺失时不影响运行
    pass


# ---------------------------------------------------------------------------
# 常量与默认配置
# ---------------------------------------------------------------------------
DEFAULT_GLM_MODEL = "glm-4v-plus"
DEFAULT_COOLDOWN_SEC = 60.0
DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CAMERAS = "0"
DEFAULT_CONF = 0.35

# 同一摄像头相邻两次告警画面的平均像素差阈值，低于该值视为同一事件的连续帧
SAME_FRAME_DIFF = 3.0
# 连续多少帧没有检测到行人后重新布防（允许同一场景再次告警）
CLEAR_STREAK_FRAMES = 30
# 摄像头重连与读帧失败的重试间隔（秒），避免忙等
RECONNECT_DELAY_SEC = 2.0
READ_FAIL_RETRY_SEC = 0.5
# 模型原始回复写入日志时的截断长度
REPLY_MAX_CHARS = 120

ALERT_TEXT = {
    0: "环境误报（光影/杂物）",
    1: "真实行人闯入",
    2: "画面模糊，人工复核",
}
# 云端返回非 0/1/2 时统一按“需人工复核”处理
UNKNOWN_ALERT_TYPE = 2
# 云端不可用时的本地降级告警类型（本地确实检出了行人）
FALLBACK_ALERT_TYPE = 1

LOG_COLUMNS = [
    "批次ID",
    "帧号",
    "时间",
    "图片路径",
    "告警类型",
    "类型说明",
    "置信度",
    "框数量",
    "复核来源",
    "模型回复",
    "错误信息",
]

GLM_PROMPT = (
    "判断图中是否有人闯入厂区，只输出数字："
    "0=没人/杂物，1=有人，2=画面看不清"
)


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------
def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[警告] 环境变量 {name}={raw!r} 不是有效数字，改用默认值 {default}")
        return default
    if minimum is not None and value < minimum:
        print(f"[警告] 环境变量 {name}={value} 小于下限 {minimum}，改用 {minimum}")
        return minimum
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_cameras(raw: str) -> list:
    """把 "0,http://ip:8080/video" 解析成 [0, "http://ip:8080/video"]。"""
    cameras = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        cameras.append(int(item) if item.lstrip("-").isdigit() else item)
    return cameras or [0]


def describe_alert(alert_type: int) -> str:
    """告警类型文字映射，未知类型回退到 2，避免 KeyError。"""
    return ALERT_TEXT.get(alert_type, ALERT_TEXT[UNKNOWN_ALERT_TYPE])


# ---------------------------------------------------------------------------
# 云端复核
# ---------------------------------------------------------------------------
def build_client(api_key: str, timeout_sec: float):
    """创建智谱客户端；老版本 SDK 不支持 timeout 参数时自动降级。"""
    try:
        return ZhipuAI(api_key=api_key, timeout=timeout_sec)
    except TypeError:
        return ZhipuAI(api_key=api_key)


def img_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_alert_type(reply: str) -> int:
    """从模型回复中解析 0/1/2；解析不到或超出范围一律按 2 处理。"""
    for ch in reply:
        if ch in "012":
            return int(ch)
    return UNKNOWN_ALERT_TYPE


def judge_alert(
    client, img_path: str, model_name: str
) -> Tuple[Optional[int], str, str]:
    """调用 GLM-4V 判定告警类型。

    返回 (alert_type, raw_reply, error)：
    - alert_type 为 0/1/2：云端判定成功；
    - alert_type 为 None：云端不可用（网络/鉴权/返回异常），调用方应降级为本地告警。
    本函数不会抛出异常，保证主循环不中断。
    """
    if client is None:
        return None, "", "智谱客户端未初始化"

    try:
        b64_data = img_to_base64(img_path)
    except OSError as exc:
        return None, "", f"读取截图失败：{exc}"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - 边界处兜底，任何 SDK/网络异常都降级
        return None, "", f"{type(exc).__name__}: {exc}"

    try:
        raw = (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError) as exc:
        return None, "", f"模型响应结构异常：{type(exc).__name__}: {exc}"

    return parse_alert_type(raw), raw, ""


# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------
def get_camera(cameras, quiet: bool = False):
    """按顺序尝试摄像头源，全部失败返回 None。"""
    for source in cameras:
        cap = cv2.VideoCapture(source)
        if cap is not None and cap.isOpened():
            if not quiet:
                print(f"[配置] 已打开摄像头源：{source}")
            return cap
        if cap is not None:
            cap.release()
        if not quiet:
            print(f"[警告] 摄像头源不可用：{source}")
    return None


# ---------------------------------------------------------------------------
# 告警抑制
# ---------------------------------------------------------------------------
def frame_signature(frame) -> np.ndarray:
    """生成 64x48 灰度缩略签名，用于判断是否为同一事件的连续帧。"""
    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


@dataclass
class CameraState:
    """单个摄像头源的告警状态。"""

    last_call_ts: float = 0.0
    last_sig: Optional[np.ndarray] = None
    empty_streak: int = 0

    def allow_call(self, now: float, sig: np.ndarray, cooldown: float) -> Tuple[bool, str]:
        remaining = self.last_call_ts + cooldown - now
        if remaining > 0:
            return False, f"冷却窗口内（剩余 {remaining:.0f}s）"
        if (
            self.last_sig is not None
            and float(np.abs(sig - self.last_sig).mean()) < SAME_FRAME_DIFF
        ):
            return False, "与上次告警画面几乎相同（连续帧去重）"
        return True, ""


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def save_snapshot(frame, screenshot_dir: str, stamp: str, frame_idx: int) -> str:
    """保存告警截图，文件名带毫秒与帧号，避免同一秒内互相覆盖。"""
    os.makedirs(screenshot_dir, exist_ok=True)
    millis = int((time.time() % 1) * 1000)
    filename = f"{stamp}_{millis:03d}_f{frame_idx:06d}_告警截图.jpg"
    path = os.path.join(screenshot_dir, filename)
    if not cv2.imwrite(path, frame):
        raise OSError(f"截图写入失败：{path}")
    return path


def append_log(log_file: str, row: dict) -> None:
    """追加一行告警日志；写盘失败只告警，不中断监控。"""
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            old_df = pd.read_csv(log_file, encoding="utf-8-sig")
            merged = pd.concat(
                [old_df.reindex(columns=LOG_COLUMNS), row_df], ignore_index=True
            )
        else:
            merged = row_df
        merged.to_csv(log_file, index=False, encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        print(f"[警告] 告警日志写入失败（{type(exc).__name__}: {exc}），本条记录如下：")
        print(row)


class PreviewWindow:
    """可选的可视化窗口；无图形环境时自动关闭预览而不是崩溃。"""

    def __init__(self, title: str, enabled: bool = True):
        self.title = title
        self.enabled = enabled
        self._warned = False
        self._created = False

    def show(self, frame) -> None:
        if not self.enabled:
            return
        try:
            cv2.imshow(self.title, frame)
            self._created = True
        except cv2.error as exc:
            self.enabled = False
            if not self._warned:
                self._warned = True
                print(f"[警告] 无法显示预览窗口（{exc}），已切换为无界面运行。")

    def should_quit(self) -> bool:
        if not self.enabled:
            return False
        try:
            return cv2.waitKey(1) & 0xFF == ord("q")
        except cv2.error:
            self.enabled = False
            return False

    def close(self) -> None:
        if self._created:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        print("[错误] 未找到环境变量 ZHIPU_API_KEY，无法调用智谱多模态接口。", file=sys.stderr)
        print("       请先在 .env 或系统环境变量中配置（参考 .env.example / README）。", file=sys.stderr)
        return 2

    glm_model = env_str("GLM_MODEL", DEFAULT_GLM_MODEL)
    weights = env_str("YOLO_WEIGHTS", DEFAULT_WEIGHTS)
    cooldown = env_float("ALERT_COOLDOWN", DEFAULT_COOLDOWN_SEC, minimum=0.0)
    det_conf = env_float("YOLO_CONF", DEFAULT_CONF, minimum=0.0)
    cameras = parse_cameras(env_str("CAMERA_SOURCES", DEFAULT_CAMERAS))
    screenshot_dir = env_str("SCREENSHOT_DIR", "alert_screenshot")
    log_file = env_str("LOG_CSV", "告警日志.csv")
    glm_max_failures = int(env_float("GLM_MAX_FAILURES", 3, minimum=1))
    preview = PreviewWindow("厂区安防监控", enabled=env_bool("ENABLE_PREVIEW", True))

    os.makedirs(screenshot_dir, exist_ok=True)

    client = None
    try:
        client = build_client(api_key, env_float("GLM_TIMEOUT", 30.0, minimum=1.0))
    except Exception as exc:  # noqa: BLE001 - 客户端初始化失败也走本地降级
        print(f"[警告] 智谱客户端初始化失败（{type(exc).__name__}: {exc}），本次仅做本地告警。")

    print(f"[配置] 复核模型={glm_model} 冷却窗口={cooldown:g}s 摄像头={cameras} 权重={weights}")

    try:
        model = YOLO(weights)
    except Exception as exc:  # noqa: BLE001 - 权重加载失败无法继续
        print(f"[错误] 加载 YOLO 权重失败：{weights}（{type(exc).__name__}: {exc}）", file=sys.stderr)
        print("       首次运行会自动下载 yolov8n.pt，请检查网络或通过 YOLO_WEIGHTS 指定本地权重。", file=sys.stderr)
        return 1

    cap = get_camera(cameras)
    if cap is None:
        print(f"[错误] 无法打开任何摄像头（CAMERA_SOURCES={cameras}）。", file=sys.stderr)
        print("       本机摄像头请填 0；手机/IP 摄像头请确认与本机同网段且地址可访问。", file=sys.stderr)
        return 1

    batch_id = f"B{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    state = CameraState()
    total_person = 0
    frames = 0
    frame_idx = 0
    alerts = 0
    suppressed = 0
    glm_failures = 0
    glm_disabled = False
    glm_disabled_reason = ""
    fail_streak = 0
    start_time = time.time()

    print("监控启动：检测到行人自动截图并调用云端复核，按 Q 关闭。")
    try:
        while True:
            if cap is None:
                # 重连失败后等待再试，避免空转占满 CPU
                time.sleep(RECONNECT_DELAY_SEC)
                cap = get_camera(cameras, quiet=True)
                continue

            ret, frame = cap.read()
            if not ret:
                fail_streak += 1
                if fail_streak == 1 or fail_streak % 50 == 0:
                    print(f"[警告] 读取画面失败（连续 {fail_streak} 次），0.5s 后重连摄像头。")
                cap.release()
                cap = None
                time.sleep(READ_FAIL_RETRY_SEC)
                continue

            fail_streak = 0
            frame_idx += 1
            frames += 1

            # verbose=False 关闭逐帧刷屏日志
            results = model(frame, classes=[0], conf=det_conf, verbose=False)
            boxes = results[0].boxes if len(results) else None
            box_count = 0 if boxes is None else len(boxes)

            if box_count > 0:
                total_person += 1
                state.empty_streak = 0
                confidence = float(max(box.conf.item() for box in boxes))
                stamp = time.strftime("%Y%m%d_%H%M%S")
                now = time.time()
                sig = frame_signature(frame)
                allowed, reason = state.allow_call(now, sig, cooldown)

                if not allowed:
                    # 冷却窗口内 / 同一事件的连续帧：只计数，不写日志、不调用云端
                    suppressed += 1
                else:
                    alert_type = None
                    raw_reply = ""
                    error = ""
                    source = "GLM"
                    save_path = ""

                    # 截图无论如何都保存，降级时也要留下现场证据
                    try:
                        save_path = save_snapshot(frame, screenshot_dir, stamp, frame_idx)
                    except OSError as exc:
                        error = str(exc)
                        print(f"[警告] {exc}")

                    if not save_path:
                        pass  # 无截图则无法复核，直接走本地降级
                    elif glm_disabled:
                        error = f"云端复核已熔断：{glm_disabled_reason}"
                    else:
                        alert_type, raw_reply, error = judge_alert(
                            client, save_path, glm_model
                        )

                    state.last_call_ts = now
                    state.last_sig = sig

                    if alert_type is None:
                        # 云端不可用：降级为仅本地 YOLO 告警，主循环继续
                        alert_type = FALLBACK_ALERT_TYPE
                        source = "LOCAL_FALLBACK"
                        desc = "本地YOLO行人告警（云端未复核，需人工确认）"
                        glm_failures += 1
                        print(f"[警告] 云端复核失败：{error}；已降级为仅本地 YOLO 告警。")
                        if not glm_disabled and glm_failures >= glm_max_failures:
                            glm_disabled = True
                            glm_disabled_reason = (
                                f"连续失败 {glm_failures} 次，停止调用云端接口"
                            )
                            print(f"[警告] {glm_disabled_reason}，后续运行仅使用本地检测。")
                    else:
                        glm_failures = 0
                        desc = describe_alert(alert_type)

                    alerts += 1
                    append_log(
                        log_file,
                        {
                            "批次ID": batch_id,
                            "帧号": frame_idx,
                            "时间": stamp,
                            "图片路径": save_path,
                            "告警类型": alert_type,
                            "类型说明": desc,
                            "置信度": round(confidence, 3),
                            "框数量": box_count,
                            "复核来源": source,
                            "模型回复": raw_reply[:REPLY_MAX_CHARS],
                            "错误信息": error[:REPLY_MAX_CHARS],
                        },
                    )
                    print(
                        f"【告警】{stamp} | 类型 {alert_type} {desc} | "
                        f"置信度 {confidence:.2f} | 框数 {box_count} | 来源 {source}"
                    )
            else:
                state.empty_streak += 1
                if state.empty_streak >= CLEAR_STREAK_FRAMES and state.last_sig is not None:
                    # 画面持续无行人后重新布防，允许同一场景再次告警
                    state.last_sig = None

            preview.show(frame)
            if preview.should_quit():
                break
    except KeyboardInterrupt:
        print("\n[信息] 收到 Ctrl+C，正在退出。")
    finally:
        if cap is not None:
            cap.release()
        preview.close()

    run_sec = time.time() - start_time
    fps = frames / run_sec if run_sec > 0 else 0.0
    print(
        f"\n监控结束：运行 {run_sec:.1f}s，处理 {frames} 帧，平均 {fps:.1f} FPS，"
        f"含行人帧 {total_person}，告警 {alerts}，抑制 {suppressed}（冷却/连续帧去重）"
    )
    if glm_disabled:
        print("[提示] 本次运行云端复核已熔断，告警结果均来自本地 YOLO。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
