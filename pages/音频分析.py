#!/usr/bin/env python3
"""
音频频谱分析系统 - Streamlit 版本
功能与 Web 版完全一致：上传 → STFT/FFT 分析 → 缺陷检测 → AI 诊断 → HTML 报告下载

运行方式：
    pip install streamlit numpy scipy plotly kaleido soundfile openai
    streamlit run audio_analyzer.py --server.port 8501
"""

import io
import json
import math
import os
import re
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import get_window, stft as scipy_stft
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore")

# ─────────────────────────── 页面配置 ───────────────────────────

st.set_page_config(
    page_title="音频频谱分析",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 现代浅色主题（干净明亮 + 圆角卡片 + 精致阴影）──
# st.markdown("""<style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
# * { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; }

# .stApp { background:#f8fafc !important; min-height:100vh; overflow:visible !important; }
# .main > div { padding:2rem 3rem !important; max-width:1100px !important; margin:0 auto !important; overflow:visible !important; }
# .stAppDeployButton { display:none !important; }

# [data-testid="stSidebar"] { background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%) !important; border-right:1px solid #e2e8f0 !important; }
# h1 { color:#0f172a !important; font-weight:800 !important; letter-spacing:-0.6px !important; }
# h2 { color:#4f46e5 !important; font-weight:700 !important; margin-top:32px !important; }
# p { color:#475569 !important; line-height:1.7 !important; }

# .stFileUploader { background:#fff !important; border:2px dashed #cbd5e1 !important; border-radius:16px !important; padding:36px 20px !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); transition:all 0.25s; }
# .stFileUploader:hover { border-color:#818cf8 !important; background:#fafbff !important; box-shadow:0 4px 12px rgba(99,102,241,0.10) !important; }
# .stFileUploader > label { color:#64748b !important; font-size:15px !important; font-weight:500 !important; }
# /* 隐藏 uploader 内部重复文字，避免与自定义 label 重叠 */
# .stFileUploader span[data-testid="stMarkdownContainer"],
# .stFileUploader .stFileUploaderDropContainer > span:not(:first-child):not([data-baseweb="visually-hidden"]),
# .stFileUploader div[data-testid="stCaptionContainer"] + div,
# .stFileUploader [data-baseweb="file-uploader"] > div:first-child > span:nth-of-type(2),
# .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }

# .stTextInput input,.stTextArea textarea { background:#f8fafc !important; border:1.5px solid #cbd5e1 !important; color:#0f172a !important; border-radius:11px !important; padding:10px 14px !important; transition:all 0.2s; }
# .stTextInput:focus-within input,.stTextArea:focus-within textarea { border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,0.08) !important; }

# .stButton button { border-radius:11px !important; font-weight:600 !important; font-size:14px !important; padding:9px 22px !important; transition:all 0.2s ease !important; }
# .stButton > button[kind='primary'] { background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important; border:none !important; color:#fff !important; box-shadow:0 3px 10px rgba(79,70,229,0.28) !important; }
# .stButton > button[kind='primary']:hover { transform:translateY(-1px) !important; box-shadow:0 5px 18px rgba(79,70,229,0.38) !important; }
# .stButton > button:not([kind='primary']) { background:#fff !important; border:1.5px solid #e2e8f0 !important; color:#334155 !important; }
# .stButton > button:not([kind='primary']):hover { background:#f8fafc !important; border-color:#a5b4fc !important; }

# .stProgress div[role='progressbar'] { height:7px !important; border-radius:999px !important; background:linear-gradient(90deg,#4f46e5,#a78bfa,#4f46e5) !important; background-size:200% 100% !important; animation:shimmer 2s infinite linear !important; }
# @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
# .stProgress div > div:first-child { height:7px !important; border-radius:999px !important; background:#e2e8f0; }
# .stProgress { margin-bottom:8px !important; }
# [data-testid="stStatusWidget"], [data-testid="stAlert"] { overflow:visible !important; }

# .stTabs [role='tablist'] { gap:4px !important; background:#f1f5f9 !important; border-radius:11px !important; padding:5px !important; }
# .stTabs [role='tab'] { border-radius:8px !important; font-weight:500 !important; font-size:13px !important; color:#64748b !important; transition:all 0.2s !important; }
# .stTabs [role='tab'][aria-selected='true'] { background:#fff !important; color:#4f46e5 !important; box-shadow:0 1px 4px rgba(0,0,0,0.08) !important; }
# .stTabs [aria-selected='false']:hover { color:#334155 !important; background:#e2e8f0 !important; }

# .stDataFrame { border-radius:12px !important; overflow:hidden !important; background:#fff !important; border:1px solid #e2e8f0 !important; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
# .stDataFrame table thead th { background:#f8fafc !important; color:#475569 !important; font-weight:600 !important; font-size:12px !important; text-transform:uppercase !important; letter-spacing:0.4px !important; padding:11px 16px !important; border-bottom:2px solid #e2e8f0 !important; }
# .stDataFrame table td { color:#1e293b !important; font-size:13px !important; padding:10px 16px !important; border-bottom:1px solid #f1f5f9 !important; }
# .stDataFrame table tr:hover td { background:#fafbff !important; }

# .stMetric { background:#fff !important; border:1px solid #e2e8f0 !important; border-radius:14px !important; padding:18px 14px !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
# .stMetric label { color:#94a3b8 !important; font-size:12px !important; font-weight:600 !important; text-transform:uppercase !important; }
# .stMetric div[data-testid='stMetricValue'] { color:#0f172a !important; font-weight:700 !important; font-size:20px !important; }

# .stExpander { border:1px solid #e2e8f0 !important; border-radius:14px !important; background:#fff !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
# .stExpander > summary { color:#4f46e5 !important; font-weight:600 !important; font-size:14px !important; }

# .stCaption { color:#94a3b8 !important; font-size:12px !important; }
# hr, hr + * { display:none !important; }
# .stInfo { background:#eff6ff !important; border:1px solid #bfdbfe !important; border-radius:12px !important; padding:16px 20px !important; }
# .stInfo p,.stInfo span { color:#2563eb !important; font-size:15px !important; }
# .stSuccess { background:#ecfdf5 !important; border:1px solid #a7f3d0 !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
# .stSuccess p, .stSuccess span { color:#059669 !important; font-size:16px !important; font-weight:700 !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }
# .stWarning { background:#fffbeb !important; border:1px solid #fde68a !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
# .stWarning p, .stWarning span { color:#d97706 !important; font-size:16px !important; font-weight:600 !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }
# .stError { background:#fef2f2 !important; border:1px solid #fecaca !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
# .stError p, .stError span { color:#dc2626 !important; font-size:15px !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }

# /* 强制状态消息容器不被裁剪 */
# div[data-testid="stStatusWidget"] { min-height:auto !important; max-height:none !important; height:auto !important; overflow:visible !important; }
# [data-testid="stAlertContainer"] > div { height:auto !important; min-height:auto !important; max-height:none !important; overflow:visible !important; }
# </style>""", unsafe_allow_html=True)


# 初始化 session_state
for key in ["analysis_result", "uploaded_file", "analysis_history"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_history" else []

# ──────────────────────── 音频解码引擎 ────────────────────────


def decode_wav(buffer_bytes: bytes) -> tuple[np.ndarray, int]:
    """解码 WAV 字节数据，返回 (samples, sample_rate)"""
    buffer = io.BytesIO(buffer_bytes)
    data, sr = sf.read(buffer, dtype="float32")
    # 转为单声道（取均值）
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def _decode_via_ffmpeg(file_bytes: bytes) -> tuple[np.ndarray, int] | None:
    """使用 ffmpeg 子进程解码音频（支持 MP4 等容器格式）"""
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_in_path = tmp_in.name
        cmd = [
            "ffmpeg", "-i", tmp_in_path,
            "-vn",               # 忽略视频轨
            "-ac", "1",          # 单声道
            "-ar", "44100",      # 采样率
            "-f", "f32le",       # 32-bit float PCM
            "-acodec", "pcm_f32le",
            "pipe:1",            # 输出到 stdout
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        os.unlink(tmp_in_path)
        if result.returncode != 0 or len(result.stdout) < 4:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.float32)
        return samples, 44100
    except FileNotFoundError:
        # ffmpeg 未安装
        return None
    except Exception:
        return None


def try_decode_audio(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """尝试解码音频文件，优先 WAV → soundfile → ffmpeg，失败则抛出异常"""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "wav":
            return decode_wav(file_bytes)
        else:
            buffer = io.BytesIO(file_bytes)
            data, sr = sf.read(buffer, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, sr
    except Exception:
        pass
    # soundfile 失败时，尝试 ffmpeg 解码（支持 MP4 等容器格式）
    ffmpeg_result = _decode_via_ffmpeg(file_bytes)
    if ffmpeg_result is not None:
        return ffmpeg_result
    # 所有解码方式均失败，直接报错而非生成模拟数据
    raise ValueError(
        f"无法解码音频文件 '{filename}'。"
        f"请确认文件格式有效且未损坏。支持格式：WAV/MP3/OGG/FLAC/AAC/M4A/MP4。"
        f"如为视频格式，建议先转换为 WAV 后再上传。"
    )


# ──────────────────────── DSP 核心算法 ────────────────────────


def compute_rms(samples: np.ndarray) -> float:
    """RMS 电平 (dBFS)"""
    rms = np.sqrt(np.mean(samples**2))
    return float(20 * np.log10(max(rms, 1e-10)))


def compute_peak(samples: np.ndarray) -> float:
    """峰值电平 (dBFS)"""
    peak = np.max(np.abs(samples))
    return float(20 * np.log10(max(peak, 1e-10)))


def next_power_of_2(n: int) -> int:
    """向上取最近的 2 的幂"""
    p = 1
    while p < n:
        p *= 2
    return p


def perform_fft_analysis(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    执行 FFT 频谱分析（Welch 法：全量信号分段平均功率谱）

    不再只截取开头零点几秒，而是将整段音频分成多个重叠段，
    对每段做加窗 FFT 后取平均，得到统计上更稳定的功率谱估计。
    这就是 Welch 法的核心思想——用空间换精度，用平均换稳定。

    返回: (magnitudes, freqs, fft_size)
    """
    fft_size = next_power_of_2(min(len(samples), 8192))
    hop_size = fft_size // 2  # 50% 重叠
    window = get_window("hann", fft_size)

    # 计算可以切出多少个完整段
    n_segments = max(1, (len(samples) - fft_size) // hop_size + 1)

    # 累加各段功率谱
    avg_power = np.zeros(fft_size // 2)
    for i in range(n_segments):
        start = i * hop_size
        segment = samples[start:start + fft_size].astype(np.float64)
        if len(segment) < fft_size:
            segment = np.pad(segment, (0, fft_size - len(segment)))
        windowed = segment * window
        spectrum = fft(windowed)
        avg_power += np.abs(spectrum[:fft_size // 2]) ** 2

    # 归一化：除以段数和窗函数修正因子
    window_correction = np.sum(window ** 2)
    avg_power /= (n_segments * window_correction * sample_rate)
    magnitudes = np.sqrt(avg_power)  # 幅度谱 = 功率谱的平方根

    freqs = fftfreq(fft_size, d=1.0 / sample_rate)[:fft_size // 2]
    return magnitudes, freqs, fft_size


def find_spectrum_peaks(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """检测频谱峰值，返回前 N 个显著峰"""
    max_freq_idx = min(len(magnitudes), np.searchsorted(freqs, 20000))
    peaks_data = []
    mag_max = np.max(magnitudes[1:max_freq_idx]) if max_freq_idx > 1 else 1.0
    if mag_max == 0:
        mag_max = 1.0

    for i in range(2, min(max_freq_idx - 2, len(magnitudes))):
        mag = magnitudes[i]
        if (
            mag > magnitudes[i - 1]
            and mag > magnitudes[i - 2]
            and mag > magnitudes[i + 1]
            and mag > magnitudes[i + 2]
            and mag > 0.01 * mag_max
        ):
            freq = freqs[i]
            norm_mag = float(mag / mag_max)

            if freq < 300:
                label = "低频"
            elif freq < 1000:
                label = "中低频"
            elif freq < 4000:
                label = "中频"
            elif freq < 8000:
                label = "中高频"
            else:
                label = "高频"

            peaks_data.append({
                "frequency": float(freq),
                "magnitude": norm_mag,
                "label": label,
                "raw_magnitude": float(mag),
            })

    peaks_data.sort(key=lambda x: x["magnitude"], reverse=True)

    # 去重：同一 50Hz bucket 只保留最强
    seen = set()
    filtered = []
    for pk in peaks_data:
        bucket = round(pk["frequency"] / 50) * 50
        if bucket not in seen:
            seen.add(bucket)
            filtered.append(pk)
        if len(filtered) >= top_n:
            break

    return filtered


def detect_defects(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    peaks: list[dict],
    rms_level: float,
    peak_level: float,
    fft_size: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    """
    缺陷检测引擎：
    - 共振现象（高 Q 值峰）
    - 谐波失真（基频的整数倍）
    - 低频噪声本底
    - 削波失真（电平过高）
    - 互调失真（差频/和频）
    """
    defects: list[dict[str, Any]] = []
    freq_resolution = sample_rate / fft_size
    mag_max = np.max(magnitudes) or 1.0

    def _bin_at(freq):
        idx = int(round(freq / freq_resolution))
        return max(0, min(idx, len(magnitudes) - 1))

    def _mag_at(freq):
        return magnitudes[_bin_at(freq)] / mag_max

    def _bandwidth(center_bin, threshold_ratio=0.707):
        threshold = magnitudes[center_bin] * threshold_ratio
        left = center_bin
        right = center_bin
        while left > 0 and magnitudes[left] > threshold:
            left -= 1
        while right < len(magnitudes) - 1 and magnitudes[right] > threshold:
            right += 1
        return right - left

    # 1. 共振检测
    for pk in peaks:
        if pk["magnitude"] > 0.7:
            bin_idx = _bin_at(pk["frequency"])
            bw = _bandwidth(bin_idx)
            q = pk["frequency"] / (bw * freq_resolution + 1e-10)
            if q > 10:
                severity = "high" if pk["magnitude"] > 0.85 else "medium"
                defects.append({
                    "type": "共振现象",
                    "description": (
                        f"在 {pk['frequency']:.1f} Hz 处检测到高Q值共振峰 "
                        f"(Q≈{q:.1f})，可能由结构固有频率激发引起。"
                        f"建议检查该频率对应的机械部件是否存在松动或结构设计缺陷。"
                    ),
                    "severity": severity,
                    "frequency": pk["frequency"],
                })

    # 2. 谐波失真
    fundamentals = [p for p in peaks if p["magnitude"] > 0.3 and p["frequency"] < 2000]
    for fund in fundamentals:
        for harmonic in [2, 3, 4, 5]:
            hf = fund["frequency"] * harmonic
            hm = _mag_at(hf)
            if hm > 0.15 and harmonic < 4:
                severity = "high" if hm > 0.4 else "medium"
                defects.append({
                    "type": "谐波失真",
                    "description": (
                        f"检测到基频 {fund['frequency']:.1f} Hz 的第{harmonic}次谐波"
                        f"({hf:.1f} Hz)，幅度比 {(hm*100):.1f}%。"
                        f"可能存在非线性失真或部件装配缺陷。"
                    ),
                    "severity": severity,
                    "frequency": hf,
                })

    # 3. 低频噪声
    low_freq_end = min(20, len(magnitudes))
    avg_noise = np.mean(magnitudes[1:low_freq_end]) / mag_max
    if avg_noise > 0.1:
        severity = "high" if avg_noise > 0.3 else "low"
        defects.append({
            "type": "低频噪声",
            "description": (
                f"低频段(0-{low_freq_end * freq_resolution:.0f} Hz)存在较高噪声本底，"
                f"可能来源于电源干扰(50/60Hz)、机械振动或环境噪声。"
                f"建议排查电源质量和设备接地状况。"
            ),
            "severity": severity,
            "frequency": low_freq_end * freq_resolution,
        })

    # 4. 削波检测（使用峰值电平而非 RMS）
    # 满量程纯正弦波的 RMS = -3.01 dBFS，方波 RMS 可达 0 dBFS，
    # 因此 RMS 无法可靠检测削波。应看峰值电平是否逼近 0 dBFS。
    if peak_level >= -0.1:
        defects.append({
            "type": "信号削波",
            "description": (
                "音频信号电平过高，接近或超过0dBFS，存在削波失真。"
                "建议降低录音增益或增加系统动态余量。"
            ),
            "severity": "high",
            "frequency": 0,
        })

    # 5. 互调失真
    if len(peaks) >= 2:
        f1, f2 = peaks[0]["frequency"], peaks[1]["frequency"]
        for im_freq in [abs(f1 - f2), f1 + f2]:
            if 20 < im_freq < 20000:
                im_m = _mag_at(im_freq)
                if im_m > 0.1:
                    severity = "medium" if im_m > 0.3 else "low"
                    defects.append({
                        "type": "互调失真",
                        "description": (
                            f"在 {im_freq:.1f} Hz 处检测到互调产物"
                            f"(由 {f1:.0f} Hz 与 {f2:.0f} Hz 互调产生)，"
                            f"幅度比 {(im_m*100):.1f}%。"
                            f"可能由多个振动源耦合引起。"
                        ),
                        "severity": severity,
                        "frequency": im_freq,
                    })
    return defects


def compute_quality_score(
    rms_level: float, dynamic_range: float, defects: list[dict]
) -> int:
    """音质评分 0-100"""
    score = 100
    if rms_level > -3:
        score -= 30
    elif rms_level > -6:
        score -= 15
    elif rms_level < -40:
        score -= 10

    if dynamic_range < 10:
        score -= 20
    elif dynamic_range < 20:
        score -= 10

    for d in defects:
        if d["severity"] == "high":
            score -= 15
        elif d["severity"] == "medium":
            score -= 8
        else:
            score -= 3

    return max(0, min(100, score))


def get_quality_grade(score: int) -> str:
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "中等"
    elif score >= 60:
        return "合格"
    return "不合格"


# ────────────────────── 音频比对算法（机械噪声优化） ──────────────────────────
#
# 【给非声学专业工程师的导读】
#
# 本模块实现"噪声指纹比对"——判断两段音频是否来自同一个噪声源。
# 类比：就像指纹识别通过纹路特征判断"是不是同一个人"，噪声指纹通过
# 频谱特征判断"是不是同一台设备/同一个声源"。
#
# ● 声音 = 不同频率正弦波的叠加
#   任何声音都可以拆成许多不同频率的"纯音"（正弦波）叠加。
#   频谱图就是把声音拆开，横轴是频率（Hz），纵轴是该频率的强度。
#   → 低频（<500Hz）：电机转速、齿轮啮合、结构共振的能量区
#   → 中频（500-3000Hz）：人耳最敏感区，也是语音所在区
#   → 高频（>3000Hz）：轴承缺陷、气蚀、摩擦的高频特征
#
# ● 什么是"声纹/噪声指纹"？
#   一台电机以 1500rpm 运行，会在 25Hz（=1500/60）及其倍频（50、75、100Hz…）
#   处产生能量峰值——这就是这台电机的"指纹"。不同设备的指纹不同。
#   如果两段音频在相同频率处有相似的峰值结构，它们大概率来自同类声源。
#
# ● 本模块的比对流程（5 个维度）：
#   ┌─────────────┬────────────────────────────────────────────┬──────┐
#   │  维度        │  含义                                      │ 权重 │
#   ├─────────────┼────────────────────────────────────────────┼──────┤
#   │  LFCC       │  增强统计分布：均值+分位数+协方差+Wasserstein│ 15%  │
#   │  频谱特征   │  子带能量+质心+带宽+平坦度：整体能量分布     │ 20%  │
#   │  峰结构     │  抗漂移峰间距比+倍频关系：频率偏移鲁棒       │ 20%  │
#   │  谐波指纹   │  特征频率峰+谐波比：机械噪声最核心的识别指标 │ 25%  │
#   │  包络谱     │  Hilbert包络频谱：幅度调制节奏指纹           │ 20%  │
#   └─────────────┴────────────────────────────────────────────┴──────┘
#
# ● MFCC vs LFCC——为什么用 LFCC？
#   MFCC（梅尔频率倒谱系数）是语音识别的标准特征，它的滤波器组按人耳听觉
#   特性设计：在 300-3400Hz（人声区）频率分辨率最高，低频和高频则很稀疏。
#   这对语音识别很好，但对机械噪声极不利——机械噪声主能量常在 20-500Hz，
#   而 MFCC 在那个区域几乎不采样。
#   LFCC（线性频率倒谱系数）使用等间距滤波器，0-20kHz 全频段均匀采样，
#   不会遗漏任何频段的机械特征。
#


def compute_lfcc(
    samples: np.ndarray, sample_rate: int, n_lfcc: int = 13, n_fft: int = 2048
) -> np.ndarray:
    """
    计算线性频率倒谱系数 (LFCC) —— 机械噪声的"音色指纹"。

    【通俗解释】
    把声音想象成一束光，通过三棱镜（FFT）后分解成不同颜色（频率）的光。
    LFCC 就是量化这束光中每种颜色的强度模式，生成一组数字来代表"音色"。
    两段音色相同的噪声，其 LFCC 数字序列会很接近。

    与 MFCC 的关键区别：
    - MFCC 的"三棱镜"在语音区（300-3400Hz）分辨率最高，低频/高频粗糙
    - LFCC 的"三棱镜"全频段等间距，每个频率区间都有相同的分辨率
    → 电机振动（25-200Hz）、轴承异响（3-8kHz）在 LFCC 中都能被精确捕获

    【PCEN 增强】
    传统 LFCC 用 np.log(filter_output) 取对数，这只能压缩动态范围，
    无法补偿手机麦克风近场/远场导致的频谱倾斜（EQ 频响差异）。
    PCEN 对每个频率通道做指数移动平均归一化：
      PCEN(t,f) = (S(t,f) / (ε + M(t,f)))^α - δ)^r
    让同一设备在 1m 和 3m 处录到的频谱形状更一致。

    参数说明：
    - n_lfcc: 输出多少维系数。13维是常用值，前几维代表频谱包络（整体能量分布），
      后几维代表精细结构（谐波纹理）。维度越多描述越精细，但也越容易受噪声干扰。
    - n_fft: FFT 窗长（采样点数）。窗长越长→频率分辨率越细→能区分更接近的频率，
      但时间分辨率变差。2048 点 @44100Hz ≈ 46ms 窗宽，频率分辨率 ≈ 21.5Hz。

    返回: shape=(n_lfcc, n_frames) 的二维数组
    - 行: 13 个倒谱系数（从低频包络到高频纹理）
    - 列: 按时间顺序的帧（每帧约 23ms，帧间重叠 50%）
    """
    # ── 预加重 ──
    pre_emphasis = 0.5
    emphasized = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

    # ── 分帧 ──
    frame_len = n_fft
    hop_len = frame_len // 2
    n_frames = max(1, 1 + (len(emphasized) - frame_len) // hop_len)

    # ── 加窗 ──
    window = get_window("hann", frame_len)

    # ── 线性间距滤波器组 ──
    n_filters = 30
    low_bin = 0
    high_bin = n_fft // 2 + 1
    filter_edges = np.linspace(low_bin, high_bin, n_filters + 2).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left, center, right = filter_edges[i], filter_edges[i + 1], filter_edges[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fbank[i, j] = (right - j) / max(right - center, 1)

    # ── 阶段一：逐帧 FFT → 滤波器组 → 收集 filterbank 矩阵 ──
    # 必须先把所有帧的滤波器组输出收集成矩阵，才能做跨帧的倾斜消除
    filter_bank_matrix = np.zeros((n_filters, n_frames))

    for frame_idx in range(n_frames):
        start = frame_idx * hop_len
        end = min(start + frame_len, len(emphasized))
        frame = np.zeros(frame_len)
        frame[: end - start] = emphasized[start:end]
        frame *= window

        spectrum = np.abs(fft(frame))[: n_fft // 2 + 1]
        power_spec = (spectrum ** 2) / n_fft
        filter_output = np.dot(fbank, power_spec)
        filter_bank_matrix[:, frame_idx] = np.maximum(filter_output, 1e-10)

    # ── 阶段二：log 域频谱倾斜消除 ──
    # 传统 LFCC 用 np.log(filter_output)，但手机麦克风近场/远场会导致
    # 频谱整体倾斜（高频随距离衰减），使同一设备在不同距离录到的 LFCC 偏移。
    #
    # 【为什么不用 PCEN？】
    # PCEN 公式 (S/M)^α - δ)^r 设计初衷是"增强瞬态、抑制背景"（鸟类声学），
    # 对准稳态机械噪声：S ≈ M → S/M ≈ 1 → (1-δ) 为负 → 负数分数次幂 = NaN。
    # 即使修复 NaN，δ=2 也让稳态信号 PCEN 输出全零，LFCC 完全失去区分度。
    #
    # 【正确做法：log 域一阶倾斜消除】
    # 频谱倾斜在对数域表现为频率的线性函数（斜率 = 衰减系数），
    # 对每帧的 log 滤波器组输出拟合 1 阶多项式并减去，只保留非线性分量
    # （峰、谷、谐波结构）——这些才是设备特征，倾斜只是录音距离的伪影。
    log_filter_matrix = np.log(filter_bank_matrix)

    # 逐帧消除线性倾斜：fit y = a*x + b, 保留 residual = y - (a*x + b)
    freq_axis = np.arange(n_filters, dtype=float)
    for t in range(n_frames):
        coeffs = np.polyfit(freq_axis, log_filter_matrix[:, t], deg=1)
        log_filter_matrix[:, t] -= np.polyval(coeffs, freq_axis)

    # ── 阶段三：DCT → LFCC ──
    lfcc_features = np.zeros((n_lfcc, n_frames))
    dct_basis = np.array([
        np.cos(np.pi * k * (np.arange(n_filters) + 0.5) / n_filters)
        for k in range(n_lfcc)
    ]) * np.sqrt(2.0 / n_filters)

    for frame_idx in range(n_frames):
        lfcc_features[:, frame_idx] = dct_basis @ log_filter_matrix[:, frame_idx]

    return lfcc_features


def compute_harmonic_peaks(
    samples: np.ndarray, sample_rate: int, n_peaks: int = 10, min_prominence: float = 0.05,
    *,
    welch_freqs: np.ndarray | None = None, welch_power: np.ndarray | None = None,
) -> list[dict]:
    """
    检测频谱中的谐波峰值 —— 机械噪声的"DNA条形码"。

    【通俗解释】
    机械设备的运转部件（电机、齿轮、轴承、泵）都有固定的运动频率，
    这些频率在频谱上表现为"尖峰"——就像条形码的条纹。
    同一台设备在不同时间录音，条形码形状不变；不同设备则完全不同。

    举例：
    - 一台 4 极异步电机转速 1500rpm → 基频 = 1500/60 = 25Hz
    - 频谱上会出现 25Hz, 50Hz, 75Hz, 100Hz... 的尖峰（转频的倍频）
    - 齿轮啮合频率 = 齿数 × 转频，也在频谱上留下独立的峰
    - 这些峰的频率组合就是这台设备的"DNA"

    参数：
    - n_peaks: 最多返回多少个峰（按强度排序取前 N）
    - min_prominence: 峰的最小显著度（相对最大峰的比值），
      0.05 表示只保留强度 ≥ 最大峰 5% 的峰，过滤噪声伪峰
    - welch_freqs/welch_power: 可选的预计算 Welch 功率谱，
      传入后跳过内部 FFT 计算，避免 compute_noise_signature 重复运算

    返回: 按幅度降序排列的峰列表 [{freq, magnitude, label}, ...]
    """
    if welch_freqs is not None and welch_power is not None:
        freqs, power = welch_freqs, welch_power
    else:
        # 使用 scipy.signal.welch 计算全量信号的平均功率谱
        # 窗长 16384 点 @44100Hz → 频率分辨率 ≈ 2.7Hz，足以分辨机械特征频率
        from scipy.signal import welch as scipy_welch
        nperseg = min(len(samples), 16384)
        freqs, power = scipy_welch(
            samples, fs=sample_rate, window="hann", nperseg=nperseg,
            noverlap=nperseg // 2, scaling="spectrum",
        )

    # ── 局部峰值检测（scipy.signal.find_peaks + prominence）──
    # 传统手写 diff 循环找"全局最大值"会被低频风噪或高频电磁啸叫统治，
    # 掩盖真实的机械峰。find_peaks 的 prominence 参数衡量峰相对于
    # 局部基线（两侧最低点中的较高者）的高度，而非相对于全局最大值，
    # 因此能有效过滤风噪/啸叫造成的虚假宽峰，只保留真正突出的窄峰。
    from scipy.signal import find_peaks as scipy_find_peaks

    # prominence 的绝对阈值：功率谱最大值的 min_prominence 比例
    max_power = np.max(power) + 1e-10
    prom_threshold = min_prominence * max_power

    peak_indices, peak_props = scipy_find_peaks(
        power,
        prominence=prom_threshold,
        distance=max(1, int(5 / (freqs[1] - freqs[0] + 1e-10))),  # 最小间距 ~5Hz
    )

    if len(peak_indices) == 0:
        return []

    # 按 prominence 降序排序取前 N 个——最显著的峰最有特征价值
    prominences = peak_props["prominences"]
    sorted_order = np.argsort(prominences)[::-1][:n_peaks]

    peaks = []
    for order_idx in sorted_order:
        idx = peak_indices[order_idx]
        freq = freqs[idx]
        mag = power[idx] / max_power
        label = _freq_label(freq)
        peaks.append({"freq": float(freq), "magnitude": float(mag), "label": label})

    return peaks


def _freq_label(freq: float) -> str:
    """
    根据频率范围标注物理来源——帮助工程师理解"这个频率意味着什么"。

    【频段与噪声源的对应关系】
    ┌──────────────┬───────────┬──────────────────────────────────────────┐
    │  频段名称    │  频率范围  │  典型噪声源                               │
    ├──────────────┼───────────┼──────────────────────────────────────────┤
    │  次声波      │  <20 Hz   │  大型结构共振、地震波、风扇叶片通过频率     │
    │  低频振动    │  20-100   │  电机转频（如 25/30/50/60Hz）、泵脉冲       │
    │  低频机械    │  100-300  │  齿轮啮合低阶谐波、轴承外圈缺陷低阶通过频率 │
    │  中频机械    │  300-1k   │  齿轮啮合频率、轴承内圈缺陷频率             │
    │  语音区      │  1k-3k    │  人声基频及谐波（机械分析时应关注但不依赖）  │
    │  高频机械    │  3k-6k    │  轴承滚动体缺陷、气蚀噪声、电磁噪声谐波     │
    │  超高频      │  >6k Hz   │  摩擦/磨损金属声、高压泄漏、超声波频段      │
    └──────────────┴───────────┴──────────────────────────────────────────┘

    关键概念：机械噪声的频率与设备转速/结构参数直接相关：
    - 转频 = 转速(rpm) / 60    例：1500rpm → 25Hz
    - 齿轮啮合频率 = 齿数 × 转频   例：30齿×25Hz = 750Hz
    - 轴承缺陷频率 = (球径/节径) × 转频 × 球数（取决于轴承型号）
    """
    if freq < 20:
        return "次声波"
    elif freq < 100:
        return "低频振动"
    elif freq < 300:
        return "低频机械"
    elif freq < 1000:
        return "中频机械"
    elif freq < 3000:
        return "语音区"
    elif freq < 6000:
        return "高频机械"
    else:
        return "超高频"


@st.cache_data(hash_funcs={np.ndarray: lambda arr: arr.tobytes()}, show_spinner=False)
def compute_noise_signature(samples: np.ndarray, sample_rate: int) -> dict:
    """
    提取机械噪声的特征签名 —— 噪声源的"全身CT扫描"。

    【通俗解释】
    如果说 compute_harmonic_peaks 是拍"骨骼X光"（找关键频率峰），
    那 compute_noise_signature 就是做"全身CT"——全面扫描能量分布。
    它回答的问题是："这台设备的噪声能量主要集中在哪里？"

    与 compute_spectral_features（用于单音频分析）的区别：
    - 更多子带（10个 vs 6个），低频细分到 50Hz 粒度
    - 新增谐波频率比——如果基频25Hz有2×、3×倍频，比值就是 2.0、3.0
    - 专门为噪声源比对设计，不是通用频谱描述

    返回的特征字典包含：
    - band_energies: 10个频段的能量占比（加起来=1.0），类似"能量饼图"
    - harmonic_peaks: 谐波峰列表（来自 compute_harmonic_peaks）
    - harmonic_ratios: 基频与各谐波的频率比——同一设备的谐波比稳定
    - centroid/bandwidth/flatness: 频谱整体特征（质心=重心频率，带宽=分散程度）
    """
    # 使用 scipy.signal.welch 计算全量信号的平均功率谱
    # nperseg=16384 → 频率分辨率 ≈ 2.7Hz @44100Hz
    # noverlap=50% → Welch 法标准重叠率，兼顾频率分辨率和统计稳定性
    # scaling="spectrum" → 返回功率谱（V²），与旧代码行为一致
    from scipy.signal import welch as scipy_welch
    nperseg = min(len(samples), 16384)
    freqs, power = scipy_welch(
        samples, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, scaling="spectrum",
    )

    total_power = np.sum(power) + 1e-10

    # ── 10 频段子带能量分布 ──
    # 为什么低频要细分？因为机械噪声的"身份信息"主要在低频：
    #   50Hz 以下：大型设备的转频、电源频率（国内50Hz工频干扰）
    #   50-100Hz：常见电机转频的2倍频、泵的脉冲频率
    #   100-200Hz：齿轮啮合低阶、轴承外圈缺陷
    #   200-500Hz：中低速设备的啮合频率
    # 高频段粗分是因为：高频通常是低频的倍频/谐波的延伸，
    # 核心区分信息已在低频段捕获。
    band_defs = [
        ("0-50Hz 超低频", 0, 50),       # 电源工频、大型电机转频
        ("50-100Hz 低频A", 50, 100),     # 2倍电源频率、中速电机转频
        ("100-200Hz 低频B", 100, 200),   # 齿轮低阶啮合
        ("200-500Hz 中低频", 200, 500),  # 轴承外圈缺陷、中速啮合
        ("500-1kHz 中频A", 500, 1000),   # 高速齿轮啮合
        ("1k-2kHz 中频B", 1000, 2000),   # 轴承内圈缺陷
        ("2k-3kHz 语音区", 2000, 3000),  # 人声干扰区（需过滤）
        ("3k-5kHz 中高频", 3000, 5000),  # 轴承滚动体缺陷
        ("5k-8kHz 高频", 5000, 8000),    # 气蚀、电磁噪声
        ("8k+ 超高频", 8000, sample_rate // 2),  # 金属摩擦、超声波
    ]

    # 计算每个子带内的能量占总能量的比例
    # 类比：把噪声能量想象成一块蛋糕，10个频段各自切走一部分
    # 同类设备的"切法"相似，不同设备的"切法"不同
    band_energies = []
    band_names = []
    for name, lo, hi in band_defs:
        lo_idx = np.searchsorted(freqs, lo)
        hi_idx = np.searchsorted(freqs, hi)
        energy = np.sum(power[lo_idx:hi_idx]) / total_power
        band_energies.append(float(energy))
        band_names.append(name)

    # ── 谐波峰检测 ──
    # 直接传入已算好的 Welch 功率谱，避免重复 FFT
    harmonic_peaks = compute_harmonic_peaks(
        samples, sample_rate, n_peaks=10,
        welch_freqs=freqs, welch_power=power,
    )

    # ── 谐波频率比 ──
    # 基频（最大峰）与各谐波峰的频率比。
    # 例如：基频 25Hz，第二峰 50Hz → 比值 2.0；第三峰 75Hz → 比值 3.0
    # 关键洞察：同一台设备的谐波比是稳定的（由物理结构决定），
    # 即使录音环境、距离、增益变化，频率比不变。
    # 这就是谐波比作为"同源判断核心指标"的原因。
    harmonic_ratios = []
    if len(harmonic_peaks) >= 2:
        fundamental = harmonic_peaks[0]["freq"]
        if fundamental > 1:
            for p in harmonic_peaks[1:]:
                harmonic_ratios.append(round(p["freq"] / fundamental, 2))

    # ── 频谱整体特征 ──
    # 频谱质心：功率谱的"重心"频率。质心越高→高频能量越强→噪声越"尖锐"
    # 类比：质心就像一个跷跷板的支点，如果高频成分重，支点就往高频端移
    centroid = float(np.sum(freqs * power) / total_power)

    # 频谱带宽：质心周围的频率分散程度。带宽越大→频率分布越宽→噪声越"丰富"
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))

    # 频谱平坦度：几何平均/算术平均
    # → 接近1：各频率能量均匀（白噪声/宽带噪声）
    # → 接近0：能量集中在少数频率（纯音/窄带噪声，如电机嗡嗡声）
    # 类比：平坦度高的声音像"沙沙"声，低的像"嗡嗡"声
    geometric_mean = np.exp(np.mean(np.log(np.maximum(power, 1e-10))))
    arithmetic_mean = np.mean(power) + 1e-10
    flatness = float(geometric_mean / arithmetic_mean)

    return {
        "band_energies": band_energies,
        "band_names": band_names,
        "harmonic_peaks": harmonic_peaks,
        "harmonic_ratios": harmonic_ratios,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "flatness": flatness,
        "spectrum": power / (np.max(power) + 1e-10),
        "freqs": freqs,
    }


def remove_speech_band(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    智能去除语音——VAD 帧丢弃法：检测到语音的帧直接从时域丢弃，
    只保留无人声的帧拼接起来算特征。

    【为什么不使用带阻滤波？】
    之前的方案对语音帧做 300-3400Hz 带阻滤波，会在频域"挖洞"。
    如果音频A有人说话被挖了频域大洞，音频B没人说话没有洞，
    那么在计算子带能量、LFCC、包络谱时，音频A在 300-3400Hz 是真空的，
    对比相似度会暴跌。

    【帧丢弃法的优势】
    直接丢弃语音帧，保留的非语音帧频谱完整（300-3400Hz 的机械特征完好）。
    虽然时间不连续了，但对于稳态的机械频谱分析（Welch 法）来说，
    只要总时长足够，结果远比挖个频域大洞要准得多。

    【VAD 原理：短时能量语音活动检测】
    将信号分帧，计算每帧在 300-3400Hz 语音频带内的能量占比。
    如果某帧的语音频带能量显著高于全频带平均水平，判定为语音帧。
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = sample_rate / 2.0
    low_norm = 300.0 / nyquist
    high_norm = 3400.0 / nyquist

    # ── 步骤1：短时能量 VAD ──
    frame_len = int(0.03 * sample_rate)   # 30ms 帧长
    hop_len = frame_len // 2              # 50% 重叠
    n_frames = max(1, 1 + (len(samples) - frame_len) // hop_len)

    # 提取语音频带（300-3400Hz）用于 VAD
    sos_speech = butter(4, [low_norm, high_norm], btype='bandpass', output='sos')
    speech_band = sosfiltfilt(sos_speech, samples)

    # 计算每帧全频带和语音频带的 RMS
    frame_rms_full = np.zeros(n_frames)
    frame_rms_speech = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(samples))
        frame_rms_full[i] = np.sqrt(np.mean(samples[start:end] ** 2)) + 1e-10
        frame_rms_speech[i] = np.sqrt(np.mean(speech_band[start:end] ** 2)) + 1e-10

    # 语音活动判定：语音频带能量占全频带能量比例超过阈值
    speech_ratio = frame_rms_speech / frame_rms_full
    threshold = max(float(np.median(speech_ratio)) * 1.5, 0.35)
    is_speech = speech_ratio > threshold

    # 如果没有检测到语音帧，直接返回原信号
    if not np.any(is_speech):
        return samples.copy()

    # ── 步骤2：丢弃语音帧，只保留非语音帧拼接 ──
    non_speech_segments = []
    for i in range(n_frames):
        if not is_speech[i]:
            start = i * hop_len
            end = min(start + frame_len, len(samples))
            non_speech_segments.append(samples[start:end])

    if not non_speech_segments:
        # 所有帧都被判定为语音，返回原信号（避免空信号）
        return samples.copy()

    result = np.concatenate(non_speech_segments)

    # 安全检查：如果丢弃后时长太短（<0.2s），返回原信号
    if len(result) < int(0.2 * sample_rate):
        return samples.copy()

    return result


def compute_envelope_spectrum(
    samples: np.ndarray, sample_rate: int
) -> dict:
    """
    提取包络频谱特征——机械噪声的"低频节奏指纹"。

    【通俗解释】
    如果说 FFT 看的是"声音里有哪些频率成分"，
    那包络频谱看的是"声音的幅度在怎样波动"——即"节奏"。
    很多机械噪声有周期性的幅度调制（如电机转动导致的嗡嗡声振幅起伏），
    包络频谱能捕捉这种低频调制特征，与普通 FFT 形成互补。

    【频率范围 3-150Hz 的依据】
    包络谱（调制谱）的核心信息集中在极低频——转频及其倍频：
    - 典型电机转频：25Hz（1500rpm）、50Hz（3000rpm）
    - 轴承缺陷频率：通常在 5-100Hz
    - 齿轮啮合的幅度调制频率：通常 < 100Hz
    0-3Hz 被屏蔽（手持手机晃动的包络），>150Hz 主要是随机宽带噪声的包络，
    引入无用变量会稀释相似度得分。

    返回: {"envelope_spectrum": np.ndarray, "envelope_freqs": np.ndarray,
           "envelope_peaks": list[dict]}
    """
    from scipy.signal import hilbert, find_peaks

    # 希尔伯特变换取解析信号 → 取绝对值 = 包络
    analytic = hilbert(samples)
    envelope = np.abs(analytic)

    # 去均值后做 Welch 功率谱
    envelope_centered = envelope - np.mean(envelope)
    from scipy.signal import welch as scipy_welch
    nperseg = min(len(envelope_centered), 4096)
    freqs, power = scipy_welch(
        envelope_centered, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, scaling="spectrum",
    )

    # 只保留 3-150Hz 范围（核心调制信息，屏蔽超低频手持晃动和高频随机噪声）
    freq_mask = (freqs >= 3.0) & (freqs <= 150.0)
    env_freqs = freqs[freq_mask]
    env_power = power[freq_mask]

    # 归一化
    env_power = env_power / (np.max(env_power) + 1e-10)

    # 提取包络谱峰（调制峰结构）
    envelope_peaks = []
    if len(env_power) > 5:
        peak_indices, _ = find_peaks(
            env_power,
            height=0.15,           # 只取显著峰（归一化后 >15%）
            distance=max(1, int(2.0 / (env_freqs[1] - env_freqs[0] + 1e-10))),  # 最小峰距 2Hz
            prominence=0.05,
        )
        for idx in peak_indices:
            envelope_peaks.append({
                "freq": round(float(env_freqs[idx]), 2),
                "magnitude": round(float(env_power[idx]), 4),
            })

    return {
        "envelope_spectrum": env_power,
        "envelope_freqs": env_freqs,
        "envelope_peaks": envelope_peaks,
    }


def _compute_peak_structure_similarity(
    peaks_a: list[dict], peaks_b: list[dict]
) -> float:
    """
    峰结构相似度——抗频率漂移的频谱形状比对。

    【为什么余弦相似度不够？】
    同一台电机转速差 3%，所有特征频率整体偏移 3%。
    对余弦相似度来说，每个峰都"移位"了，得分会大幅下降。
    但实际上这是同一台设备！峰与峰之间的关系（间距比、倍频关系）没变。

    【本方法的策略】
    不比较频谱曲线本身，而是比较峰与峰之间的"关系结构"：
    1. 峰间距比（60%）：相邻峰之间的频率比是否一致
       例如 A 的峰间距比 [2.0, 1.5, 1.3]，B 的 [2.0, 1.5, 1.3] → 高度相似
       即使 A 的峰在 [25, 50, 75, 97.5] Hz，B 的在 [25.75, 51.5, 77.25, 100.4] Hz
    2. 倍频关系（40%）：谐波峰与基频的整数倍关系
       这部分与 harmonic_similarity 有重叠，但此处侧重"结构形态"而非精确匹配
    """
    if len(peaks_a) < 2 or len(peaks_b) < 2:
        return 0.0

    # ── 峰间距比 ──
    # 计算相邻峰之间的频率比（归一化，消除绝对频率偏移）
    def _peak_spacing_ratios(peaks):
        freqs = sorted([p["freq"] for p in peaks])
        if len(freqs) < 2:
            return []
        spacings = [freqs[i + 1] - freqs[i] for i in range(len(freqs) - 1)]
        # 归一化为比率：每个间距 / 最小间距
        min_sp = min(spacings) if spacings else 1
        if min_sp < 1:
            min_sp = 1
        return [round(s / min_sp, 2) for s in spacings]

    ratio_a = _peak_spacing_ratios(peaks_a)
    ratio_b = _peak_spacing_ratios(peaks_b)

    spacing_sim = 0.0
    if ratio_a and ratio_b:
        # 双向匹配
        def _one_way(src, tgt):
            scores = []
            for r in src:
                min_diff = min(abs(r - t) for t in tgt)
                scores.append(max(0, 1.0 - min_diff / max(r, 0.1)))
            return float(np.mean(scores)) if scores else 0.0
        spacing_sim = (_one_way(ratio_a, ratio_b) + _one_way(ratio_b, ratio_a)) / 2

    # ── 倍频关系 ──
    # 检测峰之间是否存在近似整数倍关系
    def _harmonic_structure(peaks):
        freqs = sorted([p["freq"] for p in peaks])
        if len(freqs) < 2 or freqs[0] < 1:
            return []
        fund = freqs[0]
        ratios = [round(f / fund, 1) for f in freqs[1:]]
        # 统计有多少是接近整数的
        integer_count = sum(1 for r in ratios if abs(r - round(r)) < 0.2)
        return ratios, integer_count / max(len(ratios), 1)

    harm_a, harm_b = _harmonic_structure(peaks_a), _harmonic_structure(peaks_b)
    harm_sim = 0.0
    if harm_a and harm_b:
        # 比较整数倍比例的相似度
        int_ratio_a = harm_a[1] if isinstance(harm_a, tuple) else 0
        int_ratio_b = harm_b[1] if isinstance(harm_b, tuple) else 0
        harm_sim = 1.0 - abs(int_ratio_a - int_ratio_b)
        # 也比较谐波比序列的相似度
        ratios_a = harm_a[0] if isinstance(harm_a, tuple) else []
        ratios_b = harm_b[0] if isinstance(harm_b, tuple) else []
        if ratios_a and ratios_b:
            def _one_way_ratio(src, tgt):
                scores = []
                for r in src:
                    min_diff = min(abs(r - t) for t in tgt)
                    scores.append(max(0, 1.0 - min_diff / 0.5))
                return float(np.mean(scores)) if scores else 0.0
            ratio_sim = (_one_way_ratio(ratios_a, ratios_b) + _one_way_ratio(ratios_b, ratios_a)) / 2
            harm_sim = 0.5 * harm_sim + 0.5 * ratio_sim

    return 0.6 * spacing_sim + 0.4 * harm_sim


def compute_confidence_score(
    samples: np.ndarray, sample_rate: int, harmonic_peaks: list[dict],
    feature_scores: list[float] | None = None,
) -> dict:
    """
    计算比对结果的置信度——"这次比对结果靠不靠谱？"

    【为什么需要置信度？】
    当音频太短、信噪比太差、人声太多或峰值太少时，
    相似度分数本身的统计意义就很弱，强行判别容易误导。
    置信度告诉用户："这个结论有多可靠"，低置信度时需谨慎参考。

    六个评估因子（各 0-1 分）：
    1. 时长充分性（20%）：音频是否足够长以提取稳定特征
    2. 信噪比（20%）：有效信号是否足够强
    3. 峰值充分性（15%）：检测到的特征峰数量是否足够
    4. 稳态程度（15%）：信号是否足够稳定（非瞬态/突发）
    5. 语音占比（10%）：人声干扰程度（占比越高越不可靠）
    6. 维度一致性（20%）：各维度得分是否一致（一致性惩罚）
       如果 std(feature_scores) > threshold → 置信度降低
       含义：各维度得分差异大 → 某些维度可能不可靠 → 整体结论不可信
    """
    duration = len(samples) / sample_rate

    # 1. 时长充分性：>=2s 满分，<0.5s 极低
    duration_score = min(duration / 2.0, 1.0)

    # 2. 信噪比：用信号 RMS 与静默段 RMS 的比值估计
    # 简化方法：信号 RMS / 底部 10% 分位段 RMS
    rms_total = np.sqrt(np.mean(samples ** 2)) + 1e-10
    frame_len = int(0.05 * sample_rate)
    n_frames = max(1, len(samples) // frame_len)
    frame_rms = np.array([
        np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2)) + 1e-10
        for i in range(n_frames)
    ])
    noise_floor = np.percentile(frame_rms, 10)
    snr_estimate = rms_total / max(noise_floor, 1e-10)
    # SNR > 20dB (≈10x) 满分，< 3dB (≈2x) 极低
    snr_score = min(max((snr_estimate - 2) / 8, 0), 1.0)

    # 3. 峰值充分性：>=5个峰满分，0个峰零分
    peak_score = min(len(harmonic_peaks) / 5.0, 1.0)

    # 4. 稳态程度：短时能量序列的变异系数（CV = std/mean）
    # CV 越小 → 信号越稳态；CV 越大 → 瞬态/突发越多
    if len(frame_rms) > 1:
        cv = np.std(frame_rms) / (np.mean(frame_rms) + 1e-10)
        # CV < 0.3 很稳态，> 1.0 很不稳定
        stationarity_score = max(0, min(1.0 - (cv - 0.3) / 0.7, 1.0))
    else:
        stationarity_score = 0.5

    # 5. 语音占比：300-3400Hz 能量占总能量的比例
    from scipy.signal import butter, sosfiltfilt
    nyquist = sample_rate / 2.0
    if nyquist > 3400:
        sos = butter(4, [300.0 / nyquist, 3400.0 / nyquist], btype='bandpass', output='sos')
        speech_band = sosfiltfilt(sos, samples)
        speech_ratio = np.sqrt(np.mean(speech_band ** 2)) / (rms_total + 1e-10)
    else:
        speech_ratio = 0.0
    # 语音占比 < 20% 很好，> 60% 很差
    speech_score = max(0, min(1.0 - (speech_ratio - 0.2) / 0.4, 1.0))

    # 6. 维度一致性惩罚（Consistency Penalty）
    # 如果各维度得分标准差太大，说明有些维度可能不可靠
    consistency_score = 1.0  # 默认满分
    if feature_scores and len(feature_scores) >= 3:
        score_std = float(np.std(feature_scores))
        # std < 0.10 → 各维度高度一致 → 满分
        # std > 0.30 → 维度间矛盾大 → 大幅扣分
        consistency_score = max(0, min(1.0 - (score_std - 0.10) / 0.20, 1.0))

    # 加权综合
    confidence = (
        0.20 * duration_score
        + 0.20 * snr_score
        + 0.15 * peak_score
        + 0.15 * stationarity_score
        + 0.10 * speech_score
        + 0.20 * consistency_score
    )
    confidence = max(0, min(1, confidence))

    return {
        "confidence": round(confidence, 3),
        "duration_score": round(duration_score, 3),
        "snr_score": round(snr_score, 3),
        "peak_score": round(peak_score, 3),
        "stationarity_score": round(stationarity_score, 3),
        "speech_score": round(speech_score, 3),
        "speech_ratio": round(float(speech_ratio), 3),
        "consistency_score": round(consistency_score, 3),
    }


def _compute_envelope_peak_similarity(
    env_peaks_a: list[dict], env_peaks_b: list[dict]
) -> float:
    """
    包络谱峰匹配——比较调制峰结构而非整体曲线形状。

    【为什么余弦相似度不够？】
    包络谱的余弦相似度会被背景能量、宽带噪声严重干扰。
    两段同源音频如果背景噪声水平不同，余弦得分会偏低。
    但它们的调制峰结构（峰频率、峰间距、倍频关系）应该高度一致。

    【本方法的策略】—— 类似 harmonic_peaks 的思路：
    1. 峰频率匹配（40%）：调制峰频率是否接近
    2. 峰间距比匹配（35%）：相邻峰之间的频率比是否一致（抗转速漂移）
    3. 倍频关系（25%）：峰之间是否存在近似整数倍关系
    """
    if len(env_peaks_a) < 1 or len(env_peaks_b) < 1:
        return 0.0

    freqs_a = [p["freq"] for p in env_peaks_a]
    freqs_b = [p["freq"] for p in env_peaks_b]

    # ── 峰频率匹配（双向） ──
    def _one_way_freq(src, tgt, tol_ratio=0.08):
        scores = []
        for f in src:
            min_diff = min(abs(f - t) for t in tgt)
            tolerance = max(f * tol_ratio, 1.5)  # 包络频率较低，容差 8% 或 1.5Hz
            scores.append(max(0, 1.0 - min_diff / tolerance))
        return float(np.mean(scores)) if scores else 0.0

    freq_sim = (_one_way_freq(freqs_a, freqs_b) + _one_way_freq(freqs_b, freqs_a)) / 2

    # ── 峰间距比匹配 ──
    def _spacing_ratios(freqs):
        if len(freqs) < 2:
            return []
        sorted_f = sorted(freqs)
        spacings = [sorted_f[i + 1] - sorted_f[i] for i in range(len(sorted_f) - 1)]
        min_sp = min(spacings) if spacings else 1
        if min_sp < 0.5:
            min_sp = 0.5
        return [round(s / min_sp, 2) for s in spacings]

    ratio_a = _spacing_ratios(freqs_a)
    ratio_b = _spacing_ratios(freqs_b)

    spacing_sim = 0.0
    if ratio_a and ratio_b:
        def _one_way_ratio(src, tgt):
            scores = []
            for r in src:
                min_diff = min(abs(r - t) for t in tgt)
                scores.append(max(0, 1.0 - min_diff / max(r, 0.1)))
            return float(np.mean(scores)) if scores else 0.0
        spacing_sim = (_one_way_ratio(ratio_a, ratio_b) + _one_way_ratio(ratio_b, ratio_a)) / 2

    # ── 倍频关系 ──
    def _harmonic_check(freqs):
        if len(freqs) < 2 or freqs[0] < 1:
            return 0.0
        fund = min(freqs)
        ratios = [f / fund for f in freqs[1:]]
        integer_count = sum(1 for r in ratios if abs(r - round(r)) < 0.25)
        return integer_count / max(len(ratios), 1)

    harm_a = _harmonic_check(freqs_a)
    harm_b = _harmonic_check(freqs_b)
    harm_sim = 1.0 - abs(harm_a - harm_b)

    return 0.40 * freq_sim + 0.35 * spacing_sim + 0.25 * harm_sim


def _compute_topk_anomaly_similarity(
    samples_a: np.ndarray, samples_b: np.ndarray, sample_rate: int, top_k: int = 5
) -> float:
    """
    Top-K 异常帧比对——捕捉"10% 关键异响"而非只看全局平均。

    【为什么需要 Top-K？】
    当前大量特征本质是全局统计平均，但机械音频经常是：90% 正常 + 10% 关键异响。
    全局平均会把异响淹没，两段含有相同异响的音频可能被判为不相似。

    【方法】
    对每段音频分帧，提取三种异常指标：
    1. RMS 最高帧——最响的瞬间
    2. 谱峭度最高帧——最"尖锐/突发"的帧
    3. 高频能量比最高帧——最"刺耳"的帧
    取每种指标的前 K 帧，对频谱取 dB 对数域，计算皮尔逊相关系数。

    【为什么用 dB + Pearson 而非线性功率 + 余弦？】
    线性功率谱值极度倾斜，一个强尖峰会统治余弦相似度得分，掩盖其他频带结构；
    手机不同的 EQ 在线性域产生不可逆扭曲。dB 域 + 皮尔逊自动中心化（去均值），
    能在数学上完美抵消手机麦克风带来的常数增益/减益（EQ 频响在对数域 = 加减常数）。
    """
    from scipy.signal import welch as scipy_welch

    frame_len = int(0.05 * sample_rate)  # 50ms 帧长
    hop_len = frame_len // 2
    n_fft = min(frame_len, 1024)

    def _extract_frame_features(samples):
        """分帧并提取 RMS、谱峭度、高频能量比、dB 频谱"""
        n_frames = max(1, 1 + (len(samples) - frame_len) // hop_len)
        frame_rms = np.zeros(n_frames)
        frame_kurtosis = np.zeros(n_frames)
        frame_hf_ratio = np.zeros(n_frames)
        frame_spectra_db = []

        for i in range(n_frames):
            start = i * hop_len
            end = min(start + frame_len, len(samples))
            frame = samples[start:end]

            # RMS
            rms = np.sqrt(np.mean(frame ** 2)) + 1e-10
            frame_rms[i] = rms

            # 简化谱峭度：帧能量的四阶矩 / 二阶矩的平方
            centered = frame - np.mean(frame)
            var = np.var(centered) + 1e-10
            kurt = np.mean(centered ** 4) / (var ** 2)
            frame_kurtosis[i] = kurt

            # Welch 功率谱
            freqs, power = scipy_welch(
                frame, fs=sample_rate, nperseg=min(len(frame), n_fft),
                noverlap=min(len(frame), n_fft) // 2, scaling="spectrum",
            )
            # 高频能量比：>4kHz 能量 / 全频带能量
            total_power = np.sum(power) + 1e-10
            hf_mask = freqs > 4000
            hf_power = np.sum(power[hf_mask]) + 1e-10
            frame_hf_ratio[i] = hf_power / total_power

            # dB 对数域频谱（用于皮尔逊相关系数计算）
            # 10 * log10(power) 将功率转到 dB 域，压缩动态范围
            power_db = 10.0 * np.log10(np.maximum(power, 1e-20))
            frame_spectra_db.append(power_db)

        return frame_rms, frame_kurtosis, frame_hf_ratio, frame_spectra_db, freqs

    rms_a, kurt_a, hf_a, spectra_db_a, freqs_a = _extract_frame_features(samples_a)
    rms_b, kurt_b, hf_b, spectra_db_b, freqs_b = _extract_frame_features(samples_b)

    # 对齐频谱长度
    min_len = min(len(freqs_a), len(freqs_b))

    def _topk_avg_db(rms, kurtosis, hf_ratio, spectra_db, k):
        """取三种指标 Top-K 帧的 dB 频谱，求均值"""
        selected_indices = set()

        # RMS Top-K
        top_rms = np.argsort(rms)[-k:]
        selected_indices.update(top_rms)

        # 谱峭度 Top-K
        top_kurt = np.argsort(kurtosis)[-k:]
        selected_indices.update(top_kurt)

        # 高频能量比 Top-K
        top_hf = np.argsort(hf_ratio)[-k:]
        selected_indices.update(top_hf)

        # 拼接所有选中帧的 dB 频谱
        spectra_list = []
        for idx in selected_indices:
            if idx < len(spectra_db):
                s = spectra_db[idx][:min_len]
                spectra_list.append(s)

        if not spectra_list:
            return None
        # 取均值 dB 频谱
        return np.mean(spectra_list, axis=0)

    topk_a = _topk_avg_db(rms_a, kurt_a, hf_a, spectra_db_a, top_k)
    topk_b = _topk_avg_db(rms_b, kurt_b, hf_b, spectra_db_b, top_k)

    if topk_a is None or topk_b is None:
        return 0.0

    # 皮尔逊相关系数（dB 域）
    # 皮尔逊 = 去均值后的余弦相似度，自动抵消手机 EQ 带来的常数增益/减益
    a_centered = topk_a - np.mean(topk_a)
    b_centered = topk_b - np.mean(topk_b)
    na, nb = np.linalg.norm(a_centered), np.linalg.norm(b_centered)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a_centered, b_centered) / (na * nb))


def apply_pcen(
    spectrogram: np.ndarray, sample_rate: int, hop_len: int,
    alpha: float = 0.98, delta: float = 2.0, r: float = 0.5,
) -> np.ndarray:
    """
    PCEN (Per-Channel Energy Normalization)——频谱倾斜补偿。

    【为什么需要 PCEN？】
    录音距离远近会改变频谱的"倾斜度"：近距离录音高频丰富，远距离高频衰减。
    普通归一化只调音量，不调频谱倾斜。PCEN 通过指数归一化，
    让每个频率通道的动态范围独立归一化，相当于"自动均衡"。

    【原理】
    PCEN(t, f) = (S(t, f) / (ε + M(t, f)))^α - δ)^r
    其中 M(t, f) 是该频率通道的指数移动平均（IIR 平滑），
    代表该通道的"背景水平"。除以背景水平后取幂，实现自适应增益控制。

    参数说明：
    - alpha: 控制增益压缩强度（0.95-0.99，越大压缩越强）
    - delta: 偏移量，防止对数域负值（通常 2-10）
    - r: 最终幂次，控制动态范围压缩（0.3-0.5）
    """
    _, n_frames = spectrogram.shape

    # IIR 平滑计算每个频率通道的背景水平 M(t, f)
    # M(t, f) = (1 - s) * M(t-1, f) + s * S(t, f)
    # 平滑系数 s 由时间常数决定
    time_constant = 0.5  # 500ms 时间常数
    s = 1.0 - np.exp(-hop_len / (time_constant * sample_rate))

    M = np.zeros_like(spectrogram)
    M[:, 0] = spectrogram[:, 0]
    for t in range(1, n_frames):
        M[:, t] = (1 - s) * M[:, t - 1] + s * spectrogram[:, t]

    # PCEN 公式
    eps = 1e-6
    pcen = (spectrogram / (eps + M)) ** alpha
    # 先 clip 再取幂：避免 (负值)^分数次幂 产生 NaN
    # 例如稳态信号 S/M≈1 → (1-δ)<0 → (-1)^0.5=NaN
    pcen = np.maximum(pcen - delta, 0)
    pcen = pcen ** r

    return pcen


def compute_audio_similarity(
    samples_a: np.ndarray, sr_a: int,
    samples_b: np.ndarray, sr_b: int,
    filter_speech: bool = False,
) -> dict:
    """
    计算两段音频的综合噪声指纹相似度 —— "两段录音是不是同一台设备发出的？"

    【整体流程图】
    ┌──────────────┐     ┌──────────────┐
    │  音频 A      │     │  音频 B      │
    └──────┬───────┘     └──────┬───────┘
           │  resample_poly→44100Hz  │  resample_poly→44100Hz
           │  (可选)VAD帧丢弃语音     │  (可选)VAD帧丢弃语音
           │  归一化幅度+PCEN补偿     │  归一化幅度+PCEN补偿
           ▼                     ▼
    ┌──────────────────────────────────────────┐
    │          六维度特征提取 & 比对              │
    │                                          │
    │  ① LFCC: 增强统计分布+分位数+Wasserstein (15%) │
    │  ② 频谱特征: PCEN补偿+子带能量+质心  (15%) │
    │  ③ 峰结构: 抗漂移的峰间距+倍频关系   (20%) │
    │  ④ 谐波指纹: 特征峰+谐波比比对      (25%) │
    │  ⑤ 包络谱: 3-150Hz峰匹配+曲线形状   (15%) │
    │  ⑥ 异常帧: Top-K RMS/峭度/高频帧    (10%) │
    │                                          │
    │       加权求和 → 综合相似度 0-100%        │
    │       + 一致性惩罚 → 置信度评估            │
    └──────────────────────────────────────────┘

    【为什么采样率统一到 44100Hz？】
    采样率决定能分析的最高频率（奈奎斯特定理：最高频率 = 采样率/2）。
    - 16kHz → 最高 8kHz，丢失 8kHz 以上的轴承/摩擦特征
    - 44100Hz → 最高 22kHz，覆盖整个可听范围 + 部分超声波
    机械异响（轴承外圈缺陷 5-12kHz、气蚀 8-16kHz）需要高频覆盖。

    【为什么幅度归一化？】
    录音时距离远近、麦克风增益都会影响音量，但同源噪声的"形状"不变。
    归一化消除音量差异，只比对频谱"形状"。类比：同一张照片亮度不同但内容一样。
    """
    TARGET_SR = 44100

    # ── 重采样：使用 resample_poly 替代 resample，减少频域伪影 ──
    # scipy.signal.resample 使用 FFT-based 方法，在信号首尾可能产生振铃伪影
    # resample_poly 基于多相滤波，抗混叠效果更好，频域更干净
    from math import gcd

    def resample(samples, sr_from, sr_to):
        """使用 scipy.signal.resample_poly 进行抗混叠重采样"""
        if sr_from == sr_to:
            return samples
        # 计算上/下采样因子（互质化）
        g = gcd(sr_to, sr_from)
        up = sr_to // g
        down = sr_from // g
        target_len = int(len(samples) * up / down)
        if target_len < 100:
            return samples
        from scipy.signal import resample_poly as scipy_resample_poly
        return scipy_resample_poly(samples, up, down)

    a = resample(samples_a, sr_a, TARGET_SR)
    b = resample(samples_b, sr_b, TARGET_SR)

    # 可选：智能语音过滤（VAD 帧丢弃法——丢弃语音帧，保留非语音帧）
    low_energy_warning = False
    if filter_speech:
        len_before_a, len_before_b = len(a), len(b)
        a = remove_speech_band(a, TARGET_SR)
        b = remove_speech_band(b, TARGET_SR)
        # 帧丢弃后时长可能大幅缩短
        if len_before_a > 1000 and len(a) / len_before_a < 0.1:
            low_energy_warning = True
        if len_before_b > 1000 and len(b) / len_before_b < 0.1:
            low_energy_warning = True

    # 归一化幅度到 [-1, 1]，消除录音音量差异的影响
    a = a / (np.max(np.abs(a)) + 1e-10)
    b = b / (np.max(np.abs(b)) + 1e-10)

    # ════════════════════════════════════════════════════════
    # 频谱倾斜补偿（已在 compute_lfcc 内部集成）
    # ════════════════════════════════════════════════════════
    # 录音距离不同会导致频谱倾斜（近场高频丰富，远场高频衰减）。
    # compute_lfcc 内部已将传统 np.log(filter_output) 替换为
    # log 域一阶倾斜消除：
    #   1. 先收集所有帧的滤波器组输出矩阵 (n_filters × n_frames)
    #   2. 取对数后，对每帧拟合 1 阶多项式并减去倾斜分量
    #   3. 保留非线性分量（峰、谷、谐波结构）再做 DCT → LFCC
    # 这比 PCEN 更适合机械噪声：PCEN 设计用于增强瞬态/抑制背景，
    # 对准稳态信号 S≈M 会导致 NaN 或全零输出。
    lfcc_a = compute_lfcc(a, TARGET_SR)
    lfcc_b = compute_lfcc(b, TARGET_SR)

    # ════════════════════════════════════════════════════════
    # 维度 1：LFCC 增强统计分布相似度（15%）
    # ════════════════════════════════════════════════════════
    # 除了均值和标准差，新增：
    # - 分位数统计（P10, P50, P90）：捕捉分布的偏态和尾部特征
    # - 协方差结构：反映 LFCC 各维度之间的耦合关系
    # - Wasserstein 距离：度量两个分布之间的"搬运成本"
    # 均值和标准差
    mean_a, std_a = np.mean(lfcc_a, axis=1), np.std(lfcc_a, axis=1)
    mean_b, std_b = np.mean(lfcc_b, axis=1), np.std(lfcc_b, axis=1)

    # 均值向量余弦相似度
    norm_ma, norm_mb = np.linalg.norm(mean_a), np.linalg.norm(mean_b)
    mean_sim = float(np.dot(mean_a, mean_b) / (norm_ma * norm_mb)) if norm_ma > 1e-10 and norm_mb > 1e-10 else 0.0

    # 标准差向量余弦相似度
    norm_sa, norm_sb = np.linalg.norm(std_a), np.linalg.norm(std_b)
    std_sim = float(np.dot(std_a, std_b) / (norm_sa * norm_sb)) if norm_sa > 1e-10 and norm_sb > 1e-10 else 0.0

    # 分位数统计（每维系数的 P10, P50, P90）
    q10_a, q50_a, q90_a = np.percentile(lfcc_a, [10, 50, 90], axis=1)
    q10_b, q50_b, q90_b = np.percentile(lfcc_b, [10, 50, 90], axis=1)
    # 分位数向量的余弦相似度
    def _cosine_sim(va, vb):
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(np.dot(va, vb) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0
    q10_sim = _cosine_sim(q10_a, q10_b)
    q50_sim = _cosine_sim(q50_a, q50_b)
    q90_sim = _cosine_sim(q90_a, q90_b)
    quantile_sim = (q10_sim + q50_sim + q90_sim) / 3

    # 协方差结构（上三角展平后的余弦相似度）
    n_coef = lfcc_a.shape[0]
    if n_coef >= 2:
        cov_a = np.cov(lfcc_a)
        cov_b = np.cov(lfcc_b)
        triu_idx = np.triu_indices(n_coef, k=1)
        cov_vec_a = cov_a[triu_idx]
        cov_vec_b = cov_b[triu_idx]
        cov_sim = _cosine_sim(cov_vec_a, cov_vec_b)
    else:
        cov_sim = 0.0

    # Wasserstein 距离（每维系数取平均，转为相似度）
    from scipy.stats import wasserstein_distance
    wd_per_dim = []
    for dim in range(min(lfcc_a.shape[0], lfcc_b.shape[0])):
        wd = wasserstein_distance(lfcc_a[dim], lfcc_b[dim])
        wd_per_dim.append(wd)
    mean_wd = float(np.mean(wd_per_dim)) if wd_per_dim else 1.0
    wasserstein_sim = max(0, 1.0 - mean_wd / 5.0)

    # 综合五项 LFCC 统计
    lfcc_similarity = (
        0.25 * max(mean_sim, 0)
        + 0.15 * max(std_sim, 0)
        + 0.20 * max(quantile_sim, 0)
        + 0.15 * max(cov_sim, 0)
        + 0.25 * max(wasserstein_sim, 0)
    )

    # ════════════════════════════════════════════════════════
    # 维度 2：频谱特征相似度（15%）
    # ════════════════════════════════════════════════════════
    sig_a = compute_noise_signature(a, TARGET_SR)
    sig_b = compute_noise_signature(b, TARGET_SR)

    be_a = np.array(sig_a["band_energies"])
    be_b = np.array(sig_b["band_energies"])

    # PCEN 频谱倾斜补偿：对子带能量做对数域归一化
    # 相当于对每个子带的能量做"相对于自身平均"的归一化，
    # 消除近场/远场导致的整体能量倾斜
    def _pcen_band_normalize(band_energies):
        """对子带能量做简化的 PCEN 归一化"""
        be_log = np.log1p(band_energies * 1000)  # 对数压缩
        be_mean = np.mean(be_log) + 1e-10
        return np.clip((be_log / be_mean - 1.0) * 2 + 0.5, 0, 1)

    be_a_norm = _pcen_band_normalize(be_a)
    be_b_norm = _pcen_band_normalize(be_b)
    band_sim = float(np.dot(be_a_norm, be_b_norm) / (np.linalg.norm(be_a_norm) * np.linalg.norm(be_b_norm) + 1e-10))

    def scalar_sim(va, vb):
        denom = max(abs(va), abs(vb), 1e-10)
        return 1.0 - min(abs(va - vb) / denom, 1.0)

    centroid_sim = scalar_sim(sig_a["centroid"], sig_b["centroid"])
    bandwidth_sim = scalar_sim(sig_a["bandwidth"], sig_b["bandwidth"])
    flatness_sim = scalar_sim(sig_a["flatness"], sig_b["flatness"])

    spectral_similarity = float(np.mean([band_sim, centroid_sim, bandwidth_sim, flatness_sim]))

    # ════════════════════════════════════════════════════════
    # 维度 3：峰结构相似度（20%）—— 抗频率漂移
    # ════════════════════════════════════════════════════════
    peaks_a = sig_a["harmonic_peaks"]
    peaks_b = sig_b["harmonic_peaks"]
    peak_structure_similarity = _compute_peak_structure_similarity(peaks_a, peaks_b)

    # ════════════════════════════════════════════════════════
    # 维度 4：谐波指纹相似度（25%）—— 机械噪声最核心的识别指标
    # ════════════════════════════════════════════════════════
    harmonic_similarity = _compute_harmonic_similarity(peaks_a, peaks_b)

    # ════════════════════════════════════════════════════════
    # 维度 5：包络谱相似度（15%）—— 幅度调制的"节奏指纹"
    # ════════════════════════════════════════════════════════
    # 包络谱范围：3-150Hz（核心调制信息，屏蔽手持晃动 0-3Hz 和高频噪声 >150Hz）
    # 比较方法：包络峰匹配（峰频率+峰间距+倍频关系），而非余弦相似度
    env_a = compute_envelope_spectrum(a, TARGET_SR)
    env_b = compute_envelope_spectrum(b, TARGET_SR)

    # 包络峰匹配（主方法）
    env_peaks_a = env_a.get("envelope_peaks", [])
    env_peaks_b = env_b.get("envelope_peaks", [])
    envelope_peak_sim = _compute_envelope_peak_similarity(env_peaks_a, env_peaks_b)

    # 包络曲线余弦相似度（辅助方法，权重较低）
    env_spec_a = env_a["envelope_spectrum"]
    env_spec_b = env_b["envelope_spectrum"]
    min_env_len = min(len(env_spec_a), len(env_spec_b))
    if min_env_len > 1:
        envelope_curve_sim = float(
            np.dot(env_spec_a[:min_env_len], env_spec_b[:min_env_len])
            / (np.linalg.norm(env_spec_a[:min_env_len]) * np.linalg.norm(env_spec_b[:min_env_len]) + 1e-10)
        )
    else:
        envelope_curve_sim = 0.0

    # 综合包络相似度：峰匹配为主（70%），曲线形状为辅（30%）
    if env_peaks_a and env_peaks_b:
        envelope_similarity = 0.70 * envelope_peak_sim + 0.30 * envelope_curve_sim
    else:
        # 没有包络峰时退化为纯曲线相似度
        envelope_similarity = envelope_curve_sim

    # ════════════════════════════════════════════════════════
    # 维度 6：Top-K 异常帧相似度（10%）—— 捕捉关键异响
    # ════════════════════════════════════════════════════════
    # 机械音频经常是 90% 正常 + 10% 关键异响，全局平均会淹没异响。
    # 提取 RMS 最高帧、谱峭度最高帧、高频能量最高帧，单独比较。
    topk_similarity = _compute_topk_anomaly_similarity(a, b, TARGET_SR, top_k=5)

    # ════════════════════════════════════════════════════════
    # 综合加权评分（六维）
    # ════════════════════════════════════════════════════════
    # 权重分配逻辑：
    # - 谐波指纹 25%：最可靠的机械特征
    # - 峰结构 20%：抗漂移的频谱形状比对
    # - LFCC 15%：增强统计分布，补充音色信息
    # - 频谱特征 15%：子带能量+标量特征（PCEN 补偿后）
    # - 包络谱 15%：包络峰匹配为主，幅度调制节奏指纹
    # - Top-K 异常 10%：捕捉关键异响，补充全局平均的盲区
    feature_scores = [
        max(lfcc_similarity, 0),
        max(spectral_similarity, 0),
        max(peak_structure_similarity, 0),
        max(harmonic_similarity, 0),
        max(envelope_similarity, 0),
        max(topk_similarity, 0),
    ]
    overall = (
        0.15 * max(lfcc_similarity, 0)
        + 0.15 * max(spectral_similarity, 0)
        + 0.20 * max(peak_structure_similarity, 0)
        + 0.25 * max(harmonic_similarity, 0)
        + 0.15 * max(envelope_similarity, 0)
        + 0.10 * max(topk_similarity, 0)
    )
    overall_pct = max(0, min(100, overall * 100))

    # ════════════════════════════════════════════════════════
    # 置信度评估（含维度一致性惩罚）
    # ════════════════════════════════════════════════════════
    conf_a = compute_confidence_score(a, TARGET_SR, peaks_a, feature_scores)
    conf_b = compute_confidence_score(b, TARGET_SR, peaks_b, feature_scores)
    # 取两段音频置信度的较低值（木桶效应）
    confidence = min(conf_a["confidence"], conf_b["confidence"])

    # ── 同源判断阈值 ──
    low_confidence = confidence < 0.4
    consistency_penalty = min(conf_a.get("consistency_score", 1.0), conf_b.get("consistency_score", 1.0))
    if overall_pct >= 75:
        conclusion = "高度同源"
        conclusion_detail = "六维噪声指纹高度吻合，极大概率来自同一噪声源或具有相同的振动机理。"
    elif overall_pct >= 65:
        conclusion = "可能同源"
        conclusion_detail = "多数特征匹配但部分维度存在差异，可能来自同一类噪声源或相似机理，需结合其他证据进一步确认。"
    elif overall_pct >= 45:
        conclusion = "相似度低"
        conclusion_detail = "个别特征有相似之处但整体差异较大，两段音频大概率不是来自同一噪声源。"
    else:
        conclusion = "大概率不同源"
        conclusion_detail = "六维噪声指纹差异显著，来自不同噪声源的可能性很大。"

    if low_confidence:
        conclusion_detail += " ⚠️ 置信度较低（音频质量/时长/信噪比不足），结论仅供参考，建议补充更高质量的录音。"
    if consistency_penalty < 0.5:
        conclusion_detail += " ⚠️ 维度一致性低（各特征维度得分差异较大），可能存在录音条件不一致，结论需谨慎参考。"

    return {
        "lfcc_similarity": round(max(lfcc_similarity * 100, 0), 1),
        "spectral_similarity": round(max(spectral_similarity * 100, 0), 1),
        "peak_structure_similarity": round(max(peak_structure_similarity * 100, 0), 1),
        "harmonic_similarity": round(max(harmonic_similarity * 100, 0), 1),
        "envelope_similarity": round(max(envelope_similarity * 100, 0), 1),
        "topk_similarity": round(max(topk_similarity * 100, 0), 1),
        "overall_similarity": round(overall_pct, 1),
        "conclusion": conclusion,
        "conclusion_detail": conclusion_detail,
        "lfcc_a": lfcc_a,
        "lfcc_b": lfcc_b,
        "sig_a": sig_a,
        "sig_b": sig_b,
        "env_a": env_a,
        "env_b": env_b,
        "target_sr": TARGET_SR,
        "len_a": len(a),
        "len_b": len(b),
        "duration_a": len(a) / TARGET_SR,
        "duration_b": len(b) / TARGET_SR,
        "filter_speech": filter_speech,
        "low_energy_warning": low_energy_warning,
        "confidence": round(confidence, 3),
        "consistency_penalty": round(consistency_penalty, 3),
        "conf_a": conf_a,
        "conf_b": conf_b,
    }


def _compute_harmonic_similarity(peaks_a: list[dict], peaks_b: list[dict]) -> float:
    """
    计算两个谐波峰集合的相似度 —— "两台设备的条形码是否匹配"。

    【比对策略——两层匹配】
    第1层：频率匹配（60% 权重）
      对 A 的每个峰，在 B 中找频率最接近的峰。如果差距在 5% 以内认为匹配。
      类比：两台设备的频谱像两个条形码，逐条扫描看有多少条纹位置重合。

    第2层：谐波比匹配（40% 权重）
      比较基频与各谐波的频率倍数关系。例如 A 的谐波比 [2.0, 3.0, 4.0]，
      B 的谐波比 [2.0, 3.1, 4.1]，则匹配度很高。
      类比：不管设备转速快慢（基频不同），只要"倍频模式"一样就是同源。
      → 这是为什么两台不同转速的同型电机也能被识别为同源的关键。

    为什么频率匹配权重更高？
      频率匹配验证了"绝对位置"，谐波比验证了"相对结构"。
      同源噪声通常两者都匹配，所以 60/40 的分配已足够。
    """
    if not peaks_a or not peaks_b:
        return 0.0

    # 提取频率列表
    freqs_a = np.array([p["freq"] for p in peaks_a])
    freqs_b = np.array([p["freq"] for p in peaks_b])

    # ── 第1层：频率匹配（双向对称） ──
    # 分别从 A→B 和 B→A 两个方向做匹配，取平均，
    # 消除"峰多的一方遍历时多出无匹配峰"导致的不对称。
    def _one_way_freq_match(src_freqs, tgt_freqs):
        scores = []
        for f in src_freqs:
            min_diff = np.min(np.abs(tgt_freqs - f))
            tolerance = max(f * 0.05, 10)
            if min_diff < tolerance:
                scores.append(1.0 - min_diff / tolerance)
            else:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    freq_match = (_one_way_freq_match(freqs_a, freqs_b)
                  + _one_way_freq_match(freqs_b, freqs_a)) / 2

    # ── 第2层：谐波比匹配 ──
    # 谐波比 = 各峰频率 / 基频，例如 [1.0, 2.0, 3.0, 4.5]
    # 同型设备即使转速不同，谐波比也近似（由物理结构决定）
    ratio_a, ratio_b = [], []
    if len(peaks_a) >= 2:
        fund_a = peaks_a[0]["freq"]
        if fund_a > 1:
            ratio_a = [round(p["freq"] / fund_a, 1) for p in peaks_a[1:6]]
    if len(peaks_b) >= 2:
        fund_b = peaks_b[0]["freq"]
        if fund_b > 1:
            ratio_b = [round(p["freq"] / fund_b, 1) for p in peaks_b[1:6]]

    ratio_match = 0.0
    if ratio_a and ratio_b:
        # 双向对称匹配：A→B 和 B→A 各算一次，取平均
        def _one_way_ratio_match(src, tgt):
            scores = []
            for r in src:
                min_diff = min(abs(r - rb) for rb in tgt)
                scores.append(max(0, 1.0 - min_diff / 0.3))
            return float(np.mean(scores)) if scores else 0.0

        ratio_match = (_one_way_ratio_match(ratio_a, ratio_b)
                       + _one_way_ratio_match(ratio_b, ratio_a)) / 2

    # 频率匹配 60% + 谐波比匹配 40%
    # 如果没有谐波比信息（单峰），退化为纯频率匹配
    if ratio_match > 0:
        return 0.6 * freq_match + 0.4 * ratio_match
    else:
        return freq_match


# ────────────────────── 可视化生成（Plotly） ──────────────────────────


def plot_waveform(
    samples: np.ndarray, sample_rate: int, title: str = "时域波形"
) -> go.Figure:
    """绘制包络波形图，返回 Plotly Figure"""
    duration = samples.shape[0] / sample_rate
    abs_max = np.max(np.abs(samples)) or 0.01

    # 包络降采样
    num_cols = 600
    step = max(1, len(samples) // num_cols)
    cols = []
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step]
        if len(chunk) > 0:
            cols.append((np.min(chunk), np.max(chunk)))

    xs = np.linspace(0, duration, len(cols))
    uppers = [c[1] / abs_max for c in cols]
    lowers = [c[0] / abs_max for c in cols]

    fig = go.Figure()
    # 填充区域
    fig.add_trace(go.Scatter(
        x=list(xs) + list(xs[::-1]),
        y=uppers + lowers[::-1],
        fill="toself",
        fillcolor="rgba(59,130,246,0.30)",
        line=dict(color="rgba(59,130,246,0)", width=0),
        hoverinfo="skip",
        showlegend=False,
    ))
    # 上包络
    fig.add_trace(go.Scatter(
        x=xs, y=uppers, mode="lines",
        line=dict(color="#60a5fa", width=1.2),
        name="上包络", hovertemplate="时间: %{x:.3f}s<br>幅度: %{y:.3f}<extra></extra>",
    ))
    # 下包络
    fig.add_trace(go.Scatter(
        x=xs, y=lowers, mode="lines",
        line=dict(color="#60a5fa", width=1.2),
        name="下包络", hovertemplate="时间: %{x:.3f}s<br>幅度: %{y:.3f}<extra></extra>",
    ))
    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        title=dict(text=title, x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="时间 (s)",
        yaxis_title="幅度",
        xaxis_range=[0, duration],
        yaxis_range=[-1.05, 1.05],
        height=260,
        margin=dict(l=50, r=20, t=40, b=45),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
        xaxis=dict(showgrid=False, zerolinecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
        showlegend=False,
    )
    return fig


def plot_spectrum(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    sample_rate: int,
    title: str = "频率谱",
) -> go.Figure:
    """绘制频率谱柱状图，返回 Plotly Figure"""
    max_freq = min(8000, sample_rate // 2)
    cutoff = min(len(magnitudes), np.searchsorted(freqs, max_freq))
    mags = magnitudes[:cutoff]
    frqs = freqs[:cutoff]

    norm = np.max(mags) or 1.0
    bar_colors = []
    for m in mags:
        r = m / norm
        if r > 0.7:
            bar_colors.append("#ef4444")
        elif r > 0.4:
            bar_colors.append("#f59e0b")
        elif r > 0.2:
            bar_colors.append("#4f46e5")
        else:
            bar_colors.append("#93c5fd")

    fig = go.Figure(data=[
        go.Bar(
            x=frqs, y=mags,
            marker_color=bar_colors,
            marker_opacity=0.85,
            marker_line_width=0,
            hovertemplate="频率: %{x:.1f} Hz<br>幅度: %{y:.4f}<extra></extra>",
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="频率 (Hz)",
        yaxis_title="幅度",
        xaxis_range=[0, max_freq],
        height=300,
        margin=dict(l=50, r=20, t=40, b=45),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
        xaxis=dict(showgrid=False, zerolinecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
        bargap=0,
        showlegend=False,
    )
    return fig


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int,
    title: str = "STFT 时频谱图",
) -> go.Figure:
    """使用 scipy.signal.stft 绘制时频谱图，返回 Plotly Figure"""
    hop = max(fft_size // 4, 64)
    win = "hann"
    nperseg = fft_size

    f_vals, t_vals, Zxx = scipy_stft(
        samples, fs=sample_rate, window=win, nperseg=nperseg, noverlap=nperseg - hop
    )

    power = np.abs(Zxx)
    log_power = np.log10(power * 1000 + 1) / 3

    max_freq = min(8000, sample_rate // 2)
    freq_mask = f_vals <= max_freq

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=log_power[freq_mask, :][::-1],  # 翻转Y轴使低频在底部
        x=t_vals,
        y=f_vals[freq_mask][::-1],
        colorscale=[[0, "#000000"], [0.25, "#8B0000"], [0.5, "#FF4500"], [0.75, "#FFD700"], [1, "#FFFF00"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title=dict(text="幅度 (dB)", side="right", font_size=10), thickness=12, len=0.92,
                     tickfont=dict(size=8)),
        hovertemplate="时间: %{x:.3f}s<br>频率: %{y:.1f} Hz<br>幅度: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="时间 (s)",
        yaxis_title="频率 (Hz)",
        height=360,
        margin=dict(l=55, r=35, t=40, b=45),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig





# ────────────────────── 比对可视化 ──────────────────────────


def plot_lfcc_heatmap(lfcc: np.ndarray, _sample_rate: int, title: str = "LFCC") -> go.Figure:
    """绘制 LFCC 热力图（线性频率倒谱系数，适合机械噪声分析）"""
    fig = go.Figure(go.Heatmap(
        z=lfcc,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="系数值", thickness=10, len=0.9),
        hovertemplate="帧: %{x}<br>LFCC序号: %{y}<br>值: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="帧序号",
        yaxis_title="LFCC 系数",
        height=280,
        margin=dict(l=60, r=25, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
    )
    return fig


def plot_spectrum_overlay(feat_a: dict, feat_b: dict, name_a: str, name_b: str) -> go.Figure:
    """叠加绘制两段音频的归一化频谱"""
    spec_a, freqs_a = feat_a["spectrum"], feat_a["freqs"]
    spec_b, freqs_b = feat_b["spectrum"], feat_b["freqs"]

    max_freq = min(16000, min(freqs_a[-1], freqs_b[-1]))
    mask_a = freqs_a <= max_freq
    mask_b = freqs_b <= max_freq

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs_a[mask_a], y=spec_a[mask_a], mode="lines",
        line=dict(color="#4f46e5", width=1.5), name=name_a[:20],
        hovertemplate="%{x:.1f} Hz | 幅度: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=freqs_b[mask_b], y=spec_b[mask_b], mode="lines",
        line=dict(color="#f59e0b", width=1.5), name=name_b[:20],
        hovertemplate="%{x:.1f} Hz | 幅度: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="频谱叠加对比", x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="频率 (Hz)",
        yaxis_title="归一化幅度",
        height=300,
        margin=dict(l=50, r=20, t=40, b=45),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
        xaxis=dict(showgrid=False, zerolinecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return fig


def plot_similarity_radar(comp: dict) -> go.Figure:
    """绘制多维度相似度雷达图"""
    categories = ["LFCC", "频谱特征", "峰结构", "谐波指纹", "包络谱", "异常帧", "综合相似度"]
    values = [
        comp["lfcc_similarity"],
        comp["spectral_similarity"],
        comp["peak_structure_similarity"],
        comp["harmonic_similarity"],
        comp["envelope_similarity"],
        comp.get("topk_similarity", 0),
        comp["overall_similarity"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(79,70,229,0.15)",
        line=dict(color="#4f46e5", width=2),
        marker=dict(size=6, color="#4f46e5"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
        ),
        height=360,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="white",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=11, color="#475569"),
        showlegend=False,
    )
    return fig


def plot_band_energy_comparison(sig_a: dict, sig_b: dict, name_a: str, name_b: str) -> go.Figure:
    """绘制两段音频的子带能量分布对比柱状图"""
    names_a = sig_a["band_names"]
    energies_a = sig_a["band_energies"]
    energies_b = sig_b["band_energies"]

    # 简化标签
    short_labels = [n.split(" ", 1)[-1] if " " in n else n for n in names_a]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a[:20],
        x=short_labels,
        y=energies_a,
        marker_color="#4f46e5",
        marker_opacity=0.8,
        hovertemplate="%{x}<br>能量占比: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=name_b[:20],
        x=short_labels,
        y=energies_b,
        marker_color="#f59e0b",
        marker_opacity=0.8,
        hovertemplate="%{x}<br>能量占比: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="子带能量分布对比", x=1, xanchor="right", font_size=13, font_color="#334155"),
        xaxis_title="频段",
        yaxis_title="能量占比",
        yaxis_tickformat=".0%",
        height=350,
        margin=dict(l=50, r=20, t=40, b=60),
        paper_bgcolor="white",
        plot_bgcolor="#fafbff",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", size=10, color="#475569"),
        xaxis=dict(showgrid=False, zerolinecolor="#e2e8f0", tickangle=30),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return fig


# ──────────────────────── AI 分析 ────────────────────────────


def generate_ai_analysis(
    file_name: str,
    sample_rate: int,
    duration: float,
    channels: int,
    rms_level: float,
    peak_level: float,
    dynamic_range: float,
    peaks: list[dict],
    defects: list[dict],
    quality_score: int,
    api_key: str,
) -> str:
    """调用 DeepSeek API 进行 AI 智能分析"""
    if not api_key or len(api_key.strip()) < 10:
        return _fallback_analysis(
            file_name, sample_rate, duration, rms_level, dynamic_range,
            peaks, defects, quality_score,
        )

    peaks_info = "、".join([
        f"{p['frequency']:.1f}Hz({p['label']},幅度{(p['magnitude']*100):.1f}%)"
        for p in peaks
    ])

    defects_info = "\n".join([
        f"{d['type']}: {d['description']} [严重度:{d['severity']},频率:{d['frequency']:.1f}Hz]"
        for d in defects
    ]) or "未检测到明显缺陷"

    prompt = f"""你是一位资深的声学与振动分析专家。请根据以下音频频谱分析数据，提供专业、详细的智能分析报告。

当前日期: {datetime.now().strftime("%Y年%m月%d日")}
注意：报告中如需填写日期，请使用当前日期，不要编造日期。

## 音频基本信息
- 文件名: {file_name}
- 采样率: {sample_rate} Hz
- 时长: {duration:.2f} 秒
- 声道数: {channels}
- RMS电平: {rms_level:.2f} dB
- 峰值电平: {peak_level:.2f} dB
- 动态范围: {dynamic_range:.2f} dB
- 音质评分: {quality_score}/100

## 频谱峰值
{peaks_info}

## 检测到的缺陷
{defects_info}

请从以下角度进行详细分析：

### 1. 信号特征解读
分析频谱峰值的含义，解释各个频率成分可能的来源和物理意义。

### 2. 共振与异响分析
针对检测到的共振现象或异常频率，分析其可能的产生机理。

### 3. 振动与噪声源定位
基于频谱特征，推断可能的振动源和噪声源位置及传播路径。

### 4. 部件装配缺陷评估
评估是否存在因装配不当导致的异常振动特征。

### 5. 音质评估与改进建议
综合评价音频信号质量，给出具体的改进措施和优化方向。

请使用专业但易于理解的语言，条理清晰，逻辑严谨。"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
        )
        return response.choices[0].message.content or "AI分析返回为空"
    except ImportError:
        # fallback to requests
        import urllib.request
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 2048,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get("choices", [{}])[0].get("message", {}).get("content", "AI返回空")
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 错误: {e}")
    except Exception as e:
        raise RuntimeError(f"AI分析失败: {e}")


def _fallback_analysis(
    file_name, sample_rate, duration, rms_level, dynamic_range, peaks, defects, quality_score,
) -> str:
    """无 API Key 时的模板化分析"""
    peaks_str = "、".join(f"{p['frequency']:.1f}Hz" for p in peaks)

    has_resonance = any(d["type"] == "共振现象" for d in defects)
    has_low_noise = any(d["type"] == "低频噪声" for d in defects)
    has_distortion = any(d["type"] in ("谐波失真", "互调失真") for d in defects)

    return f"""## 音频分析报告 - {file_name}

### 1. 信号特征解读
该音频采样率{sample_rate}Hz，时长{duration:.2f}秒，RMS电平{rms_level:.2f}dB，
动态范围{dynamic_range:.2f}dB。主要频率成分分布在{peaks_str}，
频谱能量分布{'较广' if len(peaks) > 3 else '集中'}。

### 2. 共振与异响分析
{'检测到共振现象，表现为特定频率处的窄带能量集中，可能由结构固有频率与激励频率耦合导致。' if has_resonance else '未检测到明显的共振现象，频谱分布较为均匀。'}

### 3. 振动与噪声源定位
{'低频段存在噪声本底升高，可能来源于电源干扰(50/60Hz)或机械振动。' if has_low_noise else '噪声本底水平正常，无明显异常噪声源。'}

### 4. 部件装配缺陷评估
{'检测到谐波或互调失真，可能暗示部件装配存在间隙或不平衡。建议进一步检查运动部件的对中性和紧固状态。' if has_distortion else '未发现明显的装配缺陷特征。'}

### 5. 音质评估
综合评分: {quality_score}/100。{'音频质量良好。' if quality_score >= 80 else '音频质量一般，存在改进空间。' if quality_score >= 60 else '音频质量较差，建议重新录制或处理。'}"""


# ──────────────────────── HTML 报告生成 ────────────────────────


def generate_report_html(result: dict) -> str:
    """生成完整 HTML 报告（内嵌 Plotly 交互式图表，无需 kaleido）"""

    def fig_to_html_div(fig) -> str:
        """将 Plotly Figure 转为独立 HTML div（含 Plotly.js CDN）"""
        try:
            return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})
        except Exception:
            return "<p style='color:#94a3b8;text-align:center;padding:20px;'>图表渲染失败</p>"

    def sev_cn(s):
        return {"high": "严重", "medium": "中等", "low": "轻微"}[s]

    def sev_color(s):
        return {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}[s]

    # 将 Plotly Figure 转为内嵌 HTML（无需 kaleido）
    _wf = fig_to_html_div(result["waveform_fig"])
    _sf = fig_to_html_div(result["spectrum_fig"])
    _stf = fig_to_html_div(result["spectrogram_fig"])

    defect_rows = "\n".join([
        f"""<tr>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:{sev_color(d['severity'])};font-weight:600">{sev_cn(d['severity'])}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b">{d['type']}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">{d['frequency']:.1f} Hz</td>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b;font-size:13px">{d['description']}</td>
    </tr>"""
        for d in result["defects"]
    ])

    peak_rows = "\n".join([
        f"""<tr>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b">{p['frequency']:.1f} Hz</td>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">{(p['magnitude']*100):.1f}%</td>
      <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">{p['label']}</td>
    </tr>"""
        for p in result["spectrum_peaks"]
    ])

    score_color = "#059669" if result["quality_score"] >= 80 else ("#d97706" if result["quality_score"] >= 60 else "#dc2626")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>音频频谱分析报告 - {result['file_name']}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#f8fafc; color:#1e293b; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; line-height:1.6; padding:24px; max-width:860px; margin:0 auto; }}
h1 {{ font-size:26px; color:#0f172a; margin-bottom:6px; letter-spacing:-0.5px; }}
h2 {{ font-size:17px; color:#4f46e5; margin:28px 0 14px; border-left:3px solid #4f46e5; padding-left:14px; }}
.meta {{ color:#64748b; font-size:13px; margin-bottom:24px; padding:8px 0; border-bottom:1px solid #e2e8f0; }}
.card {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:22px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
.score {{ font-size:52px; font-weight:bold; }}
.grade {{ font-size:22px; color:#0f172a; font-weight:600; }}
.stat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:18px; }}
.stat-item {{ background:#f8fafc; padding:16px; border-radius:10px; text-align:center; border:1px solid #f1f5f9; }}
.stat-label {{ font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }}
.stat-value {{ font-size:18px; color:#0f172a; font-weight:600; }}
.chart-img {{ width:100%; border-radius:10px; display:block; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ padding:12px 14px; text-align:left; border-bottom:2px solid #e2e8f0; color:#64748b; font-size:13px; font-weight:600; background:#f8fafc; }}
td {{ padding:10px 14px; border-bottom:1px solid #f1f5f9; }}
.ai-content {{ white-space:pre-wrap; line-height:1.85; color:#334155; font-size:14px; }}
.footer {{ text-align:center; color:#94a3b8; font-size:12px; margin-top:36px; padding-top:18px; border-top:1px solid #e2e8f0; }}
</style>
</head>
<body>
<h1>🎵 音频频谱分析报告</h1>
<div class="meta">📁 文件: {result['file_name']} &nbsp;|&nbsp; 🔈 采样率: {result['sample_rate']}Hz &nbsp;|&nbsp; ⏱️ 时长: {result['duration']:.2f}s &nbsp;|&nbsp; 📅 {now_str}</div>

<div class="card">
<h2>📊 音质评估</h2>
<div style="display:flex;align-items:center;gap:24px;margin-bottom:18px;">
  <div class="score" style="color:{score_color}">{result['quality_score']}</div>
  <div><div class="grade">{result['quality_grade']}</div><div style="color:#94a3b8;font-size:13px;">综合音质评级</div></div>
</div>
<div class="stat-grid">
  <div class="stat-item"><div class="stat-label">RMS 电平</div><div class="stat-value">{result['rms_level']:.2f} dB</div></div>
  <div class="stat-item"><div class="stat-label">峰值电平</div><div class="stat-value">{result['peak_level']:.2f} dB</div></div>
  <div class="stat-item"><div class="stat-label">动态范围</div><div class="stat-value">{result['dynamic_range']:.2f} dB</div></div>
  <div class="stat-item"><div class="stat-label">检测缺陷</div><div class="stat-value">{len(result['defects'])} 项</div></div>
</div>
</div>

<div class="card">
<h2>📈 时域波形（包络）</h2>
{_wf}
</div>

<div class="card">
<h2>🔊 频率谱 (FFT)</h2>
{_sf}
</div>

<div class="card">
<h2>🌡️ STFT 时频谱图</h2>
{_stf}
<p style="margin-top:8px;color:#64748b;font-size:12px;">
算法: Hann窗短时傅里叶变换 | 窗长={result.get('fft_size','N/A')} | 重叠率≈75% | 对数压缩
</p>
</div>

<div class="card">
<h2>🔍 频谱峰值</h2>
<table>
<thead><tr><th>频率</th><th>归一化幅度</th><th>频段分类</th></tr></thead>
<tbody>{peak_rows}</tbody>
</table>
</div>

<div class="card">
<h2>⚠️ 缺陷诊断</h2>
{f'<table><thead><tr><th>严重度</th><th>类型</th><th>频率</th><th>描述</th></tr></thead><tbody>{defect_rows}</tbody></table>' if result['defects'] else '<p style="color:#10b981;text-align:center;padding:24px;">✅ 未检测到异常，音频信号质量良好</p>'}
</div>

<div class="card">
<h2>🤖 AI 智能分析</h2>
<div class="ai-content">{result['ai_analysis']}</div>
</div>

<div class="footer">音频频谱分析系统 (Streamlit版) | 分析报告由 DeepSeek AI 辅助生成</div>
</body>
</html>"""


# ──────────────────────── 主分析流程 ────────────────────────


def run_full_analysis(
    file_bytes: bytes, file_name: str, api_key: str = "", progress=None
) -> dict:
    """执行完整分析流程，返回结果字典"""

    if progress:
        progress(5, "正在解码音频文件...")

    samples, sample_rate = try_decode_audio(file_bytes, file_name)
    duration = float(samples.shape[0]) / sample_rate
    channels = 1

    if progress:
        progress(25, "计算 RMS / 峰值...")

    rms_level = compute_rms(samples)
    peak_level = compute_peak(samples)
    dynamic_range = peak_level - rms_level

    if progress:
        progress(40, "执行 FFT 频谱分析...")

    magnitudes, freqs, fft_size = perform_fft_analysis(samples, sample_rate)
    peaks = find_spectrum_peaks(magnitudes, freqs)

    if progress:
        progress(55, "运行缺陷检测引擎...")

    defects = detect_defects(magnitudes, freqs, peaks, rms_level, peak_level, fft_size, sample_rate)
    quality_score = compute_quality_score(rms_level, dynamic_range, defects)
    quality_grade = get_quality_grade(quality_score)

    if progress:
        progress(68, "生成可视化图表...")

    waveform_img = plot_waveform(samples, sample_rate)
    spectrum_img = plot_spectrum(magnitudes, freqs, sample_rate)
    spectrogram_img = plot_spectrogram(samples, sample_rate, fft_size)

    if progress:
        progress(80, "调用 AI 分析模型...")

    ai_text = ""
    try:
        ai_text = generate_ai_analysis(
            file_name, sample_rate, duration, channels,
            rms_level, peak_level, dynamic_range,
            peaks, defects, quality_score, api_key,
        )
    except Exception as e:
        st.warning(f"AI分析失败: {e}，已降级为模板分析")
        ai_text = _fallback_analysis(
            file_name, sample_rate, duration, rms_level, dynamic_range,
            peaks, defects, quality_score,
        )

    if progress:
        progress(95, "组装报告...")

    result = {
        "id": str(uuid.uuid4()),
        "file_name": file_name,
        "file_size": len(file_bytes),
        "sample_rate": sample_rate,
        "duration": duration,
        "channels": channels,
        "rms_level": rms_level,
        "peak_level": peak_level,
        "dynamic_range": dynamic_range,
        "fft_size": fft_size,
        "spectrum_peaks": peaks,
        "defects": defects,
        "quality_score": quality_score,
        "quality_grade": quality_grade,
        "waveform_fig": waveform_img,
        "spectrum_fig": spectrum_img,
        "spectrogram_fig": spectrogram_img,
        "ai_analysis": ai_text,
        "report_html": None,  # 延迟到下载时生成（避免 kaleido 慢）
    }

    if progress:
        progress(100, "完成")

    return result


# ──────────────────────── Streamlit UI ────────────────────────


def main():
    # ════════════════════ 侧边栏 ════════════════════
    with st.sidebar:
        # Logo 区域
        st.markdown("""
        <div style='text-align:center;margin-bottom:6px;'>
            <div style="
                display:inline-flex;align-items:center;justify-content:center;
                width:48px;height:48px;border-radius:14px;
                background:linear-gradient(135deg,#6366f1,#8b5cf6);
                font-size:24px;box-shadow:0 4px 16px rgba(99,102,241,0.35);
            ">🎵</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="text-align:center;font-weight:700;font-size:15px;color:#1e293b;letter-spacing:-0.3px;">Audio Analyzer</p>', unsafe_allow_html=True)
        st.caption('专业级音频频谱分析系统')

        st.markdown("---")

        # API Key 配置
        st.markdown("**🔑 AI 接口**")
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-xxxxx...",
            help="platform.deepseek.com → API Keys",
        )
        _key_ok = bool(api_key and len(api_key.strip()) > 10)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:6px;padding:6px 12px;
                    border-radius:8px;background:{'rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2)' if _key_ok else 'rgba(250,204,21,0.08);border:1px solid rgba(250,204,21,0.2)'};margin-top:4px;'>
            <span>{'✅ 已就绪' if _key_ok else '⚠️ 未配置'}</span>
            <span style='color:#64748b;font-size:11px;'>{'— DeepSeek 分析可用' if _key_ok else '— 将使用规则模板'}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🗑️ 清除所有缓存", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.uploaded_file = None
            st.session_state.analysis_history = []
            st.session_state.comparison_result = None
            st.session_state.comp_report_html = None
            st.success("缓存已清除!")
            st.rerun()

        st.markdown("---")
        st.markdown("""<details>
<summary style='cursor:pointer;color:#94a3b8;font-size:13px;font-weight:500;'>&#x2139; 使用说明</summary>
<div style='padding:10px 4px;color:#64748b;font-size:13px;line-height:2;'>
1. 选择功能标签页（单音频分析 / 双音频比对）<br/>
2. 上传音频文件<br/>
3. 点击操作按钮<br/>
4. 查看结果与可视化图表<br/>
5. 下载 HTML 报告<br/><br/>
<b>支持格式:</b> WAV / MP3 / OGG / FLAC / AAC / M4A / MP4<br/>
<b>文件大小:</b> 最大 50 MB
</div></details>""", unsafe_allow_html=True)

    # ════════════════════ 主区域 Header ════════════════════
    st.markdown("""
    <div style='text-align:center;padding:18px 0 4px;'>
        <h1 style='font-size:28px;margin-bottom:4px;color:#0f172a;letter-spacing:-0.5px;'>
            🎵 音频频谱分析系统
        </h1>
        <p style='color:#94a3b8;font-size:13px;letter-spacing:0.3px;'>
            STFT &middot; FFT &middot; 缺陷检测 &middot; 噪声指纹比对 &middot; AI 诊断 &middot; 一键导出
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════ 主 Tab 导航 ════════════════════
    tab_analyze, tab_compare = st.tabs(["📊 单音频分析", "🔗 双音频比对"])

    # ════════════════════ Tab 1：单音频分析 ════════════════════
    with tab_analyze:
        # ── 上传区卡片 ──
        st.markdown("""
        <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
            padding:24px;margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>
                <span style='font-size:20px;'>📤</span>
                <span style='font-size:16px;font-weight:700;color:#0f172a;'>上传音频文件</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p style="color:#d97706;font-size:13px;margin-bottom:2px;">'
            '💡 若上传 MP4 等视频格式，建议先前往 '
            '<a href="https://online-audio-converter.com" target="_blank" style="color:#4f46e5;text-decoration:underline;">'
            'online-audio-converter.com</a> 转换为 WAV 格式后再上传，分析效果更佳。</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "**拖拽或点击上传**",
            type=["wav", "mp3", "ogg", "flac", "aac", "m4a", "mp4"],
            help="支持 MP3 / WAV / OGG / FLAC / AAC / M4A，最大 50MB",
        )

        # 操作按钮行
        c_left, c_right = st.columns([1.2, 3])

        with c_left:
            analyze_clicked = st.button(
                "▶  开始分析",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_file,
            )

        with c_right:
            _has_result = bool(st.session_state.analysis_result)
            if _has_result:
                result = st.session_state.analysis_result
                if not result.get("report_html"):
                    with st.spinner("正在生成 HTML 报告..."):
                        result["report_html"] = generate_report_html(result)
                safe_name = (
                    re.sub(r"[^\w\u4e00-\u9fff_-]", "_", Path(result["file_name"]).stem)
                    + "_音频分析报告_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".html"
                )
                st.download_button(
                    label="⬇  导出 HTML 报告",
                    data=result["report_html"].encode("utf-8"),
                    file_name=safe_name,
                    mime="text/html;charset=utf-8",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button("⬇  导出 HTML 报告", disabled=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)  # 关闭上传区卡片

        # ── 执行分析 ──
        if analyze_clicked and uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            progress_bar = st.progress(0, "准备分析...")
            status = st.empty()

            def on_progress(pct, msg):
                progress_bar.progress(int(pct), msg)

            try:
                result = run_full_analysis(file_bytes, file_name, api_key.strip(), on_progress)
                st.session_state.analysis_result = result
                st.session_state.uploaded_file = uploaded_file
                # 记录历史
                history_entry = {
                    "name": file_name,
                    "size": len(file_bytes),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "score": result["quality_score"],
                    "grade": result["quality_grade"],
                }
                st.session_state.analysis_history.insert(0, history_entry)
                st.session_state.analysis_history = st.session_state.analysis_history[:20]

                progress_bar.empty()
                status.success("✅ 分析完成！向下滚动查看详细结果")
            except Exception as e:
                status.error(f"❌ 分析失败: {e}")
                st.exception(e)

        # ── 结果展示区 ──
        result = st.session_state.analysis_result
        if result:
            _old_keys = ("waveform_img", "spectrum_img", "spectrogram_img")
            if any(_k in result for _k in _old_keys) and "waveform_fig" not in result:
                st.session_state.analysis_result = None
                st.warning("⚠️ 检测到旧版本分析结果缓存，请重新上传文件并分析")
            else:
                render_results(result)

    # ════════════════════ Tab 2：双音频比对 ════════════════════
    with tab_compare:
        # ── 上传区卡片 ──
        st.markdown("""
        <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
            padding:24px;margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>
                <span style='font-size:20px;'>📤</span>
                <span style='font-size:16px;font-weight:700;color:#0f172a;'>上传两段音频进行比对</span>
            </div>
        """, unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            file_a = st.file_uploader(
                "🎵 音频 A",
                key="compare_file_a",
                type=["wav", "mp3", "ogg", "flac", "aac", "m4a", "mp4"],
            )
        with comp_col2:
            file_b = st.file_uploader(
                "🎵 音频 B",
                key="compare_file_b",
                type=["wav", "mp3", "ogg", "flac", "aac", "m4a", "mp4"],
            )

        # 语音过滤选项
        filter_speech = st.checkbox(
            "🔇 过滤语音频带（300-3400Hz）",
            value=False,
            help="勾选后会去除人声频段的能量，使比对更专注于机械噪声特征。适合音频中混有人声干扰的场景。",
        )

        # 比对按钮
        compare_clicked = st.button(
            "🔬  开始比对",
            type="primary",
            use_container_width=True,
            disabled=not (file_a and file_b),
        )

        st.markdown("</div>", unsafe_allow_html=True)  # 关闭上传区卡片

        # ── 知识小贴士 ──
        with st.expander("📖 噪声指纹比对入门 — 什么是'同源'？", expanded=False):
            st.markdown("""
**🔍 什么叫做"来自同一噪声源"？**

同一台电机在不同时间、不同距离录制的声音，虽然听起来有差别（远近不同、环境噪声不同），
但它的"特征频率组合"——即频谱上的峰值位置和谐波比例——是稳定不变的。
就像同一个人的指纹不会因为换了环境就变化一样，这就是"噪声指纹"。

---

**🧬 五维比对原理**

| 维度 | 权重 | 比什么 | 通俗理解 |
|------|------|--------|----------|
| LFCC | 15% | 增强统计分布（均值+分位数+协方差+Wasserstein） | 对比"全身体检报告" |
| 频谱特征 | 20% | 质心、带宽、平坦度等统计量 | 比对"整体体型指标" |
| 峰结构 | 20% | 峰间距比 + 倍频关系（抗频率漂移） | 比对"旋律音程关系" |
| 谐波指纹 | 25% | 特征峰位置 + 谐波频率比 | 比对"DNA 条形码" |
| 包络谱 | 20% | Hilbert 包络频谱（幅度调制节奏） | 比对"鼓点节奏" |

---

**🎯 相似度怎么看？**

| 范围 | 结论 | 含义 |
|------|------|------|
| ≥ 75% | 高度同源 | 五维特征高度吻合，极大概率是同一台设备 |
| 65-75% | 可能同源 | 多数特征匹配，需结合其他信息进一步判断 |
| 45-65% | 相似度低 | 个别特征相似但整体差异大，大概率不同源 |
| < 45% | 大概率不同源 | 四维特征差异显著，不是同一设备 |

---

**🔇 何时开启"过滤语音频带"？**

当音频中混有人声对话时（如车间内有人说话），人声能量集中在 300-3400Hz，
会"遮盖"该频段的机械特征。勾选后会用频域遮罩将此频段"抹去"再比对，
使结果更专注于机械噪声本身。如果音频中没有人声，则无需开启。
        """)

        # ── 执行比对 ──
        if compare_clicked and file_a and file_b:
            with st.spinner("正在提取噪声指纹并计算相似度..."):
                try:
                    bytes_a, bytes_b = file_a.getvalue(), file_b.getvalue()
                    name_a, name_b = file_a.name, file_b.name

                    samples_a, sr_a = try_decode_audio(bytes_a, name_a)
                    samples_b, sr_b = try_decode_audio(bytes_b, name_b)

                    comp = compute_audio_similarity(samples_a, sr_a, samples_b, sr_b, filter_speech=filter_speech)
                    # 保存到 session_state，供导出报告使用
                    st.session_state.comparison_result = comp
                    st.session_state.comparison_name_a = name_a
                    st.session_state.comparison_name_b = name_b
                    st.session_state.comp_report_html = None  # 清除旧报告缓存
                    render_comparison(comp, name_a, name_b)
                except Exception as e:
                    st.error(f"❌ 比对失败: {e}")

        # ── 恢复上次比对结果（页面 rerun 时） ──
        _cached_comp = st.session_state.get("comparison_result")
        if _cached_comp and not compare_clicked:
            # 防御：旧版缓存可能缺少新字段，检测到则清除
            _required_keys = ("peak_structure_similarity", "envelope_similarity", "confidence", "topk_similarity")
            if any(k not in _cached_comp for k in _required_keys):
                st.session_state.comparison_result = None
            else:
                render_comparison(
                    _cached_comp,
                    st.session_state.get("comparison_name_a", ""),
                    st.session_state.get("comparison_name_b", ""),
                )


def render_results(result: dict):
    """渲染分析结果面板（卡片式布局）"""

    # ════ 评分大卡片 ════
    sc = result["quality_score"]
    if sc >= 80:
        score_color, score_bg = "#059669", "#ecfdf5"
    elif sc >= 60:
        score_color, score_bg = "#d97706", "#fffbeb"
    else:
        score_color, score_bg = "#dc2626", "#fef2f2"

    _html = f"""<div style='background:{score_bg};border:1.5px solid {score_color}30;border-radius:18px;
        padding:28px;text-align:center;margin:0 0 24px;box-shadow:0 2px 12px rgba(0,0,0,0.05);'>
      <div style='font-size:56px;font-weight:800;color:{score_color};letter-spacing:-2px;line-height:1;'>
          {sc}<span style='font-size:26px;font-weight:600;color:#94a3b8;'>/100</span>
      </div>
      <div style='font-size:22px;font-weight:700;color:#0f172a;margin-top:6px;'>{result['quality_grade']}</div>
      <div style='color:#475569;font-size:13px;margin-top:10px;display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;'>
          <span>🔊 采样率 {result['sample_rate']}Hz</span><span style='color:#cbd5e1;'>|</span>
          <span>⏱️ 时长 {result['duration']:.2f}s</span><span style='color:#cbd5e1;'>|</span>
          <span>📁 {result['file_name']}</span>
      </div>
    </div>"""
    st.markdown(_html, unsafe_allow_html=True)

    # ════ 统计指标行 ════
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("RMS 电平", f"{result['rms_level']:.1f} dB", delta_color="off",
                 help="均方根电平 — 反映音频信号的整体能量/响度。值越大声音越响，正常语音约 -20~-10dB，音乐约 -15~-5dB。0dBFS 为满幅上限。")
    with c2:
        st.metric("峰值电平", f"{result['peak_level']:.1f} dB", delta_color="off",
                 help="最大振幅电平 — 采样点中的绝对最大值。接近 0dBFS 时可能存在削波失真（波形顶部被截断），一般应保留至少 3dB 余量。")
    with c3:
        st.metric("动态范围", f"{result['dynamic_range']:.1f} dB", delta_color="off",
                 help='峰值与 RMS 的差值 — 衡量信号的起伏幅度。大动态范围说明有明显的强弱对比；过小说明信号比较"平"，缺乏层次感。')
    with c4:
        st.metric("检测缺陷", f"{len(result['defects'])} 项", delta_color="off")

    # ════ 声学知识小贴士 ════
    with st.expander("📖 声学分析入门 — 工程师快速理解指南", expanded=False):
        st.markdown("""
**🔊 什么是频谱？为什么需要它？**

声音本质上是空气的振动。时域波形只展示了"振幅随时间变化"，就像只看到心电图曲线。
而频谱分析相当于给声音做"棱镜分光"——把混合的振动拆解成不同频率的成分，
每个频率有多强一目了然，就像彩虹把白光拆成不同颜色。

---

**📐 坐标轴的物理含义**

| 图表 | 横轴 | 纵轴 | 通俗理解 |
|------|------|------|----------|
| 时域波形 | 时间（秒） | 振幅 | 某一瞬间的声音"大小" |
| 频率谱 | 频率（Hz） | 幅度（dB） | 该频率的"音量"有多强 |
| 时频谱 | 时间×频率 | 颜色深浅 | 某个时刻、某个频率有多响 |

> **dB（分贝）**：对数刻度。+6dB = 振幅翻倍，+20dB = 振幅 ×10。
> 人耳对响度的感知接近对数，所以 dB 更符合听觉直觉。

---

**🎛️ 关键频段与物理来源**

| 频段范围 | 典型来源 | 类比 |
|----------|----------|------|
| 0-100 Hz | 电机转速、电源工频 | 大鼓的低沉"嗡嗡"声 |
| 100-500 Hz | 齿轮啮合低次谐波 | 风扇的"呼呼"声 |
| 500-2kHz | 轴承内圈缺陷 | 摩擦的"沙沙"声 |
| 2k-6kHz | 轴承外圈/滚动体缺陷 | 尖锐的"嘶嘶"声 |
| 6k-16kHz | 气蚀、高频振动 | 刺耳的"嘶鸣"声 |

> **转速公式**：电机转速频率 = RPM ÷ 60。例如 3000RPM → 50Hz 基频。
> **啮合频率**：齿数 × 转速频率。例如 20齿 × 50Hz = 1000Hz。

---

**🌡️ 时频谱是什么？**

时频谱（STFT Spectrogram）是频谱的"电影版"——普通频谱是某一时刻的"照片"，
时频谱则是把整段音频切成很多小段，每段各画一张频谱，再按时间排列。
横轴是时间，纵轴是频率，颜色越亮表示该频率在该时刻越强。
它能帮你发现"哪个频率在什么时刻出现"，对间歇性故障特别有用。
        """)

    # ════ 图表区卡片 ════
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>📉 频谱可视化</div>
    """, unsafe_allow_html=True)

    tab_wave, tab_spec, tab_stft = st.tabs(["📊 时域波形", "🎚️ 频率谱 (FFT)", "🌡️ STFT 时频谱"])

    with tab_wave:
        st.plotly_chart(result["waveform_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption("包络波形 — 像素级 min/max 检测，按峰峰值归一化")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看波形图：</b>横轴是时间，纵轴是声音振幅。<br>
• 波形越"胖"→ 声音越响；波形越"瘦"→ 声音越轻<br>
• 波形上下对称 → 正常；被"削顶"→ 削波失真（录音音量过大）<br>
• 密集细密波形 → 高频成分多；稀疏平缓波形 → 低频成分多
</div>
        """, unsafe_allow_html=True)

    with tab_spec:
        st.plotly_chart(result["spectrum_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption(f"FFT 频率谱 · Hann 窗 (N={result['fft_size']})")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看频率谱：</b>横轴是频率（Hz），纵轴是该频率的强度（dB）。<br>
• 峰值 = 该频率处有突出的振动成分，可能对应某种机械故障或共振<br>
• 低频段（左端）高耸 → 大质量部件振动（转子、轴承座）<br>
• 高频段（右端）高耸 → 小质量部件或气蚀（滚动体、润滑不良）<br>
• 平坦无峰 → 宽带噪声（如风噪、流体噪声），无明显特征频率
</div>
        """, unsafe_allow_html=True)

    with tab_stft:
        st.plotly_chart(result["spectrogram_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption("STFT 时频谱图 · Hann 窗 · ~75% 重叠 · 对数幅度压缩")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看时频谱：</b>横轴=时间，纵轴=频率，颜色越亮=越响。<br>
• 水平亮线 → 持续存在的固定频率（稳态振动，如电机基频）<br>
• 垂直亮线 → 某一时刻的全频带冲击（撞击、启动瞬态）<br>
• 斜线 → 频率随时间变化（变速运行、啁啾信号）<br>
• 散点亮斑 → 间歇性故障（如轴承周期性打滑）
</div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # 关闭图表区卡片

    # ════ 频谱峰值 ════
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>📋 频谱峰值</div>
    """, unsafe_allow_html=True)
    if result["spectrum_peaks"]:
        peak_data = [
            {"频率": f"{p['frequency']:.1f} Hz", "相对强度": f"{p['magnitude']*100:.1f}%", "频段": p["label"]}
            for p in result["spectrum_peaks"]
        ]
        st.dataframe(peak_data, use_container_width=True, hide_index=True)
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:8px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 峰值解读：</b>频谱中的"尖峰"就像指纹的"纹路"——是设备独有的特征频率。<br>
• 同一台设备在不同时间录制，特征频率位置基本不变（只是幅度可能变化）<br>
• 不同设备的峰值位置和组合通常不同，这就是"声纹识别"的基本原理<br>
• "频段"列标识了该频率可能对应的物理来源（电机/齿轮/轴承等）
</div>
        """, unsafe_allow_html=True)
    else:
        st.info("未检测到明显峰值信号")
    st.markdown("</div>", unsafe_allow_html=True)  # 关闭峰值卡片

    # ════ 缺陷诊断 ════
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>⚠️ 缺陷诊断</div>
    """, unsafe_allow_html=True)
    if result["defects"]:
        defect_data = []
        for d in result["defects"]:
            sev_label = {"high": "严重", "medium": "中等", "low": "轻微"}[d["severity"]]
            defect_data.append({
                "严重度": sev_label,
                "类型": d["type"],
                "频率": f"{d['frequency']:.1f}" if d["frequency"] > 0 else "—",
                "描述": d["description"],
            })
        st.dataframe(defect_data, use_container_width=True, hide_index=True)
        st.markdown("""
<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:12px 16px;margin-top:8px;color:#7f1d1d;font-size:13px;line-height:1.7;'>
<b>⚠️ 缺陷诊断说明：</b>系统根据频谱异常特征自动判定可能的缺陷。<br>
• <b>削波</b>：录音音量过大导致波形截断，信息丢失不可恢复，需降低增益重新录制<br>
• <b>直流偏移</b>：波形整体偏离零线，通常由采集设备问题导致，不影响分析但需校准<br>
• <b>低频隆起</b>：超低频段能量异常偏高，可能是风噪、手震或电源干扰<br>
• <b>高频衰减</b>：高频段能量明显偏低，可能是采样率不足或传感器高频响应差
</div>
        """, unsafe_allow_html=True)
    else:
        st.success("未检测到异常缺陷，音频信号质量良好")
    st.markdown("</div>", unsafe_allow_html=True)  # 关闭缺陷卡片

    # ════ AI 分析 ════
    with st.expander("🤖  AI 智能分析报告", expanded=False):
        _ai_html = f"<div style='background:#fafbff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;" \
                   f"white-space:pre-wrap;line-height:1.9;color:#334155;font-size:14px;'>" \
                   f"{result['ai_analysis']}</div>"
        st.markdown(_ai_html, unsafe_allow_html=True)

    # ════ 分析历史 ════
    history = st.session_state.analysis_history or []
    if history:
        st.markdown("""
        <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
            padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
            <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>🕐 分析历史</div>
        """, unsafe_allow_html=True)
        n_cols = min(len(history), 5)
        hist_cols = st.columns(n_cols)
        for i, entry in enumerate(history):
            with hist_cols[i % n_cols]:
                _sc = entry.get("score", 0)
                _c = "#059669" if _sc >= 80 else ("#d97706" if _sc >= 60 else "#dc2626")
                _hist = f"""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px;text-align:center;'>
<div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>{entry['time']}</div>
<div style='font-size:12.5px;color:#1e293b;font-weight:500;' title="{entry['name']}">{entry['name'][:14]}{'...' if len(entry['name']) > 14 else ''}</div>
<div style='font-size:14px;font-weight:700;color:{_c};margin-top:6px;'>{entry['grade']} &middot; {_sc}分</div>
</div>"""
                st.markdown(_hist, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # 关闭历史卡片


# ──────────────────────── 音频比对 UI ────────────────────────


def generate_comparison_report_html(comp: dict, name_a: str, name_b: str) -> str:
    """生成音频噪声指纹比对 HTML 报告（内嵌 Plotly 交互式图表，离线可打开）"""

    def fig_to_html_div(fig) -> str:
        """将 Plotly Figure 转为独立 HTML div（含 Plotly.js CDN）"""
        try:
            return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})
        except Exception:
            return "<p style='color:#94a3b8;text-align:center;padding:20px;'>图表渲染失败</p>"

    overall = comp["overall_similarity"]
    if overall >= 75:
        color, bg, icon = "#059669", "#ecfdf5", "✅"
    elif overall >= 65:
        color, bg, icon = "#2563eb", "#eff6ff", "🔵"
    elif overall >= 45:
        color, bg, icon = "#d97706", "#fffbeb", "🟡"
    else:
        color, bg, icon = "#dc2626", "#fef2f2", "🔴"

    # 生成图表
    fig_radar = plot_similarity_radar(comp)
    fig_lfcc_a = plot_lfcc_heatmap(comp["lfcc_a"], comp["target_sr"], f"LFCC — {name_a[:18]}")
    fig_lfcc_b = plot_lfcc_heatmap(comp["lfcc_b"], comp["target_sr"], f"LFCC — {name_b[:18]}")
    fig_overlay = plot_spectrum_overlay(comp["sig_a"], comp["sig_b"], name_a, name_b)
    fig_bands = plot_band_energy_comparison(comp["sig_a"], comp["sig_b"], name_a, name_b)

    radar_html = fig_to_html_div(fig_radar)
    lfcc_a_html = fig_to_html_div(fig_lfcc_a)
    lfcc_b_html = fig_to_html_div(fig_lfcc_b)
    overlay_html = fig_to_html_div(fig_overlay)
    bands_html = fig_to_html_div(fig_bands)

    # 谐波峰表格
    def peaks_table_rows(peaks, max_n=8):
        rows = []
        for i, p in enumerate(peaks[:max_n]):
            rows.append(f"""<tr>
                <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b;font-weight:600">{i+1}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b">{p['freq']:.1f} Hz</td>
                <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">{p['magnitude']*100:.1f}%</td>
                <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b">{p['label']}</td>
            </tr>""")
        return "\n".join(rows) if rows else '<tr><td colspan="4" style="padding:16px;text-align:center;color:#94a3b8;">未检测到明显谐波峰</td></tr>'

    sig_a, sig_b = comp["sig_a"], comp["sig_b"]
    peaks_a = sig_a.get("harmonic_peaks", [])
    peaks_b = sig_b.get("harmonic_peaks", [])
    ratios_a = sig_a.get("harmonic_ratios", [])
    ratios_b = sig_b.get("harmonic_ratios", [])

    # 子带能量对比表
    band_names = [
        "0-50Hz 超低频", "50-100Hz 低频A", "100-200Hz 低频B", "200-500Hz 中低频",
        "500-1kHz 中频A", "1k-2kHz 中频B", "2k-3kHz 语音区", "3k-5kHz 中高频",
        "5k-8kHz 高频", "8k+ 超高频",
    ]
    be_a = sig_a.get("band_energies", [])
    be_b = sig_b.get("band_energies", [])

    band_rows = []
    for i, bn in enumerate(band_names):
        va = f"{be_a[i]*100:.1f}%" if i < len(be_a) else "—"
        vb = f"{be_b[i]*100:.1f}%" if i < len(be_b) else "—"
        band_rows.append(f"""<tr>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b;font-weight:500">{bn}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b;text-align:center">{va}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b;text-align:center">{vb}</td>
        </tr>""")
    band_table_rows = "\n".join(band_rows)

    # 标量特征对比
    scalar_features = [
        ("频谱质心", sig_a.get("centroid", 0), sig_b.get("centroid", 0), "Hz"),
        ("频谱带宽", sig_a.get("bandwidth", 0), sig_b.get("bandwidth", 0), "Hz"),
        ("频谱平坦度", sig_a.get("flatness", 0), sig_b.get("flatness", 0), ""),
    ]
    scalar_rows = []
    for label, va, vb, unit in scalar_features:
        u = f" {unit}" if unit else ""
        scalar_rows.append(f"""<tr>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b;font-weight:500">{label}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b;text-align:center">{va:.1f}{u}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#64748b;text-align:center">{vb:.1f}{u}</td>
        </tr>""")
    scalar_table_rows = "\n".join(scalar_rows)

    speech_filter_note = ""
    if comp.get("filter_speech"):
        speech_filter_note = '<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;margin:12px 0;color:#1e40af;font-size:13px;">🔊 已启用智能语音过滤——VAD 帧丢弃法：检测到语音的帧直接丢弃，只保留无语音帧计算特征，避免频域挖洞。</div>'

    # 置信度提示
    confidence = comp.get("confidence", 1.0)
    conf_pct = f"{confidence:.0%}"
    if confidence < 0.4:
        conf_color, conf_note = "#dc2626", "极低——结论很可能不可靠，请勿以此做判断"
    elif confidence < 0.6:
        conf_color, conf_note = "#d97706", "偏低——结论仅供参考，建议补充更高质量的录音"
    elif confidence < 0.8:
        conf_color, conf_note = "#2563eb", "中等——有一定参考价值，建议结合实际工况综合判断"
    else:
        conf_color, conf_note = "#059669", "良好——音频条件满足比对要求，结论可靠度较高"
    confidence_note = f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin:12px 0;color:#475569;font-size:13px;">📊 置信度：<b style="color:{conf_color};">{conf_pct}</b> — {conf_note}</div>'

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>音频噪声指纹比对报告 - {name_a} vs {name_b}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#f8fafc; color:#1e293b; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; line-height:1.6; padding:24px; max-width:920px; margin:0 auto; }}
h1 {{ font-size:26px; color:#0f172a; margin-bottom:6px; letter-spacing:-0.5px; }}
h2 {{ font-size:17px; color:#4f46e5; margin:28px 0 14px; border-left:3px solid #4f46e5; padding-left:14px; }}
.meta {{ color:#64748b; font-size:13px; margin-bottom:24px; padding:8px 0; border-bottom:1px solid #e2e8f0; }}
.card {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:22px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
.score-big {{ font-size:56px; font-weight:800; letter-spacing:-3px; line-height:1; }}
.grade {{ font-size:24px; font-weight:700; }}
.stat-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-top:18px; }}
.stat-item {{ background:#f8fafc; padding:16px; border-radius:10px; text-align:center; border:1px solid #f1f5f9; }}
.stat-label {{ font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }}
.stat-value {{ font-size:20px; color:#0f172a; font-weight:700; }}
.chart-img {{ width:100%; border-radius:10px; display:block; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ padding:12px 14px; text-align:left; border-bottom:2px solid #e2e8f0; color:#64748b; font-size:13px; font-weight:600; background:#f8fafc; }}
td {{ padding:10px 14px; border-bottom:1px solid #f1f5f9; }}
.footer {{ text-align:center; color:#94a3b8; font-size:12px; margin-top:36px; padding-top:18px; border-top:1px solid #e2e8f0; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.tip {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px 16px; margin-top:12px; color:#475569; font-size:13px; line-height:1.7; }}
</style>
</head>
<body>
<h1>🔗 音频噪声指纹比对报告</h1>
<div class="meta">
    📁 音频 A: {name_a}（时长 {comp['duration_a']:.2f}s）&nbsp;|&nbsp;
    📁 音频 B: {name_b}（时长 {comp['duration_b']:.2f}s）&nbsp;|&nbsp;
    📅 {now_str}
</div>

{speech_filter_note}

{confidence_note}

<div class="card" style="background:{bg};border-color:{color}30;">
<h2 style="color:{color};border-color:{color};">🎯 比对结论</h2>
<div style="text-align:center;padding:18px 0;">
    <div class="score-big" style="color:{color};">{overall}<span style="font-size:28px;font-weight:600;color:#94a3b8;">%</span></div>
    <div class="grade" style="color:{color};margin-top:8px;">{icon} {comp['conclusion']}</div>
    <div style="color:#475569;font-size:14px;margin-top:10px;max-width:560px;margin-left:auto;margin-right:auto;">{comp['conclusion_detail']}</div>
</div>
</div>

<div class="card">
<h2>📊 特征相似度得分</h2>
<div class="stat-grid">
    <div class="stat-item">
        <div class="stat-label">LFCC</div>
        <div class="stat-value">{comp['lfcc_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 15%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">频谱特征</div>
        <div class="stat-value">{comp['spectral_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 15%（PCEN 补偿）</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">峰结构</div>
        <div class="stat-value">{comp['peak_structure_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 20%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">谐波指纹</div>
        <div class="stat-value" style="color:{color};">{comp['harmonic_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 25%（核心）</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">包络谱</div>
        <div class="stat-value">{comp['envelope_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 15%（3-150Hz 峰匹配）</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">异常帧</div>
        <div class="stat-value">{comp.get('topk_similarity', 0)}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 10%</div>
    </div>
</div>
<div class="tip">
<b>📖 六项指标通俗解读：</b><br>
🔸 <b>LFCC</b>（15%）：增强统计分布比对——均值+标准差+分位数+协方差结构+Wasserstein距离。<br>
🔸 <b>频谱特征</b>（15%）：PCEN 补偿后子带能量+质心+带宽+平坦度比对，消除录音距离影响。<br>
🔸 <b>峰结构</b>（20%）：抗频率漂移——比较峰间距比和倍频关系，即使转速差3%也能识别同源。<br>
🔸 <b>谐波指纹</b>（25%）：权重最高。DNA 条形码比对——每台设备有独特的特征频率组合。<br>
🔸 <b>包络谱</b>（15%）：3-150Hz 包络峰匹配为主，捕捉转频、轴承缺陷等低频调制特征。<br>
🔸 <b>异常帧</b>（10%）：Top-K 异常帧比对——提取最响/最尖锐/最高频的帧单独比较，捕捉关键异响。
</div>
</div>

<div class="card">
<h2>🎯 相似度雷达图</h2>
{radar_html}
<div class="tip"><b>💡 阅读方法：</b>六个轴代表六个比对维度，蓝色多边形越"大"越"圆"→ 两段音频越相似。谐波指纹轴突出说明特征频率匹配好。</div>
</div>

<div class="card">
<h2>📊 LFCC 对比热力图</h2>
<div class="two-col">
    <div>
        <h3 style="font-size:14px;color:#475569;margin-bottom:8px;">音频 A — {name_a[:20]}</h3>
        {lfcc_a_html}
    </div>
    <div>
        <h3 style="font-size:14px;color:#475569;margin-bottom:8px;">音频 B — {name_b[:20]}</h3>
        {lfcc_b_html}
    </div>
</div>
<div class="tip"><b>💡 阅读方法：</b>纵轴=13个频率系数，横轴=时间帧，颜色=系数值。两图颜色分布越像 → LFCC 相似度越高。与 MFCC 不同：LFCC 用线性频率间隔，不会忽略高频机械特征。</div>
</div>

<div class="card">
<h2>🎚️ 频谱叠加对比</h2>
{overlay_html}
<div class="tip"><b>💡 阅读方法：</b>两条曲线越重合 → 频率分布越相似。峰值位置对齐说明特征频率一致；曲线高度不同但形状相似仅表示音量差异。</div>
</div>

<div class="card">
<h2>📈 子带能量分布对比</h2>
{bands_html}
<div class="tip"><b>💡 阅读方法：</b>10个频段的能量占比对比。低频细分便于识别机械噪声特征。同组柱高接近 → 该频段相似；差距大 → 有显著差异。</div>
</div>

<div class="card">
<h2>📋 子带能量明细</h2>
<table>
<thead><tr><th>频段</th><th style="text-align:center;">音频 A</th><th style="text-align:center;">音频 B</th></tr></thead>
<tbody>{band_table_rows}</tbody>
</table>
</div>

<div class="card">
<h2>📏 频谱标量特征对比</h2>
<table>
<thead><tr><th>特征</th><th style="text-align:center;">音频 A</th><th style="text-align:center;">音频 B</th></tr></thead>
<tbody>{scalar_table_rows}</tbody>
</table>
<div class="tip">
<b>📖 特征含义：</b><br>
• <b>频谱质心</b>：功率谱的"重心"频率。越高→高频能量越强→噪声越"尖锐"<br>
• <b>频谱带宽</b>：能量围绕质心的分散程度。越大→频率成分越丰富<br>
• <b>频谱平坦度</b>：几何均值/算术均值。≈1→白噪声（平坦）；≪1→有明显峰值（周期性）
</div>
</div>

<div class="card">
<h2>🎵 谐波峰详情</h2>
<div class="two-col">
    <div>
        <h3 style="font-size:14px;color:#475569;margin-bottom:8px;">音频 A — {name_a[:20]}</h3>
        <table>
        <thead><tr><th>#</th><th>频率</th><th>相对幅度</th><th>频段</th></tr></thead>
        <tbody>{peaks_table_rows(peaks_a)}</tbody>
        </table>
        {f'<div style="margin-top:8px;color:#64748b;font-size:12px;">谐波比: {", ".join(f"{r}×" for r in ratios_a[:5])}</div>' if ratios_a else ''}
    </div>
    <div>
        <h3 style="font-size:14px;color:#475569;margin-bottom:8px;">音频 B — {name_b[:20]}</h3>
        <table>
        <thead><tr><th>#</th><th>频率</th><th>相对幅度</th><th>频段</th></tr></thead>
        <tbody>{peaks_table_rows(peaks_b)}</tbody>
        </table>
        {f'<div style="margin-top:8px;color:#64748b;font-size:12px;">谐波比: {", ".join(f"{r}×" for r in ratios_b[:5])}</div>' if ratios_b else ''}
    </div>
</div>
<div class="tip">
<b>📖 谐波峰解读：</b>频谱中的"尖峰"是设备的独有特征频率，就像指纹的纹路。<br>
• 同一设备在不同时间录制，特征频率位置基本不变<br>
• "频段"标识了该频率可能对应的物理来源（电机/齿轮/轴承等）<br>
• <b>谐波比</b>：基频与各谐波峰的频率比。同一设备的谐波比稳定（由物理结构决定），即使录音条件变化也不受影响
</div>
</div>

<div class="footer">音频噪声指纹比对系统 | 报告生成时间 {now_str}</div>
</body>
</html>"""


def render_comparison(comp: dict, name_a: str, name_b: str):
    """渲染音频比对结果（卡片式布局）"""

    # ── 结论大卡片 ──
    overall = comp["overall_similarity"]
    if overall >= 75:
        color, bg = "#059669", "#ecfdf5"
        icon = "✅"
    elif overall >= 65:
        color, bg = "#2563eb", "#eff6ff"
        icon = "🔵"
    elif overall >= 45:
        color, bg = "#d97706", "#fffbeb"
        icon = "🟡"
    else:
        color, bg = "#dc2626", "#fef2f2"
        icon = "🔴"

    _card = f"""<div style='background:{bg};border:1.5px solid {color}30;border-radius:18px;
        padding:28px;text-align:center;margin:0 0 20px;box-shadow:0 2px 12px rgba(0,0,0,0.05);'>
      <div style='font-size:16px;color:#64748b;margin-bottom:8px;'>综合噪声指纹相似度</div>
      <div style='font-size:64px;font-weight:800;color:{color};letter-spacing:-3px;line-height:1;'>
          {overall}<span style='font-size:28px;font-weight:600;color:#94a3b8;'>%</span>
      </div>
      <div style='font-size:24px;font-weight:700;color:{color};margin-top:8px;'>
          {icon} {comp['conclusion']}
      </div>
      <div style='color:#475569;font-size:14px;margin-top:10px;max-width:520px;margin-left:auto;margin-right:auto;'>
          {comp['conclusion_detail']}
      </div>
    </div>"""
    st.markdown(_card, unsafe_allow_html=True)

    # ── 语音过滤提示 ──
    if comp.get("filter_speech"):
        if comp.get("low_energy_warning"):
            st.warning(
                "⚠️ 语音频带过滤后剩余能量极低（语音频带占原始信号 99% 以上）。"
                "滤波后信号主要是噪声底噪，归一化后不同音频可能呈现虚假高相似度。"
                "建议关闭语音频带过滤，或使用包含更多机械噪声成分的音频。"
            )
        else:
            st.info("🔊 已启用智能语音过滤——检测到语音的帧直接丢弃（VAD 帧丢弃法），只保留无语音帧进行特征计算，避免频域挖洞导致的信息损失。")

    # ── 置信度提示 ──
    confidence = comp.get("confidence", 1.0)
    if confidence < 0.4:
        st.error(f"🔴 置信度极低 ({confidence:.0%})——音频质量/时长/信噪比严重不足，结论很可能不可靠，请勿以此做判断。")
    elif confidence < 0.6:
        st.warning(f"🟡 置信度偏低 ({confidence:.0%})——音频条件欠佳，结论仅供参考，建议补充更高质量的录音再比对。")
    elif confidence < 0.8:
        st.info(f"🔵 置信度中等 ({confidence:.0%})——结论有一定参考价值，但建议结合实际工况综合判断。")
    else:
        st.success(f"🟢 置信度良好 ({confidence:.0%})——音频条件满足比对要求，结论可靠度较高。")

    # ── 分项指标 ──
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>📋 特征相似度得分</div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("LFCC", f"{comp['lfcc_similarity']}%",
                  help="增强统计分布比对：均值+标准差+分位数+协方差结构+Wasserstein距离。全频段等权重，适合机械噪声。")
    with c2:
        st.metric("频谱特征", f"{comp['spectral_similarity']}%",
                  help="PCEN 倾斜补偿后的子带能量+质心+带宽+平坦度比对 — 消除录音距离对频谱倾斜的影响。")
    with c3:
        st.metric("峰结构", f"{comp['peak_structure_similarity']}%",
                  help="抗频率漂移的峰间距比+倍频关系比对 — 即使同型电机转速差3%，峰结构仍高度相似。")
    with c4:
        st.metric("谐波指纹", f"{comp['harmonic_similarity']}%",
                  help="特征频率峰的匹配 + 谐波频率比结构比对 — 核心指标，类似 DNA 比对，即使两段录音转速不同也能识别同源设备。")
    with c5:
        st.metric("包络谱", f"{comp['envelope_similarity']}%",
                  help="3-150Hz 包络峰匹配（70%）+ 曲线相似度（30%）— 捕捉转频、轴承缺陷频率等低频调制特征。")
    with c6:
        st.metric("异常帧", f"{comp.get('topk_similarity', 0)}%",
                  help="Top-K 异常帧比对——提取 RMS/谱峭度/高频能量最高的帧单独比较，捕捉关键异响而非只看全局平均。")
    st.markdown("</div>", unsafe_allow_html=True)  # 关闭指标卡片

    # ── 分项指标通俗解读 ──
    st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;margin:8px 0 16px;color:#475569;font-size:13px;line-height:1.8;'>
<b>📖 六项指标通俗解读：</b><br>
🔸 <b>LFCC</b>（15%）：像"对比两个人的全身体检报告"——不只看平均指标，还看波动范围、上下限、指标间关联性。<br>
🔸 <b>频谱特征</b>（15%）：PCEN 补偿后比对子带能量、质心等 — 消除近场/远场导致的频谱倾斜差异。<br>
🔸 <b>峰结构</b>（20%）：像"对比两段旋律的音程关系"——不管整体偏高偏低，只要音符间距模式一样就算相似。<br>
🔸 <b>谐波指纹</b>（25%）：权重最高。像"DNA 条形码比对"——每台设备有独特的特征频率组合。<br>
🔸 <b>包络谱</b>（15%）：像"对比两首歌的鼓点节奏"——看 3-150Hz 调制峰结构是否一致，而非整体曲线形状。<br>
🔸 <b>异常帧</b>（10%）：像"专门比较最响/最尖锐的瞬间"——捕捉全局平均会淹没的关键异响。
</div>
    """, unsafe_allow_html=True)

    # ── 谐波峰详情 ──
    sig_a, sig_b = comp["sig_a"], comp["sig_b"]
    peaks_a, peaks_b = sig_a["harmonic_peaks"], sig_b["harmonic_peaks"]
    if peaks_a or peaks_b:
        st.markdown("""
        <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
            padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
            <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>🎵 谐波峰详情</div>
        """, unsafe_allow_html=True)
        pa_cols, pb_cols = st.columns(2)
        with pa_cols:
            st.markdown(f"**音频 A — {name_a[:20]}**")
            if peaks_a:
                for i, p in enumerate(peaks_a[:8]):
                    st.markdown(f"  {i+1}. **{p['freq']:.1f} Hz** — {p['label']} (幅度 {p['magnitude']*100:.1f}%)")
                if sig_a["harmonic_ratios"]:
                    st.caption(f"谐波比: {', '.join(f'{r}×' for r in sig_a['harmonic_ratios'][:5])}")
            else:
                st.caption("未检测到明显谐波峰")
        with pb_cols:
            st.markdown(f"**音频 B — {name_b[:20]}**")
            if peaks_b:
                for i, p in enumerate(peaks_b[:8]):
                    st.markdown(f"  {i+1}. **{p['freq']:.1f} Hz** — {p['label']} (幅度 {p['magnitude']*100:.1f}%)")
                if sig_b["harmonic_ratios"]:
                    st.caption(f"谐波比: {', '.join(f'{r}×' for r in sig_b['harmonic_ratios'][:5])}")
            else:
                st.caption("未检测到明显谐波峰")
        st.markdown("</div>", unsafe_allow_html=True)  # 关闭谐波峰卡片

    # ── 音频信息卡片 ──
    st.markdown(f"""
    <div style='display:flex;gap:16px;justify-content:center;margin:16px 0 24px;flex-wrap:wrap;'>
      <div style='background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;padding:16px 22px;text-align:center;min-width:180px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
        <div style='font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;'>音频 A</div>
        <div style='font-size:14px;font-weight:600;color:#1e293b;margin-top:4px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title="{name_a}">{name_a[:24]}</div>
        <div style='font-size:12px;color:#64748b;margin-top:2px;'>⏱️ 时长 {comp['duration_a']:.2f}s</div>
      </div>
      <div style='display:flex;align-items:center;color:#94a3b8;font-size:20px;'>⟷</div>
      <div style='background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;padding:16px 22px;text-align:center;min-width:180px;box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
        <div style='font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;'>音频 B</div>
        <div style='font-size:14px;font-weight:600;color:#1e293b;margin-top:4px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title="{name_b}">{name_b[:24]}</div>
        <div style='font-size:12px;color:#64748b;margin-top:2px;'>⏱️ 时长 {comp['duration_b']:.2f}s</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 图表区卡片 ──
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>📉 可视化对比</div>
    """, unsafe_allow_html=True)

    tab_radar, tab_lfcc, tab_overlay, tab_bands = st.tabs(
        ["🎯 雷达图", "📊 LFCC", "🎚️ 频谱叠加", "📈 子带能量"]
    )

    with tab_radar:
        fig_radar = plot_similarity_radar(comp)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.caption("多维度噪声指纹相似度雷达图 — 越接近外圈表示越相似")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看雷达图：</b>六个轴代表六个比对维度，蓝色多边形越"大"越"圆"→ 两段音频越相似。<br>
• 谐波指纹的"尖角"突出 → 特征频率匹配好，大概率是同源设备<br>
• 某轴明显内缩 → 该维度差异大，需关注具体原因（如录音环境/传感器不同）
</div>
        """, unsafe_allow_html=True)

    with tab_lfcc:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_lfcc_a = plot_lfcc_heatmap(comp["lfcc_a"], comp["target_sr"], f"LFCC — {name_a[:18]}")
            st.plotly_chart(fig_lfcc_a, use_container_width=True, config={"displayModeBar": False})
        with col_b:
            fig_lfcc_b = plot_lfcc_heatmap(comp["lfcc_b"], comp["target_sr"], f"LFCC — {name_b[:18]}")
            st.plotly_chart(fig_lfcc_b, use_container_width=True, config={"displayModeBar": False})
        st.caption("线性频率倒谱系数热力图 — 全频段等权重，适合机械噪声分析，纵轴为 13 维系数，横轴为时间帧")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看 LFCC 热力图：</b>纵轴是 13 个"频率系数"（低维在上、高维在下），横轴是时间。<br>
• 颜色代表该系数在该帧的值——两图的颜色分布越像，LFCC 相似度越高<br>
• 第 1 行（最低维）反映频谱整体能量，下面各行反映越来越精细的频率细节<br>
• 与 MFCC 不同：LFCC 用线性频率间隔，不会忽略高频机械特征
</div>
        """, unsafe_allow_html=True)

    with tab_overlay:
        fig_overlay = plot_spectrum_overlay(comp["sig_a"], comp["sig_b"], name_a, name_b)
        st.plotly_chart(fig_overlay, use_container_width=True, config={"displayModeBar": False})
        st.caption("两段音频的归一化频谱叠加对比 — 曲线越重合表示频谱形状越相似")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看频谱叠加图：</b>两条曲线越重合 → 频率分布越相似。<br>
• 峰值位置对齐 → 特征频率一致，说明可能来自同型号或同台设备<br>
• 曲线高度不同但形状相似 → 仅音量差异，"峰结构"指标仍会较高（抗漂移）<br>
• 完全不重合 → 两段音频的噪声来源几乎不同
</div>
        """, unsafe_allow_html=True)

    with tab_bands:
        fig_bands = plot_band_energy_comparison(comp["sig_a"], comp["sig_b"], name_a, name_b)
        st.plotly_chart(fig_bands, use_container_width=True, config={"displayModeBar": False})
        st.caption("10 频段子带能量分布对比 — 低频细分便于识别机械噪声特征频率")
        st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-top:4px;color:#475569;font-size:13px;line-height:1.7;'>
<b>💡 如何看子带能量图：</b>把全频段切成 10 个子带，比较每个子带的能量占比。<br>
• 低频段（0-200Hz）细分了 3 个子带——因为机械噪声的核心特征集中在低频<br>
• 同组柱子高度接近 → 该频段能量分布相似；差距大 → 该频段有显著差异<br>
• 能量"重心"偏左 → 低频为主（大质量振动）；偏右 → 高频为主（小部件/气蚀）
</div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # 关闭图表区卡片

    # ── 导出 HTML 报告 ──
    st.markdown("---")
    comp_key = "comp_report_html"
    if comp_key not in st.session_state:
        st.session_state[comp_key] = None

    _has_comp = bool(st.session_state.get("comparison_result"))
    if _has_comp:
        _comp = st.session_state.comparison_result
        _name_a = st.session_state.get("comparison_name_a", "")
        _name_b = st.session_state.get("comparison_name_b", "")
        if not st.session_state[comp_key]:
            with st.spinner("正在生成比对 HTML 报告..."):
                st.session_state[comp_key] = generate_comparison_report_html(_comp, _name_a, _name_b)
        safe_name_a = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", Path(_name_a).stem)
        safe_name_b = re.sub(r"[^\w\u4e00-\u9fff_-]", "_", Path(_name_b).stem)
        report_filename = (
            f"{safe_name_a}_vs_{safe_name_b}_噪声指纹比对_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".html"
        )
        st.download_button(
            label="⬇  导出比对 HTML 报告",
            data=st.session_state[comp_key].encode("utf-8"),
            file_name=report_filename,
            mime="text/html;charset=utf-8",
            use_container_width=True,
            type="primary",
        )
    else:
        st.button("⬇  导出比对 HTML 报告", disabled=True, use_container_width=True)





if __name__ == "__main__":
    main()
