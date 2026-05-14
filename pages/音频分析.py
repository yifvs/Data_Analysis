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
from scipy.fft import fft, fftfreq, ifft

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

# .css-1lcbxhc { background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%) !important; border-right:1px solid #e2e8f0 !important; }
# .css-154b5vr { padding:24px 16px 12px !important; }
# .css-154bvr h3,.css-154bvr label { color:#1e293b !important; font-weight:600 !important; }
# h1 { color:#0f172a !important; font-weight:800 !important; letter-spacing:-0.6px !important; }
# h2 { color:#4f46e5 !important; font-weight:700 !important; margin-top:32px !important; }
# p { color:#475569 !important; line-height:1.7 !important; }

# .stFileUploader { background:#fff !important; border:2px dashed #cbd5e1 !important; border-radius:16px !important; padding:36px 20px !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); transition:all 0.25s; }
# .stFileUploader:hover { border-color:#818cf8 !important; background:#fafbff !important; box-shadow:0 4px 12px rgba(99,102,241,0.10) !important; }
# .stFileUploader > label { color:#64748b !important; font-size:15px !important; font-weight:500 !important; }
# /* 云端环境隐藏 uploader 内部重复文字，避免与自定义 label 重叠 */
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
    """尝试解码音频文件，优先 WAV → soundfile → ffmpeg，最终模拟兜底"""
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
    # 最终兜底：生成模拟数据
    print(f"[警告] 无法解码 {filename}，使用模拟信号演示")
    duration = min(5.0, len(file_bytes) * 2 / 44100)
    t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.15 * np.sin(2 * np.pi * 880 * t)
        + 0.08 * np.sin(2 * np.pi * 1320 * t)
        + 0.05 * np.sin(2 * np.pi * 2200 * t)
        + 0.02 * (np.random.random(len(t)) - 0.5)
        + 0.1 * np.sin(2 * np.pi * 120 * t)
    )
    return signal.astype(np.float32), 44100


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
) -> tuple[np.ndarray, int]:
    """
    执行 FFT 频谱分析
    返回: (magnitudes, fft_size)
    """
    fft_size = next_power_of_2(min(len(samples), 8192))
    segment = samples[:fft_size].astype(np.float64)
    # 窗函数减少频谱泄漏
    window = get_window("hann", fft_size)
    windowed = segment * window
    spectrum = fft(windowed)
    n_half = fft_size // 2
    magnitudes = np.abs(spectrum[:n_half]) / fft_size
    freqs = fftfreq(fft_size, d=1.0 / sample_rate)[:n_half]
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

    # 4. 削波检测
    if rms_level > -3:
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
# ● 本模块的比对流程（4 个维度）：
#   ┌─────────────┬────────────────────────────────────────────┬──────┐
#   │  维度        │  含义                                      │ 权重 │
#   ├─────────────┼────────────────────────────────────────────┼──────┤
#   │  LFCC       │  线性频率倒谱系数：全频段等权重的"音色指纹"  │ 20%  │
#   │  频谱特征   │  子带能量+质心+带宽+平坦度：整体能量分布     │ 25%  │
#   │  频谱形状   │  归一化频谱曲线的相似度：频率分布的形态匹配  │ 25%  │
#   │  谐波指纹   │  特征频率峰+谐波比：机械噪声最核心的识别指标 │ 30%  │
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
    # 对信号做一阶差分滤波：y[n] = x[n] - α·x[n-1]
    # 作用：适度提升高频成分。语音处理通常用 α=0.97（大幅提升高频），
    # 但机械噪声低频能量丰富，用 α=0.5 只做轻微提升，避免压低低频信息。
    pre_emphasis = 0.5
    emphasized = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

    # ── 分帧 ──
    # 声音信号随时间变化，需要切成短帧（每帧内近似稳定）。
    # frame_len = n_fft = 2048 点 ≈ 46ms @44100Hz
    # hop_len = 1024 点（50% 重叠），相邻帧共享一半数据，保证时间连续性
    frame_len = n_fft
    hop_len = frame_len // 2
    n_frames = max(1, 1 + (len(emphasized) - frame_len) // hop_len)

    # ── 加窗 ──
    # 直接截断信号会在帧边缘产生频谱泄漏（虚假频率成分）。
    # Hann 窗让帧两端平滑衰减到零，抑制泄漏。类比：拍照时用渐变滤镜
    # 避免边缘硬裁切产生的伪影。
    window = get_window("hann", frame_len)

    # ── 线性间距滤波器组（LFCC 与 MFCC 的根本区别） ──
    # 滤波器组就像一组"频段筛子"，把连续频谱筛成若干频道的能量。
    # MFCC 用"梅尔刻度"——低频密、高频疏（模拟人耳）；
    # LFCC 用"线性刻度"——等间距排列，每个频道宽度相同。
    # 类比：MFCC 像用放大镜看低频、用望远镜看高频；
    #       LFCC 像用同一把尺子量遍全频段。
    n_filters = 30  # 30 个三角滤波器，比 MFCC 常用的 26 个更多，覆盖更细
    low_bin = 0
    high_bin = n_fft // 2 + 1
    # 等间距分布滤波器中心频率——这就是"线性"的含义
    filter_edges = np.linspace(low_bin, high_bin, n_filters + 2).astype(int)

    # 构建三角滤波器矩阵：每个滤波器在中心频率处增益为1，向两侧线性衰减到0
    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left, center, right = filter_edges[i], filter_edges[i + 1], filter_edges[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / max(center - left, 1)   # 上升沿
        for j in range(center, right):
            fbank[i, j] = (right - j) / max(right - center, 1)  # 下降沿

    lfcc_features = np.zeros((n_lfcc, n_frames))

    for frame_idx in range(n_frames):
        start = frame_idx * hop_len
        end = min(start + frame_len, len(emphasized))
        frame = np.zeros(frame_len)
        frame[: end - start] = emphasized[start:end]
        frame *= window  # 加窗

        # ── FFT（快速傅里叶变换） ──
        # 把时域信号转换为频域信号，结果是复数数组。
        # 取绝对值得到幅度谱（每个频率分量的强度），取前一半（频谱对称，后半是镜像）。
        # 类比：FFT 像一台频谱仪，输入声音波形，输出"每个频率有多响"。
        spectrum = np.abs(fft(frame))[: n_fft // 2 + 1]
        # 功率谱 = 幅度²/窗长，更常用于能量分析（dB 刻度的基准）
        power_spec = (spectrum ** 2) / n_fft

        # ── 线性滤波 ──
        # 用滤波器组矩阵乘以功率谱，得到每个滤波器频道的输出能量。
        # 输出是 30 维向量，代表 30 个等宽频道的能量分布。
        filter_output = np.dot(fbank, power_spec)
        filter_output = np.maximum(filter_output, 1e-10)  # 防止 log(0)
        log_filter = np.log(filter_output)  # 取对数：人耳对响度的感知是对数的

        # ── DCT（离散余弦变换）→ LFCC ──
        # DCT 把 30 维对数能量向量压缩成 13 维倒谱系数。
        # 类比：DCT 就像"主成分分析"，把高维数据压缩到低维，
        # 保留主要信息（低阶系数=频谱包络，高阶系数=精细纹理）。
        # 去相关性：DCT 后各系数之间近似独立，便于后续比对。
        for k in range(n_lfcc):
            lfcc_features[k, frame_idx] = np.sum(
                log_filter * np.cos(np.pi * k * (np.arange(n_filters) + 0.5) / n_filters)
            ) * np.sqrt(2.0 / n_filters)

    return lfcc_features


def compute_harmonic_peaks(
    samples: np.ndarray, sample_rate: int, n_peaks: int = 10, min_prominence: float = 0.05
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

    返回: 按幅度降序排列的峰列表 [{freq, magnitude, label}, ...]
    """
    # FFT 计算——窗长取 16384 点的最近 2^N，获得更细的频率分辨率
    # 频率分辨率 = 采样率/窗长，例如 44100/16384 ≈ 2.7Hz
    # 即相邻两个频率点间隔 2.7Hz，足以分辨机械特征频率
    fft_size = next_power_of_2(min(len(samples), 16384))
    segment = samples[:fft_size]
    window = get_window("hann", fft_size)
    windowed = segment * window
    spectrum = np.abs(fft(windowed))[: fft_size // 2]
    freqs = fftfreq(fft_size, d=1.0 / sample_rate)[: fft_size // 2]
    power = spectrum ** 2  # 功率谱：幅度的平方，更直观反映能量

    # ── 局部峰值检测 ──
    # 对功率谱做差分：如果某点左侧在上升（diff>0）、右侧在下降（diff≤0），
    # 则该点就是局部极大值——即一个"峰"。
    # 类比：登山时从上升变为下降的那个点就是山顶。
    diff = np.diff(power)
    peaks_mask = np.zeros(len(power), dtype=bool)
    for i in range(1, len(diff)):
        if diff[i - 1] > 0 and diff[i] <= 0 and power[i] > 0:
            peaks_mask[i] = True

    # ── 显著度过滤 ──
    # 不是每个微小的起伏都值得关注。只保留强度达到最大峰 5% 以上的峰，
    # 排除由随机噪声引起的虚假峰值（就像过滤掉山丘，只留山峰）。
    max_power = np.max(power) + 1e-10
    threshold = min_prominence * max_power
    peak_indices = np.where(peaks_mask & (power > threshold))[0]

    # 按幅度排序取前 N 个——最强的峰最有特征价值
    if len(peak_indices) == 0:
        return []

    sorted_idx = peak_indices[np.argsort(power[peak_indices])[::-1]][:n_peaks]

    peaks = []
    for idx in sorted_idx:
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
    fft_size = next_power_of_2(min(len(samples), 16384))
    segment = samples[:fft_size]
    window = get_window("hann", fft_size)
    windowed = segment * window
    spectrum = np.abs(fft(windowed))[: fft_size // 2]
    freqs = fftfreq(fft_size, d=1.0 / sample_rate)[: fft_size // 2]
    power = spectrum ** 2
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
    harmonic_peaks = compute_harmonic_peaks(samples, sample_rate, n_peaks=10)

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
        "spectrum": spectrum / (np.max(spectrum) + 1e-10),
        "freqs": freqs,
    }


def remove_speech_band(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    去除语音频带（300-3400Hz），保留机械噪声主能量区。

    【为什么需要过滤语音频带？】
    在工业现场录音时，人声干扰是常见问题——操作员对话、广播通知等。
    人声的能量集中在 300-3400Hz（基频 80-400Hz + 谐波延伸到 3kHz 以上），
    而这个频段恰好与许多机械噪声的中频段重叠，会干扰比对结果。

    过滤掉语音频带后，比对只依赖：
    - <300Hz：低频机械振动（电机转频、齿轮啮合低阶）
    - >3400Hz：高频机械特征（轴承缺陷、气蚀、摩擦）
    这两个区域人声能量很弱，机械特征相对"纯净"。

    【技术原理：频域滤波】
    1. FFT 把时域信号转到频域（每个频率点一个复数值）
    2. 在 300-3400Hz 范围把复数值乘以 0（该频段静音）
    3. IFFT 把频域信号转回时域
    类比：就像图片处理中用橡皮擦擦掉某个区域，只不过这里擦的是"频率"。

    【过渡带：避免硬截断】
    不能直接在 300Hz 和 3400Hz 处一刀切（0/1 跳变），
    那样会产生"振铃"（Gibbs 现象）——在截止频率附近出现虚假波纹。
    用余弦函数在截止频率两侧做 100Hz 宽度的平滑过渡，
    类比：图片编辑中用"羽化"而不是硬边选区。
    """
    fft_size = next_power_of_2(len(samples))
    padded = np.zeros(fft_size)
    padded[:len(samples)] = samples

    # FFT 转频域
    freq_data = fft(padded)
    freqs = fftfreq(fft_size, d=1.0 / sample_rate)

    # ── 构建频域遮罩 ──
    # 初始全1（全部保留），然后在语音频段置0
    mask = np.ones(fft_size, dtype=float)
    speech_low, speech_high = 300, 3400
    mask[(np.abs(freqs) >= speech_low) & (np.abs(freqs) <= speech_high)] = 0.0

    # ── 过渡带平滑（余弦过渡，防止振铃） ──
    transition_width = 100  # Hz，过渡带宽度
    for side in [1, -1]:
        for edge in [speech_low, speech_high]:
            transition = (np.abs(freqs) >= edge - transition_width) & (np.abs(freqs) <= edge + transition_width)
            dist = (np.abs(freqs[transition]) - edge) / transition_width
            dist = np.clip(dist, -1, 1)
            # 半余弦过渡：从 1 平滑降到 0，再平滑升回 1
            smooth = 0.5 + 0.5 * np.cos(np.pi * (1 - np.abs(dist)))
            mask[transition] *= smooth

    # 频域乘以遮罩 → IFFT 回时域
    filtered = freq_data * mask
    result = np.real(ifft(filtered))[:len(samples)]
    return result


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
           │  重采样→44100Hz     │  重采样→44100Hz
           │  (可选)过滤语音     │  (可选)过滤语音
           │  归一化幅度         │  归一化幅度
           ▼                     ▼
    ┌──────────────────────────────────────────┐
    │            四维度特征提取 & 比对            │
    │                                          │
    │  ① LFCC: 帧级倒谱系数 → 余弦相似度  (20%) │
    │  ② 频谱特征: 子带能量+质心+带宽    (25%) │
    │  ③ 频谱形状: 归一化频谱曲线比对    (25%) │
    │  ④ 谐波指纹: 特征峰+谐波比比对    (30%) │
    │                                          │
    │          加权求和 → 综合相似度 0-100%      │
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

    def resample(samples, sr_from, sr_to):
        """线性插值重采样：改变采样率而不改变播放速度/音高"""
        if sr_from == sr_to:
            return samples
        duration = len(samples) / sr_from
        target_len = int(duration * sr_to)
        if target_len < 100:
            return samples
        indices = np.linspace(0, len(samples) - 1, target_len).astype(int)
        return samples[indices]

    a = resample(samples_a, sr_a, TARGET_SR)
    b = resample(samples_b, sr_b, TARGET_SR)

    # 可选：过滤语音频带（勾选后去除 300-3400Hz 人声干扰）
    if filter_speech:
        a = remove_speech_band(a, TARGET_SR)
        b = remove_speech_band(b, TARGET_SR)

    # 归一化幅度到 [-1, 1]，消除录音音量差异的影响
    a = a / (np.max(np.abs(a)) + 1e-10)
    b = b / (np.max(np.abs(b)) + 1e-10)

    # ════════════════════════════════════════════════════════
    # 维度 1：LFCC 余弦相似度（20%）
    # ════════════════════════════════════════════════════════
    # LFCC 是频谱的紧凑数值表示。两段音频每帧都算出 13 维 LFCC 向量，
    # 逐帧计算向量夹角的余弦值 → 越接近 1 说明"音色"越像。
    # 取所有帧的平均作为整体 LFCC 相似度。
    lfcc_a = compute_lfcc(a, TARGET_SR)
    lfcc_b = compute_lfcc(b, TARGET_SR)

    # 对齐帧数：取较短的音频帧数，逐帧比对
    min_frames = min(lfcc_a.shape[1], lfcc_b.shape[1])
    lfcc_a_aligned = lfcc_a[:, :min_frames]
    lfcc_b_aligned = lfcc_b[:, :min_frames]

    # 逐帧余弦相似度
    # 余弦相似度 = 两个向量的内积 / (模长之积)
    # = cos(θ)，θ 是向量夹角。θ=0°→完全相同(1)，θ=90°→完全正交(0)
    frame_sims = []
    for f in range(min_frames):
        va, vb = lfcc_a_aligned[:, f], lfcc_b_aligned[:, f]
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a > 1e-10 and norm_b > 1e-10:
            frame_sims.append(float(np.dot(va, vb) / (norm_a * norm_b)))
    lfcc_similarity = float(np.mean(frame_sims)) if frame_sims else 0.0

    # ════════════════════════════════════════════════════════
    # 维度 2：频谱特征相似度（25%）
    # ════════════════════════════════════════════════════════
    # 包含4个子指标的平均值：
    # - 子带能量分布：10个频段各自的能量占比是否一致
    # - 频谱质心：能量重心频率是否接近
    # - 频谱带宽：频率分散程度是否接近
    # - 频谱平坦度：噪声"纯度"是否接近（嗡嗡 vs 沙沙）
    sig_a = compute_noise_signature(a, TARGET_SR)
    sig_b = compute_noise_signature(b, TARGET_SR)

    # 子带能量分布相似度（余弦相似度）
    # 类比：两个饼图的扇形比例是否一致
    be_a = np.array(sig_a["band_energies"])
    be_b = np.array(sig_b["band_energies"])
    band_sim = float(np.dot(be_a, be_b) / (np.linalg.norm(be_a) * np.linalg.norm(be_b) + 1e-10))

    # 标量特征相似度（归一化差值法）
    # 1.0 - |A-B|/max(|A|,|B|)：值越接近越相似，1.0=完全相同
    def scalar_sim(va, vb):
        denom = max(abs(va), abs(vb), 1e-10)
        return 1.0 - min(abs(va - vb) / denom, 1.0)

    centroid_sim = scalar_sim(sig_a["centroid"], sig_b["centroid"])
    bandwidth_sim = scalar_sim(sig_a["bandwidth"], sig_b["bandwidth"])
    flatness_sim = scalar_sim(sig_a["flatness"], sig_b["flatness"])

    spectral_similarity = float(np.mean([band_sim, centroid_sim, bandwidth_sim, flatness_sim]))

    # ════════════════════════════════════════════════════════
    # 维度 3：频谱形状相似度（25%）
    # ════════════════════════════════════════════════════════
    # 把频谱曲线归一化后计算余弦相似度。
    # 归一化消除了绝对音量，只比较"曲线形状"——
    # 哪些频率有峰、哪些频率有谷，形态是否一致。
    # 类比：两条股票走势线，虽然价格不同但走势一样=高度相关
    spec_a = sig_a["spectrum"]
    spec_b = sig_b["spectrum"]
    min_len = min(len(spec_a), len(spec_b))
    spec_a, spec_b = spec_a[:min_len], spec_b[:min_len]
    spectrum_similarity = float(
        np.dot(spec_a, spec_b) / (np.linalg.norm(spec_a) * np.linalg.norm(spec_b) + 1e-10)
    )

    # ════════════════════════════════════════════════════════
    # 维度 4：谐波指纹相似度（30%）——机械噪声最核心的识别指标
    # ════════════════════════════════════════════════════════
    # 机械设备的特征频率由物理参数决定（转速、齿数、轴承型号等），
    # 这些频率在频谱上形成"谐波系列"——基频 + 整数倍频率的峰群。
    # 同一设备不管在哪录、音量多大，这些峰的频率和频率比是不变的。
    # 这是"同源判断"最可靠的维度——即使其他维度受环境干扰波动，
    # 谐波指纹也能精准识别。
    peaks_a = sig_a["harmonic_peaks"]
    peaks_b = sig_b["harmonic_peaks"]
    harmonic_similarity = _compute_harmonic_similarity(peaks_a, peaks_b)

    # ════════════════════════════════════════════════════════
    # 综合加权评分
    # ════════════════════════════════════════════════════════
    # 权重分配逻辑：
    # - 谐波指纹 30%：最可靠的机械特征，权重最高
    # - 频谱特征 25%：子带能量+标量特征，反映整体能量分布
    # - 频谱形状 25%：曲线形态匹配，对环境变化有一定鲁棒性
    # - LFCC 20%：补充音色信息，权重较低因为对增益/距离敏感
    overall = (
        0.20 * max(lfcc_similarity, 0)
        + 0.25 * max(spectral_similarity, 0)
        + 0.25 * max(spectrum_similarity, 0)
        + 0.30 * max(harmonic_similarity, 0)
    )
    overall_pct = max(0, min(100, overall * 100))

    # ── 同源判断阈值 ──
    # 这些阈值基于经验设定，实际应用中可根据具体场景微调：
    # - ≥75%：谐波峰频率高度重合 + 频谱形状非常吻合 → 高度同源
    # - 55-75%：部分特征相似，但不完全吻合 → 可能同源，需佐证
    # - 35-55%：个别特征相似，但整体差异明显 → 相似度低
    # - <35%：各维度差异都大 → 不同声源
    if overall_pct >= 75:
        conclusion = "高度同源"
        conclusion_detail = "两段音频的噪声指纹高度吻合，极大概率来自同一噪声源或具有相同的振动机理。"
    elif overall_pct >= 55:
        conclusion = "可能同源"
        conclusion_detail = "两段音频存在一定相似性，可能来自同一类噪声源或相似机理，但需结合其他证据进一步确认。"
    elif overall_pct >= 35:
        conclusion = "相似度低"
        conclusion_detail = "两段音频噪声特征差异较明显，不太可能来自同一噪声源。"
    else:
        conclusion = "非同源"
        conclusion_detail = "两段音频噪声指纹显著不同，来自不同噪声源。"

    return {
        "lfcc_similarity": round(max(lfcc_similarity * 100, 0), 1),
        "spectral_similarity": round(max(spectral_similarity * 100, 0), 1),
        "spectrum_similarity": round(max(spectrum_similarity * 100, 0), 1),
        "harmonic_similarity": round(max(harmonic_similarity * 100, 0), 1),
        "overall_similarity": round(overall_pct, 1),
        "conclusion": conclusion,
        "conclusion_detail": conclusion_detail,
        "lfcc_a": lfcc_a_aligned,
        "lfcc_b": lfcc_b_aligned,
        "sig_a": sig_a,
        "sig_b": sig_b,
        "target_sr": TARGET_SR,
        "len_a": len(a),
        "len_b": len(b),
        "duration_a": len(a) / TARGET_SR,
        "duration_b": len(b) / TARGET_SR,
        "filter_speech": filter_speech,
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

    # ── 第1层：频率匹配 ──
    # 对 A 的每个峰，找 B 中频率最接近的峰，计算归一化差距
    match_scores = []
    for fa in freqs_a:
        diffs = np.abs(freqs_b - fa)
        min_diff = np.min(diffs)
        # 容差设计：5% 相对容差 + 10Hz 绝对容差（取较大者）
        # 相对容差适应高频峰（如 5000Hz ± 250Hz），
        # 绝对容差适应低频峰（如 25Hz ± 10Hz 比用 5% 更合理）
        tolerance = max(fa * 0.05, 10)
        if min_diff < tolerance:
            match_scores.append(1.0 - min_diff / tolerance)
        else:
            match_scores.append(0.0)

    freq_match = float(np.mean(match_scores)) if match_scores else 0.0

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
        # 对 A 的每个谐波比，找 B 中最近的，容差 0.3
        # 例如 A=2.0 vs B=2.1 → 差0.1，在容差内，得分 = 1-0.1/0.3 = 0.67
        ratio_scores = []
        for ra in ratio_a:
            min_diff = min(abs(ra - rb) for rb in ratio_b)
            ratio_scores.append(max(0, 1.0 - min_diff / 0.3))
        ratio_match = float(np.mean(ratio_scores))

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
    categories = ["LFCC 相似度", "频谱特征", "频谱形状", "谐波指纹", "综合相似度"]
    values = [
        comp["lfcc_similarity"],
        comp["spectral_similarity"],
        comp["spectrum_similarity"],
        comp["harmonic_similarity"],
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

    defects = detect_defects(magnitudes, freqs, peaks, rms_level, fft_size, sample_rate)
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

**🧬 四维比对原理**

| 维度 | 权重 | 比什么 | 通俗理解 |
|------|------|--------|----------|
| LFCC | 20% | 逐帧频率子带能量分布 | 逐段对比"频谱截图" |
| 频谱特征 | 25% | 质心、带宽、平坦度等统计量 | 比对"整体体型指标" |
| 频谱形状 | 25% | 归一化频谱曲线轮廓 | 比对"剪影轮廓" |
| 谐波指纹 | 30% | 特征峰位置 + 谐波频率比 | 比对"DNA 条形码" |

---

**🎯 相似度怎么看？**

| 范围 | 结论 | 含义 |
|------|------|------|
| ≥ 75% | 很可能同源 | 特征高度吻合，大概率是同一台设备 |
| 55-75% | 可能同源 | 部分特征匹配，需结合其他信息判断 |
| 35-55% | 不确定 | 特征差异较大，可能同型号不同个体 |
| < 35% | 大概率不同源 | 特征几乎不匹配，不是同一设备 |

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
        if st.session_state.get("comparison_result") and not compare_clicked:
            render_comparison(
                st.session_state.comparison_result,
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
    elif overall >= 55:
        color, bg, icon = "#2563eb", "#eff6ff", "🔵"
    elif overall >= 35:
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
        speech_filter_note = '<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;margin:12px 0;color:#1e40af;font-size:13px;">🔊 已启用语音频带过滤（300-3400Hz），比对结果排除了人声干扰，专注于机械噪声特征。</div>'

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
.stat-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; margin-top:18px; }}
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
        <div class="stat-label">LFCC 相似度</div>
        <div class="stat-value">{comp['lfcc_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 20%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">频谱特征</div>
        <div class="stat-value">{comp['spectral_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 25%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">频谱形状</div>
        <div class="stat-value">{comp['spectrum_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 25%</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">谐波指纹</div>
        <div class="stat-value" style="color:{color};">{comp['harmonic_similarity']}%</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">权重 30%（核心）</div>
    </div>
</div>
<div class="tip">
<b>📖 四项指标通俗解读：</b><br>
🔸 <b>LFCC 相似度</b>：像"逐段对比两首歌的频谱截图"——每帧切一小段，看各频率子带能量分布像不像。全频段等权重，不偏向人声。<br>
🔸 <b>频谱特征</b>：像"比较两个人的身高、体重、BMI"——不看细节，只比对几个整体统计量（能量重心在哪、频率分布有多宽/多平）。<br>
🔸 <b>频谱形状</b>：像"比较两张剪影的轮廓"——忽略谁响谁轻，只看频率分布的"形状"像不像。<br>
🔸 <b>谐波指纹</b>：权重最高（30%）。像"DNA 条形码比对"——每台设备有独特的特征频率组合，不管录音条件怎么变，比例不变。
</div>
</div>

<div class="card">
<h2>🎯 相似度雷达图</h2>
{radar_html}
<div class="tip"><b>💡 阅读方法：</b>五个轴代表五个比对维度，蓝色多边形越"大"越"圆"→ 两段音频越相似。谐波指纹轴突出说明特征频率匹配好。</div>
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
    elif overall >= 55:
        color, bg = "#2563eb", "#eff6ff"
        icon = "🔵"
    elif overall >= 35:
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
        st.info("🔊 已启用语音频带过滤（300-3400Hz），比对结果排除了人声干扰，专注于机械噪声特征。")

    # ── 分项指标 ──
    st.markdown("""
    <div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:16px;
        padding:20px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <div style='font-size:15px;font-weight:700;color:#0f172a;margin-bottom:12px;'>📋 特征相似度得分</div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("LFCC 相似度", f"{comp['lfcc_similarity']}%",
                  help="线性频率倒谱系数 — 把整段音频拆成等间隔频率子带，逐帧比较各子带能量分布的相似程度。全频段等权重，不偏向人声频段，适合机械噪声。")
    with c2:
        st.metric("频谱特征", f"{comp['spectral_similarity']}%",
                  help="子带能量、质心、带宽、平坦度等'整体统计量'的比对 — 像比较两个人的身高体重，而非逐像素比照片。")
    with c3:
        st.metric("频谱形状", f"{comp['spectrum_similarity']}%",
                  help="归一化频谱曲线的形状匹配 — 忽略绝对响度差异，只看'频率分布的轮廓'像不像。录音音量不同不影响此指标。")
    with c4:
        st.metric("谐波指纹", f"{comp['harmonic_similarity']}%",
                  help="特征频率峰的匹配 + 谐波频率比结构比对 — 最核心的指标，类似 DNA 比对，即使两段录音转速不同也能识别同源设备。")
    st.markdown("</div>", unsafe_allow_html=True)  # 关闭指标卡片

    # ── 分项指标通俗解读 ──
    st.markdown("""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;margin:8px 0 16px;color:#475569;font-size:13px;line-height:1.8;'>
<b>📖 四项指标通俗解读：</b><br>
🔸 <b>LFCC 相似度</b>：像"逐段对比两首歌的频谱截图"——每帧切一小段，看各频率子带能量分布像不像。全频段等权重，不偏向人声。<br>
🔸 <b>频谱特征</b>：像"比较两个人的身高、体重、BMI"——不看细节，只比对几个整体统计量（能量重心在哪、频率分布有多宽/多平）。<br>
🔸 <b>频谱形状</b>：像"比较两张剪影的轮廓"——忽略谁响谁轻，只看频率分布的"形状"像不像。即使音量不同也不影响。<br>
🔸 <b>谐波指纹</b>：权重最高（30%）。像"DNA 条形码比对"——每台设备有独特的特征频率组合（峰值的间距比例），不管录音条件怎么变，比例不变。
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
<b>💡 如何看雷达图：</b>五个轴代表五个比对维度，蓝色多边形越"大"越"圆"→ 两段音频越相似。<br>
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
• 曲线高度不同但形状相似 → 仅音量差异，"频谱形状"指标仍会较高<br>
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
