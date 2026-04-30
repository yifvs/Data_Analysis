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
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; }

.stApp { background:#f8fafc !important; min-height:100vh; overflow:visible !important; }
.main > div { padding:2rem 3rem !important; max-width:1100px !important; margin:0 auto !important; overflow:visible !important; }
.stAppDeployButton { display:none !important; }

.css-1lcbxhc { background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%) !important; border-right:1px solid #e2e8f0 !important; }
.css-154b5vr { padding:24px 16px 12px !important; }
.css-154bvr h3,.css-154bvr label { color:#1e293b !important; font-weight:600 !important; }
h1 { color:#0f172a !important; font-weight:800 !important; letter-spacing:-0.6px !important; }
h2 { color:#4f46e5 !important; font-weight:700 !important; margin-top:32px !important; }
p { color:#475569 !important; line-height:1.7 !important; }

.stFileUploader { background:#fff !important; border:2px dashed #cbd5e1 !important; border-radius:16px !important; padding:36px 20px !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); transition:all 0.25s; }
.stFileUploader:hover { border-color:#818cf8 !important; background:#fafbff !important; box-shadow:0 4px 12px rgba(99,102,241,0.10) !important; }
.stFileUploader > label { color:#64748b !important; font-size:15px !important; font-weight:500 !important; }
/* 云端环境隐藏 uploader 内部重复文字，避免与自定义 label 重叠 */
.stFileUploader span[data-testid="stMarkdownContainer"],
.stFileUploader .stFileUploaderDropContainer > span:not(:first-child):not([data-baseweb="visually-hidden"]),
.stFileUploader div[data-testid="stCaptionContainer"] + div,
.stFileUploader [data-baseweb="file-uploader"] > div:first-child > span:nth-of-type(2),
.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }

.stTextInput input,.stTextArea textarea { background:#f8fafc !important; border:1.5px solid #cbd5e1 !important; color:#0f172a !important; border-radius:11px !important; padding:10px 14px !important; transition:all 0.2s; }
.stTextInput:focus-within input,.stTextArea:focus-within textarea { border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,0.08) !important; }

.stButton button { border-radius:11px !important; font-weight:600 !important; font-size:14px !important; padding:9px 22px !important; transition:all 0.2s ease !important; }
.stButton > button[kind='primary'] { background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important; border:none !important; color:#fff !important; box-shadow:0 3px 10px rgba(79,70,229,0.28) !important; }
.stButton > button[kind='primary']:hover { transform:translateY(-1px) !important; box-shadow:0 5px 18px rgba(79,70,229,0.38) !important; }
.stButton > button:not([kind='primary']) { background:#fff !important; border:1.5px solid #e2e8f0 !important; color:#334155 !important; }
.stButton > button:not([kind='primary']):hover { background:#f8fafc !important; border-color:#a5b4fc !important; }

.stProgress div[role='progressbar'] { height:7px !important; border-radius:999px !important; background:linear-gradient(90deg,#4f46e5,#a78bfa,#4f46e5) !important; background-size:200% 100% !important; animation:shimmer 2s infinite linear !important; }
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
.stProgress div > div:first-child { height:7px !important; border-radius:999px !important; background:#e2e8f0; }
.stProgress { margin-bottom:8px !important; }
[data-testid="stStatusWidget"], [data-testid="stAlert"] { overflow:visible !important; }

.stTabs [role='tablist'] { gap:4px !important; background:#f1f5f9 !important; border-radius:11px !important; padding:5px !important; }
.stTabs [role='tab'] { border-radius:8px !important; font-weight:500 !important; font-size:13px !important; color:#64748b !important; transition:all 0.2s !important; }
.stTabs [role='tab'][aria-selected='true'] { background:#fff !important; color:#4f46e5 !important; box-shadow:0 1px 4px rgba(0,0,0,0.08) !important; }
.stTabs [aria-selected='false']:hover { color:#334155 !important; background:#e2e8f0 !important; }

.stDataFrame { border-radius:12px !important; overflow:hidden !important; background:#fff !important; border:1px solid #e2e8f0 !important; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.stDataFrame table thead th { background:#f8fafc !important; color:#475569 !important; font-weight:600 !important; font-size:12px !important; text-transform:uppercase !important; letter-spacing:0.4px !important; padding:11px 16px !important; border-bottom:2px solid #e2e8f0 !important; }
.stDataFrame table td { color:#1e293b !important; font-size:13px !important; padding:10px 16px !important; border-bottom:1px solid #f1f5f9 !important; }
.stDataFrame table tr:hover td { background:#fafbff !important; }

.stMetric { background:#fff !important; border:1px solid #e2e8f0 !important; border-radius:14px !important; padding:18px 14px !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.stMetric label { color:#94a3b8 !important; font-size:12px !important; font-weight:600 !important; text-transform:uppercase !important; }
.stMetric div[data-testid='stMetricValue'] { color:#0f172a !important; font-weight:700 !important; font-size:20px !important; }

.stExpander { border:1px solid #e2e8f0 !important; border-radius:14px !important; background:#fff !important; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.stExpander > summary { color:#4f46e5 !important; font-weight:600 !important; font-size:14px !important; }

.stCaption { color:#94a3b8 !important; font-size:12px !important; }
hr, hr + * { display:none !important; }
.stInfo { background:#eff6ff !important; border:1px solid #bfdbfe !important; border-radius:12px !important; padding:16px 20px !important; }
.stInfo p,.stInfo span { color:#2563eb !important; font-size:15px !important; }
.stSuccess { background:#ecfdf5 !important; border:1px solid #a7f3d0 !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
.stSuccess p, .stSuccess span { color:#059669 !important; font-size:16px !important; font-weight:700 !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }
.stWarning { background:#fffbeb !important; border:1px solid #fde68a !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
.stWarning p, .stWarning span { color:#d97706 !important; font-size:16px !important; font-weight:600 !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }
.stError { background:#fef2f2 !important; border:1px solid #fecaca !important; border-radius:12px !important; margin:16px 0 !important; padding:18px 20px !important; min-height:52px !important; height:auto !important; box-sizing:border-box !important; overflow:visible !important; }
.stError p, .stError span { color:#dc2626 !important; font-size:15px !important; line-height:1.6 !important; white-space:normal !important; word-break:break-word !important; display:block !important; }

/* 强制状态消息容器不被裁剪 */
div[data-testid="stStatusWidget"] { min-height:auto !important; max-height:none !important; height:auto !important; overflow:visible !important; }
[data-testid="stAlertContainer"] > div { height:auto !important; min-height:auto !important; max-height:none !important; overflow:visible !important; }
/* ═════════════════  云端环境 Material Icon 泄露文本修复  ═════════════════
   根因：云端环境 Material Icons 字体未加载，icon ligature 文本泄露为可见文字。
   三处泄露：① Uploader 内 "upload"  ② Expander 内 "_arrow"  ③ Sidebar 内 "keyboard_double_..."
   治本：引入 Google Fonts Material Icons   治标：对三处容器做核弹级隐藏 */

/* A. @import Material Icons 字体（治本：让 ligature 渲染为图标而非文本） */
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

/* B1. FileUploader: 核弹级隐藏内部按钮（upload 泄露源），恢复上传区域 */
.stFileUploader [data-baseweb="file-uploader"] > div:first-child,
.stFileUploader [data-baseweb="file-uploader"] > div:first-child > *,
.stFileUploader [data-baseweb="file-uploader"] button,
.stFileUploader [data-baseweb="file-uploader"] button *,
.stFileUploader [role="button"]:not(.stFileUploaderDropContainer) {
  font-size: 0 !important;
  line-height: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: hidden !important;
  visibility: hidden !important;
  display: none !important;
}
/* 但恢复上传区域本身的可见性 */
.stFileUploader .stFileUploaderDropContainer,
.stFileUploader .stFileUploaderDropContainer * {
  font-size: inherit !important;
  line-height: inherit !important;
  visibility: visible !important;
  display: block !important;
  overflow: visible !important;
}

/* B2. Expander: summary 中除第一个子元素外全部隐藏（_arrow 泄露源） */
.stExpander > summary > :not(:first-child),
.stExpander > summary > :not(:first-child) * {
  display: none !important;
  font-size: 0 !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}

/* B3. Sidebar 导航：隐藏所有非首项的次要文字（keyboard_double_ 泄露源） */
.stSidebarNav [class*="MenuItems"] span:not(:first-child),
.stSidebarNav button span:not(:first-child),
.stSidebarNav a span:not(:first-child),
.stSidebarNav div[data-testid*="navitem"] span:nth-child(n+2),
.stSidebarNav [class*="element-container"] span:empty,
.stSidebarNav [class*="container"] span[aria-hidden],
.stApp [class*="sidebar"] svg + span,
.stApp [class*="sidebar"] span:has(+ svg) {
  display: none !important;
  font-size: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
}

/* B4. 全局兜底：所有含 aria-hidden 或空内容的可疑 span */
.stApp span[aria-hidden="true"],
.stApp span:empty:not([data-testid]),
button span[aria-hidden],
label span[aria-hidden] {
  display: none !important;
  font-size: 0 !important;
}
</style>""", unsafe_allow_html=True)


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


def try_decode_audio(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """尝试解码音频文件，优先 WAV，其他格式用 soundfile 兜底"""
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
            st.success("缓存已清除!")
            st.rerun()

        st.markdown("---")
        st.markdown("""<details>
<summary style='cursor:pointer;color:#94a3b8;font-size:13px;font-weight:500;'>&#x2139; 使用说明</summary>
<div style='padding:10px 4px;color:#64748b;font-size:13px;line-height:2;'>
1. 上传音频文件<br/>
2. 点击「开始分析」按钮<br/>
3. 查看图表与诊断结果<br/>
4. 下载完整 HTML 报告<br/><br/>
<b>支持格式:</b> WAV / MP3 / OGG / FLAC / AAC / M4A<br/>
<b>文件大小:</b> 最大 50 MB
</div></details>""", unsafe_allow_html=True)

    # ════════════════════ 主区域 Header ════════════════════
    st.markdown("""
    <div style='text-align:center;padding:28px 0 12px;'>
        <h1 style='font-size:32px;margin-bottom:6px;color:#0f172a;'>
            音频频谱分析系统
        </h1>
        <p style='color:#94a3b8;font-size:14px;letter-spacing:0.3px;'>
            STFT &middot; FFT &middot; 缺陷检测 &middot; AI 智能诊断 &middot; 一键导出
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════ 文件上传区 ════════════════════
    uploaded_file = st.file_uploader(
        "**拖拽或点击上传**",
        type=["wav", "mp3", "ogg", "flac", "aac", "m4a"],
        help="支持 MP3 / WAV / OGG / FLAC / AAC / M4A，最大 50MB",
    )

    # 操作按钮行
    c_left, c_right = st.columns([1.2, 3])
    analyze_clicked = False
    download_clicked = False

    with c_left:
        analyze_clicked = st.button(
            "▶  开始分析",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_file,
        )

    with c_right:
        download_clicked = st.button(
            "⬇  导出 HTML 报告",
            use_container_width=True,
            disabled=not st.session_state.analysis_result,
        )

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

    # ── 下载按钮（按需生成报告，避免 kaleido 阻塞主流程）──
    if download_clicked and st.session_state.analysis_result:
        result = st.session_state.analysis_result
        with st.spinner("正在生成 HTML 报告..."):
            html_content = result.get("report_html") or generate_report_html(result)
            result["report_html"] = html_content  # 缓存起来避免重复生成
        safe_name = (
            re.sub(r"[^\w\u4e00-\u9fff_-]", "_", Path(result["file_name"]).stem)
            + "_音频分析报告_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".html"
        )
        st.download_button(
            label="📄 点击下载报告文件",
            data=html_content.encode("utf-8"),
            file_name=safe_name,
            mime="text/html;charset=utf-8",
            use_container_width=True,
            type="primary",
        )

    # ── 结果展示区 ──
    result = st.session_state.analysis_result
    if result:
        # 检测旧版 matplotlib 缓存（base64 字符串 vs 新版 plotly Figure）
        _old_keys = ("waveform_img", "spectrum_img", "spectrogram_img")
        if any(_k in result for _k in _old_keys) and "waveform_fig" not in result:
            # 旧格式不兼容，清除缓存提示重新分析
            st.session_state.analysis_result = None
            st.warning("⚠️ 检测到旧版本分析结果缓存，请重新上传文件并分析")
        else:
            render_results(result)


def render_results(result: dict):
    """渲染分析结果面板（浅色主题）"""

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

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
                 help='峰值与 RMS 的差值 — 衡量信号的起伏幅度。大动态范围说明有明显的强弱对比（如音乐中的轻柔段落与高潮）；过小说明信号比较"平"，缺乏层次感。')
    with c4:
        st.metric("检测缺陷", f"{len(result['defects'])} 项", delta_color="off")

    # ════ 评分大卡片 ════
    sc = result["quality_score"]
    if sc >= 80:
        score_color, score_bg = "#059669", "#ecfdf5"
    elif sc >= 60:
        score_color, score_bg = "#d97706", "#fffbeb"
    else:
        score_color, score_bg = "#dc2626", "#fef2f2"

    _html = f"""<div style='background:#ffffff;border:1.5px solid #e2e8f0;border-radius:18px;
        padding:32px;text-align:center;margin:20px 0 24px;box-shadow:0 2px 12px rgba(0,0,0,0.05);'>
      <div style='font-size:56px;font-weight:800;color:{score_color};letter-spacing:-2px;line-height:1;'>
          {sc}<span style='font-size:26px;font-weight:600;color:#94a3b8;'>/100</span>
      </div>
      <div style='font-size:22px;font-weight:700;color:#0f172a;margin-top:6px;'>{result['quality_grade']}</div>
      <div style='color:#94a3b8;font-size:13px;margin-top:10px;display:flex;align-items:center;justify-content:center;gap:14px;'>
          <span>采样率 {result['sample_rate']}Hz</span><span>|</span>
          <span>时长 {result['duration']:.2f}s</span><span>|</span>
          <span>{result['file_name']}</span>
      </div>
    </div>"""
    st.markdown(_html, unsafe_allow_html=True)

    # ════ 图表 Tabs ════
    tab_wave, tab_spec, tab_stft = st.tabs(["📊 时域波形", "🎚️ 频率谱 (FFT)", "🌡️ STFT 时频谱"])

    with tab_wave:
        st.plotly_chart(result["waveform_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption("包络波形 — 像素级 min/max 检测，按峰峰值归一化")

    with tab_spec:
        st.plotly_chart(result["spectrum_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption(f"FFT 频率谱 · Hann 窗 (N={result['fft_size']})")

    with tab_stft:
        st.plotly_chart(result["spectrogram_fig"], use_container_width=True, config={"displayModeBar": False})
        st.caption("STFT 时频谱图 · Hann 窗 · ~75% 重叠 · 对数幅度压缩")

    # ════ 频谱峰值 ════
    st.subheader("📋  频谱峰值")
    if result["spectrum_peaks"]:
        peak_data = [
            {"频率": f"{p['frequency']:.1f} Hz", "相对强度": f"{p['magnitude']*100:.1f}%", "频段": p["label"]}
            for p in result["spectrum_peaks"]
        ]
        st.dataframe(peak_data, use_container_width=True, hide_index=True)
    else:
        st.info("未检测到明显峰值信号")

    # ════ 缺陷诊断 ════
    st.subheader("⚠️  缺陷诊断")
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
    else:
        st.success("未检测到异常缺陷，音频信号质量良好")

    # ════ AI 分析 ════
    with st.expander("🤖  AI 智能分析报告", expanded=False):
        _ai_html = f"<div style='background:#fafbff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;" \
                   f"white-space:pre-wrap;line-height:1.9;color:#334155;font-size:14px;'>" \
                   f"{result['ai_analysis']}</div>"
        st.markdown(_ai_html, unsafe_allow_html=True)

    # ════ 分析历史 ════
    history = st.session_state.analysis_history or []
    if history:
        st.subheader("🕐  分析历史")
        n_cols = min(len(history), 5)
        hist_cols = st.columns(n_cols)
        for i, entry in enumerate(history):
            with hist_cols[i % n_cols]:
                _sc = entry.get("score", 0)
                _c = "#059669" if _sc >= 80 else ("#d97706" if _sc >= 60 else "#dc2626")
                _hist = f"""
<div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04);'>
<div style='font-size:11px;color:#94a3b8;margin-bottom:4px;'>{entry['time']}</div>
<div style='font-size:12.5px;color:#1e293b;font-weight:500;' title="{entry['name']}">{entry['name'][:14]}{'...' if len(entry['name']) > 14 else ''}</div>
<div style='font-size:14px;font-weight:700;color:{_c};margin-top:6px;'>{entry['grade']} &middot; {_sc}分</div>
</div>"""
                st.markdown(_hist, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
