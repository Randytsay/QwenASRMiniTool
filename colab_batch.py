import os
import time
from pathlib import Path
import numpy as np
import torch
import librosa
from qwen_asr import Qwen3ASRModel
import onnxruntime as ort
import opencc

# ── 設定 ──────────────────────────────────────────────────
# 支援從 Colab 環境變數中讀取，方便在 Notebook 內即時修改
# 可以將反斜線替換為正斜線以符合 Colab (Linux) 環境
TARGET_DIR = os.environ.get("TARGET_DIR", "/content/drive/MyDrive/01 美安/01 態度與知識/01 GMTSS課程/09 產品專題會/2023美安台灣產品專題會/公司提供錄音檔")

# 支援的格式
EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".mp4", ".mkv", ".mov"}
# 模型路徑 (Colab 上的相對路徑)
BASE_DIR      = Path("/content/QwenASRMiniTool")
GPU_MODEL_DIR = BASE_DIR / "GPUModel"
OV_MODEL_DIR  = BASE_DIR / "ov_models"
VAD_PATH      = GPU_MODEL_DIR / "silero_vad_v4.onnx"

# ── 初始化 ────────────────────────────────────────────────
print("🔄 正在初始模型...")
def load_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = Qwen3ASRModel.from_pretrained(
        str(GPU_MODEL_DIR / "Qwen3-ASR-1.7B"), 
        device_map=device, 
        dtype=dtype
    )
    vad_sess = ort.InferenceSession(str(VAD_PATH), providers=["CPUExecutionProvider"])
    cc = opencc.OpenCC("s2twp")
    return {"model": model, "vad_sess": vad_sess, "cc": cc, "device": device}

eng = load_engine()

def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def _detect_speech_groups(audio: np.ndarray, vad_sess) -> list[tuple[float, float, np.ndarray]]:
    VAD_CHUNK = 512; VAD_THRESHOLD = 0.5; SAMPLE_RATE = 16000; MAX_GROUP_SEC = 20
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    
    MIN_CH = 16; PAD = 5; MERGE = 16
    raw = []
    in_sp = False; s0 = 0
    for i, p in enumerate(probs):
        if p >= VAD_THRESHOLD and not in_sp:
            s0 = i; in_sp = True
        elif p < VAD_THRESHOLD and in_sp:
            if i - s0 >= MIN_CH: raw.append((max(0, s0-PAD), min(n, i+PAD)))
            in_sp = False
    
    if not raw: return []
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= MERGE: merged[-1][1] = e
        else: merged.append([s, e])
    
    groups = []; gs = merged[0][0] * VAD_CHUNK; ge = merged[0][1] * VAD_CHUNK
    mx_samp = MAX_GROUP_SEC * SAMPLE_RATE
    for seg in merged[1:]:
        s = seg[0] * VAD_CHUNK; e = seg[1] * VAD_CHUNK
        if e - gs > mx_samp:
            groups.append((gs, ge)); gs = s
        ge = e
    groups.append((gs, ge))
    
    res = []
    for gs, ge in groups:
        ns = max(1, int((ge - gs) // SAMPLE_RATE))
        ch = audio[gs: gs + ns * SAMPLE_RATE].astype(np.float32)
        if len(ch) >= SAMPLE_RATE:
            res.append((gs / SAMPLE_RATE, gs / SAMPLE_RATE + ns, ch))
    return res

def process_file_in_place(file_path, eng):
    path = Path(file_path)
    srt_path = path.with_suffix(".srt")
    txt_path = path.with_suffix(".txt")
    
    if srt_path.exists():
        print(f"⏭️  跳過 (已存在): {path.name}")
        return

    print(f"🎙️  處理中: {path.name}...")
    try:
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        groups = _detect_speech_groups(audio, eng["vad_sess"])
        
        all_subs = []
        full_text = []
        for g0, g1, chunk in groups:
            results = eng["model"].transcribe([(chunk, 16000)])
            text = eng["cc"].convert(results[0].text.strip()) if results else ""
            if text:
                full_text.append(text)
                all_subs.append((g0, g1, text))
        
        if not all_subs: return

        srt_lines = []
        for i, (s, e, t) in enumerate(all_subs, 1):
            srt_lines.append(f"{i}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{t}\n")
            
        with open(srt_path, "w", encoding="utf-8") as f: f.write("\n".join(srt_lines))
        with open(txt_path, "w", encoding="utf-8") as f: f.write(" ".join(full_text))
        print(f"✅ 完成: {path.name}")
    except Exception as e:
        print(f"❌ 錯誤 ({path.name}): {e}")

# ── 執行掃描 ──────────────────────────────────────────
print(f"🔍 開始掃描: {TARGET_DIR}")
for root, dirs, files in os.walk(TARGET_DIR):
    for file in files:
        if Path(file).suffix.lower() in EXTENSIONS:
            process_file_in_place(os.path.join(root, file), eng)

print("✅ 所有任務完成！")
