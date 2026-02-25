from huggingface_hub import snapshot_download
from pathlib import Path

target_dir = Path("GPUModel") / "Qwen3-ASR-0.6B"
print(f"開始下載 Qwen3-ASR-0.6B 模型至 {target_dir}...")

snapshot_download(
    repo_id="Qwen/Qwen3-ASR-0.6B",
    local_dir=target_dir,
    allow_patterns=["*.bin", "*.json", "*.safetensors", "*.model", "*.txt", "*.jsonnet"],
    ignore_patterns=["*.msgpack", "*.h5", "*.gguf", "*.onnx", "*.pb"],
)

print(f"✅ Qwen3-ASR-0.6B 下載完成！請在 GUI 中重新載入模型。")
