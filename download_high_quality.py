import downloader
from pathlib import Path

model_dir = Path("ov_models")
print("=== 下載 1.7B 高級模型 ===")
downloader.download_1p7b(model_dir, progress_cb=downloader._cli_bar)
print("\n=== 下載說話者分離模型 ===")
downloader.download_diarization(model_dir / "diarization", progress_cb=downloader._cli_bar)
print("\n✅ 所有高級模型下載完成")
