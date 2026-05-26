import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Popup window to monitor U-Net fitting.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--refresh-ms", type=int, default=2000)
    parser.add_argument("--window-title", type=str, default="U-Net Fitting Monitor")
    return parser.parse_args()


def read_history(history_path: Path) -> list[dict[str, float]]:
    if not history_path.exists():
        return []

    rows: list[dict[str, float]] = []
    with history_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value not in (None, ""):
                    parsed[key] = float(value)
            if parsed:
                rows.append(parsed)
    return rows


def latest_val_visual(visuals_dir: Path) -> Path | None:
    files = sorted(visuals_dir.glob("val_epoch_*.png"))
    if files:
        return files[-1]

    fallbacks = sorted(visuals_dir.glob("*samples*.png"))
    if fallbacks:
        return fallbacks[-1]
    return None


def summarize_fit(history: list[dict[str, float]]) -> tuple[str, str]:
    if not history:
        return "waiting", "还没有训练记录，先启动训练。"

    latest = history[-1]
    epoch = int(latest["epoch"])
    train_loss = latest.get("train_loss", 0.0)
    val_loss = latest.get("val_loss", 0.0)
    train_dice = latest.get("train_dice", 0.0)
    val_dice = latest.get("val_dice", 0.0)
    gap = train_dice - val_dice

    if epoch < 3:
        return "warming up", f"第 {epoch} 轮，模型还在起步阶段。先继续看 3-5 个 epoch。"
    if gap > 0.12 and val_loss > train_loss:
        return "possible overfitting", f"训练 Dice 比验证 Dice 高 {gap:.3f}，而且 val_loss 更高，可能开始过拟合。"
    if train_dice < 0.55 and val_dice < 0.50:
        return "underfitting", f"train/val Dice 还比较低 ({train_dice:.3f}/{val_dice:.3f})，更像还没学够。"
    if val_dice >= 0.70 and gap < 0.08:
        return "healthy fitting", f"验证 Dice {val_dice:.3f}，训练验证差距 {gap:.3f}，目前拟合比较健康。"
    return "still improving", f"验证 Dice {val_dice:.3f}，训练验证差距 {gap:.3f}，先继续观察趋势。"


class MonitorApp:
    def __init__(self, run_dir: Path, refresh_ms: int, window_title: str) -> None:
        self.run_dir = run_dir
        self.refresh_ms = max(refresh_ms, 500)
        self.history_path = run_dir / "history.csv"
        self.visuals_dir = run_dir / "visuals"
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1480x980")
        self.root.minsize(1180, 780)

        self.status_var = tk.StringVar(value="waiting")
        self.summary_var = tk.StringVar(value="准备读取训练结果...")
        self.metrics_var = tk.StringVar(value="暂无指标")
        self.paths_var = tk.StringVar(value=f"run dir: {self.run_dir}")
        self.after_id: str | None = None

        self.curves_photo: ImageTk.PhotoImage | None = None
        self.samples_photo: ImageTk.PhotoImage | None = None

        self.build_ui()
        self.refresh()

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        title = ttk.Label(top, text="U-Net Fitting Monitor", font=("Microsoft YaHei UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Refresh", command=self.refresh).grid(row=0, column=1, sticky="e", padx=(8, 0))

        ttk.Label(top, textvariable=self.status_var, font=("Microsoft YaHei UI", 11, "bold")).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(top, textvariable=self.summary_var, wraplength=1080, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(top, textvariable=self.metrics_var, wraplength=1080, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(top, textvariable=self.paths_var, wraplength=1080, justify="left", foreground="#666666").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        content = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        curves_frame = ttk.Labelframe(content, text="Training Curves", padding=8)
        samples_frame = ttk.Labelframe(content, text="Latest Validation Visualization", padding=8)
        content.add(curves_frame, weight=1)
        content.add(samples_frame, weight=1)

        curves_frame.columnconfigure(0, weight=1)
        curves_frame.rowconfigure(0, weight=1)
        samples_frame.columnconfigure(0, weight=1)
        samples_frame.rowconfigure(0, weight=1)

        self.curves_label = ttk.Label(curves_frame, text="还没有曲线图。")
        self.curves_label.grid(row=0, column=0, sticky="nsew")

        self.samples_label = ttk.Label(samples_frame, text="还没有验证可视化图。")
        self.samples_label.grid(row=0, column=0, sticky="nsew")

    def set_image(self, label: ttk.Label, path: Path | None, max_width: int, max_height: int, slot: str) -> None:
        if path is None or not path.exists():
            label.configure(text="暂无图像", image="")
            if slot == "curves":
                self.curves_photo = None
            else:
                self.samples_photo = None
            return

        image = Image.open(path).convert("RGB")
        image.thumbnail((max_width, max_height), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        if slot == "curves":
            self.curves_photo = photo
        else:
            self.samples_photo = photo

    def refresh(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        history = read_history(self.history_path)
        state, summary = summarize_fit(history)
        self.status_var.set(f"状态: {state}")
        self.summary_var.set(summary)

        if history:
            latest = history[-1]
            metrics_text = (
                f"epoch={int(latest['epoch'])} | "
                f"train_loss={latest.get('train_loss', 0.0):.4f} val_loss={latest.get('val_loss', 0.0):.4f} | "
                f"train_dice={latest.get('train_dice', 0.0):.4f} val_dice={latest.get('val_dice', 0.0):.4f} | "
                f"train_iou={latest.get('train_iou', 0.0):.4f} val_iou={latest.get('val_iou', 0.0):.4f}"
            )
            self.metrics_var.set(metrics_text)
        else:
            self.metrics_var.set("暂无指标")

        curves_path = self.visuals_dir / "training_curves.png"
        sample_path = latest_val_visual(self.visuals_dir)
        self.paths_var.set(
            f"run dir: {self.run_dir} | curves: {curves_path if curves_path.exists() else 'missing'} | "
            f"latest sample: {sample_path if sample_path is not None else 'missing'}"
        )

        self.set_image(self.curves_label, curves_path if curves_path.exists() else None, 1380, 380, "curves")
        self.set_image(self.samples_label, sample_path, 1380, 440, "samples")
        self.after_id = self.root.after(self.refresh_ms, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    app = MonitorApp(args.run_dir, args.refresh_ms, args.window_title)
    app.run()


if __name__ == "__main__":
    main()
