"""
Thermal Camera Gradient Analysis Tool
Usage: thermal_analysis.exe <image_path>
       or double-click to open file dialog
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")          # interactive window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, RadioButtons
from pathlib import Path

try:
    import flyr
    FLYR_AVAILABLE = True
except ImportError:
    FLYR_AVAILABLE = False


# ── file dialog ──────────────────────────────────────────────────────────────

def ask_file(title="Select thermal image", filetypes=None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes or [("Image / Pickle", "*.jpg *.jpeg *.png *.tiff *.pkl"), ("All", "*.*")]
    )
    root.destroy()
    return path or None


# ── load ─────────────────────────────────────────────────────────────────────

def load_temperature(filepath: str) -> np.ndarray:
    if FLYR_AVAILABLE and filepath.lower().endswith((".jpg", ".jpeg")):
        try:
            return flyr.unpack(filepath).celsius.astype(np.float32)
        except Exception:
            pass
    raw = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {filepath}")
    return raw.astype(np.float32)


def normalize_u8(arr):
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr, dtype=np.uint8) if mx == mn else (255*(arr-mn)/(mx-mn)).astype(np.uint8)


# ── compute ───────────────────────────────────────────────────────────────────

def compute(temp: np.ndarray) -> dict:
    img     = (255 * temp / temp.max()).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (5,5), sigmaX=1.5)

    sx_raw  = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy_raw  = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sx      = cv2.convertScaleAbs(sx_raw)
    sy      = cv2.convertScaleAbs(sy_raw)

    combined    = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
    sobel_norm  = normalize_u8(combined)
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    sobel_clahe = clahe.apply(sobel_norm)

    magnitude   = np.sqrt(sx_raw**2 + sy_raw**2).astype(np.float32)
    mag_thresh  = np.where(magnitude >= np.percentile(magnitude, 75), magnitude, 0).astype(np.float32)
    angle       = np.arctan2(sy_raw, sx_raw).astype(np.float32)   # -pi to pi
    canny       = cv2.Canny(sobel_norm, 30, 100)

    orig_color  = cv2.applyColorMap(normalize_u8(temp), cv2.COLORMAP_INFERNO)
    edge_color  = cv2.applyColorMap(sobel_clahe,         cv2.COLORMAP_HOT)
    overlay_rgb = cv2.cvtColor(cv2.addWeighted(orig_color,0.6,edge_color,0.4,0), cv2.COLOR_BGR2RGB)

    return dict(
        temp           = temp,        # float32  Celsius
        blurred        = blurred,     # float32  gaussian smoothed
        sobelx_raw     = sx_raw,      # float64  signed sobel X
        sobely_raw     = sy_raw,      # float64  signed sobel Y
        sobelx         = sx,          # uint8    abs sobel X
        sobely         = sy,          # uint8    abs sobel Y
        sobel_combined = combined,    # uint8    combined abs
        sobel_clahe    = sobel_clahe, # uint8    CLAHE enhanced
        magnitude      = magnitude,   # float32  gradient magnitude
        mag_thresh     = mag_thresh,  # float32  top-25% only
        angle          = angle,       # float32  radians  -pi..pi
        canny          = canny,       # uint8    binary edges
        overlay_rgb    = overlay_rgb, # uint8    RGB overlay
    )


# ── save full gradient arrays as CSVs ────────────────────────────────────────

def save_full_csvs(data: dict, out_dir: Path, stem: str):
    """Save only temp, gradX, gradY as full-image CSVs."""
    import csv as _csv
    for key in ("temp", "sobelx_raw", "sobely_raw"):
        arr  = data[key]
        path = out_dir / f"{stem}_{key}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            for row in arr:
                w.writerow([f"{v:.6f}" for v in row])
        print(f"  [saved] {path.name}  {arr.shape}")


# ── save static PNGs ──────────────────────────────────────────────────────────

def save_all_pngs(data, out_dir, stem):
    panels = {
        "original"         : (data["temp"],        "inferno"),
        "magnitude"        : (data["magnitude"],    "hot"),
        "magnitude_thresh" : (data["mag_thresh"],   "hot"),
        "overlay"          : (data["overlay_rgb"],  None),
        "gradient_angle"   : (data["angle"],        "hsv"),
    }
    for name, (img, cmap) in panels.items():
        fig, ax = plt.subplots(figsize=(8,6), facecolor="#0e0e0e")
        ax.imshow(img, cmap=cmap);  ax.axis("off")
        ax.set_title(name.replace("_"," ").title(), color="white", fontsize=10)
        p = out_dir / f"{stem}_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
        plt.close(fig)
        print(f"  [saved] {p.name}")

    # combined grid
    fig, axes = plt.subplots(2, 3, figsize=(15,10), facecolor="#111")
    for ax, (title, (img, cmap)) in zip(axes.flatten(), panels.items()):
        ax.imshow(img, cmap=cmap);  ax.axis("off")
        ax.set_title(title.replace("_"," ").title(), color="white", fontsize=9)
    axes.flatten()[-1].set_visible(False)
    fig.suptitle(f"Thermal Analysis — {stem}", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    p = out_dir / f"{stem}_grid.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"  [saved] {p.name}")


# ── quiver builder ────────────────────────────────────────────────────────────

def build_quiver_fig(data, subsample=15):
    temp = data["temp"]; sx = data["sobelx_raw"]; sy = data["sobely_raw"]
    magnitude = data["magnitude"]
    strong = magnitude > np.percentile(magnitude, 75)
    h, w   = temp.shape
    ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
    u = sx[ys,xs].astype(float);  v = -sy[ys,xs].astype(float)
    mask  = strong[ys,xs]
    norm  = np.sqrt(u**2+v**2)+1e-8
    u_n, v_n = u/norm, v/norm
    colors = plt.cm.hsv((np.arctan2(v_n,u_n)+np.pi)/(2*np.pi))

    fig, ax = plt.subplots(figsize=(12,8), facecolor="black")
    ax.imshow(temp, cmap="inferno", alpha=0.85)
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            if mask[i,j]:
                ax.annotate("",
                    xy    =(xs[i,j]+u_n[i,j]*6, ys[i,j]+v_n[i,j]*6),
                    xytext=(xs[i,j],              ys[i,j]),
                    arrowprops=dict(arrowstyle="->", color=colors[i,j], lw=1.1))
    ax_w = fig.add_axes([0.80,0.76,0.16,0.16], projection="polar")
    theta = np.linspace(0,2*np.pi,256); r = np.linspace(0.4,1,10)
    T, R  = np.meshgrid(theta, r)
    ax_w.pcolormesh(T, R, T, cmap="hsv", shading="auto")
    ax_w.set_yticklabels([])
    ax_w.set_xticklabels(["E","","N","","W","","S",""], fontsize=7, color="white")
    ax_w.set_title("Direction", fontsize=8, color="white", pad=2)
    ax_w.spines["polar"].set_visible(False); ax_w.set_facecolor("black")
    ax.set_title("Gradient Direction (arrow=heat flow, colour=angle)", color="white", fontsize=11)
    ax.axis("off"); fig.patch.set_facecolor("black")
    return fig


# ── interactive viewer ────────────────────────────────────────────────────────

PANEL_DEFS = [
    ("Temperature (°C)",      "temp",        "inferno"),
    ("Gradient Magnitude",    "magnitude",    "hot"),
    ("Magnitude Thresholded", "mag_thresh",   "hot"),
    ("Gradient Angle (rad)",  "angle",        "hsv"),
    ("Overlay",               "overlay_rgb",  None),
]


def ask_real_distance_dialog(default=10.0):
    """Small tkinter popup asking for real-world distance in cm."""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk(); root.withdraw()
    val = simpledialog.askfloat(
        "Calibration",
        "Enter the REAL distance between the two points (cm):",
        initialvalue=default, minvalue=0.01
    )
    root.destroy()
    return val  # None if cancelled


def circle_stats(arr2d, cx, cy, radius_px):
    """Return stats of arr2d pixels within a circle."""
    h, w = arr2d.shape
    Y, X  = np.ogrid[:h, :w]
    mask  = (X - cx)**2 + (Y - cy)**2 <= radius_px**2
    vals  = arr2d[mask].ravel().astype(float)
    if vals.size == 0:
        return {}
    return dict(n=vals.size, mn=vals.min(), mx=vals.max(),
                mean=vals.mean(), std=vals.std(),
                p25=np.percentile(vals,25), p75=np.percentile(vals,75))


def save_roi_data(data: dict, cx: int, cy: int, r_px: float, r_cm: float,
                  px_cm: float, direc: dict, out_dir: Path, stem: str) -> Path:
    """
    Extract every pixel inside the circle from ALL arrays and save as a single CSV.
    Columns: pixel coords, real-world coords, all scalar arrays, plus a metadata
    header block at the top of the file.
    """
    import csv

    h, w  = data["temp"].shape
    Y, X  = np.ogrid[:h, :w]
    mask  = (X - cx)**2 + (Y - cy)**2 <= r_px**2

    ys, xs = np.where(mask)
    dx_cm  = (xs - cx) / px_cm
    dy_cm  = (ys - cy) / px_cm

    scalar_keys = ["temp", "sobelx_raw", "sobely_raw"]

    tag      = f"roi_x{cx}_y{cy}_r{r_cm:.1f}cm"
    csv_path = out_dir / f"{stem}_{tag}.csv"

    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)

        # ── metadata header (prefixed with #) ────────────────────────────────
        w_csv.writerow(["# METADATA"])
        w_csv.writerow(["# stem",        stem])
        w_csv.writerow(["# centre_px",   f"{cx},{cy}"])
        w_csv.writerow(["# centre_cm",   f"{cx/px_cm:.4f},{cy/px_cm:.4f}"])
        w_csv.writerow(["# radius_cm",   f"{r_cm:.4f}"])
        w_csv.writerow(["# radius_px",   f"{r_px:.2f}"])
        w_csv.writerow(["# px_per_cm",   f"{px_cm:.6f}"])
        w_csv.writerow(["# n_pixels",    int(mask.sum())])
        w_csv.writerow(["#"])
        w_csv.writerow(["# DIRECTIONAL GRADIENTS (mean magnitude)"])
        for d in ["N", "S", "E", "W"]:
            w_csv.writerow([f"# grad_{d}", f"{direc[d]:.6f}", f"per_cm", f"{direc[d]/r_cm:.6f}"])
        w_csv.writerow(["# net_EW", f"{direc['net_x']:+.6f}"])
        w_csv.writerow(["# net_NS", f"{direc['net_y']:+.6f}"])
        w_csv.writerow(["#"])
        w_csv.writerow(["# ARRAY STATS (over circle)"])
        for k in scalar_keys:
            arr = data[k]
            if arr.ndim == 2:
                v = arr[mask].astype(float)
                w_csv.writerow([f"# {k}", f"min={v.min():.4f}", f"max={v.max():.4f}",
                                f"mean={v.mean():.4f}", f"std={v.std():.4f}"])
        w_csv.writerow(["#"])

        # ── column headers ────────────────────────────────────────────────────
        cols = ["px_x", "px_y", "dx_cm", "dy_cm"] + scalar_keys
        w_csv.writerow(cols)

        # ── per-pixel rows ────────────────────────────────────────────────────
        arrays = {k: data[k][mask].ravel() for k in scalar_keys if data[k].ndim == 2}
        for i in range(len(xs)):
            row = [xs[i], ys[i], f"{dx_cm[i]:.4f}", f"{dy_cm[i]:.4f}"]
            row += [f"{arrays[k][i]:.6f}" for k in scalar_keys]
            w_csv.writerow(row)

    # ── summary txt ───────────────────────────────────────────────────────────
    txt_path = out_dir / f"{stem}_{tag}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"ROI Summary  —  {stem}\n{'='*50}\n")
        f.write(f"Centre      : ({cx}, {cy}) px  =  ({cx/px_cm:.2f}, {cy/px_cm:.2f}) cm\n")
        f.write(f"Radius      : {r_cm:.2f} cm  ({r_px:.1f} px)\n")
        f.write(f"Pixels      : {int(mask.sum())}\n\n")
        f.write("Directional gradient (mean magnitude / /cm)\n")
        for d in ["N","S","E","W"]:
            f.write(f"  {d}  :  {direc[d]:.6f}   ({direc[d]/r_cm:.6f} /cm)\n")
        f.write(f"  net E-W  :  {direc['net_x']:+.6f}\n")
        f.write(f"  net N-S  :  {direc['net_y']:+.6f}\n\n")
        f.write("Per-array stats (inside circle)\n")
        for k in scalar_keys:
            arr = data[k]
            if arr.ndim == 2:
                v = arr[mask].astype(float)
                f.write(f"  {k:<20}  min={v.min():>10.4f}  max={v.max():>10.4f}"
                        f"  mean={v.mean():>10.4f}  std={v.std():>10.4f}\n")

    print(f"  [ROI saved]  {csv_path.name}")
    print(f"  [ROI saved]  {txt_path.name}")
    return csv_path


def directional_grad(mag, sx_raw, sy_raw, cx, cy, r_px):
    """
    Mean gradient magnitude in each of the 4 cardinal half-circles.
    N = pixels above centre, S = below, E = right, W = left.
    Also computes the net X (E-W) and Y (N-S) gradient component.
    """
    h, w   = mag.shape
    Y, X   = np.ogrid[:h, :w]
    in_c   = (X-cx)**2 + (Y-cy)**2 <= r_px**2   # inside circle mask
    dx = X - cx;  dy = Y - cy                    # relative coords

    masks = {
        "N": in_c & (dy < 0),   # image y increases downward → N = dy<0
        "S": in_c & (dy > 0),
        "E": in_c & (dx > 0),
        "W": in_c & (dx < 0),
    }
    result = {}
    for name, m in masks.items():
        v = mag[m].ravel().astype(float)
        result[name] = float(v.mean()) if v.size else 0.0

    # net signed component from raw sobel (mean over full circle)
    in_flat = in_c.ravel()
    net_x = float(sx_raw.ravel()[in_flat].mean())   # + = E dominant
    net_y = float(sy_raw.ravel()[in_flat].mean())   # + = S dominant (image coords)
    result["net_x"] = net_x
    result["net_y"] = net_y
    return result


def show_stats_popup(cx, cy, r_cm, r_px, px_cm, panel_title, s_panel, s_mag, direc):
    """Open a small standalone figure with all measurement stats."""
    import matplotlib.patches as mpatches

    fig_s, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                               gridspec_kw={"width_ratios": [1, 1]},
                               num=f"Stats  ({cx},{cy})  r={r_cm:.1f}cm")
    fig_s.patch.set_facecolor("#141414")
    for ax in axes: ax.set_facecolor("#141414"); ax.axis("off")

    # ── LEFT: text stats ──────────────────────────────────────────────────────
    ax_txt = axes[0]
    lines = [
        ("CIRCLE", "#888888"),
        (f"Centre  :  ({cx}, {cy}) px", "white"),
        (f"         ({cx/px_cm:.2f}, {cy/px_cm:.2f}) cm", "#aaaaaa"),
        (f"Radius  :  {r_cm:.1f} cm  ({r_px:.0f} px)", "white"),
        (f"Pixels  :  {s_panel.get('n',0)}", "white"),
        ("", None),
        (f"── {panel_title} ──", "#888888"),
        (f"Min    {s_panel['mn']:>10.4f}", "#cccccc"),
        (f"Max    {s_panel['mx']:>10.4f}", "#cccccc"),
        (f"Mean   {s_panel['mean']:>10.4f}", "white"),
        (f"Std    {s_panel['std']:>10.4f}", "#aaaaaa"),
        ("", None),
        ("── Gradient Magnitude ──", "#888888"),
        (f"Mean   {s_mag['mean']:>10.4f}", "white"),
        (f"Max    {s_mag['mx']:>10.4f}", "#cccccc"),
        (f"∇/cm   {s_mag['mean']/r_cm:>10.4f}", "#88ffcc"),
        ("", None),
        ("── Directional (mean mag) ──", "#888888"),
        (f"North  {direc['N']:>10.4f}  /cm  {direc['N']/r_cm:.4f}", "#88ccff"),
        (f"South  {direc['S']:>10.4f}  /cm  {direc['S']/r_cm:.4f}", "#ff8888"),
        (f"East   {direc['E']:>10.4f}  /cm  {direc['E']/r_cm:.4f}", "#88ff88"),
        (f"West   {direc['W']:>10.4f}  /cm  {direc['W']/r_cm:.4f}", "#ffcc88"),
        ("", None),
        ("── Net signed component ──", "#888888"),
        (f"E-W    {direc['net_x']:>+10.4f}  (+E / -W)", "#88ff88"),
        (f"N-S    {direc['net_y']:>+10.4f}  (+S / -N)", "#ff8888"),
    ]
    y0 = 0.97; dy = 0.037
    for txt, col in lines:
        if col:
            ax_txt.text(0.04, y0, txt, transform=ax_txt.transAxes,
                        fontsize=8, color=col, va="top", family="monospace")
        y0 -= dy

    # ── RIGHT: compass rose bar chart ─────────────────────────────────────────
    ax_rose = axes[1]
    ax_rose.set_facecolor("#1a1a1a")
    ax_rose.axis("on")
    ax_rose.set_facecolor("#1a1a1a")
    for spine in ax_rose.spines.values(): spine.set_color("#444")
    ax_rose.tick_params(colors="#888")

    dirs   = ["N", "E", "S", "W"]
    vals   = [direc[d] for d in dirs]
    colors = ["#88ccff","#88ff88","#ff8888","#ffcc88"]
    bars   = ax_rose.bar(dirs, vals, color=colors, width=0.5, zorder=3)
    ax_rose.set_ylim(0, max(vals)*1.3 if max(vals) > 0 else 1)
    ax_rose.set_ylabel("Mean gradient magnitude", color="#aaa", fontsize=8)
    ax_rose.set_title("Directional Gradient", color="white", fontsize=10)
    ax_rose.yaxis.label.set_color("#aaa")
    ax_rose.tick_params(axis="both", colors="#aaa", labelsize=8)
    for bar, val in zip(bars, vals):
        ax_rose.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="white")

    # dominant direction arrow
    dom = dirs[vals.index(max(vals))]
    arrow_map = {"N":(0,-0.15), "S":(0,0.15), "E":(0.15,0), "W":(-0.15,0)}
    dx_, dy_ = arrow_map[dom]
    ax_rose.annotate("", xy=(0.5+dx_, 0.12+dy_), xytext=(0.5, 0.12),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax_rose.text(0.5, 0.02, f"dominant: {dom}", transform=ax_rose.transAxes,
                 ha="center", fontsize=8, color="white")

    plt.tight_layout(pad=1.5)
    fig_s.show()


def launch_interactive(data: dict, stem: str, out_dir: Path):
    plt.rcParams.update({
        "figure.facecolor": "#141414", "axes.facecolor": "#141414",
        "text.color": "white", "axes.labelcolor": "white",
        "xtick.color": "#888", "ytick.color": "#888",
    })

    state = {
        "idx"            : 0,
        "cmap_override"  : None,
        "mode"           : "normal",   # normal | calib1 | calib2 | measure
        "calib_pt1"      : None,
        "calib_px_per_cm": None,
        "calib_artists"  : [],        # all calibration artists to clear
        "meas_artists"   : [],        # circle + crosshair artists
        "radius_cm"      : 5.0,
    }

    # ── figure layout: image (left 72%) | controls (right 28%) ───────────────
    fig = plt.figure(figsize=(15, 8), num=f"Thermal Analysis — {stem}")
    fig.patch.set_facecolor("#141414")

    ax_main = fig.add_axes([0.01, 0.06, 0.65, 0.90])   # image
    ax_cb   = fig.add_axes([0.67, 0.06, 0.012, 0.90])  # colorbar
    ax_main.axis("off")

    im_obj = [None]

    # status bar bottom-centre of image
    status_text = fig.text(0.34, 0.005, "Ready — calibrate first, then measure",
                           fontsize=11, color="#ffcc00", ha="center", va="bottom", family="monospace")

    coord_text = fig.text(0.34, 0.055, "", fontsize=10, color="#888",
                          ha="center", va="bottom", family="monospace")

    # ── RIGHT PANEL controls (stacked, fixed positions) ───────────────────────
    R = 0.705   # left edge of right panel
    W = 0.27    # width
    BH = 0.048  # button height

    def btn(rect, label, bg="#1e1e1e", hover="#333"):
        ax  = fig.add_axes(rect)
        b   = Button(ax, label, color=bg, hovercolor=hover)
        b.label.set_color("white"); b.label.set_fontsize(11)
        return b

    # panel title label
    panel_title_txt = fig.text(R, 0.975, "", fontsize=12, color="white",
                               va="top", ha="left", fontweight="bold")

    # ── navigation
    b_prev = btn([R,        0.915, W/2-0.005, BH], "◀ Prev")
    b_next = btn([R+W/2+0.005, 0.915, W/2-0.005, BH], "Next ▶")

    # ── quiver
    b_quiv = btn([R, 0.855, W, BH], "⬡ Quiver arrows", "#0d2b0d", "#1a4a1a")

    # ── separator label
    fig.text(R, 0.840, "─── Spatial Calibration ───────────────",
             fontsize=9, color="#555", va="top", ha="left", family="monospace")

    # ── calibrate
    b_cal = btn([R, 0.790, W, BH], "📏 Calibrate  (click 2 pts)", "#1a1a3a", "#2a2a5a")

    calib_status = fig.text(R, 0.778, "Not calibrated", fontsize=10,
                            color="#ff6666", va="top", ha="left", family="monospace")

    # ── measure
    fig.text(R, 0.748, "─── Measurement ────────────────────────",
             fontsize=9, color="#555", va="top", ha="left", family="monospace")

    b_meas = btn([R, 0.698, W, BH], "⊙ Measure circle  (click centre)", "#2a1a0a", "#4a3a1a")

    # radius text box
    from matplotlib.widgets import TextBox
    fig.text(R, 0.678, "Radius (cm):", fontsize=10, color="#aaa", va="top", ha="left")
    ax_tb = fig.add_axes([R+0.09, 0.655, W-0.09, 0.028])
    ax_tb.set_facecolor("#2a2a2a")
    tb_r  = TextBox(ax_tb, "", initial="5.0", color="#2a2a2a", hovercolor="#3a3a3a")
    tb_r.text_disp.set_color("white")
    tb_r.text_disp.set_fontsize(11)

    # ── colormap
    fig.text(R, 0.638, "─── Colormap ───────────────────────────",
             fontsize=9, color="#555", va="top", ha="left", family="monospace")

    ax_radio = fig.add_axes([R, 0.475, W, 0.155])
    ax_radio.set_facecolor("#1a1a1a")
    radio = RadioButtons(ax_radio,
                         ("default","hot","inferno","viridis","gray","hsv","plasma"),
                         activecolor="#ff6600")
    for lbl in radio.labels: lbl.set_color("white"); lbl.set_fontsize(10)

    # keyboard hint
    fig.text(R, 0.460, "Keys: ← →  navigate    Esc  cancel",
             fontsize=9, color="#444", va="top", ha="left", family="monospace")

    # panel index quick-ref
    idx_txt = "\n".join(f"  [{i}] {t}" for i,(t,_,_) in enumerate(PANEL_DEFS))
    fig.text(R, 0.440, idx_txt, fontsize=9, color="#444",
             va="top", ha="left", family="monospace")

    # ── draw image ────────────────────────────────────────────────────────────
    def draw(idx):
        title, key, cmap = PANEL_DEFS[idx]
        arr  = data[key]
        cmap = state["cmap_override"] or cmap
        ax_main.cla(); ax_cb.cla(); ax_cb.set_visible(bool(cmap))
        ax_main.axis("off")
        kwargs = dict(cmap=cmap) if cmap else {}
        im = ax_main.imshow(arr, **kwargs)
        im_obj[0] = im
        ax_main.set_title(title, color="white", fontsize=14, pad=6)
        if cmap:
            cb = fig.colorbar(im, cax=ax_cb)
            cb.ax.yaxis.set_tick_params(color="#888", labelcolor="#aaa", labelsize=10)
        panel_title_txt.set_text(title)
        state["meas_artists"] = []
        state["calib_artists"] = []
        fig.canvas.draw_idle()

    # ── wire up controls ──────────────────────────────────────────────────────
    def prev(_): state["idx"]=(state["idx"]-1)%len(PANEL_DEFS); draw(state["idx"])
    def nxt(_):  state["idx"]=(state["idx"]+1)%len(PANEL_DEFS); draw(state["idx"])
    b_prev.on_clicked(prev); b_next.on_clicked(nxt)

    def show_q(_): build_quiver_fig(data).show()
    b_quiv.on_clicked(show_q)

    def start_calib(_):
        state["mode"] = "calib1"
        status_text.set_text("CALIBRATE  →  click point 1")
        fig.canvas.draw_idle()
    b_cal.on_clicked(start_calib)

    def get_radius():
        try:
            v = float(tb_r.text)
            return max(0.1, v)
        except ValueError:
            return 5.0

    def start_meas(_):
        if state["calib_px_per_cm"] is None:
            status_text.set_text("⚠  Calibrate first!")
            fig.canvas.draw_idle(); return
        state["mode"] = "measure"
        r = get_radius()
        status_text.set_text(f"MEASURE  →  click centre   (r = {r:.2f} cm)")
        fig.canvas.draw_idle()
    b_meas.on_clicked(start_meas)

    def set_cmap(lbl):
        state["cmap_override"] = None if lbl == "default" else lbl; draw(state["idx"])
    radio.on_clicked(set_cmap)

    # ── click handler ─────────────────────────────────────────────────────────
    def on_click(ev):
        if ev.inaxes != ax_main or ev.button != 1: return
        x, y = int(ev.xdata), int(ev.ydata)
        mode = state["mode"]

        if mode == "calib1":
            state["calib_pt1"] = (x, y)
            state["mode"]      = "calib2"
            for a in state["calib_artists"]:
                try: a.remove()
                except: pass
            state["calib_artists"] = []
            d, = ax_main.plot(x, y, "o", color="#00ffff", ms=7, zorder=6)
            state["calib_artists"].append(d)
            status_text.set_text("CALIBRATE  →  click point 2")
            fig.canvas.draw_idle()

        elif mode == "calib2":
            x1, y1 = state["calib_pt1"]
            dist_px = np.hypot(x-x1, y-y1)
            real_cm = ask_real_distance_dialog()
            if not real_cm:
                state["mode"] = "normal"
                status_text.set_text("Calibration cancelled")
                fig.canvas.draw_idle(); return
            state["calib_px_per_cm"] = dist_px / real_cm
            state["mode"]            = "normal"
            ln, = ax_main.plot([x1,x],[y1,y], color="#00ffff", lw=1.5, zorder=5)
            d2, = ax_main.plot(x, y, "o", color="#00ffff", ms=7, zorder=6)
            state["calib_artists"] += [ln, d2]
            px_cm = state["calib_px_per_cm"]
            calib_status.set_text(f"✓  {px_cm:.2f} px/cm")
            calib_status.set_color("#66ff66")
            status_text.set_text(f"Calibrated  ✓  {px_cm:.2f} px/cm   |   1 px = {1/px_cm:.3f} cm")
            fig.canvas.draw_idle()

        elif mode == "measure":
            px_cm = state["calib_px_per_cm"]
            r_cm  = get_radius()
            r_px  = r_cm * px_cm

            # clear old overlays
            for a in state["meas_artists"]:
                try: a.remove()
                except: pass
            state["meas_artists"] = []

            import matplotlib.patches as mpatches
            circ = mpatches.Circle((x,y), r_px, fill=False,
                                   edgecolor="#ffcc00", lw=2, zorder=7)
            ax_main.add_patch(circ)
            lx, = ax_main.plot([x-r_px,x+r_px],[y,y],   color="#ffcc00", lw=0.8, ls="--", zorder=6)
            ly, = ax_main.plot([x,x],[y-r_px,y+r_px],   color="#ffcc00", lw=0.8, ls="--", zorder=6)
            # N/S/E/W labels on circle
            off = r_px * 1.08
            artists = [circ, lx, ly]
            for txt, tx, ty in [("N",x,y-off),("S",x,y+off),("E",x+off,y),("W",x-off,y)]:
                t = ax_main.text(tx, ty, txt, color="#ffcc00", fontsize=8,
                                 ha="center", va="center", fontweight="bold", zorder=8)
                artists.append(t)
            state["meas_artists"] = artists

            # compute stats
            arr    = data[PANEL_DEFS[state["idx"]][1]]
            title  = PANEL_DEFS[state["idx"]][0]
            s_p    = circle_stats(arr,             x, y, r_px)
            s_m    = circle_stats(data["magnitude"], x, y, r_px)
            direc  = directional_grad(data["magnitude"],
                                      data["sobelx_raw"], data["sobely_raw"],
                                      x, y, r_px)

            state["mode"] = "normal"
            dom = max("NSEW", key=lambda d: direc[d])
            status_text.set_text(f"Circle @ ({x},{y})  r={r_cm:.1f}cm  |  dominant: {dom}  |  saving…")
            fig.canvas.draw_idle()

            # save all ROI data
            save_roi_data(data, x, y, r_px, r_cm, px_cm, direc, out_dir, stem)
            status_text.set_text(f"Circle @ ({x},{y})  r={r_cm:.1f}cm  |  dominant: {dom}  |  ✓ saved to {out_dir.name}/")
            fig.canvas.draw_idle()

            # open stats popup
            show_stats_popup(x, y, r_cm, r_px, px_cm, title, s_p, s_m, direc)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── hover ─────────────────────────────────────────────────────────────────
    def on_move(ev):
        if ev.inaxes != ax_main or im_obj[0] is None: return
        x, y = int(ev.xdata or 0), int(ev.ydata or 0)
        arr  = data[PANEL_DEFS[state["idx"]][1]]
        h, w = arr.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val   = arr[y,x]
            px_cm = state["calib_px_per_cm"]
            cm_s  = f"  =  ({x/px_cm:.2f}, {y/px_cm:.2f}) cm" if px_cm else ""
            coord_text.set_text(f"pixel ({x}, {y}){cm_s}   val = {val:.4f}")
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # ── keyboard ──────────────────────────────────────────────────────────────
    def on_key(ev):
        if ev.key in ("right","n"):  nxt(None)
        elif ev.key in ("left","p"): prev(None)
        elif ev.key == "escape":
            state["mode"] = "normal"
            status_text.set_text("Ready")
            fig.canvas.draw_idle()
        elif ev.key.isdigit():
            i = int(ev.key)
            if i < len(PANEL_DEFS): state["idx"] = i; draw(i)
    fig.canvas.mpl_connect("key_press_event", on_key)

    draw(0)
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = ask_file()

    if not filepath:
        print("No file selected."); input("Press Enter…"); sys.exit(0)

    filepath = str(Path(filepath).resolve())
    if not os.path.isfile(filepath):
        print(f"Not found: {filepath}"); input("Press Enter…"); sys.exit(1)

    stem    = Path(filepath).stem
    out_dir = Path(filepath).parent / f"{stem}_analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}\n  Thermal Analysis Tool\n{'='*55}")
    print(f"  Input  : {filepath}\n  Output : {out_dir}\n")

    print("Loading temperature data…")
    temp = load_temperature(filepath)
    print(f"  Shape {temp.shape}  |  {temp.min():.1f} – {temp.max():.1f} °C\n")

    print("Computing gradients…")
    data = compute(temp)

    print("\nSaving gradient CSVs…")
    save_full_csvs(data, out_dir, stem)

    print("\nSaving PNGs…")
    save_all_pngs(data, out_dir, stem)

    print("\nSaving quiver PNG…")
    qfig = build_quiver_fig(data)
    qp   = out_dir / f"{stem}_gradient_direction.png"
    qfig.savefig(qp, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(qfig)
    print(f"  [saved] {qp.name}")

    print(f"\n✓ Saved arrays: {list(data.keys())}")
    print(f"✓ Output folder: {out_dir}")
    print("\nOpening interactive viewer…")
    launch_interactive(data, stem, out_dir)


if __name__ == "__main__":
    main()
