"""
ThermalSight Backend — headless JSON API
Called by Electron main.js via child_process.spawn

Commands
--------
  analyze  <imagePath> <outputDir>
      Load image, compute all gradients, save PNGs + CSVs.
      Prints a single JSON object to stdout.

  roi  <imagePath> <cx> <cy> <r_cm> <px_per_cm> <outputDir>
      Recompute gradients, extract ROI circle stats, save ROI CSV.
      Prints a single JSON object to stdout.

All output goes to stdout as JSON.
All log/debug goes to stderr (never stdout, or the JSON parse breaks).
Exit 0 = ok, Exit 1 = error (error key in JSON).
"""

import sys, os, json, warnings, csv
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # NO GUI — headless only
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import flyr
    FLYR_AVAILABLE = True
except ImportError:
    FLYR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, file=sys.stderr, flush=True)

def emit(obj):
    """Print JSON to stdout and exit 0."""
    print(json.dumps(obj), flush=True)
    sys.exit(0)

def fail(msg):
    print(json.dumps({"error": msg}), flush=True)
    sys.exit(1)


def normalize_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return (255 * (arr - mn) / (mx - mn)).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Load temperature
# ─────────────────────────────────────────────────────────────────────────────

def load_temperature(filepath: str) -> np.ndarray:
    if FLYR_AVAILABLE and filepath.lower().endswith((".jpg", ".jpeg")):
        try:
            log("Trying flyr FLIR unpack…")
            return flyr.unpack(filepath).celsius.astype(np.float32)
        except Exception as e:
            log(f"flyr failed ({e}), falling back to grayscale")
    raw = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        fail(f"Cannot open image: {filepath}")
    return raw.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Compute gradients
# ─────────────────────────────────────────────────────────────────────────────

def compute(temp: np.ndarray) -> dict:
    img     = (255 * temp / temp.max()).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

    sx_raw = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy_raw = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sx     = cv2.convertScaleAbs(sx_raw)
    sy     = cv2.convertScaleAbs(sy_raw)

    combined    = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
    sobel_norm  = normalize_u8(combined)
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    sobel_clahe = clahe.apply(sobel_norm)

    magnitude  = np.sqrt(sx_raw**2 + sy_raw**2).astype(np.float32)
    mag_thresh = np.where(magnitude >= np.percentile(magnitude, 75),
                          magnitude, 0).astype(np.float32)
    angle      = np.arctan2(sy_raw, sx_raw).astype(np.float32)

    orig_color  = cv2.applyColorMap(normalize_u8(temp), cv2.COLORMAP_INFERNO)
    edge_color  = cv2.applyColorMap(sobel_clahe, cv2.COLORMAP_HOT)
    overlay_rgb = cv2.cvtColor(
        cv2.addWeighted(orig_color, 0.6, edge_color, 0.4, 0),
        cv2.COLOR_BGR2RGB
    )

    return dict(
        temp        = temp,
        sobelx_raw  = sx_raw,
        sobely_raw  = sy_raw,
        magnitude   = magnitude,
        mag_thresh  = mag_thresh,
        angle       = angle,
        overlay_rgb = overlay_rgb,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Save PNGs
# ─────────────────────────────────────────────────────────────────────────────

def save_panel_png(arr, cmap, path, title):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0e0e0e")
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0e0e0e")
    plt.close(fig)


def save_all_pngs(data: dict, out_dir: Path, stem: str) -> dict:
    panels = {
        "original"        : (data["temp"],        "inferno", "Temperature (°C)"),
        "magnitude"       : (data["magnitude"],   "hot",     "Gradient Magnitude"),
        "mag_thresh"      : (data["mag_thresh"],  "hot",     "Magnitude Thresholded"),
        "angle"           : (data["angle"],       "hsv",     "Gradient Angle"),
        "overlay"         : (data["overlay_rgb"], None,      "Overlay"),
    }
    paths = {}
    for key, (arr, cmap, title) in panels.items():
        p = out_dir / f"{stem}_{key}.png"
        save_panel_png(arr, cmap, str(p), title)
        paths[key] = str(p)
        log(f"  saved {p.name}")

    # quiver arrows PNG
    quiver_path = save_quiver_png(data, out_dir, stem)
    paths["quiver"] = quiver_path

    # grid PNG
    grid_path = save_grid_png(data, panels, out_dir, stem)
    paths["grid"] = grid_path

    return paths


def save_quiver_png(data: dict, out_dir: Path, stem: str) -> str:
    temp = data["temp"]
    sx   = data["sobelx_raw"]
    sy   = data["sobely_raw"]
    mag  = data["magnitude"]

    strong = mag > np.percentile(mag, 75)
    h, w   = temp.shape
    sub    = 15
    ys, xs = np.mgrid[0:h:sub, 0:w:sub]
    u = sx[ys, xs].astype(float)
    v = -sy[ys, xs].astype(float)
    mask   = strong[ys, xs]
    nrm    = np.sqrt(u**2 + v**2) + 1e-8
    u_n, v_n = u / nrm, v / nrm
    colors = plt.cm.hsv((np.arctan2(v_n, u_n) + np.pi) / (2 * np.pi))

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
    ax.imshow(temp, cmap="inferno", alpha=0.85)
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            if mask[i, j]:
                ax.annotate("",
                    xy    =(xs[i,j] + u_n[i,j]*6, ys[i,j] + v_n[i,j]*6),
                    xytext=(xs[i,j], ys[i,j]),
                    arrowprops=dict(arrowstyle="->", color=colors[i,j], lw=1.1))

    # compass wheel
    ax_w  = fig.add_axes([0.80, 0.76, 0.16, 0.16], projection="polar")
    theta = np.linspace(0, 2*np.pi, 256)
    r_arr = np.linspace(0.4, 1, 10)
    T, R  = np.meshgrid(theta, r_arr)
    ax_w.pcolormesh(T, R, T, cmap="hsv", shading="auto")
    ax_w.set_yticklabels([])
    ax_w.set_xticklabels(["E","","N","","W","","S",""], fontsize=7, color="white")
    ax_w.set_title("Direction", fontsize=8, color="white", pad=2)
    ax_w.spines["polar"].set_visible(False)
    ax_w.set_facecolor("black")

    ax.set_title("Gradient Direction", color="white", fontsize=11)
    ax.axis("off")
    fig.patch.set_facecolor("black")

    p = out_dir / f"{stem}_quiver.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    log(f"  saved {p.name}")
    return str(p)


def save_grid_png(data, panels, out_dir: Path, stem: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#111")
    for ax, (key, (arr, cmap, title)) in zip(axes.flatten(), panels.items()):
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=9)
    axes.flatten()[-1].set_visible(False)
    fig.suptitle(f"Thermal Analysis — {stem}", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"{stem}_grid.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    log(f"  saved {p.name}")
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
#  Save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_full_csvs(data: dict, out_dir: Path, stem: str) -> dict:
    csv_paths = {}
    for key in ("temp", "sobelx_raw", "sobely_raw"):
        arr  = data[key]
        path = out_dir / f"{stem}_{key}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            for row in arr:
                w.writerow([f"{v:.6f}" for v in row])
        csv_paths[key] = str(path)
        log(f"  saved {path.name}  {arr.shape}")
    return csv_paths


# ─────────────────────────────────────────────────────────────────────────────
#  ROI helpers
# ─────────────────────────────────────────────────────────────────────────────

def directional_grad(mag, sx_raw, sy_raw, cx, cy, r_px):
    h, w = mag.shape
    Y, X = np.ogrid[:h, :w]
    in_c = (X - cx)**2 + (Y - cy)**2 <= r_px**2
    dx   = X - cx;  dy = Y - cy

    result = {}
    for name, m in {"N": in_c & (dy < 0), "S": in_c & (dy > 0),
                     "E": in_c & (dx > 0), "W": in_c & (dx < 0)}.items():
        v = mag[m].ravel().astype(float)
        result[name] = float(v.mean()) if v.size else 0.0

    in_flat      = in_c.ravel()
    result["net_x"] = float(sx_raw.ravel()[in_flat].mean())
    result["net_y"] = float(sy_raw.ravel()[in_flat].mean())
    return result


def circle_stats(arr2d, cx, cy, r_px):
    Y, X = np.ogrid[:arr2d.shape[0], :arr2d.shape[1]]
    mask = (X - cx)**2 + (Y - cy)**2 <= r_px**2
    v    = arr2d[mask].ravel().astype(float)
    if v.size == 0:
        return {}
    return dict(n=int(v.size), min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), std=float(v.std()),
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)))


def save_roi_csv(data, cx, cy, r_px, r_cm, px_cm, direc,
                 out_dir: Path, stem: str) -> dict:
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
        # metadata header
        w_csv.writerow(["# METADATA"])
        w_csv.writerow(["# centre_px",  f"{cx},{cy}"])
        w_csv.writerow(["# centre_cm",  f"{cx/px_cm:.4f},{cy/px_cm:.4f}"])
        w_csv.writerow(["# radius_cm",  f"{r_cm:.4f}"])
        w_csv.writerow(["# radius_px",  f"{r_px:.2f}"])
        w_csv.writerow(["# px_per_cm",  f"{px_cm:.6f}"])
        w_csv.writerow(["# n_pixels",   int(mask.sum())])
        w_csv.writerow(["#"])
        w_csv.writerow(["# DIRECTIONAL GRADIENTS"])
        for d in ["N", "S", "E", "W"]:
            w_csv.writerow([f"# grad_{d}", f"{direc[d]:.6f}",
                            "per_cm", f"{direc[d]/r_cm:.6f}"])
        w_csv.writerow(["# net_EW", f"{direc['net_x']:+.6f}"])
        w_csv.writerow(["# net_NS", f"{direc['net_y']:+.6f}"])
        w_csv.writerow(["#"])
        # data
        w_csv.writerow(["px_x","px_y","dx_cm","dy_cm"] + scalar_keys)
        arrays = {k: data[k][mask].ravel() for k in scalar_keys}
        for i in range(len(xs)):
            row = [xs[i], ys[i], f"{dx_cm[i]:.4f}", f"{dy_cm[i]:.4f}"]
            row += [f"{arrays[k][i]:.6f}" for k in scalar_keys]
            w_csv.writerow(row)

    log(f"  saved {csv_path.name}")
    return str(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_analyze(image_path: str, out_dir_str: str):
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    log(f"Loading: {image_path}")
    temp = load_temperature(image_path)
    log(f"Shape {temp.shape}  range {temp.min():.1f}–{temp.max():.1f}")

    log("Computing gradients…")
    data = compute(temp)

    log("Saving PNGs…")
    image_paths = save_all_pngs(data, out_dir, stem)

    log("Saving CSVs…")
    csv_paths = save_full_csvs(data, out_dir, stem)

    emit({
        "status"  : "ok",
        "stem"    : stem,
        "out_dir" : str(out_dir),
        "shape"   : list(temp.shape),
        "temp_min": float(temp.min()),
        "temp_max": float(temp.max()),
        "temp_mean": float(temp.mean()),
        "images"  : image_paths,
        "csvs"    : csv_paths,
    })


def cmd_roi(image_path: str, cx: int, cy: int,
            r_cm: float, px_cm: float, out_dir_str: str):
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    log(f"ROI: loading {image_path}")
    temp = load_temperature(image_path)
    data = compute(temp)

    r_px  = r_cm * px_cm
    direc = directional_grad(data["magnitude"], data["sobelx_raw"],
                             data["sobely_raw"], cx, cy, r_px)

    stats_temp = circle_stats(data["temp"],      cx, cy, r_px)
    stats_mag  = circle_stats(data["magnitude"], cx, cy, r_px)
    stats_gx   = circle_stats(data["sobelx_raw"], cx, cy, r_px)
    stats_gy   = circle_stats(data["sobely_raw"], cx, cy, r_px)

    csv_path = save_roi_csv(data, cx, cy, r_px, r_cm, px_cm,
                            direc, out_dir, stem)

    dom = max(["N","S","E","W"], key=lambda d: direc[d])

    emit({
        "status"     : "ok",
        "csv_path"   : csv_path,
        "centre_px"  : [cx, cy],
        "centre_cm"  : [round(cx / px_cm, 3), round(cy / px_cm, 3)],
        "radius_cm"  : r_cm,
        "radius_px"  : round(r_px, 1),
        "n_pixels"   : stats_temp.get("n", 0),
        "dominant"   : dom,
        "directional": {
            "N"    : direc["N"],
            "S"    : direc["S"],
            "E"    : direc["E"],
            "W"    : direc["W"],
            "net_x": direc["net_x"],
            "net_y": direc["net_y"],
        },
        "grad_per_cm": round(stats_mag.get("mean", 0) / r_cm, 6),
        "stats": {
            "temp"      : stats_temp,
            "magnitude" : stats_mag,
            "sobelx_raw": stats_gx,
            "sobely_raw": stats_gy,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        fail("Usage: analyzer.py analyze <imagePath> <outputDir>")

    command = sys.argv[1].lower()

    if command == "analyze":
        if len(sys.argv) < 4:
            fail("Usage: analyzer.py analyze <imagePath> <outputDir>")
        cmd_analyze(sys.argv[2], sys.argv[3])

    elif command == "roi":
        if len(sys.argv) < 8:
            fail("Usage: analyzer.py roi <imagePath> <cx> <cy> <r_cm> <px_per_cm> <outputDir>")
        cmd_roi(
            image_path  = sys.argv[2],
            cx          = int(sys.argv[3]),
            cy          = int(sys.argv[4]),
            r_cm        = float(sys.argv[5]),
            px_cm       = float(sys.argv[6]),
            out_dir_str = sys.argv[7],
        )

    else:
        fail(f"Unknown command: {command}. Use 'analyze' or 'roi'.")
