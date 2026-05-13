import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
# CRITICAL: Use 'Agg' to prevent Matplotlib from opening GUI windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import flyr for thermal image extraction (matches your original code)
try:
    import flyr
    FLYR_AVAILABLE = True
except ImportError:
    FLYR_AVAILABLE = False

def load_temperature(filepath: str) -> np.ndarray:
    """Loads thermal data from image."""
    if FLYR_AVAILABLE and filepath.lower().endswith((".jpg", ".jpeg")):
        try:
            return flyr.unpack(filepath).celsius.astype(np.float32)
        except Exception:
            pass
    
    # Fallback to standard grayscale load if FLIR metadata isn't available
    raw = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {filepath}")
    return raw.astype(np.float32)

def compute(temp: np.ndarray) -> dict:
    """Computes gradients (heat flow)."""
    # Calculate gradients (rate of temperature change across X and Y axes)
    gy, gx = np.gradient(temp)
    # Calculate magnitude (how fast it is changing overall)
    magnitude = np.hypot(gx, gy)
    
    return {
        "temp": temp,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude
    }

def save_full_csvs(data: dict, out_dir: str, stem: str):
    """Saves raw data to CSVs for export."""
    # Saving magnitude as an example (you can add temp, gx, gy here if needed)
    np.savetxt(os.path.join(out_dir, f"{stem}_magnitude.csv"), data["magnitude"], delimiter=",", fmt="%.3f")

def save_all_pngs(data: dict, out_dir: str, stem: str):
    """Saves heatmaps as PNGs."""
    plt.imsave(os.path.join(out_dir, f"{stem}_gradient.png"), data["magnitude"], cmap="inferno")
    plt.imsave(os.path.join(out_dir, f"{stem}_temperature.png"), data["temp"], cmap="hot")

def build_quiver_fig(data: dict):
    """Builds the heat direction arrows (Quiver plot)."""
    temp = data["temp"]
    gx = data["gx"]
    gy = data["gy"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display the temperature as the background
    im = ax.imshow(temp, cmap="hot", origin="upper")
    
    # Thin out the arrows so it isn't a solid black mess (skips pixels based on size)
    skip = max(1, temp.shape[0] // 30)
    y, x = np.mgrid[0:temp.shape[0], 0:temp.shape[1]]
    
    # Plot the arrows (quiver)
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip], 
              gx[::skip, ::skip], gy[::skip, ::skip], 
              color='white', scale_units='xy', angles='xy')
              
    ax.set_title("Heat Flow Direction")
    plt.colorbar(im, ax=ax, label="Temperature")
    
    return fig

def main():
    # 1. Check arguments from Electron
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "error": "Missing input or output directory."}))
        sys.exit(1)

    filepath = sys.argv[1]
    out_dir = sys.argv[2]

    try:
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Use the filename without extension as the base for output files
        stem = Path(filepath).stem

        # 2. Process Data
        temp = load_temperature(filepath)
        data = compute(temp)
        
        # 3. Save Outputs silently
        save_full_csvs(data, out_dir, stem)
        save_all_pngs(data, out_dir, stem)
        
        # Save Quiver plot
        qfig = build_quiver_fig(data)
        qp = os.path.join(out_dir, f"{stem}_quiver.png")
        qfig.savefig(qp, dpi=150, bbox_inches='tight')
        plt.close(qfig) # Free up memory to prevent memory leaks!

        # 4. Return success and file paths to Electron via stdout
        print(json.dumps({
            "status": "success", 
            "message": "Analysis complete",
            "output_dir": out_dir,
            "images": [f"{stem}_gradient.png", f"{stem}_quiver.png"]
        }))
        
    except Exception as e:
        # If ANYTHING goes wrong, tell Electron so the UI doesn't freeze
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()