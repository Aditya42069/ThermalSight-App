# backend/analyzer.py
import sys, os, json
import numpy as np
import cv2
import matplotlib
# Use 'Agg' to prevent interactive windows from opening
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# (Assume load_temperature, compute, save_full_csvs, etc., from your original code are defined here)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing input or output directory."}))
        sys.exit(1)

    filepath = sys.argv[1]
    out_dir = sys.argv[2]

    try:
        # 1. Load Data
        temp = load_temperature(filepath) # From original script
        
        # 2. Compute
        data = compute(temp) # From original script
        
        # 3. Save Outputs silently
        stem = "thermal_data"
        save_full_csvs(data, out_dir, stem)
        save_all_pngs(data, out_dir, stem)
        
        # Save Quiver
        qfig = build_quiver_fig(data)
        qp = os.path.join(out_dir, f"{stem}_quiver.png")
        qfig.savefig(qp, dpi=150, bbox_inches='tight')
        plt.close(qfig)

        # Return success and file paths to Electron via stdout
        print(json.dumps({
            "status": "success", 
            "message": "Analysis complete",
            "output_dir": out_dir,
            "images": [f"{stem}_gradient.png", f"{stem}_quiver.png"]
        }))
        
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()