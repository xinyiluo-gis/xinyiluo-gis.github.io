# Selected Code Samples

Small, reusable examples curated from local research utilities. Data, model weights, private assets, virtual environments, and machine-specific paths are intentionally excluded.

## Contents

- `merge_panel_csv.py` — merge yearly CSV tables and standardize column names.
- `split_csv.py` — reproducibly split one CSV file into two subsets.
- `yolo_image_inference.py` — run Ultralytics YOLO inference on an image or folder.
- `gee_landcover_rf.js` — a compact Google Earth Engine random-forest land-cover workflow.

## Quick start

```bash
pip install pandas scikit-learn ultralytics
python merge_panel_csv.py --base-dir ./data
python split_csv.py input.csv --test-size 0.3
python yolo_image_inference.py ./images --weights yolo11s.pt
```

For the Earth Engine example, paste the JavaScript into the [Google Earth Engine Code Editor](https://code.earthengine.google.com/) and replace the placeholder AOI and training collections with your own assets.


