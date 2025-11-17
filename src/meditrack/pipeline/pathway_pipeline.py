from pathlib import Path
from meditrack.config import CONFIG
from meditrack.cv.preprocessing import load_and_preprocess_image
from meditrack.cv.segmentation import segment_wound
from meditrack.cv.postprocessing import postprocess_mask
from meditrack.utils.metrics import compute_wound_area
from meditrack.utils.visualization import save_overlay

def process_single_image(image_path: Path) -> dict:
    image = load_and_preprocess_image(image_path)
    mask = segment_wound(image)
    clean_mask = postprocess_mask(mask)
    area = compute_wound_area(clean_mask, pixel_spacing_mm=0.5)
    overlay_path = CONFIG.outputs_dir / f"{image_path.stem}_overlay.png"
    save_overlay(image, clean_mask, overlay_path)
    return {
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "wound_area_mm2": area,
    }

def main():
    CONFIG.outputs_dir.mkdir(parents=True, exist_ok=True)
    for img in CONFIG.sample_wounds_dir.glob("*.png"):
        result = process_single_image(img)
        print(f"Processed {img.name}: {result['wound_area_mm2']:.2f} mm^2")

if __name__ == "__main__":
    main()
