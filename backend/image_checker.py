"""
Image credibility pipeline -- starter version.

This is a real, if simple, frequency-domain heuristic: AI-generated
images (especially GAN output) tend to have unusually strong,
regularly-spaced high-frequency energy left over from upsampling
layers, which shows up as an abnormal falloff in the radial average of
the image's 2D FFT magnitude spectrum. Real photographs from a camera
sensor have a much smoother, more natural falloff.

This heuristic is NOT a substitute for a trained CNN -- treat it as a
placeholder that gives you a working end-to-end pipeline on day one.
Once you've trained the model in /train/train_image_classifier.py,
swap load_frequency_heuristic() out for a real model.forward() call.
"""

import io
import numpy as np
from PIL import Image

# Anything below this threshold "looks natural" by this rough heuristic.
# Tune this against your own labeled dataset (see /train folder) --
# don't trust the default blindly, it hasn't been calibrated on a real
# benchmark yet.
ANOMALY_THRESHOLD = 0.35


def _radial_high_freq_ratio(gray: np.ndarray) -> float:
    """Returns the fraction of spectral energy sitting in the outer
    (high-frequency) ring of the FFT magnitude spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = radius.max()

    outer_ring = radius > (0.7 * max_radius)
    total_energy = magnitude.sum() + 1e-8
    outer_energy = magnitude[outer_ring].sum()

    return float(outer_energy / total_energy)


def evaluate_image(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    except Exception:
        return {
            "type": "image",
            "verdict": "error",
            "message": "Could not read this file as an image.",
            "score": None,
        }

    # Downscale for speed; frequency-domain shape is what matters, not
    # the exact resolution.
    img = img.resize((512, 512))
    arr = np.asarray(img, dtype=np.float32)

    score = _radial_high_freq_ratio(arr)
    flagged = score > ANOMALY_THRESHOLD

    return {
        "type": "image",
        "verdict": "possible_synthetic" if flagged else "no_anomaly_detected",
        "message": (
            "Unusual high-frequency pattern detected -- possible sign of AI generation. "
            "This is a lightweight heuristic, not a trained model, so treat it as a hint, not proof."
            if flagged else
            "No strong frequency-domain anomaly detected. This does not guarantee the image is real."
        ),
        "score": round(score, 4),
        "threshold": ANOMALY_THRESHOLD,
    }
