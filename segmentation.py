"""
Segments a handwritten Bangla word drawn on a canvas into individual character
images, ready to be fed one-by-one into the single-character classifier.

Algorithm overview
~~~~~~~~~~~~~~~~~~
1. Convert the canvas RGBA/RGB image to a binary mask (ink vs background).
2. Find every connected component (blob of ink pixels) via scipy.
3. Merge components that are close to each other horizontally – this handles
   characters whose strokes nearly touch (common in Bangla cursive writing).
4. Sort the resulting bounding boxes left-to-right (Bangla is LTR).
5. Crop each bounding box from the original image, add square padding, and
   return the list of PIL Images for the predictor.
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def segment_characters(
    canvas_image: Image.Image,
    merge_threshold_ratio: float = 0.04,
    min_area_ratio: float = 0.001,
    padding_ratio: float = 0.15,
) -> list[Image.Image]:
    """
    Segment a drawn Bangla word into individual character images.

    Parameters
    ----------
    canvas_image : PIL.Image.Image
        The raw canvas output (RGBA or RGB, white ink on black background).
    merge_threshold_ratio : float
        How close two component bounding boxes must be (horizontally) to be
        merged, expressed as a fraction of the image width.
        Default 0.04 → 4 % of width (≈ 12 px on a 300-px canvas).
    min_area_ratio : float
        Components whose pixel area is smaller than this fraction of total
        image area are dropped as noise.
        Default 0.001 → 0.1 % of total pixels.
    padding_ratio : float
        Fraction of the bounding-box size to add as padding on each side
        before returning the crop.  Default 0.15 → 15 %.

    Returns
    -------
    list of PIL.Image.Image
        Character crops, sorted left-to-right, ready for the predictor.
        Returns an empty list if no ink is detected.
    """
    # ------------------------------------------------------------------
    # 1. Build binary mask  (True = ink pixel)
    # ------------------------------------------------------------------
    binary = _to_binary(canvas_image)

    if binary.sum() == 0:
        return []           # canvas is blank

    h, w = binary.shape
    min_area = min_area_ratio * h * w
    merge_gap = merge_threshold_ratio * w

    # ------------------------------------------------------------------
    # 2. Connected-component labelling
    # ------------------------------------------------------------------
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    # Collect bounding boxes for every component (skip label 0 = background)
    boxes = []  # each entry: (x_min, y_min, x_max, y_max)
    for label_id in range(1, num_features + 1):
        component = labeled == label_id
        if component.sum() < min_area:
            continue        # too small — noise
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    if not boxes:
        return []

    # ------------------------------------------------------------------
    # 3. Sort left-to-right, then merge horizontally close boxes
    # ------------------------------------------------------------------
    boxes.sort(key=lambda b: b[0])
    merged = _merge_boxes(boxes, gap=merge_gap)

    # ------------------------------------------------------------------
    # 4. Crop each merged box with padding → PIL Image
    # ------------------------------------------------------------------
    # Work on a plain RGB version of the canvas
    rgb = canvas_image.convert("RGB")
    rgb_arr = np.array(rgb)

    character_images = []
    for (x_min, y_min, x_max, y_max) in merged:
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)
        pad = int(max(bw, bh) * padding_ratio)

        # Clamp to image boundaries
        cx1 = max(0, x_min - pad)
        cy1 = max(0, y_min - pad)
        cx2 = min(w, x_max + pad + 1)
        cy2 = min(h, y_max + pad + 1)

        crop = rgb_arr[cy1:cy2, cx1:cx2]

        # Make the crop square by padding the shorter side with black
        crop_img = _make_square(crop)
        character_images.append(crop_img)

    return character_images


def get_segmentation_debug_image(
    canvas_image: Image.Image,
    merge_threshold_ratio: float = 0.04,
    min_area_ratio: float = 0.001,
) -> Image.Image:
    """
    Return a copy of the canvas with coloured bounding boxes drawn around
    each detected character segment.  Useful for debugging in the Streamlit UI.
    """
    import PIL.ImageDraw as ImageDraw

    binary = _to_binary(canvas_image)
    h, w = binary.shape
    min_area = min_area_ratio * h * w
    merge_gap = merge_threshold_ratio * w

    labeled, num_features = ndimage.label(binary)

    boxes = []
    for label_id in range(1, num_features + 1):
        component = labeled == label_id
        if component.sum() < min_area:
            continue
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    boxes.sort(key=lambda b: b[0])
    merged = _merge_boxes(boxes, gap=merge_gap)

    debug_img = canvas_image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug_img)
    colors = ["#FF4B4B", "#4BFF4B", "#4B4BFF", "#FFD700", "#FF69B4", "#00FFFF"]
    for i, (x_min, y_min, x_max, y_max) in enumerate(merged):
        color = colors[i % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

    return debug_img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_binary(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to a 2-D boolean array where True = ink (bright pixel).
    Handles RGBA, RGB, and grayscale inputs.
    The canvas uses *white* ink on a *black* background, so we threshold at
    a relatively low value to catch anti-aliased edges as well.
    """
    img = image.convert("L")            # collapse to grayscale
    arr = np.array(img, dtype=np.uint8)
    return arr > 30                     # threshold: pixels brighter than 30/255


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    gap: float,
) -> list[tuple[int, int, int, int]]:
    """
    Iteratively merge bounding boxes whose horizontal gap is ≤ `gap` pixels.
    The list must be sorted by x_min before calling this function.

    A single pass may not be enough if three or more boxes chain-merge, so we
    repeat until no further merges happen.
    """
    if not boxes:
        return []

    changed = True
    while changed:
        changed = False
        merged = [boxes[0]]
        for cur in boxes[1:]:
            prev = merged[-1]
            # Gap between prev's right edge and cur's left edge
            horizontal_gap = cur[0] - prev[2]
            if horizontal_gap <= gap:
                # Merge: take the union bounding box
                merged[-1] = (
                    min(prev[0], cur[0]),
                    min(prev[1], cur[1]),
                    max(prev[2], cur[2]),
                    max(prev[3], cur[3]),
                )
                changed = True
            else:
                merged.append(cur)
        boxes = merged

    return boxes


def _make_square(crop_arr: np.ndarray) -> Image.Image:
    """
    Pad a crop array to a square shape (using black pixels) and return a
    PIL Image.  The content is centred within the square.
    """
    h, w = crop_arr.shape[:2]
    side = max(h, w)
    if h == w:
        return Image.fromarray(crop_arr)

    channels = crop_arr.shape[2] if crop_arr.ndim == 3 else 1
    if channels > 1:
        square = np.zeros((side, side, channels), dtype=crop_arr.dtype)
    else:
        square = np.zeros((side, side), dtype=crop_arr.dtype)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop_arr

    return Image.fromarray(square)
