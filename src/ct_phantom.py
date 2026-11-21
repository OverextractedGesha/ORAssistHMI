"""ct_phantom.py - 3D Head CT Volume Loader, Synthetic Phantom & Surgical Image Processor.

Supports both the synthetic 60-slice 380x380 head CT phantom and real clinical CT volumes
(.npz and .nii/.nii.gz) in Hounsfield Units (HU), clinical windowing (Brain, Bone, Subdural),
and surgical reticle rendering.
"""
import os
import sys
import numpy as np
import cv2

N_SLICES = 60
SIZE = 380

WINDOWS = {
    "Brain": (40, 80),        # L: 40 HU, W: 80 HU (Parenchyma, edema, hemorrhage)
    "Bone": (500, 2000),      # L: 500 HU, W: 2000 HU (Cranial vault, base, fractures)
    "Subdural": (70, 200),    # L: 70 HU, W: 200 HU (Acute hematoma vs bone distinction)
}


def make_phantom() -> np.ndarray:
    """Generates a realistic 3D synthetic head CT volume (60, 380, 380) in Hounsfield Units (HU).
    
    Tissues modeled:
    - Air background: -1000 HU
    - Skull vault & diploe: +1000 HU / +450 HU
    - Brain parenchyma: +35 HU (with gray/white matter contrast)
    - Ventricles & CSF spaces: +5 HU
    - Falx cerebri / midline: +50 HU
    - Distinct hyperdense hemorrhage/lesion (slices 34..43, center ~(0.32, -0.22)): +78 HU core, +140 HU rim
    """
    volume = np.full((N_SLICES, SIZE, SIZE), -1000.0, dtype=np.float32)

    # Coordinate grids normalized to [-1.0, 1.0]
    y, x = np.ogrid[-1:1:SIZE * 1j, -1:1:SIZE * 1j]

    for z in range(N_SLICES):
        # Vertical height normalized from -1 (skull base) to +1 (vertex)
        z_norm = (z - 30.0) / 28.0
        if abs(z_norm) > 0.95:
            # Beyond head top or bottom
            continue

        # Skull elliptical radius varies with height
        head_radius = np.sqrt(max(0.0, 1.0 - z_norm**2))
        a = 0.68 * head_radius  # lateral radius (x)
        b = 0.84 * head_radius  # anterior-posterior radius (y)

        if a < 0.15 or b < 0.15:
            continue

        ellipse_outer = (x / a)**2 + (y / b)**2
        skull_thickness = 0.075 / max(head_radius, 0.4)
        inner_a = max(0.05, a - skull_thickness)
        inner_b = max(0.05, b - skull_thickness)
        ellipse_inner = (x / inner_a)**2 + (y / inner_b)**2

        # 1. Skull outer shell
        skull_mask = (ellipse_outer <= 1.0) & (ellipse_inner > 1.0)
        volume[z][skull_mask] = 1000.0

        # Diploic space (marrow within thick cranial bones)
        mid_a = inner_a + skull_thickness * 0.35
        mid_b = inner_b + skull_thickness * 0.35
        diploe_mask = (ellipse_outer <= 0.98) & ((x / mid_a)**2 + (y / mid_b)**2 > 1.0)
        volume[z][diploe_mask & skull_mask] = 480.0

        # 2. Brain Parenchyma (inside inner skull)
        brain_mask = ellipse_inner <= 1.0
        # Basal parenchyma (~ +33 to +38 HU)
        dist_from_cortex = 1.0 - np.sqrt(np.clip(ellipse_inner, 0.0, 1.0))
        # Cortex is slightly more hyperdense (+38 HU) than deep white matter (+30 HU)
        parenchyma = 30.0 + 9.0 * np.clip(dist_from_cortex * 2.5, 0.0, 1.0)
        volume[z][brain_mask] = parenchyma[brain_mask]

        # 3. Midline falx cerebri (+50 HU)
        falx_mask = brain_mask & (np.abs(x) < 0.018) & (y > -0.65) & (y < 0.65)
        volume[z][falx_mask] = 52.0

        # 4. Ventricular system (CSF: +5 HU)
        if 18 <= z <= 44:
            # Lateral ventricles (butterfly anterior & posterior horns)
            # Left & Right horns
            for sign in [-1.0, 1.0]:
                ant_horn = ((x - sign * 0.14) / 0.07)**2 + ((y - 0.12) / 0.22)**2
                post_horn = ((x - sign * 0.20) / 0.09)**2 + ((y + 0.25) / 0.20)**2
                v_mask = brain_mask & ((ant_horn <= 1.0) | (post_horn <= 0.8))
                volume[z][v_mask] = 5.0

            # 3rd ventricle (slit along midline)
            third_v = brain_mask & (np.abs(x) < 0.022) & (np.abs(y) < 0.16)
            volume[z][third_v] = 4.0

        # 5. Cortical sulci / Sylvian fissures
        sylvian_l = brain_mask & (((x + 0.38) / 0.06)**2 + ((y - 0.05) / 0.28)**2 <= 0.35)
        sylvian_r = brain_mask & (((x - 0.38) / 0.06)**2 + ((y - 0.05) / 0.28)**2 <= 0.35)
        volume[z][sylvian_l | sylvian_r] = 7.0

        # 6. Surgical Target Lesion (Slices 34 to 43, Right Parietotemporal)
        if 34 <= z <= 43:
            # Lesion center
            lx0, ly0 = 0.32, -0.22
            # Size peaks at slice 38
            z_factor = 1.0 - abs(z - 38.5) / 6.0
            r_max = 0.16 * z_factor
            if r_max > 0.03:
                r_dist = np.sqrt((x - lx0)**2 + (y - ly0)**2)

                # Perilesional vasogenic edema rim (hypodense +20 HU)
                edema_mask = brain_mask & (r_dist <= r_max * 1.35) & (r_dist > r_max)
                volume[z][edema_mask] = 20.0

                # Hyperdense contrast / calcified capsule rim (+140 HU)
                rim_mask = brain_mask & (r_dist <= r_max) & (r_dist > r_max * 0.72)
                volume[z][rim_mask] = 140.0

                # Acute hyperdense hemorrhagic lesion core (+78 HU)
                core_mask = brain_mask & (r_dist <= r_max * 0.72)
                volume[z][core_mask] = 78.0

    return volume


def get_hu_at(volume: np.ndarray, slice_idx: int, u: float, v: float) -> float:
    """Reads exact Hounsfield Units (HU) at normalized coordinates (u, v) in [0, 1] on slice_idx."""
    n_slices, h, w = volume.shape
    z = int(np.clip(slice_idx, 0, n_slices - 1))
    px = int(np.clip(u * w, 0, w - 1))
    py = int(np.clip(v * h, 0, h - 1))
    return float(volume[z, py, px])


def window_slice_grayscale(slice_hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Applies linear HU windowing and returns a clean (H, W) uint8 grayscale array."""
    min_hu = level - width / 2.0
    norm = np.clip((slice_hu - min_hu) / max(width, 1.0), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def to_display_rgb(slice_hu: np.ndarray,
                   level: float,
                   width: float,
                   cursor_uv=None,
                   markers=None,
                   is_bookmarked: bool = False,
                   dwell_progress: float = 0.0) -> np.ndarray:
    """Converts a 2D HU slice (H, W) to an RGB display image with windowing, crosshair, and annotations.
    
    Args:
        slice_hu: (H, W) float32 array in Hounsfield units.
        level: Window center (HU).
        width: Window width (HU).
        cursor_uv: Optional (u, v) normalized tuple in [0.0, 1.0] for laser pointer / mouse reticle.
        markers: Optional list of tuples [(u, v, label, hu_val), ...] for marked surgical targets.
        is_bookmarked: Boolean indicating if this slice is saved in bookmarks.
        dwell_progress: Float 0.0..1.0 for dwell lock progress ring.
        
    Returns:
        (H, W, 3) uint8 RGB numpy array suitable for display or PIL ImageTk.
    """
    h, w = slice_hu.shape[:2]
    gray = window_slice_grayscale(slice_hu, level, width)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 1. Draw existing ROI markers on this slice
    if markers:
        for idx, item in enumerate(markers):
            if len(item) == 4:
                mu, mv, mlabel, mhu = item
            else:
                mu, mv = item[:2]
                mlabel = f"ROI-{idx+1}"
                mhu = 0.0

            mx = int(np.clip(mu * w, 0, w - 1))
            my = int(np.clip(mv * h, 0, h - 1))

            # Surgical target reticle: Yellow circle + crosshair
            cv2.circle(rgb, (mx, my), 12, (0, 255, 255), 2)  # Yellow
            cv2.circle(rgb, (mx, my), 4, (0, 0, 255), -1)     # Red center
            cv2.line(rgb, (mx - 18, my), (mx - 6, my), (0, 255, 255), 2)
            cv2.line(rgb, (mx + 6, my), (mx + 18, my), (0, 255, 255), 2)
            cv2.line(rgb, (mx, my - 18), (mx, my - 6), (0, 255, 255), 2)
            cv2.line(rgb, (mx, my + 6), (mx, my + 18), (0, 255, 255), 2)

            # Badge tag text
            tag_text = f"{mlabel} ({mhu:+.0f} HU)"
            cv2.putText(rgb, tag_text, (mx + 14, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Active Cursor / Laser Pointer Reticle
    if cursor_uv is not None:
        cu, cv_coord = cursor_uv
        cx = int(np.clip(cu * w, 0, w - 1))
        cy = int(np.clip(cv_coord * h, 0, h - 1))

        # Cyan surgical crosshair
        cv2.circle(rgb, (cx, cy), 9, (0, 255, 255), 2)       # Cyan ring
        cv2.circle(rgb, (cx, cy), 2, (255, 255, 255), -1)    # White center dot
        cv2.line(rgb, (cx - 15, cy), (cx - 4, cy), (0, 255, 255), 1)
        cv2.line(rgb, (cx + 4, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(rgb, (cx, cy - 15), (cx, cy - 4), (0, 255, 255), 1)
        cv2.line(rgb, (cx, cy + 4), (cx, cy + 15), (0, 255, 255), 1)

        # Dwell progress circular arc (0.0 .. 1.0 -> 0 .. 360 deg)
        if dwell_progress > 0.0:
            angle_end = int(np.clip(dwell_progress, 0.0, 1.0) * 360.0)
            cv2.ellipse(rgb, (cx, cy), (18, 18), -90, 0, angle_end, (0, 255, 0), 3)  # Bright green arc
            pct_txt = f"{int(dwell_progress * 100)}%"
            cv2.putText(rgb, pct_txt, (cx + 14, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # 3. Draw Bookmarked Badge
    if is_bookmarked:
        # Green ribbon badge in top-right
        cv2.rectangle(rgb, (w - 128, 8), (w - 8, 30), (0, 160, 80), -1)
        cv2.rectangle(rgb, (w - 128, 8), (w - 8, 30), (0, 255, 128), 1)
        cv2.putText(rgb, "★ BOOKMARKED", (w - 124, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Orientation markers (Anterior, Posterior, Left, Right)
    cv2.putText(rgb, "A", (w // 2 - 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "P", (w // 2 - 5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "R", (10, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "L", (w - 22, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return rgb


def _find_asset(filename: str) -> str | None:
    """Locates a bundled asset across PyInstaller and source directory layouts."""
    candidates: list[str] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "assets", filename))
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, "assets", filename))
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.abspath(os.path.join(cur_dir, "..", "assets", filename)))
    candidates.append(os.path.join(cur_dir, "assets", filename))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _npz_scalar(data, key: str, default):
    """Extracts a scalar or list metadata field from an NpzFile archive."""
    if key not in data:
        return default
    val = data[key]
    if isinstance(val, np.ndarray):
        return val.item() if val.ndim == 0 else val.tolist()
    return val


def _parse_lesion_span(raw, n_slices: int) -> tuple[int, int] | None:
    """Normalizes an optional 2-element lesion slice span within [0, n_slices - 1]."""
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 2:
        try:
            lo, hi = int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None
        if 0 <= lo <= hi < n_slices:
            return (lo, hi)
    return None


def _parse_lesion_uv(raw) -> tuple[float, float] | None:
    """Normalizes an optional 2-element (u, v) lesion coordinate pair."""
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 2:
        try:
            u, v = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
        if np.isfinite(u) and np.isfinite(v):
            return (u, v)
    return None


def load_real_ct(npz_path: str | None = None) -> tuple[np.ndarray, dict]:
    """Loads bundled real clinical Cranial CT (.npz) or falls back to the synthetic phantom."""
    path = npz_path if npz_path is not None else _find_asset("real_head_ct.npz")
    if not path or not os.path.isfile(path):
        return make_phantom(), {
            "is_real": False,
            "patient_id": "OR-702 (Synthetic Phantom)",
            "series_desc": "CRANIAL PHANTOM",
            "source": "Synthetic",
            "spacing_mm": [2.5, 0.65, 0.65],
            "default_slice": 38,
            "n_slices": N_SLICES,
            "height": SIZE,
            "width": SIZE,
            "lesion_span": None,
            "lesion_marker_uv": None,
            "lesion_label": "Hyperdense Lesion",
        }

    with np.load(path, allow_pickle=False) as data:
        vol_key = "volume" if "volume" in data else data.files[0]
        volume = np.ascontiguousarray(data[vol_key], dtype=np.float32)
        n_slices, height, width = volume.shape
        spacing_raw = _npz_scalar(data, "spacing_mm", [1.0, 1.0, 1.0])
        spacing_mm = [float(x) for x in np.asarray(spacing_raw).ravel()[:3]]
        default_slice = int(np.clip(int(_npz_scalar(data, "default_slice", n_slices // 2)), 0, n_slices - 1))
        lesion_span = _parse_lesion_span(_npz_scalar(data, "lesion_span", None), n_slices)
        lesion_marker_uv = _parse_lesion_uv(_npz_scalar(data, "lesion_marker_uv", None))
        lesion_label_raw = _npz_scalar(data, "lesion_label", "Focal Lesion")
        lesion_label = str(lesion_label_raw) if lesion_label_raw is not None else "Focal Lesion"
        meta = {
            "is_real": True,
            "patient_id": str(_npz_scalar(data, "patient_id", "REAL-HEAD-CT")),
            "series_desc": str(_npz_scalar(data, "series_desc", "CRANIAL CT")),
            "source": str(_npz_scalar(data, "source", os.path.basename(path))),
            "spacing_mm": spacing_mm,
            "default_slice": default_slice,
            "n_slices": int(n_slices),
            "height": int(height),
            "width": int(width),
            "lesion_span": lesion_span,
            "lesion_marker_uv": lesion_marker_uv,
            "lesion_label": lesion_label,
        }
    return volume, meta


def load_nifti_ct(nii_path: str) -> tuple[np.ndarray, dict]:
    """Loads a .nii/.nii.gz CT volume, reorients to radiological axial (Z, Y, X), and pads to square."""
    import nibabel as nib

    img = nib.load(nii_path)
    canon = nib.as_closest_canonical(img)
    arr = canon.get_fdata(dtype=np.float32)
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3 or min(arr.shape) <= 0:
        raise ValueError(f"Expected non-empty 3D NIfTI volume, got shape {arr.shape}")

    # RAS+ (X: L->R, Y: P->A, Z: I->S) -> Axial radiological (Z, Y: A->P top-down, X: R->L left-right)
    vol = np.transpose(arr, (2, 1, 0))[:, ::-1, ::-1]
    if not np.isfinite(vol).all():
        vol = np.nan_to_num(vol, nan=-1024.0, posinf=3071.0, neginf=-1024.0)
    n_slices, h, w = vol.shape
    s = max(h, w)
    if h != s or w != s:
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top
        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        vol = np.pad(
            vol,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=-1024.0,
        )
    vol = np.ascontiguousarray(vol, dtype=np.float32)

    zooms_raw = canon.header.get_zooms()
    zooms = [
        float(zooms_raw[i])
        if i < len(zooms_raw) and np.isfinite(zooms_raw[i]) and float(zooms_raw[i]) > 0.0
        else 1.0
        for i in range(3)
    ]
    spacing_mm = [zooms[2], zooms[1], zooms[0]]

    fname = os.path.basename(nii_path)
    patient_id = fname
    for ext in (".nii.gz", ".nii"):
        if patient_id.lower().endswith(ext):
            patient_id = patient_id[:-len(ext)]
            break

    meta = {
        "is_real": True,
        "patient_id": patient_id,
        "series_desc": f"NIfTI CT ({fname})",
        "source": nii_path,
        "spacing_mm": spacing_mm,
        "default_slice": int(n_slices // 2),
        "n_slices": int(n_slices),
        "height": int(s),
        "width": int(s),
    }
    return vol, meta
