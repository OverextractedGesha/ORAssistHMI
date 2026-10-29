"""ct_phantom.py - Synthetic 3D Head CT Phantom & Surgical Image Processor for OR-SterileAssist.

Generates a realistic 60-slice 380x380 3D synthetic head CT volume in Hounsfield Units (HU),
handles clinical windowing (Brain, Bone, Subdural), and renders surgical laser crosshairs,
dwell progress rings, and lesion target reticles.
"""
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
    z = int(np.clip(slice_idx, 0, N_SLICES - 1))
    px = int(np.clip(u * SIZE, 0, SIZE - 1))
    py = int(np.clip(v * SIZE, 0, SIZE - 1))
    return float(volume[z, py, px])


def to_display_rgb(slice_hu: np.ndarray,
                   level: float,
                   width: float,
                   cursor_uv=None,
                   markers=None,
                   is_bookmarked: bool = False,
                   dwell_progress: float = 0.0) -> np.ndarray:
    """Converts a 2D HU slice (380, 380) to an RGB display image with windowing, crosshair, and annotations.
    
    Args:
        slice_hu: (380, 380) float32 array in Hounsfield units.
        level: Window center (HU).
        width: Window width (HU).
        cursor_uv: Optional (u, v) normalized tuple in [0.0, 1.0] for laser pointer / mouse reticle.
        markers: Optional list of tuples [(u, v, label, hu_val), ...] for marked surgical targets.
        is_bookmarked: Boolean indicating if this slice is saved in bookmarks.
        dwell_progress: Float 0.0..1.0 for dwell lock progress ring.
        
    Returns:
        (380, 380, 3) uint8 BGR/RGB numpy array suitable for display or PIL ImageTk.
    """
    min_hu = level - width / 2.0
    max_hu = level + width / 2.0

    # Linear window contrast mapping
    norm = np.clip((slice_hu - min_hu) / max(width, 1.0), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
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

            mx = int(np.clip(mu * SIZE, 0, SIZE - 1))
            my = int(np.clip(mv * SIZE, 0, SIZE - 1))

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
        cx = int(np.clip(cu * SIZE, 0, SIZE - 1))
        cy = int(np.clip(cv_coord * SIZE, 0, SIZE - 1))

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
        cv2.rectangle(rgb, (SIZE - 128, 8), (SIZE - 8, 30), (0, 160, 80), -1)
        cv2.rectangle(rgb, (SIZE - 128, 8), (SIZE - 8, 30), (0, 255, 128), 1)
        cv2.putText(rgb, "★ BOOKMARKED", (SIZE - 124, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Orientation markers (Anterior, Posterior, Left, Right)
    cv2.putText(rgb, "A", (SIZE // 2 - 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "P", (SIZE // 2 - 5, SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "R", (10, SIZE // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(rgb, "L", (SIZE - 22, SIZE // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return rgb
