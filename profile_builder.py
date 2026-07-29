import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from scipy.interpolate import interp1d
import datetime
import io
#import tkinter as tk
#from tkinter import filedialog


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Custom Discrete Colormap Configuration (Reversed Order)
HEX_COLORS = [
    '#BCBD37',  # Olive-yellow (bottom)
    '#FFFFCC',  # Cream/Light yellow
    '#FFED6F',  # Yellow
    '#F47C20',  # Orange
    '#D7301F',  # Deep orange-red
    '#800000'   # Dark maroon/red (top)
]

SILT_BINS = [0, 1, 2, 4, 6, 8, 10, 20, 25]
D50_BINS = [0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 1]

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(round(rgb[0])))),
        max(0, min(255, int(round(rgb[1])))),
        max(0, min(255, int(round(rgb[2]))))
    )

def interpolate_colors(hex_colors, n_out):
    if n_out <= 1:
        return [hex_colors[0]]
    n_in = len(hex_colors)
    rgbs = [hex_to_rgb(c) for c in hex_colors]
    out_colors = []
    for i in range(n_out):
        t = i / (n_out - 1)
        pos = t * (n_in - 1)
        idx = int(pos)
        if idx >= n_in - 1:
            out_colors.append(hex_colors[-1])
        else:
            frac = pos - idx
            r = rgbs[idx][0] + (rgbs[idx+1][0] - rgbs[idx][0]) * frac
            g = rgbs[idx][1] + (rgbs[idx+1][1] - rgbs[idx][1]) * frac
            b = rgbs[idx][2] + (rgbs[idx+1][2] - rgbs[idx][2]) * frac
            out_colors.append(rgb_to_hex((r, g, b)))
    return out_colors

def create_discrete_colorscale(bins, hex_colors):
    n_bins = len(bins) - 1
    bin_colors = interpolate_colors(hex_colors, n_bins)
    
    b_min, b_max = bins[0], bins[-1]
    norm_bins = [(b - b_min) / (b_max - b_min) for b in bins]
    
    colorscale = []
    for i in range(n_bins):
        colorscale.append([norm_bins[i], bin_colors[i]])
        colorscale.append([norm_bins[i+1], bin_colors[i]])
        
    return colorscale

def map_values_to_equal_bins(vals, bins):
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    n_bins = len(bins) - 1
    
    mask = np.isnan(arr)
    v_clipped = np.clip(arr, bins[0], bins[-1])
    
    indices = np.digitize(v_clipped, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    
    b_low = bins[indices]
    b_high = bins[indices + 1]
    
    bin_widths = b_high - b_low
    bin_widths = np.where(bin_widths == 0, 1e-9, bin_widths)
    
    frac = (v_clipped - b_low) / bin_widths
    norm_val = (indices + frac) / n_bins
    norm_val[mask] = np.nan
    return norm_val

def create_equal_discrete_colorscale(n_bins, hex_colors):
    bin_colors = interpolate_colors(hex_colors, n_bins)
    colorscale = []
    for i in range(n_bins):
        colorscale.append([i / n_bins, bin_colors[i]])
        colorscale.append([(i + 1) / n_bins, bin_colors[i]])
    return colorscale

def get_bin_color(val, bins, colors):
    """Return color for a value based on discrete bins."""
    if pd.isna(val):
        return '#999999'
    for i in range(len(bins) - 1):
        if val < bins[i + 1]:
            return colors[min(i, len(colors) - 1)]
    return colors[-1]

def get_bathymetry_polygon(bath_file_bytes=None):
    """Returns (poly_x, poly_y) in EPSG:25831 representing the bathymetry raster boundary polygon."""
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from pyproj import Transformer
        import os

        src = None
        if bath_file_bytes is not None:
            memfile = MemoryFile(bath_file_bytes)
            src = memfile.open()
        elif os.path.exists("25NZE4376ml9_1.img"):
            src = rasterio.open("25NZE4376ml9_1.img")

        if src is not None:
            left, bottom, right, top = src.bounds
            crs = src.crs
            src.close()

            bx = [left, right, right, left, left]
            by = [bottom, bottom, top, top, bottom]

            if crs is not None and str(crs).upper() != "EPSG:25831":
                try:
                    to_25831 = Transformer.from_crs(crs, "EPSG:25831", always_xy=True)
                    poly_x, poly_y = to_25831.transform(bx, by)
                    return list(poly_x), list(poly_y)
                except Exception:
                    pass
            return bx, by
    except Exception:
        pass
    return None, None

def get_profile_bathymetry(profile_bh_list, df_coords, bath_file_bytes=None, bath_filename=None, step_m=1.0):
    """
    Samples bathymetry continuously along the polyline connecting profile_bh_list.
    Dynamically tests candidate CRS transformations (Raster CRS, EPSG:25831, EPSG:32631, EPSG:28992)
    to guarantee valid sampling regardless of uploaded raster CRS or coordinate system.
    Returns list of (dist, depth) pairs where depth is positive depth below LAT (m).
    """
    if not profile_bh_list or len(profile_bh_list) < 2:
        return []

    bh_map = {}
    for bh in profile_bh_list:
        r = df_coords[df_coords['Boornummer'] == bh]
        if not r.empty:
            bh_map[bh] = (r['X'].iloc[0], r['Y'].iloc[0])

    if len(bh_map) < 2:
        return []

    coords_x = []
    coords_y = []
    seg_dists = [0.0]
    last_x, last_y = None, None
    cum = 0.0

    for bh in profile_bh_list:
        if bh in bh_map:
            x, y = bh_map[bh]
            coords_x.append(x)
            coords_y.append(y)
            if last_x is not None:
                cum += np.sqrt((x - last_x)**2 + (y - last_y)**2)
            seg_dists.append(cum)
            last_x, last_y = x, y

    total_dist = cum
    if total_dist <= 0:
        return []

    seg_dists = np.array(seg_dists[1:])
    coords_x = np.array(coords_x)
    coords_y = np.array(coords_y)

    n_pts = max(50, int(np.ceil(total_dist / step_m)))
    dist_arr = np.linspace(0, total_dist, n_pts)

    try:
        fx = interp1d(seg_dists, coords_x, kind='linear')
        fy = interp1d(seg_dists, coords_y, kind='linear')
        line_x = fx(dist_arr)
        line_y = fy(dist_arr)
    except Exception:
        return []

    sampled = None
    nodata = None
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import os

        src = None
        if bath_file_bytes is not None:
            fn = bath_filename if bath_filename else "bathymetry.img"
            memfile = MemoryFile(bath_file_bytes, filename=fn)
            src = memfile.open()
        elif os.path.exists("25NZE4376ml9_1.img"):
            src = rasterio.open("25NZE4376ml9_1.img")

        if src is not None:
            nodata = src.nodata
            raster_crs = src.crs

            coords_candidates = []
            if raster_crs is not None:
                try:
                    to_raster_crs = Transformer.from_crs("EPSG:32631", raster_crs, always_xy=True)
                    rx, ry = to_raster_crs.transform(line_x, line_y)
                    coords_candidates.append(list(zip(rx, ry)))
                except Exception:
                    pass

            try:
                to_25831 = Transformer.from_crs("EPSG:32631", "EPSG:25831", always_xy=True)
                rx, ry = to_25831.transform(line_x, line_y)
                coords_candidates.append(list(zip(rx, ry)))
            except Exception:
                pass

            coords_candidates.append(list(zip(line_x, line_y)))

            try:
                to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
                rx, ry = to_rd.transform(line_x, line_y)
                coords_candidates.append(list(zip(rx, ry)))
            except Exception:
                pass

            for coords in coords_candidates:
                try:
                    s_try = [v[0] for v in src.sample(coords)]
                    valid = [v for v in s_try if v is not None and not np.isnan(v) and (nodata is None or not np.isclose(v, nodata)) and -9000 < v < 9000]
                    if len(valid) > 0:
                        sampled = s_try
                        break
                except Exception:
                    continue

            src.close()

        if sampled is not None:
            valid_d = []
            valid_depth = []
            for d, v in zip(dist_arr, sampled):
                if v is not None and not np.isnan(v) and (nodata is None or not np.isclose(v, nodata)) and -9000 < v < 9000:
                    valid_d.append(float(d))
                    valid_depth.append(float(abs(v) if v < 0 else v))

            if len(valid_d) > 0:
                if len(valid_d) == 1:
                    interp_depths = np.full_like(dist_arr, valid_depth[0])
                else:
                    interp_depths = np.interp(dist_arr, valid_d, valid_depth)
                return [(float(d), float(dep)) for d, dep in zip(dist_arr, interp_depths)]
    except Exception:
        pass

    return []

@st.cache_data
def get_bathymetry_mapbox_layer(raster_path_or_bytes, bath_filename=None, max_size=500, colormap='white_deep_blue'):
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import base64
        import io
        import os

        src = None
        if isinstance(raster_path_or_bytes, bytes):
            fn = bath_filename if bath_filename else "bathymetry.img"
            memfile = MemoryFile(raster_path_or_bytes, filename=fn)
            src = memfile.open()
        elif os.path.exists(str(raster_path_or_bytes)):
            src = rasterio.open(raster_path_or_bytes)
        
        if src is None:
            return None, None

        nodata = src.nodata
        raster_crs = src.crs if src.crs is not None else "EPSG:25831"
        data = src.read(1).astype(np.float32)

        # Reproject 4 corner coordinates to EPSG:4326 (WGS84 lon/lat)
        to_wgs = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

        nw_lon, nw_lat = to_wgs.transform(left, top)
        ne_lon, ne_lat = to_wgs.transform(right, top)
        se_lon, se_lat = to_wgs.transform(right, bottom)
        sw_lon, sw_lat = to_wgs.transform(left, bottom)

        coords = [
            [nw_lon, nw_lat],
            [ne_lon, ne_lat],
            [se_lon, se_lat],
            [sw_lon, sw_lat]
        ]

        # Mask nodata & invalid values
        mask = (data == nodata) | np.isnan(data) | (data > 9000) | (data < -9000)
        valid_data = data[~mask]

        if len(valid_data) == 0:
            src.close()
            return None, None

        depth_data = np.abs(data)
        vmin = float(np.nanmin(depth_data[~mask]))
        vmax = float(np.nanmax(depth_data[~mask]))

        # Downsample array if necessary for fast web rendering
        h, w = depth_data.shape
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            new_h, new_w = int(h * scale), int(w * scale)
            img_pil = Image.fromarray(depth_data)
            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            depth_data = np.array(img_resized)

            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_resized = mask_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
            mask = np.array(mask_resized) > 128

        src.close()

        # Apply colormap to depth data (white shallow -> deep blue deep)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if colormap in ['white_deep_blue', 'white_to_blue', 'viridis_r']:
            cmap_func = LinearSegmentedColormap.from_list(
                "white_deep_blue", 
                ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
            )
        else:
            cmap_func = plt.colormaps.get_cmap(colormap)
        colored = cmap_func(norm(depth_data))  # RGBA float in [0,1]

        # Apply transparency to nodata pixels
        colored[mask, 3] = 0.0

        rgba_uint8 = (colored * 255).astype(np.uint8)
        img_out = Image.fromarray(rgba_uint8, mode='RGBA')

        buf = io.BytesIO()
        img_out.save(buf, format='PNG')
        png_bytes = buf.getvalue()

        b64_str = "data:image/png;base64," + base64.b64encode(png_bytes).decode('utf-8')

        return b64_str, coords
    except Exception:
        return None, None

@st.cache_data
def get_bathymetry_mapbox_points(raster_path_or_bytes, bath_filename=None, grid_size=40):
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import os

        src = None
        if isinstance(raster_path_or_bytes, bytes):
            fn = bath_filename if bath_filename else "bathymetry.img"
            memfile = MemoryFile(raster_path_or_bytes, filename=fn)
            src = memfile.open()
        elif os.path.exists(str(raster_path_or_bytes)):
            src = rasterio.open(raster_path_or_bytes)
            
        if src is None:
            return None, None, None

        nodata = src.nodata
        raster_crs = src.crs if src.crs is not None else "EPSG:25831"
        
        bounds = src.bounds
        xs = np.linspace(bounds.left, bounds.right, grid_size)
        ys = np.linspace(bounds.bottom, bounds.top, grid_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        
        sampled = [v[0] for v in src.sample(list(zip(flat_x, flat_y)))]
        src.close()
        
        to_wgs = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        lons, lats = to_wgs.transform(flat_x, flat_y)
        
        valid_lons = []
        valid_lats = []
        valid_depths = []
        
        for lo, la, v in zip(lons, lats, sampled):
            if v is not None and not np.isnan(v) and (nodata is None or not np.isclose(v, nodata)) and -9000 < v < 9000:
                valid_lons.append(float(lo))
                valid_lats.append(float(la))
                valid_depths.append(float(abs(v) if v < 0 else v))
                
        return valid_lats, valid_lons, valid_depths
    except Exception:
        return None, None, None

# 1. Page Configuration
st.set_page_config(
    page_title="Geotechnical Profile Builder",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Theme Toggle State
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Styling (Design System)
bg_color = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card_color = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border_color = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
accent = "#4f46e5"
accent_muted = "#4338ca"
green_color = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red_color = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

css_styles = f"""
<style>
    :root {{
        --bg: {bg_color};
        --bg-subtle: {bg_subtle};
        --card: {card_color};
        --card-hover: {card_hover};
        --border: {border_color};
        --border-subtle: {border_subtle};
        --text: {text_color};
        --text-muted: {text_muted};
        --text-dim: {text_dim};
        --accent: {accent};
        --accent-muted: {accent_muted};
        --green: {green_color};
        --green-muted: {green_muted};
        --red: {red_color};
        --red-muted: {red_muted};
        --shadow: {shadow};
        --radius: 12px;
    }}
    
    /* Hide Streamlit default components */
    header[data-testid="stHeader"], footer {{
        display: none !important;
    }}
    
    /* Global App Styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    .block-container {{
        padding: 1.5rem 2rem 2rem !important;
        max-width: 1440px !important;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-subtle) !important;
        border-right: 1px solid var(--border) !important;
    }}
    .sidebar-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Card design */
    .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .card:hover {{
        border-color: var(--text-dim);
    }}
    
    /* Metric Card (KPI) */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: var(--accent);
    }}
    .metric-label {{
        font-size: 0.72rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        margin-top: 0.2rem;
    }}
    
    /* Chart Container */
    .chart-wrap {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }}
    .chart-header {{
        margin-bottom: 1rem;
    }}
    .chart-title {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
    }}
    .chart-subtitle {{
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.1rem;
    }}
    
    /* Data Table (HTML) */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 1rem;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border);
        background: var(--bg-subtle);
    }}
    .data-table td {{
        padding: 0.75rem 1rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    .data-table tr:hover td {{
        background-color: var(--card-hover);
    }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
    }}
    .badge-blue {{
        color: var(--accent);
        background: rgba(79, 70, 229, 0.1);
    }}
    .badge-gray {{
        color: var(--text-muted);
        background: var(--border-subtle);
    }}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# 4. Data Ingestion & Caching
@st.cache_data
def load_data(uploaded_file, bath_file_bytes=None):
    df = pd.read_excel(uploaded_file, sheet_name='Locaties_einddiepte_LAT_3')
    
    # Coordinate Conversion:
    # Source coordinates X, Y in the sheet are in UTM Zone 31N (EPSG:32631).
    # We transform them to Lat/Lon (EPSG:4326) for Mapbox.
    # We also transform them to RD coordinates (EPSG:28992) for local display.
    # We transform them to EPSG:25831 for bathymetry raster lookup.
    to_wgs84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    to_25831 = Transformer.from_crs("EPSG:32631", "EPSG:25831", always_xy=True)
    
    lons, lats = to_wgs84.transform(df['X'].values, df['Y'].values)
    x_rd, y_rd = to_rd.transform(df['X'].values, df['Y'].values)
    x_25831, y_25831 = to_25831.transform(df['X'].values, df['Y'].values)
    
    df['lon'] = lons
    df['lat'] = lats
    df['X_RD'] = x_rd
    df['Y_RD'] = y_rd
    df['X_32631'] = df['X']
    df['Y_32631'] = df['Y']
    df['X_25831'] = x_25831
    df['Y_25831'] = y_25831

    # Standardize X, Y to EPSG:25831 for all spatial plots & calculations
    df['X'] = x_25831
    df['Y'] = y_25831

    # Sample Bathymetry from map on EPSG:25831
    sampled_vals = None
    try:
        import rasterio
        from rasterio.io import MemoryFile
        import os

        # First attempt: reprojected EPSG:25831 coordinates
        coords_reproj = list(zip(x_25831, y_25831))
        # Second attempt: direct X, Y coordinates
        coords_direct = list(zip(df['X'].values, df['Y'].values))

        src_ctx = None
        if bath_file_bytes is not None:
            memfile = MemoryFile(bath_file_bytes)
            src = memfile.open()
        elif os.path.exists("25NZE4376ml9_1.img"):
            src = rasterio.open("25NZE4376ml9_1.img")
        else:
            src = None

        if src is not None:
            nodata = src.nodata
            s_reproj = [val[0] for val in src.sample(coords_reproj)]
            valid_reproj = [v for v in s_reproj if v is not None and not np.isnan(v) and (nodata is None or not np.isclose(v, nodata)) and -9000 < v < 9000]

            if len(valid_reproj) > 0:
                sampled_vals = s_reproj
            else:
                s_direct = [val[0] for val in src.sample(coords_direct)]
                valid_direct = [v for v in s_direct if v is not None and not np.isnan(v) and (nodata is None or not np.isclose(v, nodata)) and -9000 < v < 9000]
                if len(valid_direct) > 0:
                    sampled_vals = s_direct
                else:
                    sampled_vals = s_reproj

            src.close()
        
        if sampled_vals is not None:
            bath_depths = []
            for v in sampled_vals:
                if v is None or np.isnan(v) or (nodata is not None and np.isclose(v, nodata)) or v > 9000 or v < -9000:
                    bath_depths.append(np.nan)
                else:
                    bath_depths.append(abs(v) if v < 0 else v)
            df['bathymetry'] = bath_depths
        else:
            df['bathymetry'] = np.nan
    except Exception as e:
        df['bathymetry'] = np.nan

    return df

# 4. Data Ingestion & Upload Section
col_up1, col_up2 = st.columns([1.8, 1.2])
with col_up1:
    uploaded_file = st.file_uploader("📂 Upload Boringen Excel-bestand (.xlsx)", type=["xlsx"])
with col_up2:
    use_bath_map = st.checkbox("🗺️ Bathymetriekaart Modus Inschakelen", value=True, help="Vink uit om bathymetriekaart sampling uit te schakelen en maaiveldhoogtes rechtstreeks uit het Excel-bestand te gebruiken.")
    uploaded_bath_file = st.file_uploader("📂 Upload Aangepaste Bathymetriekaart (.img / .tif)", type=["img", "tif", "tiff"])

if uploaded_file is None:
    st.info("👋 **Welkom bij Borehole Profile Builder!** Upload hierboven uw Boringen Excel-bestand (`.xlsx`) om te beginnen.")
    st.stop()

# Verify Bathymetry Raster availability when Bathymetry Mode is enabled
import os
bath_raster_available = False
if use_bath_map:
    if uploaded_bath_file is not None:
        bath_raster_available = True
    elif os.path.exists("25NZE4376ml9_1.img"):
        bath_raster_available = True
    else:
        st.info("ℹ️ **Bathymetry Map Not Provided**: Bathymetry mode is enabled, but no custom `.img` / `.tif` map was uploaded and default `25NZE4376ml9_1.img` was not found. Please upload a bathymetry map file above, or uncheck Bathymetry Mode.")

try:
    bath_bytes = uploaded_bath_file.getvalue() if (use_bath_map and uploaded_bath_file is not None) else None
    df = load_data(uploaded_file, bath_file_bytes=bath_bytes if bath_raster_available else None)
except Exception as e:
    st.error(f"Failed to load Excel dataset: {e}")
    st.stop()

# 5. Extract Unique Coordinates and Boreholes
if 'df' in locals() or 'df' in globals():
    df_coords = df[['Boornummer', 'X', 'Y', 'lat', 'lon', 'X_RD', 'Y_RD', 'X_25831', 'Y_25831', 'bathymetry']].drop_duplicates().sort_values('Boornummer')
    boreholes = list(df_coords['Boornummer'].unique())
else:
    df_coords = pd.DataFrame()
    boreholes = []

# Initialize session state variables
if "custom_profile" not in st.session_state:
    st.session_state.custom_profile = []

if "sidebar_sel" not in st.session_state:
    st.session_state.sidebar_sel = []

if "run_interpolation" not in st.session_state:
    st.session_state.run_interpolation = False

def trigger_interpolation():
    st.session_state.run_interpolation = True

# Clean up custom profile list just in case of data refresh
st.session_state.custom_profile = [bh for bh in st.session_state.custom_profile if bh in boreholes]

# Callback functions for the path sequence editor buttons
def move_up_callback(idx):
    st.session_state.custom_profile[idx], st.session_state.custom_profile[idx-1] = \
        st.session_state.custom_profile[idx-1], st.session_state.custom_profile[idx]
    st.session_state.sidebar_sel = st.session_state.custom_profile

def move_down_callback(idx):
    st.session_state.custom_profile[idx], st.session_state.custom_profile[idx+1] = \
        st.session_state.custom_profile[idx+1], st.session_state.custom_profile[idx]
    st.session_state.sidebar_sel = st.session_state.custom_profile

def remove_bh_callback(bh):
    st.session_state.custom_profile.remove(bh)
    st.session_state.sidebar_sel = st.session_state.custom_profile

def clear_path_callback():
    st.session_state.custom_profile = []
    st.session_state.sidebar_sel = []
    st.session_state.run_interpolation = False


# ── Inpainting interpolation ──────────────────────────────────────────────────
def interpolate_profile(prof_df, cum_dist, top_points, bottom_points, dx, dy, prop_col,
                        interp_method="OpenCV TELEA inpainting", anis_x=1.0, anis_y=1.0):
    """Rasterise borehole layers onto a 2-D grid then fill the gaps.

    Parameters
    ----------
    prof_df        : DataFrame with columns Boornummer, Tra_van_lat, Tra_tot_lat, <prop_col>
    cum_dist      : dict {borehole_name: cumulative_distance_m}
    top_points    : list of (dist, depth) pairs for top surface
    bottom_points : list of (dist, depth) pairs for borehole bottom
    dx, dy        : grid resolution in metres (horizontal, depth)
    prop_col      : column name for the property to interpolate

    Returns
    -------
    x_arr  : 1-D array of x positions
    y_arr  : 1-D array of depth positions (increasing = deeper)
    grid   : 2-D float array, NaN outside envelope
    """
    total_dist = max(v for v in cum_dist.values())
    y_min = min(p[1] for p in top_points)
    y_max = max(p[1] for p in bottom_points)

    # build regular grid
    x_arr = np.arange(0, total_dist + dx, dx)
    y_arr = np.arange(y_min, y_max + dy, dy)
    nx, ny = len(x_arr), len(y_arr)

    # ── 1. Filter out NaN values and stamp known borehole layer values ────
    valid_prof_df = prof_df[prof_df[prop_col].notna() & prof_df['Tra_van_lat'].notna() & prof_df['Tra_tot_lat'].notna()].copy()
    if valid_prof_df.empty:
        return x_arr, y_arr, np.full((ny, nx), np.nan)

    val_min = valid_prof_df[prop_col].min()
    val_max = valid_prof_df[prop_col].max()
    val_range = val_max - val_min if val_max != val_min else 1.0

    img    = np.zeros((ny, nx), dtype=np.float32)   # will hold normalised values
    mask   = np.ones((ny, nx),  dtype=np.uint8) * 255  # 255 = unknown pixel

    for bh, dist in cum_dist.items():
        bh_df = valid_prof_df[valid_prof_df['Boornummer'] == bh]
        # nearest x column
        xi = int(round(dist / dx)) if dx > 0 else 0
        xi = max(0, min(xi, nx - 1))

        for _, row in bh_df.iterrows():
            z_top = row['Tra_van_lat']
            z_bot = row['Tra_tot_lat']
            val   = row[prop_col]
            norm  = (val - val_min) / val_range  # 0-1

            yi_top = int(np.searchsorted(y_arr, z_top))
            yi_bot = int(np.searchsorted(y_arr, z_bot))
            yi_top = max(0, min(yi_top, ny - 1))
            yi_bot = max(0, min(yi_bot, ny - 1))
            if yi_top > yi_bot:
                yi_top, yi_bot = yi_bot, yi_top

            img[yi_top:yi_bot + 1, xi] = norm
            mask[yi_top:yi_bot + 1, xi] = 0   # known

    # ── 2. Build envelope mask ────────────────────────────────────────────
    # Linearly interpolate top and bottom surfaces across x_arr
    top_xs   = np.array([p[0] for p in top_points])
    top_ys   = np.array([p[1] for p in top_points])
    bot_xs   = np.array([p[0] for p in bottom_points])
    bot_ys   = np.array([p[1] for p in bottom_points])

    top_interp = interp1d(top_xs, top_ys, bounds_error=False,
                          fill_value=(top_ys[0], top_ys[-1]))
    bot_interp = interp1d(bot_xs, bot_ys, bounds_error=False,
                          fill_value=(bot_ys[0], bot_ys[-1]))

    top_surface = top_interp(x_arr)   # shape (nx,)
    bot_surface = bot_interp(x_arr)   # shape (nx,)

    # pixels that are permanently invalid (outside envelope) — vectorised
    # y_arr[:,None] shape (ny,1), top/bot_surface shape (nx,) → broadcast to (ny,nx)
    outside = (y_arr[:, None] < top_surface[None, :]) | (y_arr[:, None] > bot_surface[None, :])

    # also mark outside pixels as known-but-zero so inpainting ignores them
    mask[outside] = 0

    # ── 3. Fill using the chosen method ──────────────────────────────────
    # Collect known (inside-envelope) pixel coordinates and values
    known_mask = (mask == 0) & (~outside)          # stamped borehole pixels
    known_rows, known_cols = np.where(known_mask)
    known_vals = img[known_rows, known_cols]

    all_cols, all_rows = np.meshgrid(np.arange(nx), np.arange(ny))
    query_pts = np.column_stack([all_cols.ravel(), all_rows.ravel()])  # (n, 2)
    train_pts = np.column_stack([known_cols, known_rows])              # (m, 2)

    if interp_method.startswith("OpenCV") and HAS_CV2:
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        inpaint_mask = mask.copy()
        if anis_x != 1.0 or anis_y != 1.0:
            new_nx = max(3, int(round(nx / anis_x)))
            new_ny = max(3, int(round(ny / anis_y)))
            img_u8_res = cv2.resize(img_u8, (new_nx, new_ny), interpolation=cv2.INTER_NEAREST)
            inpaint_mask_res = cv2.resize(inpaint_mask, (new_nx, new_ny), interpolation=cv2.INTER_NEAREST)
            _, inpaint_mask_res = cv2.threshold(inpaint_mask_res, 127, 255, cv2.THRESH_BINARY)
            
            inpaint_radius = max(int(round(min(new_nx, new_ny) * 0.05)), 3)
            filled_u8_res = cv2.inpaint(img_u8_res, inpaint_mask_res, inpaint_radius, cv2.INPAINT_TELEA)
            filled_u8 = cv2.resize(filled_u8_res, (nx, ny), interpolation=cv2.INTER_LINEAR)
        else:
            inpaint_radius = max(int(round(min(nx, ny) * 0.05)), 3)
            filled_u8 = cv2.inpaint(img_u8, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        filled = filled_u8.astype(np.float32) / 255.0

    elif interp_method.startswith("scipy"):
        from scipy.interpolate import griddata as scipy_griddata
        sci_method = "linear" if "linear" in interp_method else \
                     "cubic"  if "cubic"  in interp_method else "nearest"
        if len(known_vals) > 0:
            train_pts_scaled = train_pts * np.array([anis_x, anis_y])
            query_pts_scaled = query_pts * np.array([anis_x, anis_y])
            filled_flat = scipy_griddata(
                train_pts_scaled, known_vals, query_pts_scaled, method=sci_method,
                fill_value=float(np.nanmean(known_vals))
            )
            filled = filled_flat.reshape(ny, nx).astype(np.float32)
        else:
            filled = img.copy()

    elif interp_method.startswith("sklearn – RBF"):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler
        # Use thin-plate spline via scipy RBFInterpolator (sklearn-style interface)
        from scipy.interpolate import RBFInterpolator
        if len(known_vals) > 0:
            # Normalise coords so x and y have comparable scales
            scaler = StandardScaler().fit(train_pts)
            X_train = scaler.transform(train_pts) * np.array([anis_x, anis_y])
            X_query = scaler.transform(query_pts) * np.array([anis_x, anis_y])
            filled_flat = RBFInterpolator(
                X_train, known_vals,
                kernel='thin_plate_spline'
            )(X_query)
            filled = filled_flat.reshape(ny, nx).astype(np.float32)
        else:
            filled = img.copy()

    elif interp_method.startswith("sklearn – Gaussian Process"):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        if len(known_vals) > 0 and len(known_vals) <= 2000:
            scaler = StandardScaler().fit(train_pts)
            X_train = scaler.transform(train_pts) * np.array([anis_x, anis_y])
            X_query = scaler.transform(query_pts) * np.array([anis_x, anis_y])
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0,
                                           normalize_y=True)
            gpr.fit(X_train, known_vals)
            filled_flat = gpr.predict(X_query)
            filled = np.clip(filled_flat, 0, 1).reshape(ny, nx).astype(np.float32)
        elif len(known_vals) > 2000:
            # Too many points – fall back to RBF for performance
            from scipy.interpolate import RBFInterpolator
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(train_pts)
            X_train = scaler.transform(train_pts) * np.array([anis_x, anis_y])
            X_query = scaler.transform(query_pts) * np.array([anis_x, anis_y])
            filled_flat = RBFInterpolator(
                X_train, known_vals,
                kernel='thin_plate_spline'
            )(X_query)
            filled = filled_flat.reshape(ny, nx).astype(np.float32)
        else:
            filled = img.copy()

    elif interp_method.startswith("skimage – biharmonic"):
        try:
            from skimage.restoration import inpaint_biharmonic
            # inpaint_biharmonic expects a float image in [0,1] and a bool mask
            # where True = pixel to be inpainted (unknown)
            bih_mask = (mask == 255).astype(bool)   # True = gap to fill
            # Do not inpaint pixels that are outside the surface envelope
            bih_mask[outside] = False
            
            if (anis_x != 1.0 or anis_y != 1.0) and HAS_CV2:
                new_nx = max(3, int(round(nx / anis_x)))
                new_ny = max(3, int(round(ny / anis_y)))
                img_res = cv2.resize(img.astype(np.float32), (new_nx, new_ny), interpolation=cv2.INTER_NEAREST)
                bih_mask_res = cv2.resize(bih_mask.astype(np.uint8), (new_nx, new_ny), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                filled_res = inpaint_biharmonic(
                    img_res.astype(np.float64), bih_mask_res
                ).astype(np.float32)
                
                filled = cv2.resize(filled_res, (nx, ny), interpolation=cv2.INTER_LINEAR)
            else:
                filled = inpaint_biharmonic(
                    img.astype(np.float64), bih_mask
                ).astype(np.float32)
        except ImportError:
            st.warning("⚠️ `scikit-image` not found – run `pip install scikit-image` or choose another method.")
            filled = img.copy()

    else:
        # Default nearest-neighbour fallback
        from scipy.interpolate import griddata as scipy_griddata
        if len(known_vals) > 0:
            train_pts_scaled = train_pts * np.array([anis_x, anis_y])
            query_pts_scaled = query_pts * np.array([anis_x, anis_y])
            filled_flat = scipy_griddata(
                train_pts_scaled, known_vals, query_pts_scaled, method='nearest',
                fill_value=float(np.nanmean(known_vals))
            )
            filled = filled_flat.reshape(ny, nx).astype(np.float32)
        else:
            filled = img.copy()

    # ── 4. Un-normalise and apply envelope mask ───────────────────────────
    result = filled * val_range + val_min
    result[outside] = np.nan

    return x_arr, y_arr, result

def interpolate_spatial_2d(df, df_coords, depth_lo, depth_hi, value_col, dx=25.0, dy=25.0, interp_method="Linear (TIN / Delaunay)", anis_x=1.0, anis_y=1.0):
    """
    Interpolates 2D spatial property data across the coordinate bounding box of all boreholes.
    Includes ALL available non-NaN data from ALL boreholes at depth_lo <= depth <= depth_hi.
    Supports all sidebar interpolation methods: Linear, Nearest, IDW, Cubic, RBF, Gaussian Process, Biharmonic.
    Returns (x_arr, y_arr, z_grid_2d).
    """
    # Filter dataset for layer overlap with depth range [depth_lo, depth_hi]
    df_sub = df[(df['Tra_van_lat'] <= depth_hi) & (df['Tra_tot_lat'] >= depth_lo) & df[value_col].notna()].copy()
    if df_sub.empty:
        return None, None, None

    # Group by borehole location to get average non-NaN value per borehole at this depth slice
    valid_data = df_sub.groupby(['Boornummer', 'X', 'Y'])[value_col].mean().reset_index()
    if len(valid_data) < 2:
        return None, None, None

    x_min, x_max = df_coords['X'].min() - 50.0, df_coords['X'].max() + 50.0
    y_min, y_max = df_coords['Y'].min() - 50.0, df_coords['Y'].max() + 50.0

    dx_use = max(1.0, float(dx))
    dy_use = max(1.0, float(dy if dy >= 1.0 else dx))

    x_arr = np.arange(x_min, x_max + dx_use, dx_use)
    y_arr = np.arange(y_min, y_max + dy_use, dy_use)
    grid_X, grid_Y = np.meshgrid(x_arr, y_arr)

    train_pts = valid_data[['X', 'Y']].values
    known_vals = valid_data[value_col].values

    ax_s = max(0.01, float(anis_x))
    ay_s = max(0.01, float(anis_y))
    train_scaled = train_pts * np.array([1.0 / ax_s, 1.0 / ay_s])
    query_scaled = np.column_stack([grid_X.ravel(), grid_Y.ravel()]) * np.array([1.0 / ax_s, 1.0 / ay_s])

    nx = len(x_arr)
    ny = len(y_arr)

    try:
        if interp_method.startswith("IDW"):
            diff_x = query_scaled[:, 0:1] - train_scaled[:, 0].T
            diff_y = query_scaled[:, 1:2] - train_scaled[:, 1].T
            dists = np.sqrt(diff_x**2 + diff_y**2)
            dists = np.maximum(dists, 1e-6)
            weights = 1.0 / (dists ** 2)
            z_flat = np.sum(weights * known_vals, axis=1) / np.sum(weights, axis=1)
            grid_Z = z_flat.reshape((ny, nx))

        elif interp_method.startswith("sklearn – RBF"):
            from scipy.interpolate import RBFInterpolator
            rbf = RBFInterpolator(train_scaled, known_vals, kernel='thin_plate_spline')
            z_flat = rbf(query_scaled)
            grid_Z = z_flat.reshape((ny, nx))

        elif interp_method.startswith("sklearn – Gaussian Process"):
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            gpr = GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(1e-4), normalize_y=True)
            gpr.fit(train_scaled, known_vals)
            z_flat = gpr.predict(query_scaled)
            grid_Z = z_flat.reshape((ny, nx))

        elif interp_method.startswith("skimage – biharmonic"):
            from scipy.interpolate import RBFInterpolator
            rbf = RBFInterpolator(train_scaled, known_vals, kernel='thin_plate_spline')
            z_flat = rbf(query_scaled)
            grid_Z = z_flat.reshape((ny, nx))

        elif interp_method.startswith("Cubic"):
            from scipy.interpolate import griddata as scipy_griddata
            z_flat = scipy_griddata(train_scaled, known_vals, query_scaled, method='cubic')
            if np.isnan(z_flat).any():
                z_near = scipy_griddata(train_scaled, known_vals, query_scaled, method='nearest')
                z_flat = np.where(np.isnan(z_flat), z_near, z_flat)
            grid_Z = z_flat.reshape((ny, nx))

        elif interp_method.startswith("Nearest"):
            from scipy.interpolate import griddata as scipy_griddata
            z_flat = scipy_griddata(train_scaled, known_vals, query_scaled, method='nearest')
            grid_Z = z_flat.reshape((ny, nx))

        else:
            # Default: Linear (TIN / Delaunay)
            from scipy.interpolate import griddata as scipy_griddata
            z_flat = scipy_griddata(train_scaled, known_vals, query_scaled, method='linear')
            if np.isnan(z_flat).any():
                z_near = scipy_griddata(train_scaled, known_vals, query_scaled, method='nearest')
                z_flat = np.where(np.isnan(z_flat), z_near, z_flat)
            grid_Z = z_flat.reshape((ny, nx))

        return x_arr, y_arr, grid_Z
    except Exception:
        return None, None, None

# ══════════════════════════════════════════════════════════════════════════════
# PDF Report Exporter Generator Function
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(df, df_coords, custom_profile, interp_dx, interp_dy, interp_method,
                        anis_x, anis_y, use_bath_map, bath_raster_available, uploaded_bath_file,
                        bar_width, show_labels, selected_cmap_63, selected_cmap_d50):
    """
    Generates a high-quality multi-page PDF report in landscape orientation (A4).
    - Page 1: Cover Summary (Dataset stats, selected profile, interpolation settings & location map)
    - Page 2: Borehole Transect Profiles (%<0.063mm and d50 with seabed & end depth lines)
    - Page 3: 2D Interpolated Cross-Sections (%<0.063mm and d50 with discrete colorbars and bathymetry overlay)
    - Pages 4+: Depth-Slice Spatial Maps for every depth interval across all boreholes
    """
    import io
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MaxNLocator

    buf = io.BytesIO()

    # Pre-compute metrics & statistics
    num_total_bh = len(df_coords)
    num_sel_bh = len(custom_profile)
    
    # Calculate profile cumulative distances
    cum_dist = {}
    last_x, last_y = None, None
    current_dist = 0.0
    for bh in custom_profile:
        r = df_coords[df_coords['Boornummer'] == bh]
        if not r.empty:
            x, y = r['X'].iloc[0], r['Y'].iloc[0]
            if last_x is not None:
                current_dist += np.sqrt((x - last_x)**2 + (y - last_y)**2)
            cum_dist[bh] = current_dist
            last_x, last_y = x, y
        else:
            cum_dist[bh] = 0.0

    prof_df = df[df['Boornummer'].isin(custom_profile)].copy() if custom_profile else pd.DataFrame()

    # Get continuous bathymetry or borehole tops/bottoms
    top_points = []
    bottom_points = []
    is_bath_top = False

    if len(custom_profile) >= 2:
        bath_bytes = uploaded_bath_file.getvalue() if (use_bath_map and uploaded_bath_file is not None) else None
        bath_sampled = get_profile_bathymetry(custom_profile, df_coords, bath_file_bytes=bath_bytes if bath_raster_available else None, step_m=5.0)
        if bath_sampled and len(bath_sampled) > 1:
            top_points = bath_sampled
            is_bath_top = True
        else:
            for bh in custom_profile:
                bh_df = prof_df[prof_df['Boornummer'] == bh]
                if not bh_df.empty and bh_df['Tra_van_lat'].notna().any():
                    top_points.append((cum_dist[bh], float(bh_df['Tra_van_lat'].min())))
                else:
                    top_points.append((cum_dist[bh], 0.0))

        for bh in custom_profile:
            bh_df = prof_df[prof_df['Boornummer'] == bh]
            if not bh_df.empty and bh_df['Tra_tot_lat'].notna().any():
                bottom_points.append((cum_dist[bh], float(bh_df['Tra_tot_lat'].max())))
            else:
                d_x = cum_dist[bh]
                top_y = dict(top_points).get(d_x, 10.0)
                bottom_points.append((d_x, top_y + 5.0))

    bath_poly_x, bath_poly_y = get_bathymetry_polygon(bath_bytes)

    # Discrete Colormaps setup for Matplotlib
    n_b63 = len(SILT_BINS) - 1
    cmap63_src = mcolors.LinearSegmentedColormap.from_list("c63", HEX_COLORS)
    cols63_hex = [mcolors.to_hex(cmap63_src(i / max(1, n_b63 - 1))) for i in range(n_b63)]
    cmap63_np = mcolors.ListedColormap(cols63_hex)
    norm63_np = mcolors.BoundaryNorm(SILT_BINS, n_b63)

    n_bd50 = len(D50_BINS) - 1
    cmapd50_src = mcolors.LinearSegmentedColormap.from_list("cd50", HEX_COLORS)
    cols_d50_hex = [mcolors.to_hex(cmapd50_src(i / max(1, n_bd50 - 1))) for i in range(n_bd50)]
    cmapd50_np = mcolors.ListedColormap(cols_d50_hex)
    normd50_np = mcolors.BoundaryNorm(D50_BINS, n_bd50)

    with PdfPages(buf) as pdf:
        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1: COVER / SUMMARY & OVERVIEW MAP (Landscape A4: 11.69 x 8.27 in)
        # ══════════════════════════════════════════════════════════════════════
        fig1 = plt.figure(figsize=(11.69, 8.27))
        fig1.patch.set_facecolor('#ffffff')

        # Header Title
        plt.figtext(0.05, 0.93, "GEOTECHNISCH PROFIEL & DIEPTE-INTERVAL RAPPORT", fontsize=16, fontweight='bold', color='#0f172a')
        plt.figtext(0.05, 0.905, f"Rapport gegenereerd op: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color='#475569')
        plt.figtext(0.05, 0.895, "_" * 135, fontsize=8, color='#cbd5e1')

        # Left Column Panel (Info & Statistics text)
        ax_text = fig1.add_axes([0.05, 0.08, 0.44, 0.78])
        ax_text.axis('off')

        silt_prof_mean = prof_df['63calculated. met zoutcorrectie'].mean() if not prof_df.empty else np.nan
        silt_prof_min = prof_df['63calculated. met zoutcorrectie'].min() if not prof_df.empty else np.nan
        silt_prof_max = prof_df['63calculated. met zoutcorrectie'].max() if not prof_df.empty else np.nan

        d50_prof_mean = prof_df['d50'].mean() if not prof_df.empty else np.nan
        d50_prof_min = prof_df['d50'].min() if not prof_df.empty else np.nan
        d50_prof_max = prof_df['d50'].max() if not prof_df.empty else np.nan

        sel_bh_str = ", ".join(custom_profile) if custom_profile else "Geen"
        if len(sel_bh_str) > 60:
            sel_bh_str = sel_bh_str[:57] + "..."

        text_lines = [
            ("1. ALGEMENE INFORMATIE", True, 12, '#0f172a'),
            (f"• Totaal aantal boringen in bestand: {num_total_bh}", False, 9.5, '#334155'),
            (f"• Geselecteerde boringen in profiel ({num_sel_bh}):", False, 9.5, '#334155'),
            (f"   {sel_bh_str}", False, 9, '#1e293b'),
            (f"• Profiellengte: {current_dist:.1f} m", False, 9.5, '#334155'),
            ("", False, 6, ''),
            ("2. PROFIELSTATISTIEKEN", True, 12, '#0f172a'),
            ("• Percentage (%) < 0.063mm (Silt / Clay):", True, 9.5, '#1e293b'),
            (f"   - Gemiddelde : {silt_prof_mean:.2f} %", False, 9, '#334155'),
            (f"   - Minimum    : {silt_prof_min:.2f} %  |  Maximum: {silt_prof_max:.2f} %", False, 9, '#334155'),
            ("• Korrelgrootte d50 (mm):", True, 9.5, '#1e293b'),
            (f"   - Gemiddelde : {d50_prof_mean:.3f} mm", False, 9, '#334155'),
            (f"   - Minimum    : {d50_prof_min:.3f} mm  |  Maximum: {d50_prof_max:.3f} mm", False, 9, '#334155'),
            ("", False, 6, ''),
            ("3. INTERPOLATIE INSTELLINGEN", True, 12, '#0f172a'),
            (f"• Grid stapgrootte: dx = {interp_dx:.1f} m, dy = {interp_dy:.2f} m", False, 9.5, '#334155'),
            (f"• Algoritme / Methode: {interp_method}", False, 9.5, '#334155'),
            (f"• Anisotropie gewichten: X = {anis_x:.2f}, Y = {anis_y:.2f}", False, 9.5, '#334155'),
            (f"• Bathymetriekaart: {'Actief (EPSG:25831)' if (use_bath_map and bath_raster_available) else 'Niet actief'}", False, 9.5, '#334155'),
        ]

        y_pos = 0.98
        for text, is_bold, sz, col in text_lines:
            if not text:
                y_pos -= 0.02
                continue
            weight = 'bold' if is_bold else 'normal'
            ax_text.text(0.0, y_pos, text, transform=ax_text.transAxes,
                         fontsize=sz, fontweight=weight, color=col, va='top')
            y_pos -= (0.05 if is_bold and sz == 12 else 0.04)

        # Right Column Map (Overview map)
        ax_map = fig1.add_axes([0.53, 0.08, 0.42, 0.78])
        ax_map.set_title("Kaart Boringselectie & Profielpad", fontsize=12, fontweight='bold', pad=10, color='#0f172a')

        profile_set = set(custom_profile)
        if not df_coords.empty:
            ax_map.scatter(df_coords['X'], df_coords['Y'], color='#22c55e', s=35, zorder=3, label='Alle boringen')
            for _, row in df_coords.iterrows():
                if row['Boornummer'] not in profile_set:
                    ax_map.annotate(row['Boornummer'], (row['X'], row['Y']), fontsize=6.5, color='#334155',
                                    textcoords="offset points", xytext=(0, 4), ha='center')

        if len(custom_profile) >= 2:
            prof_xs = []
            prof_ys = []
            for bh in custom_profile:
                r = df_coords[df_coords['Boornummer'] == bh]
                if not r.empty:
                    prof_xs.append(r['X'].iloc[0])
                    prof_ys.append(r['Y'].iloc[0])
            ax_map.plot(prof_xs, prof_ys, color='#dc2626', linewidth=2.2, linestyle='-', zorder=4, label='Profielpad')
            ax_map.scatter(prof_xs, prof_ys, color='#dc2626', s=70, zorder=5)
            for idx, (px, py, pbh) in enumerate(zip(prof_xs, prof_ys, custom_profile)):
                ax_map.annotate(f"{idx+1}. {pbh}", (px, py), fontsize=7.5, fontweight='bold', color='#b91c1c',
                                textcoords="offset points", xytext=(0, 6), ha='center')

        ax_map.set_xlabel("X (EPSG:25831)", fontsize=9)
        ax_map.set_ylabel("Y (EPSG:25831)", fontsize=9)
        ax_map.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_map.tick_params(axis='x', rotation=25, labelsize=7.5)
        ax_map.ticklabel_format(useOffset=False, style='plain')
        ax_map.grid(True, linestyle='--', alpha=0.3)
        ax_map.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax_map.set_aspect('equal', 'datalim')

        pdf.savefig(fig1)
        plt.close(fig1)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2: BOREHOLE TRANSECT PROFILES (Landscape A4)
        # ══════════════════════════════════════════════════════════════════════
        if len(custom_profile) >= 2 and not prof_df.empty:
            fig2, (ax_silt, ax_d50) = plt.subplots(2, 1, figsize=(11.69, 8.27))
            fig2.suptitle("Boringen Profieldoorsneden", fontsize=15, fontweight='bold', y=0.96, color='#0f172a')
            fig2.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.08, hspace=0.35)

            plot_width = max(30.0, current_dist / max(1, len(custom_profile) * 2.5))

            # ── Subplot 1: %<0.063mm Profile ─────────────────────────────────
            silt_df = prof_df[prof_df['63calculated. met zoutcorrectie'].notna()].copy()
            if not silt_df.empty:
                silt_heights = silt_df['Tra_tot_lat'] - silt_df['Tra_van_lat']
                silt_bottoms = silt_df['Tra_van_lat']
                silt_xs = [cum_dist[bh] for bh in silt_df['Boornummer']]
                silt_norm_vals = map_values_to_equal_bins(silt_df['63calculated. met zoutcorrectie'], SILT_BINS)
                silt_layer_widths = (0.3 + 0.9 * np.nan_to_num(silt_norm_vals, nan=0.0)) * plot_width
                bar_cols = [get_bin_color(v, SILT_BINS, cols63_hex) for v in silt_df['63calculated. met zoutcorrectie']]
                
                ax_silt.bar(silt_xs, silt_heights, bottom=silt_bottoms, width=silt_layer_widths, color=bar_cols, edgecolor='black', linewidth=0.4, zorder=3)
                if show_labels:
                    for x_val, bot_val, h_val, v_val in zip(silt_xs, silt_bottoms, silt_heights, silt_df['63calculated. met zoutcorrectie']):
                        ax_silt.text(x_val, bot_val + h_val/2, f"{v_val:.1f}%", ha='center', va='center', fontsize=6.5, color='black', fontweight='bold')

            if top_points:
                ax_silt.plot([p[0] for p in top_points], [p[1] for p in top_points], color='#ef4444', linewidth=2, label='Ligging zeebodem (ALAT)', zorder=4)
            if bottom_points:
                ax_silt.plot([p[0] for p in bottom_points], [p[1] for p in bottom_points], color='#3b82f6', linewidth=1.5, linestyle='--', label='Einddiepte (ALAT)', zorder=4)

            bh_tick_vals = [cum_dist[bh] for bh in custom_profile]
            bh_tick_labels = [f"{bh}\n({cum_dist[bh]:.0f}m)" for bh in custom_profile]
            top_map = dict(top_points) if top_points else {}

            top_y_vals = [p[1] for p in top_points] if top_points else [0.0]
            bot_y_vals = [p[1] for p in bottom_points] if bottom_points else [15.0]
            if not prof_df.empty and 'Tra_tot_lat' in prof_df.columns:
                bot_y_vals.extend(prof_df['Tra_tot_lat'].dropna().tolist())

            y_min_head = min(top_y_vals) - 1.8
            y_max_foot = max(bot_y_vals) + 1.0

            ax_silt.set_title("Percentage (%) < 0.063mm", fontsize=11, fontweight='bold', loc='left')
            ax_silt.set_ylabel("Diepte t.o.v. LAT (m)", fontsize=9)
            ax_silt.set_xlabel("Lengte langs profiel (m)", fontsize=9)
            ax_silt.set_xticks(bh_tick_vals)
            ax_silt.set_xticklabels(bh_tick_labels, fontsize=7.5, fontweight='bold')
            ax_silt.set_ylim(y_max_foot, y_min_head)
            ax_silt.grid(True, linestyle='--', alpha=0.3)
            ax_silt.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=8, frameon=False)

            # Annotate borehole names directly above each column on silt profile
            for bh in custom_profile:
                d_x = cum_dist[bh]
                t_y = top_map.get(d_x, min(top_y_vals))
                ax_silt.text(d_x, t_y - 0.4, bh, fontsize=8, fontweight='bold', color='#0f172a',
                             ha='center', va='bottom', zorder=6)

            cax63 = fig2.add_axes([0.90, 0.54, 0.015, 0.34])
            cb63 = fig2.colorbar(plt.cm.ScalarMappable(norm=norm63_np, cmap=cmap63_np), cax=cax63)
            cb63.set_label("%<0.063mm", fontsize=8)
            cb63.ax.tick_params(labelsize=7)

            # ── Subplot 2: d50 Profile ───────────────────────────────────────
            d50_df = prof_df[prof_df['d50'].notna()].copy()
            if not d50_df.empty:
                d50_heights = d50_df['Tra_tot_lat'] - d50_df['Tra_van_lat']
                d50_bottoms = d50_df['Tra_van_lat']
                d50_xs = [cum_dist[bh] for bh in d50_df['Boornummer']]
                d50_norm_vals = map_values_to_equal_bins(d50_df['d50'], D50_BINS)
                d50_layer_widths = (0.3 + 0.9 * np.nan_to_num(d50_norm_vals, nan=0.0)) * plot_width
                bar_cols_d50 = [get_bin_color(v, D50_BINS, cols_d50_hex) for v in d50_df['d50']]
                
                ax_d50.bar(d50_xs, d50_heights, bottom=d50_bottoms, width=d50_layer_widths, color=bar_cols_d50, edgecolor='black', linewidth=0.4, zorder=3)
                if show_labels:
                    for x_val, bot_val, h_val, v_val in zip(d50_xs, d50_bottoms, d50_heights, d50_df['d50']):
                        ax_d50.text(x_val, bot_val + h_val/2, f"{v_val:.2f}", ha='center', va='center', fontsize=6.5, color='black', fontweight='bold')

            if top_points:
                ax_d50.plot([p[0] for p in top_points], [p[1] for p in top_points], color='#ef4444', linewidth=2, label='Ligging zeebodem (ALAT)', zorder=4)
            if bottom_points:
                ax_d50.plot([p[0] for p in bottom_points], [p[1] for p in bottom_points], color='#3b82f6', linewidth=1.5, linestyle='--', label='Einddiepte (ALAT)', zorder=4)

            ax_d50.set_title("d50 (mm)", fontsize=11, fontweight='bold', loc='left')
            ax_d50.set_ylabel("Diepte t.o.v. LAT (m)", fontsize=9)
            ax_d50.set_xlabel("Lengte langs profiel (m)", fontsize=9)
            ax_d50.set_xticks(bh_tick_vals)
            ax_d50.set_xticklabels(bh_tick_labels, fontsize=7.5, fontweight='bold')
            ax_d50.set_ylim(y_max_foot, y_min_head)
            ax_d50.grid(True, linestyle='--', alpha=0.3)
            ax_d50.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=8, frameon=False)

            # Annotate borehole names directly above each column on d50 profile
            for bh in custom_profile:
                d_x = cum_dist[bh]
                t_y = top_map.get(d_x, min(top_y_vals))
                ax_d50.text(d_x, t_y - 0.4, bh, fontsize=8, fontweight='bold', color='#0f172a',
                            ha='center', va='bottom', zorder=6)

            caxd50 = fig2.add_axes([0.90, 0.08, 0.015, 0.34])
            cbd50 = fig2.colorbar(plt.cm.ScalarMappable(norm=normd50_np, cmap=cmapd50_np), cax=caxd50)
            cbd50.set_label("d50 (mm)", fontsize=8)
            cbd50.ax.tick_params(labelsize=7)

            pdf.savefig(fig2)
            plt.close(fig2)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3: INTERPOLATED CROSS-SECTIONS (Landscape A4)
        # ══════════════════════════════════════════════════════════════════════
        if len(custom_profile) >= 2:
            try:
                top_pts = top_points if top_points else [(cum_dist[bh], 0.0) for bh in custom_profile]
                bot_pts = bottom_points if bottom_points else [(cum_dist[bh], 10.0) for bh in custom_profile]

                x63, y63, grid63 = interpolate_profile(
                    prof_df, cum_dist, top_pts, bot_pts,
                    interp_dx, interp_dy,
                    "63calculated. met zoutcorrectie",
                    interp_method=interp_method,
                    anis_x=anis_x,
                    anis_y=anis_y
                )
                xd50, yd50, gridd50 = interpolate_profile(
                    prof_df, cum_dist, top_pts, bot_pts,
                    interp_dx, interp_dy,
                    "d50",
                    interp_method=interp_method,
                    anis_x=anis_x,
                    anis_y=anis_y
                )

                fig3, (ax_inp63, ax_inpd50) = plt.subplots(2, 1, figsize=(11.69, 8.27))
                fig3.suptitle("Geïnterpoleerde Dwarsdoorsneden", fontsize=15, fontweight='bold', y=0.96, color='#0f172a')
                fig3.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.08, hspace=0.35)

                im63 = ax_inp63.pcolormesh(x63, y63, grid63, cmap=cmap63_np, norm=norm63_np, shading='auto')
                if top_points:
                    ax_inp63.plot([p[0] for p in top_points], [p[1] for p in top_points], color='#ef4444', linewidth=2, label='Ligging zeebodem (ALAT)')
                if bottom_points:
                    ax_inp63.plot([p[0] for p in bottom_points], [p[1] for p in bottom_points], color='#3b82f6', linewidth=1.5, linestyle='--', label='Einddiepte (ALAT)')

                # Annotate borehole names on interpolated heatmaps
                for bh in custom_profile:
                    d_x = cum_dist[bh]
                    t_y = top_map.get(d_x, min(top_y_vals))
                    ax_inp63.text(d_x, t_y - 0.4, bh, fontsize=8, fontweight='bold', color='#0f172a',
                                  ha='center', va='bottom', zorder=6)
                    ax_inpd50.text(d_x, t_y - 0.4, bh, fontsize=8, fontweight='bold', color='#0f172a',
                                   ha='center', va='bottom', zorder=6)

                ax_inp63.set_title("%<0.063mm – Geïnterpoleerd", fontsize=11, fontweight='bold', loc='left')
                ax_inp63.set_ylabel("Diepte t.o.v. LAT (m)", fontsize=9)
                ax_inp63.set_xlabel("Lengte langs profiel (m)", fontsize=9)
                ax_inp63.set_xticks(bh_tick_vals)
                ax_inp63.set_xticklabels(bh_tick_labels, fontsize=7.5, fontweight='bold')
                ax_inp63.set_ylim(y_max_foot, y_min_head)
                ax_inp63.grid(True, linestyle='--', alpha=0.3)
                ax_inp63.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=8, frameon=False)

                cax_i63 = fig3.add_axes([0.90, 0.54, 0.015, 0.34])
                cb_i63 = fig3.colorbar(im63, cax=cax_i63)
                cb_i63.set_label("%<0.063mm", fontsize=8)
                cb_i63.ax.tick_params(labelsize=7)

                imd50 = ax_inpd50.pcolormesh(xd50, yd50, gridd50, cmap=cmapd50_np, norm=normd50_np, shading='auto')
                if top_points:
                    ax_inpd50.plot([p[0] for p in top_points], [p[1] for p in top_points], color='#ef4444', linewidth=2, label='Ligging zeebodem (ALAT)')
                if bottom_points:
                    ax_inpd50.plot([p[0] for p in bottom_points], [p[1] for p in bottom_points], color='#3b82f6', linewidth=1.5, linestyle='--', label='Einddiepte (ALAT)')

                ax_inpd50.set_title("d50 (mm) – Geïnterpoleerd", fontsize=11, fontweight='bold', loc='left')
                ax_inpd50.set_ylabel("Diepte t.o.v. LAT (m)", fontsize=9)
                ax_inpd50.set_xlabel("Lengte langs profiel (m)", fontsize=9)
                ax_inpd50.set_xticks(bh_tick_vals)
                ax_inpd50.set_xticklabels(bh_tick_labels, fontsize=7.5, fontweight='bold')
                ax_inpd50.set_ylim(y_max_foot, y_min_head)
                ax_inpd50.grid(True, linestyle='--', alpha=0.3)
                ax_inpd50.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=8, frameon=False)

                cax_id50 = fig3.add_axes([0.90, 0.08, 0.015, 0.34])
                cb_id50 = fig3.colorbar(imd50, cax=cax_id50)
                cb_id50.set_label("d50 (mm)", fontsize=8)
                cb_id50.ax.tick_params(labelsize=7)

                pdf.savefig(fig3)
                plt.close(fig3)
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════════════════
        # PAGES 4+: DEPTH-SLICE SPATIAL MAPS (Landscape A4)
        # ══════════════════════════════════════════════════════════════════════
        df_maps = df.copy()
        df_maps['MID_LAT'] = np.round((df_maps['Tra_van_lat'] + df_maps['Tra_tot_lat']) / 2.0, 0)
        depths = np.sort(df_maps['MID_LAT'].dropna().unique())

        for depth_val in depths:
            sel = df_maps[df_maps['MID_LAT'] == depth_val].dropna(subset=['X', 'Y'])
            if sel.empty:
                continue

            depth_lo = depth_val - 0.5
            depth_hi = depth_val + 0.5

            fig_d, (ax_d63, ax_dd50) = plt.subplots(1, 2, figsize=(11.69, 8.27))
            fig_d.suptitle(f"Diepte-interval Kaart: {depth_lo:.2f} – {depth_hi:.2f} m ALAT", fontsize=14, fontweight='bold', y=0.96, color='#0f172a')
            fig_d.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.10, wspace=0.22)

            bh_at_depth = set(sel['Boornummer'].unique())
            missing_bh = df_coords[~df_coords['Boornummer'].isin(bh_at_depth)]

            # Silt plot
            if not missing_bh.empty:
                ax_d63.scatter(missing_bh['X'], missing_bh['Y'], color='#d4d4d4', s=25, label='Geen data op dit diepte interval')
            
            valid_63 = sel[sel['63calculated. met zoutcorrectie'].notna()]
            nan_63 = sel[sel['63calculated. met zoutcorrectie'].isna()]

            if not valid_63.empty:
                c63_arr = [get_bin_color(v, SILT_BINS, cols63_hex) for v in valid_63['63calculated. met zoutcorrectie']]
                ax_d63.scatter(valid_63['X'], valid_63['Y'], color=c63_arr, s=65, edgecolor='black', linewidth=0.5, label='%<0.063mm')
                for _, row in valid_63.iterrows():
                    ax_d63.annotate(f"{row['63calculated. met zoutcorrectie']:.1f}%", (row['X'], row['Y']), fontsize=6.5, fontweight='bold', ha='center', va='bottom', xytext=(0,3), textcoords='offset points')

            if not nan_63.empty:
                ax_d63.scatter(nan_63['X'], nan_63['Y'], color='#737373', marker='x', s=55, linewidth=1.2, label='Ontbrekend / NaN', zorder=4)

            # DINO boreholes hollow circle overlay
            if 'DINO' in sel.columns:
                dino_sel = sel[sel['DINO'] == 1]
                if not dino_sel.empty:
                    ax_d63.scatter(dino_sel['X'], dino_sel['Y'], s=150, facecolors='none', edgecolors='black', linewidth=1.5, label='Data uit DINO-database', zorder=5)

            # Bathymetry raster outline polygon
            if bath_poly_x is not None and bath_poly_y is not None:
                ax_d63.plot(bath_poly_x, bath_poly_y, color='#3b82f6', linestyle='--', linewidth=1.2, label='Bathymetriekaart Omtrek', zorder=2)
                ax_dd50.plot(bath_poly_x, bath_poly_y, color='#3b82f6', linestyle='--', linewidth=1.2, label='Bathymetriekaart Omtrek', zorder=2)

            ax_d63.set_title("%<0.063mm", fontsize=11, fontweight='bold')
            ax_d63.set_xlabel("X", fontsize=9)
            ax_d63.set_ylabel("Y", fontsize=9)
            ax_d63.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_d63.tick_params(axis='x', rotation=25, labelsize=7.5)
            ax_d63.ticklabel_format(useOffset=False, style='plain')
            ax_d63.grid(True, linestyle='--', alpha=0.3)
            ax_d63.legend(loc='lower right', fontsize=8, framealpha=0.9)
            ax_d63.set_aspect('equal', 'datalim')

            # d50 plot
            if not missing_bh.empty:
                ax_dd50.scatter(missing_bh['X'], missing_bh['Y'], color='#d4d4d4', s=25, label='Geen data op dit diepte interval')
            
            valid_d50 = sel[sel['d50'].notna()]
            nan_d50 = sel[sel['d50'].isna()]

            if not valid_d50.empty:
                cd50_arr = [get_bin_color(v, D50_BINS, cols_d50_hex) for v in valid_d50['d50']]
                ax_dd50.scatter(valid_d50['X'], valid_d50['Y'], color=cd50_arr, s=65, edgecolor='black', linewidth=0.5, label='d50 (mm)')
                for _, row in valid_d50.iterrows():
                    ax_dd50.annotate(f"{row['d50']:.2f}", (row['X'], row['Y']), fontsize=6.5, fontweight='bold', ha='center', va='bottom', xytext=(0,3), textcoords='offset points')

            if not nan_d50.empty:
                ax_dd50.scatter(nan_d50['X'], nan_d50['Y'], color='#737373', marker='x', s=55, linewidth=1.2, label='Ontbrekend / NaN', zorder=4)

            if 'DINO' in sel.columns:
                dino_sel = sel[sel['DINO'] == 1]
                if not dino_sel.empty:
                    ax_dd50.scatter(dino_sel['X'], dino_sel['Y'], s=150, facecolors='none', edgecolors='black', linewidth=1.5, label='Data uit DINO-database', zorder=5)

            ax_dd50.set_title("d50 (mm)", fontsize=11, fontweight='bold')
            ax_dd50.set_xlabel("X", fontsize=9)
            ax_dd50.set_ylabel("Y", fontsize=9)
            ax_dd50.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_dd50.tick_params(axis='x', rotation=25, labelsize=7.5)
            ax_dd50.ticklabel_format(useOffset=False, style='plain')
            ax_dd50.grid(True, linestyle='--', alpha=0.3)
            ax_dd50.legend(loc='lower right', fontsize=8, framealpha=0.9)
            ax_dd50.set_aspect('equal', 'datalim')

            pdf.savefig(fig_d)
            plt.close(fig_d)

    buf.seek(0)
    return buf.getvalue()

# Callback to capture clicks from the map
def handle_map_click():
    if "map_plot" not in st.session_state:
        return
    map_event = st.session_state.map_plot
    if not map_event or "selection" not in map_event:
        return
    points = map_event["selection"].get("points", [])
    if not points:
        return

    # Collect borehole names from all clicked points across all traces
    for p in points:
        clicked_bh = None
        # 1. Try customdata first
        cd = p.get("customdata")
        if cd is not None:
            clicked_bh = cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd

        # 2. Fallback to lat/lon proximity matching if customdata is missing or from background trace
        if (not clicked_bh or clicked_bh not in boreholes) and "lat" in p and "lon" in p:
            c_lat = p["lat"]
            c_lon = p["lon"]
            dists = (df_coords['lat'] - c_lat)**2 + (df_coords['lon'] - c_lon)**2
            min_idx = dists.idxmin()
            if np.sqrt(dists.loc[min_idx]) < 0.01:
                clicked_bh = df_coords.loc[min_idx, 'Boornummer']

        if clicked_bh and clicked_bh in boreholes:
            if clicked_bh not in st.session_state.custom_profile:
                st.session_state.custom_profile.append(clicked_bh)
            else:
                st.session_state.custom_profile.remove(clicked_bh)
            # Sync to sidebar multiselect
            st.session_state.sidebar_sel = list(st.session_state.custom_profile)
            break  # only process the first valid hit per click

# Sync changes from sidebar multiselect widget back to custom_profile
if st.session_state.sidebar_sel != st.session_state.custom_profile:
    st.session_state.custom_profile = st.session_state.sidebar_sel

# 6. Header Section
head_left, head_right = st.columns([9, 1])
with head_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
        <span style="font-size: 2.2rem; color: #4f46e5;">📐</span>
        <div>
            <h1 style="margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.03em;">Geotechnische Profiel Builder</h1>
            <p style="margin: 0; font-size: 0.82rem; color: #71717a;">Klik op boringen op de kaart om een profieldoorsnede te definiëren en %<0.063mm en d50 eigenschappen te tekenen</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Licht" if IS_DARK else "🌙 Donker"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# 7. Sidebar Controls
st.sidebar.markdown('<div class="sidebar-title">🧭 Profielinstellingen</div>', unsafe_allow_html=True)

# Bathymetry sampling status
has_df = ('df' in locals() or 'df' in globals())
valid_bath_count = df['bathymetry'].notna().sum() if has_df and 'bathymetry' in df.columns else 0
if valid_bath_count > 0:
    st.sidebar.caption(f"🟢 **Bathymetriekaart (EPSG:25831)**: Actief ({valid_bath_count} punten gesampeld)")
else:
    st.sidebar.caption("🟡 **Bathymetriekaart**: Geen rasterdekking voor het huidige bestand. Upload hierboven een overeenkomstige `.img` / `.tif` kaart.")

# Map zoom level setting
map_zoom_level = st.sidebar.slider(
    "🔍 Zoomniveau Kaart",
    min_value=7.0,
    max_value=18.0,
    value=11.5,
    step=0.25,
    help="Pas het standaard zoomniveau van de boringselectiekaart aan."
)

# Toggle to show value labels on profile bars
show_labels = st.sidebar.checkbox("🏷️ Toon waarden in grafiek", value=True)

# Force white plot backgrounds for exporting
force_white_plots = st.sidebar.checkbox("⚪ Exporteren geschikte grafieken (witte achtergrond, zwarte tekst)", value=False)
is_dark_plot = IS_DARK and not force_white_plots

# Plot spacing (bar width in meters)
bar_width = st.sidebar.slider(
    "📏 Staaftbreedte Boringen (m)",
    min_value=10,
    max_value=250,
    value=50,
    step=5,
)

# Colormap settings
st.sidebar.markdown('<div class="sidebar-title">🎨 Kleurbereik & Paletten</div>', unsafe_allow_html=True)

# Silt/Clay colormaps
cmap_options = ['Custom (Discrete)', 'Viridis', 'Plasma', 'Cividis', 'Inferno', 'Magma', 'Turbo', 'Rainbow', 'Spectral_r', 'coolwarm']
selected_cmap_63 = st.sidebar.selectbox("Kleurenschaal %<0.063mm", options=cmap_options, index=0, key="cmap_63")

min_63_data = float(df['63calculated. met zoutcorrectie'].min()) if has_df else 0.0
max_63_data = float(df['63calculated. met zoutcorrectie'].max()) if has_df else 25.0
limits_63 = st.sidebar.slider(
    "Bereik %<0.063mm",
    min_value=0.0,
    max_value=50.0,
    value=(min_63_data, max_63_data),
    step=0.5,
    key="limits_63_slider"
)

# d50 colormaps
selected_cmap_d50 = st.sidebar.selectbox("Kleurenschaal d50", options=cmap_options, index=0, key="cmap_d50")

min_d50_data = float(df['d50'].min()) if has_df else 0.0
max_d50_data = float(df['d50'].max()) if has_df else 1.0
limits_d50 = st.sidebar.slider(
    "Bereik d50 (mm)",
    min_value=0.0,
    max_value=10.0,
    value=(min_d50_data, max_d50_data),
    step=0.01,
    key="limits_d50_slider"
)

# Configure active colorscales, limits, and colorbars
if selected_cmap_63 == 'Custom (Discrete)':
    n_bins_63 = len(SILT_BINS) - 1
    colorscale_63 = create_equal_discrete_colorscale(n_bins_63, HEX_COLORS)
    cmin_63 = 0.0
    cmax_63 = 1.0
    silt_ticks_vals = [i / n_bins_63 for i in range(n_bins_63 + 1)]
    silt_ticks_text = [f"{b:g}" for b in SILT_BINS]
    colorbar_63 = dict(
        title="%<0.063mm",
        x=0.47,
        thickness=15,
        len=0.85,
        y=0.45,
        tickmode='array',
        tickvals=silt_ticks_vals,
        ticktext=silt_ticks_text
    )
    cb_63_heat = dict(
        title="%<0.063mm",
        thickness=15,
        tickmode='array',
        tickvals=silt_ticks_vals,
        ticktext=silt_ticks_text
    )
else:
    colorscale_63 = selected_cmap_63
    cmin_63 = limits_63[0]
    cmax_63 = limits_63[1]
    colorbar_63 = dict(
        title="%<0.063mm",
        x=0.47,
        thickness=15,
        len=0.85,
        y=0.45
    )
    cb_63_heat = dict(title="%<0.063mm", thickness=15)

if selected_cmap_d50 == 'Custom (Discrete)':
    n_bins_d50 = len(D50_BINS) - 1
    colorscale_d50 = create_equal_discrete_colorscale(n_bins_d50, HEX_COLORS)
    cmin_d50 = 0.0
    cmax_d50 = 1.0
    d50_ticks_vals = [i / n_bins_d50 for i in range(n_bins_d50 + 1)]
    d50_ticks_text = [f"{b:g}" for b in D50_BINS]
    colorbar_d50 = dict(
        title="d50 (mm)",
        x=1.02,
        thickness=15,
        len=0.85,
        y=0.45,
        tickmode='array',
        tickvals=d50_ticks_vals,
        ticktext=d50_ticks_text
    )
    cb_d50_heat = dict(
        title="d50 (mm)",
        thickness=15,
        tickmode='array',
        tickvals=d50_ticks_vals,
        ticktext=d50_ticks_text
    )
else:
    colorscale_d50 = selected_cmap_d50
    cmin_d50 = limits_d50[0]
    cmax_d50 = limits_d50[1]
    colorbar_d50 = dict(
        title="d50 (mm)",
        x=1.02,
        thickness=15,
        len=0.85,
        y=0.45
    )
    cb_d50_heat = dict(title="d50 (mm)", thickness=15)

# Active Custom Profile Editor in Sidebar
st.sidebar.markdown('<div class="sidebar-title">📍 Profielpad Beheer</div>', unsafe_allow_html=True)

selected_list = st.sidebar.multiselect(
    "Zoek & selecteer boringen:",
    options=boreholes,
    key="sidebar_sel"
)

if st.session_state.custom_profile:
    st.sidebar.markdown("**Volgorde profielpad bewerken:**")
    for idx, bh in enumerate(st.session_state.custom_profile):
        col_name, col_up, col_down, col_del = st.sidebar.columns([5, 1, 1, 1])
        with col_name:
            st.markdown(f"<span style='font-size:0.85rem; font-weight:600;'>{idx+1}. {bh}</span>", unsafe_allow_html=True)
        with col_up:
            if idx > 0:
                st.button("▲", key=f"up_{bh}_{idx}", help=f"Verplaats {bh} omhoog", on_click=move_up_callback, args=(idx,))
        with col_down:
            if idx < len(st.session_state.custom_profile) - 1:
                st.button("▼", key=f"down_{bh}_{idx}", help=f"Verplaats {bh} omlaag", on_click=move_down_callback, args=(idx,))
        with col_del:
            st.button("❌", key=f"del_{bh}_{idx}", help=f"Verwijder {bh}", on_click=remove_bh_callback, args=(bh,))
                
    st.sidebar.button("🗑️ Geselecteerd pad wissen", use_container_width=True, on_click=clear_path_callback)

# Interpolation settings sidebar
st.sidebar.markdown('<div class="sidebar-title">🧩 Interpolatie-instellingen</div>', unsafe_allow_html=True)
interp_dx = st.sidebar.number_input(
    "Horizontale stap dx (m)",
    min_value=1.0, max_value=500.0, value=10.0, step=1.0,
    help="Gridresolutie langs het profielpad, in meters."
)
interp_dy = st.sidebar.number_input(
    "Dieptestap dy (m)",
    min_value=0.05, max_value=500.0, value=0.25, step=0.05,
    help="Gridresolutie in de diepterichting, in meters."
)

_method_options = [
    "OpenCV TELEA inpainting" if HAS_CV2 else "OpenCV TELEA (niet geïnstalleerd)",
    "scipy – linear",
    "scipy – cubic",
    "scipy – nearest",
    "sklearn – RBF (thin-plate)",
    "sklearn – Gaussian Process",
    "skimage – biharmonic",
]
interp_method = st.sidebar.selectbox(
    "Interpolatiemethode",
    options=_method_options,
    index=0 if HAS_CV2 else 1,
    help="Algoritme om waarden tussen boringen op het grid op te vullen."
)
if not HAS_CV2 and interp_method.startswith("OpenCV"):
    st.sidebar.warning("`opencv-python` is niet geïnstalleerd – kies een andere methode.")

# Anisotropy / Layer Continuation Settings
st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1rem;">📐 Anisotropie (Laagcontinuïteit)</div>', unsafe_allow_html=True)
anis_x = st.sidebar.slider(
    "Horizontaal gewicht (X)",
    min_value=0.01, max_value=10.0, value=1.0, step=0.05,
    help="Lagere waarden t.o.v. Y verkleinen de horizontale coördinaat, wat laagcontinuïteit afdwingt."
)
anis_y = st.sidebar.slider(
    "Verticaal gewicht (Y)",
    min_value=0.01, max_value=10.0, value=1.0, step=0.05,
    help="Standaard verticaal gewicht. Gewoonlijk op 1.0 gehouden."
)

# 📄 PDF Export Report Section in Sidebar
st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1.2rem;">📄 Rapport Exporteren</div>', unsafe_allow_html=True)
if st.sidebar.button("⚙️ Genereer PDF-rapport", use_container_width=True, help="Genereert het volledige PDF-rapport met kaarten, profielen, interpolatie, diepte-intervallen en statistieken."):
    with st.spinner("PDF-rapport genereren…"):
        pdf_data = generate_pdf_report(
            df, df_coords, st.session_state.custom_profile,
            interp_dx, interp_dy, interp_method,
            anis_x, anis_y, use_bath_map, bath_raster_available, uploaded_bath_file,
            bar_width, show_labels, selected_cmap_63, selected_cmap_d50
        )
        st.session_state.pdf_report_bytes = pdf_data

if "pdf_report_bytes" in st.session_state and st.session_state.pdf_report_bytes:
    st.sidebar.download_button(
        label="📥 Download PDF-rapport",
        data=st.session_state.pdf_report_bytes,
        file_name=f"Geotechnisch_Profiel_Rapport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.sidebar.info("Selecteer boringen uit de keuzelijst of klik op markeringen op de kaart om uw profielpad op te bouwen.")

# Calculate indices of selected boreholes in df_coords for Plotly Mapbox selection persistence
selected_indices = []
for bh in st.session_state.custom_profile:
    idx_list = df_coords[df_coords['Boornummer'] == bh].index.tolist()
    if idx_list:
        pos_idx = df_coords.index.get_loc(idx_list[0])
        selected_indices.append(pos_idx)

# 8. Row 1: Interactive Map
fig_map = go.Figure()

# Add Bathymetry Surface Layer FIRST if Bathymetry Map Mode is enabled and raster is available
if use_bath_map and bath_raster_available:
    try:
        b_lats, b_lons, b_depths = get_bathymetry_mapbox_points(
            uploaded_bath_file.getvalue() if uploaded_bath_file is not None else "25NZE4376ml9_1.img",
            bath_filename=uploaded_bath_file.name if uploaded_bath_file is not None else None,
            grid_size=45
        )
        if b_lats and b_lons and b_depths:
            fig_map.add_trace(go.Densitymapbox(
                lat=b_lats,
                lon=b_lons,
                z=b_depths,
                radius=16,
                opacity=0.65,
                colorscale=[
                    [0.0, "#ffffff"],
                    [0.25, "#c6dbef"],
                    [0.50, "#6baed6"],
                    [0.75, "#2171b5"],
                    [1.00, "#08306b"]
                ],
                colorbar=dict(
                    title="Bathymetrie (m)",
                    x=-0.08,
                    len=0.7,
                    thickness=12,
                    title_font=dict(size=11),
                    tickfont=dict(size=10)
                ),
                hoverinfo='skip',
                name="Bathymetriekaart"
            ))
    except Exception:
        pass

# Profile path line (rendered below borehole markers)
if len(st.session_state.custom_profile) >= 2:
    prof_coords = []
    for bh in st.session_state.custom_profile:
        r = df_coords[df_coords['Boornummer'] == bh].iloc[0]
        prof_coords.append((r['lat'], r['lon']))
    fig_map.add_trace(go.Scattermapbox(
        lat=[p[0] for p in prof_coords],
        lon=[p[1] for p in prof_coords],
        mode='lines',
        line=dict(color='#ef4444', width=2),
        hoverinfo='skip',
        name='Profielpad',
    ))

# ── Build per-marker colour / size / text arrays ──────────────────────────────
profile_set = {bh: i for i, bh in enumerate(st.session_state.custom_profile)}

marker_colors = []
marker_sizes  = []
marker_texts  = []

if not df_coords.empty:
    for bh in df_coords['Boornummer']:
        if bh in profile_set:
            seq = profile_set[bh] + 1            # 1-based sequence number
            marker_colors.append('#dc2626')      # red = selected in profile path sequence
            marker_sizes.append(16)
            marker_texts.append(f"<b>{seq}. {bh}</b>")
        else:
            marker_colors.append('#22c55e')      # green dot for all available boreholes
            marker_sizes.append(10)
            marker_texts.append(bh)              # show name for unselected

# ── Borehole Markers trace (rendered ON TOP of bathymetry & profile lines) ───
if not df_coords.empty:
    fig_map.add_trace(go.Scattermapbox(
        lat=df_coords['lat'],
        lon=df_coords['lon'],
        mode='markers+text',
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
        ),
        text=marker_texts,
        textposition='top center',
        textfont=dict(
            size=10,
            color='#1e1e24' if not IS_DARK else '#e4e4e7',
        ),
        hoverinfo='text',
        hovertext=[
            f"<b>Boring: {row['Boornummer']}</b><br>"
            f"EPSG:25831 X: {row['X']:.1f}, Y: {row['Y']:.1f}<br>"
            f"RD X: {row['X_RD']:.1f}, Y: {row['Y_RD']:.1f}<br>"
            f"Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}<br>"
            + (f"Bathymetrie: {row['bathymetry']:.2f} m" if pd.notna(row.get('bathymetry')) else "Bathymetrie: N/B")
            for _, row in df_coords.iterrows()
        ],
        customdata=df_coords['Boornummer'].values,
        name='Alle boringen',
    ))

# Generate Bathymetry Mapbox Layer overlay (explicitly rendered BELOW traces)
mapbox_layers = []
if use_bath_map and bath_raster_available:
    try:
        b64_bath, coords_bath = get_bathymetry_mapbox_layer(
            uploaded_bath_file.getvalue() if uploaded_bath_file is not None else "25NZE4376ml9_1.img",
            bath_filename=uploaded_bath_file.name if uploaded_bath_file is not None else None,
            colormap='white_deep_blue'
        )
        if b64_bath and coords_bath:
            mapbox_layers.append({
                "sourcetype": "image",
                "source": b64_bath,
                "coordinates": coords_bath,
                "opacity": 0.65,
                "below": "traces"
            })
    except Exception:
        pass

fig_map.update_layout(
    mapbox=dict(
        style="open-street-map" if not IS_DARK else "carto-darkmatter",
        center=dict(lat=df_coords['lat'].mean() if not df_coords.empty else 51.5, lon=df_coords['lon'].mean() if not df_coords.empty else 3.5),
        zoom=map_zoom_level,
        layers=mapbox_layers,
        uirevision="fixed_map_ui"
    ),
    uirevision="fixed_map_ui",
    clickmode='event+select',
    margin=dict(l=0, r=0, t=0, b=0),
    height=400,
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">Kaart Boringselectie</div><div class="chart-subtitle">Klik opeenvolgend op punten om het profielpad op te bouwen. Gebruik het muiswiel, touchpad of de schuifbalk in de zijbalk om in/uit te zoomen.</div></div>', unsafe_allow_html=True)
st.plotly_chart(
    fig_map, 
    use_container_width=True, 
    config={"displayModeBar": True, "scrollZoom": True}, 
    on_select=handle_map_click, 
    key="map_plot"
)
st.markdown('</div>', unsafe_allow_html=True)


# 9. Profile Analysis & Plotting Section
if len(st.session_state.custom_profile) < 2:
    st.info("💡 **Definieer een profielpad**: Klik op **twee of meer boringen** op de bovenstaande kaart om een profieldoorsnede te genereren.")
else:
    # 10. Generate distances along the selected path
    cum_dist = {}
    last_x, last_y = None, None
    current_dist = 0.0
    
    for bh in st.session_state.custom_profile:
        bh_row = df_coords[df_coords['Boornummer'] == bh]
        if not bh_row.empty:
            x, y = bh_row['X'].iloc[0], bh_row['Y'].iloc[0]
            if last_x is not None:
                d = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                current_dist += d
            cum_dist[bh] = current_dist
            last_x, last_y = x, y
        else:
            cum_dist[bh] = 0.0
            
    # Filter dataset for selected boreholes
    prof_df = df[df['Boornummer'].isin(st.session_state.custom_profile)].copy()
    prof_df['cum_dist'] = prof_df['Boornummer'].map(cum_dist)
    
    # Calculate heights and base depths of the layers
    heights = prof_df['Tra_tot_lat'] - prof_df['Tra_van_lat']
    bottoms = prof_df['Tra_van_lat']
    
    # Create text columns for labels inside the plot
    prof_df['63calculated_text'] = prof_df['63calculated. met zoutcorrectie'].apply(lambda v: f"{v:.2f}%")
    prof_df['d50_text'] = prof_df['d50'].apply(lambda v: f"{v:.2f}")
    
    # ── 1. Top Surface (ALAT) calculation ───────────────────────────────────
    bath_points = []
    if use_bath_map and bath_raster_available:
        bath_points = get_profile_bathymetry(
            st.session_state.custom_profile,
            df_coords,
            bath_file_bytes=uploaded_bath_file.getvalue() if uploaded_bath_file is not None else None,
            bath_filename=uploaded_bath_file.name if uploaded_bath_file is not None else None,
            step_m=min(interp_dx, 1.0)
        )

    def clean_surface_points(points):
        if not points:
            return points
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points], dtype=np.float64)
        valid_mask = ~np.isnan(ys)
        
        if np.sum(valid_mask) == 0:
            return [(float(x), 0.0) for x in xs]
        elif np.sum(valid_mask) == len(ys):
            return points
        else:
            clean_ys = np.interp(xs, xs[valid_mask], ys[valid_mask])
            return [(float(x), float(y)) for x, y in zip(xs, clean_ys)]

    bottom_points = []
    for bh in st.session_state.custom_profile:
        bh_df = prof_df[prof_df['Boornummer'] == bh]
        valid_tot = bh_df['Tra_tot_lat'].dropna() if not bh_df.empty else pd.Series()
        if not valid_tot.empty:
            bottom_points.append((cum_dist[bh], float(valid_tot.max())))
        else:
            bottom_points.append((cum_dist[bh], np.nan))
    bottom_points = clean_surface_points(bottom_points)

    if use_bath_map and bath_raster_available and bath_points and len(bath_points) > 0:
        top_points = bath_points
        is_bath_top = True
    else:
        if use_bath_map:
            st.warning("⚠️ **Bathymetriekaart Buiten Bereik**: De geselecteerde coördinaten vallen buiten het bereik van de bathymetriekaart. Maaiveldhoogtes uit het Excel-bestand worden gebruikt.")
        top_points = []
        for bh in st.session_state.custom_profile:
            bh_df = prof_df[prof_df['Boornummer'] == bh]
            valid_van = bh_df['Tra_van_lat'].dropna() if not bh_df.empty else pd.Series()
            if not valid_van.empty:
                top_points.append((cum_dist[bh], float(valid_van.min())))
            else:
                top_points.append((cum_dist[bh], np.nan))
        top_points = clean_surface_points(top_points)
        is_bath_top = False
            
    # KPI Stats for the profile path
    num_bh_selected = len(st.session_state.custom_profile)
    mean_63 = prof_df['63calculated. met zoutcorrectie'].mean()
    mean_d50 = prof_df['d50'].mean()
    profile_length = current_dist
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Geselecteerde boringen</div><div class="metric-value">{num_bh_selected}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Profiellengte</div><div class="metric-value">{profile_length:.1f} m</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Gemiddelde %<0.063mm</div><div class="metric-value">{mean_63:.2f} %</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Gemiddelde d50</div><div class="metric-value">{mean_d50:.2f} mm</div></div>', unsafe_allow_html=True)

    # 11. Plotly Subplots configuration
    fig_sub = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True, 
        horizontal_spacing=0.04,
        subplot_titles=(
            "<b>Percentage (%) < 0.063mm </b><br>", 
            "<b>d50 (mm)</b><br>"
        )
    )
    
    # Trace 1: Silt/Clay profile (excluding NaN rows)
    silt_df = prof_df[prof_df['63calculated. met zoutcorrectie'].notna() & prof_df['Tra_van_lat'].notna() & prof_df['Tra_tot_lat'].notna()].copy()
    if not silt_df.empty:
        silt_heights = silt_df['Tra_tot_lat'] - silt_df['Tra_van_lat']
        silt_bottoms = silt_df['Tra_van_lat']
        silt_norm_vals = map_values_to_equal_bins(silt_df['63calculated. met zoutcorrectie'], SILT_BINS)
        silt_layer_widths = (0.3 + 0.9 * np.nan_to_num(silt_norm_vals, nan=0.0)) * bar_width
        silt_bar_color = silt_norm_vals if selected_cmap_63 == 'Custom (Discrete)' else silt_df['63calculated. met zoutcorrectie']

        fig_sub.add_trace(
            go.Bar(
                x=silt_df['cum_dist'],
                y=silt_heights,
                base=silt_bottoms,
                width=silt_layer_widths,
                marker=dict(
                    color=silt_bar_color,
                    colorscale=colorscale_63,
                    cmin=cmin_63,
                    cmax=cmax_63,
                    colorbar=colorbar_63,
                    showscale=True,
                    line=dict(color='rgba(0,0,0,0.15)' if not is_dark_plot else 'rgba(255,255,255,0.15)', width=0.4)
                ),
                text=silt_df['63calculated_text'] if show_labels else None,
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white' if is_dark_plot else 'black', size=10),
                hoverinfo='text',
                hovertext=[
                    f"Borehole: {row['Boornummer']}<br>"
                    f"Depth: {row['Tra_van_lat']:.2f} - {row['Tra_tot_lat']:.2f} m<br>"
                    f"%<0.063mm: {row['63calculated. met zoutcorrectie']:.2f}%<br>"
                    f"d50: {row['d50']:.2f} mm" if pd.notna(row.get('d50')) else f"%<0.063mm: {row['63calculated. met zoutcorrectie']:.2f}%"
                    for _, row in silt_df.iterrows()
                ],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Trace 2: d50 profile (excluding NaN rows)
    d50_df = prof_df[prof_df['d50'].notna() & prof_df['Tra_van_lat'].notna() & prof_df['Tra_tot_lat'].notna()].copy()
    if not d50_df.empty:
        d50_heights = d50_df['Tra_tot_lat'] - d50_df['Tra_van_lat']
        d50_bottoms = d50_df['Tra_van_lat']
        d50_norm_vals = map_values_to_equal_bins(d50_df['d50'], D50_BINS)
        d50_layer_widths = (0.3 + 0.9 * np.nan_to_num(d50_norm_vals, nan=0.0)) * bar_width
        d50_bar_color = d50_norm_vals if selected_cmap_d50 == 'Custom (Discrete)' else d50_df['d50']

        fig_sub.add_trace(
            go.Bar(
                x=d50_df['cum_dist'],
                y=d50_heights,
                base=d50_bottoms,
                width=d50_layer_widths,
                marker=dict(
                    color=d50_bar_color,
                    colorscale=colorscale_d50,
                    cmin=cmin_d50,
                    cmax=cmax_d50,
                    colorbar=colorbar_d50,
                    showscale=True,
                    line=dict(color='rgba(0,0,0,0.15)' if not is_dark_plot else 'rgba(255,255,255,0.15)', width=0.4)
                ),
                text=d50_df['d50_text'] if show_labels else None,
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white' if is_dark_plot else 'black', size=10),
                hoverinfo='text',
                hovertext=[
                    f"Borehole: {row['Boornummer']}<br>"
                    f"Depth: {row['Tra_van_lat']:.2f} - {row['Tra_tot_lat']:.2f} m<br>"
                    f"d50: {row['d50']:.2f} mm<br>"
                    f"%<0.063mm: {row['63calculated. met zoutcorrectie']:.2f}%" if pd.notna(row.get('63calculated. met zoutcorrectie')) else f"d50: {row['d50']:.2f} mm"
                    for _, row in d50_df.iterrows()
                ],
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Add Topography Line (connect tops of columns or follow continuous bathymetry)
    for col_idx in [1, 2]:
        fig_sub.add_trace(
            go.Scatter(
                x=[p[0] for p in top_points],
                y=[p[1] for p in top_points],
                mode='lines' if is_bath_top else 'lines+markers',
                line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2.5),
                marker=dict(size=6, color='#ef4444') if not is_bath_top else None,
                hovertemplate="Dist: %{x:.1f} m<br>Bathymetry Depth: %{y:.2f} m (LAT)<extra>Ligging zeebodem (ALAT)</extra>",
                name='Ligging zeebodem (ALAT)',
                showlegend=(col_idx == 1)  # Only show once in the legend
            ),
            row=1, col=col_idx
        )
        # Add Bottom boundary line
        fig_sub.add_trace(
            go.Scatter(
                x=[p[0] for p in bottom_points],
                y=[p[1] for p in bottom_points],
                mode='lines+markers',
                line=dict(color='#3b82f6' if not is_dark_plot else '#93c5fd', width=2, dash='dash'),
                marker=dict(size=6, color='#3b82f6'),
                name='Einddiepte boring (ALAT)',
                showlegend=(col_idx == 1)
            ),
            row=1, col=col_idx
        )
        
    # Identify boreholes that have NaN values across all property rows or no depth intervals
    nan_boreholes = []
    for bh in st.session_state.custom_profile:
        bh_df = prof_df[prof_df['Boornummer'] == bh]
        valid_rows = bh_df[bh_df['Tra_van_lat'].notna() & bh_df['Tra_tot_lat'].notna()]
        if valid_rows.empty:
            nan_boreholes.append(bh)

    # Plot placeholder dotted column & text for NaN boreholes on subplots
    top_map = dict(top_points)
    bot_map = dict(bottom_points)
    for bh in nan_boreholes:
        d_x = cum_dist[bh]
        t_y = top_map.get(d_x, 10.0)
        b_y = bot_map.get(d_x, t_y + 5.0)
        for col_idx in [1, 2]:
            fig_sub.add_trace(
                go.Scatter(
                    x=[d_x, d_x],
                    y=[t_y, b_y],
                    mode='lines+text',
                    line=dict(color='#9ca3af' if not is_dark_plot else '#6b7280', width=1.5, dash='dot'),
                    text=["", "No Data"],
                    textposition="middle center",
                    textfont=dict(size=9, color='#6b7280' if not is_dark_plot else '#9ca3af'),
                    hoverinfo='text',
                    hovertext=f"Borehole: {bh}<br>Status: No Data (All NaN)",
                    showlegend=False
                ),
                row=1, col=col_idx
            )
        
    # Annotate borehole names directly above top_points line for each selected borehole
    top_map = dict(top_points) if top_points else {}
    for bh in st.session_state.custom_profile:
        d_x = cum_dist[bh]
        t_y = top_map.get(d_x, 10.0)
        for col_idx in [1, 2]:
            fig_sub.add_trace(
                go.Scatter(
                    x=[d_x],
                    y=[t_y],
                    mode='text',
                    text=[f"<b>{bh}</b>"],
                    textposition="top center",
                    textfont=dict(size=10, color='#0f172a' if not is_dark_plot else '#f8fafc'),
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=1, col=col_idx
            )

    # Set tick marks matching the selection path
    tick_vals = [cum_dist[bh] for bh in st.session_state.custom_profile]
    tick_text = [f"<b>{bh}</b><br>{cum_dist[bh]:.0f}m" for bh in st.session_state.custom_profile]
    
    fig_sub.update_layout(
        template="plotly_dark" if is_dark_plot else "plotly_white",
        height=620,
        margin=dict(l=50, r=120, t=70, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
        plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
    )
    
    # Configure axes
    fig_sub.update_yaxes(
        title_text="Diepte t.o.v. LAT (m)", 
        autorange="reversed", 
        gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
        row=1, col=1
    )
    fig_sub.update_yaxes(
        gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
        row=1, col=2
    )
    
    fig_sub.update_xaxes(
        title_text="Lengte langs profiel (m)",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickmode='array',
        gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
        row=1, col=1
    )
    fig_sub.update_xaxes(
        title_text="Lengte langs profiel (m)",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickmode='array',
        gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
        row=1, col=2
    )

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_sub, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 13. Interpolate Profile Button & Heatmaps ─────────────────────────────
    st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">🧩 Geïnterpoleerde Dwarsdoorsnede</div><div class="chart-subtitle">Hiaten tussen boringen opvullen met inpainting. Pas dx / dy aan in de zijbalk.</div></div>', unsafe_allow_html=True)

    if not HAS_CV2 and interp_method.startswith("OpenCV"):
        st.warning(
            "⚠️ **OpenCV niet geïnstalleerd** – selecteer een andere methode in de zijbalk."
        )

    st.button(
        "🧩 Profiel Interpoleren",
        on_click=trigger_interpolation,
        use_container_width=False,
        help="Rasteriseer boringlagen op een 2D grid en vul hiaten op met inpainting.",
    )

    if st.session_state.run_interpolation:
        with st.spinner(f"Interpolatie uitvoeren ({interp_method})…"):
            try:
                x63, y63, grid63 = interpolate_profile(
                    prof_df, cum_dist, top_points, bottom_points,
                    interp_dx, interp_dy,
                    "63calculated. met zoutcorrectie",
                    interp_method=interp_method,
                    anis_x=anis_x,
                    anis_y=anis_y,
                )
                xd50, yd50, gridd50 = interpolate_profile(
                    prof_df, cum_dist, top_points, bottom_points,
                    interp_dx, interp_dy,
                    "d50",
                    interp_method=interp_method,
                    anis_x=anis_x,
                    anis_y=anis_y,
                )

                # Build tick labels matching borehole positions
                tick_vals_inp = [cum_dist[bh] for bh in st.session_state.custom_profile]
                tick_text_inp = [
                    f"<b>{bh}</b><br>{cum_dist[bh]:.0f}m"
                    for bh in st.session_state.custom_profile
                ]

                # Surface-envelope scatter data
                top_x = [p[0] for p in top_points]
                top_y = [p[1] for p in top_points]
                bot_x = [p[0] for p in bottom_points]
                bot_y = [p[1] for p in bottom_points]

                axis_common = dict(
                    tickvals=tick_vals_inp,
                    ticktext=tick_text_inp,
                    tickmode='array',
                    gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
                )
                yaxis_common = dict(
                    title="Diepte t.o.v. LAT (m)",
                    autorange="reversed",
                    gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
                )

                # ── Silt / Clay heatmap ───────────────────────────────────────
                z_63_plot = map_values_to_equal_bins(grid63, SILT_BINS) if selected_cmap_63 == 'Custom (Discrete)' else grid63
                fig63 = go.Figure()
                fig63.add_trace(go.Heatmap(
                    z=z_63_plot,
                    x=x63,
                    y=y63,
                    customdata=grid63,
                    colorscale=colorscale_63,
                    zmin=cmin_63,
                    zmax=cmax_63,
                    colorbar=cb_63_heat,
                    connectgaps=False,
                    hovertemplate="Afstand: %{x:.0f} m<br>Diepte: %{y:.2f} m<br>%<0.063mm: %{customdata:.2f}%<extra></extra>" if selected_cmap_63 == 'Custom (Discrete)' else "Afstand: %{x:.0f} m<br>Diepte: %{y:.2f} m<br>%<0.063mm: %{z:.2f}%<extra></extra>",
                ))
                fig63.add_trace(go.Scatter(
                    x=top_x, y=top_y,
                    mode='lines' if is_bath_top else 'lines+markers',
                    line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2.5),
                    marker=dict(size=6, color='#ef4444') if not is_bath_top else None,
                    hovertemplate="Afstand: %{x:.1f} m<br>Bathymetrie Diepte: %{y:.2f} m (LAT)<extra>Ligging zeebodem (ALAT)</extra>",
                    name='Ligging zeebodem (ALAT)'
                ))
                fig63.add_trace(go.Scatter(
                    x=bot_x, y=bot_y,
                    mode='lines+markers',
                    line=dict(color='#3b82f6' if not is_dark_plot else '#93c5fd', width=2, dash='dash'),
                    marker=dict(size=6, color='#3b82f6'),
                    name='Einddiepte boring (ALAT)'
                ))
                fig63.update_layout(
                    title="<b>%<0.063mm – Geïnterpoleerd</b>",
                    template="plotly_dark" if is_dark_plot else "plotly_white",
                    height=420,
                    margin=dict(l=60, r=20, t=60, b=50),
                    paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    xaxis=dict(title="Afstand langs profiel (m)", **axis_common),
                    yaxis=dict(**yaxis_common),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                )

                # ── d50 heatmap ───────────────────────────────────────────────
                z_d50_plot = map_values_to_equal_bins(gridd50, D50_BINS) if selected_cmap_d50 == 'Custom (Discrete)' else gridd50
                fig_d50 = go.Figure()
                fig_d50.add_trace(go.Heatmap(
                    z=z_d50_plot,
                    x=xd50,
                    y=yd50,
                    customdata=gridd50,
                    colorscale=colorscale_d50,
                    zmin=cmin_d50,
                    zmax=cmax_d50,
                    colorbar=cb_d50_heat,
                    connectgaps=False,
                    hovertemplate="Afstand: %{x:.0f} m<br>Diepte: %{y:.2f} m<br>d50: %{customdata:.2f} mm<extra></extra>" if selected_cmap_d50 == 'Custom (Discrete)' else "Afstand: %{x:.0f} m<br>Diepte: %{y:.2f} m<br>d50: %{z:.2f} mm<extra></extra>",
                ))
                fig_d50.add_trace(go.Scatter(
                    x=top_x, y=top_y,
                    mode='lines' if is_bath_top else 'lines+markers',
                    line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2.5),
                    marker=dict(size=6, color='#ef4444') if not is_bath_top else None,
                    hovertemplate="Afstand: %{x:.1f} m<br>Bathymetrie Diepte: %{y:.2f} m (LAT)<extra>Ligging zeebodem (ALAT)</extra>",
                    name='Ligging zeebodem (ALAT)'
                ))
                fig_d50.add_trace(go.Scatter(
                    x=bot_x, y=bot_y,
                    mode='lines+markers',
                    line=dict(color='#3b82f6' if not is_dark_plot else '#93c5fd', width=2, dash='dash'),
                    marker=dict(size=6, color='#3b82f6'),
                    name='Einddiepte boring (ALAT)'
                ))
                fig_d50.update_layout(
                    title="<b>d50 (mm) – Geïnterpoleerd</b>",
                    template="plotly_dark" if is_dark_plot else "plotly_white",
                    height=420,
                    margin=dict(l=60, r=20, t=60, b=50),
                    paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    xaxis=dict(title="Afstand langs profiel (m)", **axis_common),
                    yaxis=dict(**yaxis_common),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                )

                st.plotly_chart(fig63, use_container_width=True)
                st.plotly_chart(fig_d50, use_container_width=True)

            except Exception as e:
                st.error(f"Interpolation failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 12. Depth-Slice Spatial Maps  (inspired by make_maps.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">📍 Depth-Slice Borehole Maps</div><div class="chart-subtitle">Spatial scatter plots of %<0.063mm and d50 at each unique depth level across all boreholes.</div></div>', unsafe_allow_html=True)

show_depth_maps = st.checkbox("🗺️ Show Depth-Slice Maps", value=False, help="Generate one pair of maps per depth level showing spatial distribution of properties.")

if show_depth_maps:
    # Compute MID_LAT (midpoint depth of each layer)
    df_maps = df.copy()
    df_maps['MID_LAT'] = np.round((df_maps['Tra_van_lat'] + df_maps['Tra_tot_lat']) / 2.0,0)
    depths = np.sort(df_maps['MID_LAT'].dropna().unique())

    if len(depths) == 0:
        st.warning("No valid depth layers found in the dataset.")
    else:
        st.info(f"📊 Generating **{len(depths)}** depth-slice map pairs...")

        # Build Plotly-compatible discrete colorscale for scatter markers
        n_bins_63_map = len(SILT_BINS) - 1
        n_bins_d50_map = len(D50_BINS) - 1
        silt_map_colors = interpolate_colors(HEX_COLORS, n_bins_63_map)
        d50_map_colors = interpolate_colors(HEX_COLORS, n_bins_d50_map)

        def get_bin_color(val, bins, colors):
            """Return color for a value based on discrete bins."""
            if pd.isna(val):
                return '#999999'
            for i in range(len(bins) - 1):
                if val < bins[i + 1]:
                    return colors[min(i, len(colors) - 1)]
            return colors[-1]

        bath_bytes_input = uploaded_bath_file.getvalue() if uploaded_bath_file is not None else None
        bath_poly_x, bath_poly_y = get_bathymetry_polygon(bath_bytes_input)

        for depth_val in depths:
            sel = df_maps[df_maps['MID_LAT'] == depth_val].dropna(subset=['X', 'Y'])
            if len(sel) == 0:
                continue

            depth_lo = depth_val - 0.5
            depth_hi = depth_val + 0.5

            st.markdown(f'<h4 style="margin-top:1.5rem;">Diepte: {depth_lo:.2f} – {depth_hi:.2f} m ALAT</h4>', unsafe_allow_html=True)

            fig_depth = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "<b>%<0.063mm</b>",
                    "<b>d50 (mm)</b>"
                ),
                horizontal_spacing=0.08
            )

            # ── Bathymetry raster outline polygon ────────────────────────────
            if bath_poly_x is not None and bath_poly_y is not None:
                for col_idx in [1, 2]:
                    fig_depth.add_trace(
                        go.Scatter(
                            x=bath_poly_x,
                            y=bath_poly_y,
                            mode='lines',
                            line=dict(color='#3b82f6', width=1.8, dash='dash'),
                            fill='toself',
                            fillcolor='rgba(59, 130, 246, 0.04)',
                            name='Omtrek Bathymetriekaart',
                            showlegend=(col_idx == 1),
                            hoverinfo='skip'
                        ),
                        row=1, col=col_idx
                    )

            # ── All boreholes not at this depth → grey reference dots ─────────
            bh_at_depth = set(sel['Boornummer'].unique())
            missing_bh = df_coords[~df_coords['Boornummer'].isin(bh_at_depth)]
            if not missing_bh.empty:
                for col_idx in [1, 2]:
                    fig_depth.add_trace(
                        go.Scatter(
                            x=missing_bh['X'],
                            y=missing_bh['Y'],
                            mode='markers',
                            marker=dict(size=8, color='#d4d4d4', symbol='circle', line=dict(color='#a3a3a3', width=0.5)),
                            name='Geen data op dit diepte interval' if col_idx == 1 else '',
                            showlegend=(col_idx == 1),
                            hovertext=[f"Boring: {bh}<br>(geen laag op dit interval)" for bh in missing_bh['Boornummer']],
                            hoverinfo='text',
                        ),
                        row=1, col=col_idx
                    )

            # ── Left: Silt/Clay (%<0.063mm) ──────────────────────────────────
            valid_63 = sel[sel['63calculated. met zoutcorrectie'].notna()]
            nan_63 = sel[sel['63calculated. met zoutcorrectie'].isna()]

            if not valid_63.empty:
                silt_colors_arr = [get_bin_color(v, SILT_BINS, silt_map_colors) for v in valid_63['63calculated. met zoutcorrectie']]
                fig_depth.add_trace(
                    go.Scatter(
                        x=valid_63['X'],
                        y=valid_63['Y'],
                        mode='markers+text',
                        marker=dict(size=14, color=silt_colors_arr, line=dict(color='black', width=0.5)),
                        text=[f"{v:.1f}%" for v in valid_63['63calculated. met zoutcorrectie']],
                        textposition='top center',
                        textfont=dict(size=9),
                        hovertext=[
                            f"Boring: {row['Boornummer']}<br>%<0.063mm: {row['63calculated. met zoutcorrectie']:.2f}%"
                            for _, row in valid_63.iterrows()
                        ],
                        hoverinfo='text',
                        showlegend=False,
                    ),
                    row=1, col=1
                )

            if not nan_63.empty:
                fig_depth.add_trace(
                    go.Scatter(
                        x=nan_63['X'],
                        y=nan_63['Y'],
                        mode='markers',
                        marker=dict(size=12, color='#999999', symbol='x', line=dict(color='black', width=0.5)),
                        name='Ontbrekend / NaN',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # Check for DINO boreholes
            dino_sel = pd.DataFrame()
            if 'DINO' in sel.columns:
                dino_sel = sel[sel['DINO'] == 1]
                if not dino_sel.empty:
                    fig_depth.add_trace(
                        go.Scatter(
                            x=dino_sel['X'],
                            y=dino_sel['Y'],
                            mode='markers',
                            marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
                            name='Data uit DINO-database',
                            showlegend=True,
                        ),
                        row=1, col=1
                    )

            # ── Right: d50 (mm) ───────────────────────────────────────────────
            valid_d50 = sel[sel['d50'].notna()]
            nan_d50 = sel[sel['d50'].isna()]

            if not valid_d50.empty:
                d50_colors_arr = [get_bin_color(v, D50_BINS, d50_map_colors) for v in valid_d50['d50']]
                fig_depth.add_trace(
                    go.Scatter(
                        x=valid_d50['X'],
                        y=valid_d50['Y'],
                        mode='markers+text',
                        marker=dict(size=14, color=d50_colors_arr, line=dict(color='black', width=0.5)),
                        text=[f"{v:.3f}" for v in valid_d50['d50']],
                        textposition='top center',
                        textfont=dict(size=9),
                        hovertext=[
                            f"Boring: {row['Boornummer']}<br>d50: {row['d50']:.3f} mm"
                            for _, row in valid_d50.iterrows()
                        ],
                        hoverinfo='text',
                        showlegend=False,
                    ),
                    row=1, col=2
                )

            if not nan_d50.empty:
                fig_depth.add_trace(
                    go.Scatter(
                        x=nan_d50['X'],
                        y=nan_d50['Y'],
                        mode='markers',
                        marker=dict(size=12, color='#999999', symbol='x', line=dict(color='black', width=0.5)),
                        name='Ontbrekend / NaN',
                        showlegend=False,
                    ),
                    row=1, col=2
                )

            if 'DINO' in sel.columns and not dino_sel.empty:
                fig_depth.add_trace(
                    go.Scatter(
                        x=dino_sel['X'],
                        y=dino_sel['Y'],
                        mode='markers',
                        marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
                        showlegend=False,
                    ),
                    row=1, col=2
                )

            # ── Discrete colorbar annotations (silt & d50) ───────────────────
            # Add invisible scatter traces just for the colorbar legend
            # Silt/Clay colorbar
            silt_dummy_vals = [(SILT_BINS[i] + SILT_BINS[i+1]) / 2.0 for i in range(n_bins_63_map)]
            silt_cs_plotly = create_equal_discrete_colorscale(n_bins_63_map, HEX_COLORS)
            silt_tick_vals_map = [i / n_bins_63_map for i in range(n_bins_63_map + 1)]
            silt_tick_text_map = [f"{b:g}" for b in SILT_BINS]

            fig_depth.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=0.001,
                        color=[0.5],
                        colorscale=silt_cs_plotly,
                        cmin=0, cmax=1,
                        colorbar=dict(
                            title="%<0.063mm",
                            x=0.44, len=0.85, y=0.5, thickness=12,
                            tickmode='array',
                            tickvals=silt_tick_vals_map,
                            ticktext=silt_tick_text_map,
                        ),
                        showscale=True,
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=1, col=1
            )

            # d50 colorbar
            d50_cs_plotly = create_equal_discrete_colorscale(n_bins_d50_map, HEX_COLORS)
            d50_tick_vals_map = [i / n_bins_d50_map for i in range(n_bins_d50_map + 1)]
            d50_tick_text_map = [f"{b:g}" for b in D50_BINS]

            fig_depth.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=0.001,
                        color=[0.5],
                        colorscale=d50_cs_plotly,
                        cmin=0, cmax=1,
                        colorbar=dict(
                            title="d50 (mm)",
                            x=1.02, len=0.85, y=0.5, thickness=12,
                            tickmode='array',
                            tickvals=d50_tick_vals_map,
                            ticktext=d50_tick_text_map,
                        ),
                        showscale=True,
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=1, col=2
            )

            fig_depth.update_layout(
                title=dict(
                    text=f"<b>Diepte-interval: {depth_lo:.2f} – {depth_hi:.2f} m ALAT</b>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=15)
                ),
                template="plotly_dark" if is_dark_plot else "plotly_white",
                height=520,
                margin=dict(l=60, r=80, t=80, b=60),
                paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
                xaxis2=dict(title="X", scaleanchor="y2", scaleratio=1),
                yaxis=dict(title="Y"),
                yaxis2=dict(title=""),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig_depth, use_container_width=True, key=f"depth_slice_{depth_val}")

# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Geïnterpoleerde Diepte-interval Kaarten (2D Spatial Interpolation)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='margin-top:2.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
st.markdown("### 🌐 Geïnterpoleerde Diepte-interval Kaarten")
st.markdown("Klik op de onderstaande knop om voor **elk diepte-interval** een 2D ruimtelijke interpolatiekaart te genereren. Hierbij worden exact dezelfde interpolatie-instellingen gebruikt als ingesteld in het zijpaneel.")

col_btn1, col_btn2 = st.columns([1.5, 2.5])
with col_btn1:
    gen_depth_interp_btn = st.button(
        "🌐 Genereer Geïnterpoleerde Diepte-interval Kaarten",
        key="btn_generate_depth_interpolations",
        use_container_width=True
    )

if 'show_depth_interp_maps' not in st.session_state:
    st.session_state['show_depth_interp_maps'] = False

if gen_depth_interp_btn:
    st.session_state['show_depth_interp_maps'] = True

if st.session_state['show_depth_interp_maps']:
    st.info(f"📊 **Toegepaste Interpolatie parameters**: Methode: `{interp_method}` | Grid: `dx={interp_dx}m, dy={interp_dy}m` | Gewichten: `X={anis_x}, Y={anis_y}`")
    
    with st.spinner("2D Diepte-interval interpolatiekaarten genereren..."):
        for depth_val in depths:
            sel = df_maps[df_maps['MID_LAT'] == depth_val].dropna(subset=['X', 'Y'])
            if len(sel) == 0:
                continue

            depth_lo = depth_val - 0.5
            depth_hi = depth_val + 0.5

            # Perform 2D Spatial Interpolation for %<0.063mm and d50
            x_grid_63, y_grid_63, z_grid_63 = interpolate_spatial_2d(
                df, df_coords, depth_lo, depth_hi, "63calculated. met zoutcorrectie",
                dx=interp_dx, dy=interp_dy, interp_method=interp_method,
                anis_x=anis_x, anis_y=anis_y
            )
            x_grid_d50, y_grid_d50, z_grid_d50 = interpolate_spatial_2d(
                df, df_coords, depth_lo, depth_hi, "d50",
                dx=interp_dx, dy=interp_dy, interp_method=interp_method,
                anis_x=anis_x, anis_y=anis_y
            )

            fig_d_interp = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "<b>%<0.063mm – Geïnterpoleerd</b>",
                    "<b>d50 (mm) – Geïnterpoleerd</b>"
                ),
                horizontal_spacing=0.08
            )

            # Left Subplot: %<0.063mm Spatial Heatmap
            if x_grid_63 is not None and z_grid_63 is not None:
                fig_d_interp.add_trace(
                    go.Heatmap(
                        x=x_grid_63,
                        y=y_grid_63,
                        z=z_grid_63,
                        colorscale=silt_cs_plotly,
                        showscale=False,
                        hoverinfo='x+y+z',
                        name='%<0.063mm'
                    ),
                    row=1, col=1
                )

            # Right Subplot: d50 Spatial Heatmap
            if x_grid_d50 is not None and z_grid_d50 is not None:
                fig_d_interp.add_trace(
                    go.Heatmap(
                        x=x_grid_d50,
                        y=y_grid_d50,
                        z=z_grid_d50,
                        colorscale=d50_cs_plotly,
                        showscale=False,
                        hoverinfo='x+y+z',
                        name='d50'
                    ),
                    row=1, col=2
                )

            # Overlay Bathymetry Polygon Outline on both subplots
            if bath_poly_x is not None and bath_poly_y is not None:
                for col_idx in [1, 2]:
                    fig_d_interp.add_trace(
                        go.Scatter(
                            x=bath_poly_x,
                            y=bath_poly_y,
                            mode='lines',
                            line=dict(color='#3b82f6', width=1.8, dash='dash'),
                            name='Omtrek Bathymetriekaart',
                            showlegend=(col_idx == 1),
                            hoverinfo='skip'
                        ),
                        row=1, col=col_idx
                    )

            # Overlay Borehole Markers & Labels on %<0.063mm subplot
            valid_63 = sel[sel['63calculated. met zoutcorrectie'].notna()]
            if not valid_63.empty:
                silt_colors_arr = [get_bin_color(v, SILT_BINS, silt_map_colors) for v in valid_63['63calculated. met zoutcorrectie']]
                fig_d_interp.add_trace(
                    go.Scatter(
                        x=valid_63['X'],
                        y=valid_63['Y'],
                        mode='markers+text',
                        marker=dict(size=12, color=silt_colors_arr, line=dict(color='black', width=0.5)),
                        text=[f"{v:.1f}%" for v in valid_63['63calculated. met zoutcorrectie']],
                        textposition='top center',
                        textfont=dict(size=9),
                        hovertext=[f"Boring: {row['Boornummer']}<br>%<0.063mm: {row['63calculated. met zoutcorrectie']:.2f}%" for _, row in valid_63.iterrows()],
                        hoverinfo='text',
                        showlegend=False,
                    ),
                    row=1, col=1
                )

            # Overlay Borehole Markers & Labels on d50 subplot
            valid_d50 = sel[sel['d50'].notna()]
            if not valid_d50.empty:
                d50_colors_arr = [get_bin_color(v, D50_BINS, d50_map_colors) for v in valid_d50['d50']]
                fig_d_interp.add_trace(
                    go.Scatter(
                        x=valid_d50['X'],
                        y=valid_d50['Y'],
                        mode='markers+text',
                        marker=dict(size=12, color=d50_colors_arr, line=dict(color='black', width=0.5)),
                        text=[f"{v:.3f}" for v in valid_d50['d50']],
                        textposition='top center',
                        textfont=dict(size=9),
                        hovertext=[f"Boring: {row['Boornummer']}<br>d50: {row['d50']:.3f} mm" for _, row in valid_d50.iterrows()],
                        hoverinfo='text',
                        showlegend=False,
                    ),
                    row=1, col=2
                )

            # Add DINO boreholes outline if present
            if 'DINO' in sel.columns:
                dino_sel = sel[sel['DINO'] == 1]
                if not dino_sel.empty:
                    for col_idx in [1, 2]:
                        fig_d_interp.add_trace(
                            go.Scatter(
                                x=dino_sel['X'],
                                y=dino_sel['Y'],
                                mode='markers',
                                marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
                                name='Data uit DINO-database' if col_idx == 1 else '',
                                showlegend=(col_idx == 1),
                            ),
                            row=1, col=col_idx
                        )

            # Colorbars setup
            fig_d_interp.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(
                        size=0.001, color=[0.5], colorscale=silt_cs_plotly, cmin=0, cmax=1,
                        colorbar=dict(title="%<0.063mm", x=0.44, len=0.85, y=0.5, thickness=12, tickmode='array', tickvals=silt_tick_vals_map, ticktext=silt_tick_text_map),
                        showscale=True,
                    ),
                    showlegend=False, hoverinfo='skip',
                ),
                row=1, col=1
            )
            fig_d_interp.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(
                        size=0.001, color=[0.5], colorscale=d50_cs_plotly, cmin=0, cmax=1,
                        colorbar=dict(title="d50 (mm)", x=1.02, len=0.85, y=0.5, thickness=12, tickmode='array', tickvals=d50_tick_vals_map, ticktext=d50_tick_text_map),
                        showscale=True,
                    ),
                    showlegend=False, hoverinfo='skip',
                ),
                row=1, col=2
            )

            fig_d_interp.update_layout(
                title=dict(
                    text=f"<b>Geïnterpoleerde Diepte-interval Kaart: {depth_lo:.2f} – {depth_hi:.2f} m ALAT</b>",
                    x=0.5, xanchor="center", font=dict(size=15)
                ),
                template="plotly_dark" if is_dark_plot else "plotly_white",
                height=520,
                margin=dict(l=60, r=80, t=80, b=60),
                paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
                xaxis2=dict(title="X", scaleanchor="y2", scaleratio=1),
                yaxis=dict(title="Y"),
                yaxis2=dict(title=""),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig_d_interp, use_container_width=True, key=f"depth_slice_interp_{depth_val}")

st.markdown('</div>', unsafe_allow_html=True)
