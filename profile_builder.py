import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from scipy.interpolate import interp1d
#import tkinter as tk
#from tkinter import filedialog


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name='Locaties_einddiepte_LAT_3')
    
    # Coordinate Conversion:
    # Source coordinates X, Y in the sheet are in UTM Zone 31N (EPSG:32631).
    # We transform them to Lat/Lon (EPSG:4326) for Mapbox.
    # We also transform them to RD coordinates (EPSG:28992) for local display.
    to_wgs84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    
    lons, lats = to_wgs84.transform(df['X'].values, df['Y'].values)
    x_rd, y_rd = to_rd.transform(df['X'].values, df['Y'].values)
    
    df['lon'] = lons
    df['lat'] = lats
    df['X_RD'] = x_rd
    df['Y_RD'] = y_rd
    return df

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
else:
    st.info("Please upload an Excel file to get started.")
    st.stop()

# 5. Extract Unique Coordinates and Boreholes
df_coords = df[['Boornummer', 'X', 'Y', 'lat', 'lon', 'X_RD', 'Y_RD']].drop_duplicates().sort_values('Boornummer')
boreholes = list(df_coords['Boornummer'].unique())

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

    # ── 1. Stamp known borehole layer values ──────────────────────────────
    # image pixel dtype float32 – OpenCV requires 0-255 uint8 for inpainting;
    # we normalise, inpaint, then un-normalise.
    val_min = prof_df[prop_col].min()
    val_max = prof_df[prop_col].max()
    val_range = val_max - val_min if val_max != val_min else 1.0

    img    = np.zeros((ny, nx), dtype=np.float32)   # will hold normalised values
    mask   = np.ones((ny, nx),  dtype=np.uint8) * 255  # 255 = unknown pixel

    for bh, dist in cum_dist.items():
        bh_df = prof_df[prof_df['Boornummer'] == bh]
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
        cd = p.get("customdata")
        if cd is None:
            continue
        # customdata can arrive as a scalar or as a 1-element list
        clicked_bh = cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd
        if clicked_bh in boreholes:
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
            <h1 style="margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.03em;">Borehole Profile Builder</h1>
            <p style="margin: 0; font-size: 0.82rem; color: #71717a;">Click boreholes on the map to define a cross-section transect and draw silt/clay (63calculated) and d50 properties</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# 7. Sidebar Controls
st.sidebar.markdown('<div class="sidebar-title">🧭 Profile Settings</div>', unsafe_allow_html=True)

# Toggle to show value labels on profile bars
show_labels = st.sidebar.checkbox("🏷️ Show values on plot", value=True)

# Force white plot backgrounds for exporting
force_white_plots = st.sidebar.checkbox("⚪ Export-ready plots (white bg, black text)", value=False)
is_dark_plot = IS_DARK and not force_white_plots

# Plot spacing (bar width in meters)
bar_width = st.sidebar.slider(
    "📏 Borehole Bar Width (m)",
    min_value=10,
    max_value=150,
    value=50,
    step=5,
)

# Colormap settings
st.sidebar.markdown('<div class="sidebar-title">🎨 Color Ranges & Palettes</div>', unsafe_allow_html=True)

# Silt/Clay colormaps
cmap_options = ['Viridis', 'Plasma', 'Cividis', 'Inferno', 'Magma', 'Turbo', 'Rainbow', 'Spectral_r', 'coolwarm']
selected_cmap_63 = st.sidebar.selectbox("Silt/Clay Colormap", options=cmap_options, index=0, key="cmap_63")

min_63_data = float(df['63calculated. met zoutcorrectie'].min())
max_63_data = float(df['63calculated. met zoutcorrectie'].max())
limits_63 = st.sidebar.slider(
    "Silt/Clay Range (%)",
    min_value=0.0,
    max_value=35.0,
    value=(min_63_data, max_63_data),
    step=0.5,
    key="limits_63_slider"
)

# d50 colormaps
selected_cmap_d50 = st.sidebar.selectbox("d50 Colormap", options=cmap_options, index=1, key="cmap_d50")

min_d50_data = float(df['d50'].min())
max_d50_data = float(df['d50'].max())
limits_d50 = st.sidebar.slider(
    "d50 Range (mm)",
    min_value=0.0,
    max_value=1.0,
    value=(min_d50_data, max_d50_data),
    step=0.01,
    key="limits_d50_slider"
)

# Active Custom Profile Editor in Sidebar
st.sidebar.markdown('<div class="sidebar-title">📍 Profile Path Manager</div>', unsafe_allow_html=True)

selected_list = st.sidebar.multiselect(
    "Search & select boreholes:",
    options=boreholes,
    key="sidebar_sel"
)

if st.session_state.custom_profile:
    st.sidebar.markdown("**Path Sequence Editor:**")
    for idx, bh in enumerate(st.session_state.custom_profile):
        col_name, col_up, col_down, col_del = st.sidebar.columns([5, 1, 1, 1])
        with col_name:
            st.markdown(f"<span style='font-size:0.85rem; font-weight:600;'>{idx+1}. {bh}</span>", unsafe_allow_html=True)
        with col_up:
            if idx > 0:
                st.button("▲", key=f"up_{bh}_{idx}", help=f"Move {bh} up", on_click=move_up_callback, args=(idx,))
        with col_down:
            if idx < len(st.session_state.custom_profile) - 1:
                st.button("▼", key=f"down_{bh}_{idx}", help=f"Move {bh} down", on_click=move_down_callback, args=(idx,))
        with col_del:
            st.button("❌", key=f"del_{bh}_{idx}", help=f"Remove {bh}", on_click=remove_bh_callback, args=(bh,))
                
    st.sidebar.button("🗑️ Clear Selected Path", use_container_width=True, on_click=clear_path_callback)

# Interpolation settings sidebar
st.sidebar.markdown('<div class="sidebar-title">🧩 Interpolation Settings</div>', unsafe_allow_html=True)
interp_dx = st.sidebar.number_input(
    "Horizontal step dx (m)",
    min_value=1.0, max_value=500.0, value=10.0, step=1.0,
    help="Grid resolution along the profile path, in metres."
)
interp_dy = st.sidebar.number_input(
    "Depth step dy (m)",
    min_value=0.05, max_value=5.0, value=0.25, step=0.05,
    help="Grid resolution in the depth direction, in metres."
)

_method_options = [
    "OpenCV TELEA inpainting" if HAS_CV2 else "OpenCV TELEA (not installed)",
    "scipy – linear",
    "scipy – cubic",
    "scipy – nearest",
    "sklearn – RBF (thin-plate)",
    "sklearn – Gaussian Process",
    "skimage – biharmonic",
]
interp_method = st.sidebar.selectbox(
    "Interpolation method",
    options=_method_options,
    index=0 if HAS_CV2 else 1,
    help="Algorithm used to fill values between boreholes on the grid."
)
if not HAS_CV2 and interp_method.startswith("OpenCV"):
    st.sidebar.warning("`opencv-python` is not installed – please choose another method.")

# Anisotropy / Layer Continuation Settings
st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1rem;">📐 Anisotropy (Layer Continuation)</div>', unsafe_allow_html=True)
anis_x = st.sidebar.slider(
    "Horizontal weight (X)",
    min_value=0.01, max_value=10.0, value=1.0, step=0.05,
    help="Lower values relative to Y shrink the horizontal coordinate, enforcing layer continuation."
)
anis_y = st.sidebar.slider(
    "Vertical weight (Y)",
    min_value=0.01, max_value=10.0, value=1.0, step=0.05,
    help="Standard vertical weight. Usually kept at 1.0."
)

st.sidebar.info("Select boreholes from the dropdown or click markers on the map to start building your profile path.")

# Calculate indices of selected boreholes in df_coords for Plotly Mapbox selection persistence
selected_indices = []
for bh in st.session_state.custom_profile:
    idx_list = df_coords[df_coords['Boornummer'] == bh].index.tolist()
    if idx_list:
        pos_idx = df_coords.index.get_loc(idx_list[0])
        selected_indices.append(pos_idx)

# 8. Row 1: Interactive Map
fig_map = go.Figure()

# ── Build per-marker colour / size / text arrays ──────────────────────────────
# Encode selection order directly in the main trace so there is only ONE
# clickable layer and no overlay trace can block the click event.
profile_set = {bh: i for i, bh in enumerate(st.session_state.custom_profile)}

marker_colors = []
marker_sizes  = []
marker_texts  = []

for bh in df_coords['Boornummer']:
    if bh in profile_set:
        seq = profile_set[bh] + 1            # 1-based sequence number
        marker_colors.append('#ef4444')      # red = in profile
        marker_sizes.append(18)
        marker_texts.append(f"<b>{seq}</b>")
    else:
        marker_colors.append('#4f46e5' if not IS_DARK else '#818cf8')
        marker_sizes.append(11)
        marker_texts.append(bh)              # show name for unselected

# ── Single Scattermapbox trace ───────────────────────────────────────────────
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
        f"<b>Borehole: {row['Boornummer']}</b><br>"
        f"UTM X: {row['X']:.1f}, Y: {row['Y']:.1f}<br>"
        f"RD X: {row['X_RD']:.1f}, Y: {row['Y_RD']:.1f}<br>"
        f"Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}"
        for _, row in df_coords.iterrows()
    ],
    customdata=df_coords['Boornummer'].values,
    name='All Boreholes',
))

# Profile path line (no markers, so cannot intercept clicks)
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
        name='Profile Path',
    ))

fig_map.update_layout(
    mapbox=dict(
        style="open-street-map" if not IS_DARK else "carto-darkmatter",
        center=dict(lat=df_coords['lat'].mean(), lon=df_coords['lon'].mean()),
        zoom=11.5
    ),
    clickmode='event+select',
    margin=dict(l=0, r=0, t=0, b=0),
    height=400,
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">Boreholes Selection Map</div><div class="chart-subtitle">Click points sequentially to build your profile path. Clicking an active sequence point removes it.</div></div>', unsafe_allow_html=True)
st.plotly_chart(
    fig_map, 
    use_container_width=True, 
    config={"displayModeBar": False}, 
    on_select=handle_map_click, 
    key="map_plot"
)
st.markdown('</div>', unsafe_allow_html=True)


# 9. Profile Analysis & Plotting Section
if len(st.session_state.custom_profile) < 2:
    st.info("💡 **Define a Profile Path**: Click on **two or more boreholes** on the map above. Once selected, their side-by-side profile transect will draw below, using true geographic distances.")
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
    prof_df['63calculated_text'] = prof_df['63calculated. met zoutcorrectie'].apply(lambda v: f"{v:.1f}%")
    prof_df['d50_text'] = prof_df['d50'].apply(lambda v: f"{v:.3f}")
    
    # Build Topography and Bottom Boundaries
    top_points = []
    bottom_points = []
    for bh in st.session_state.custom_profile:
        bh_df = prof_df[prof_df['Boornummer'] == bh]
        if not bh_df.empty:
            top_points.append((cum_dist[bh], bh_df['Tra_van_lat'].min()))
            bottom_points.append((cum_dist[bh], bh_df['Tra_tot_lat'].max()))
            
    # KPI Stats for the profile path
    num_bh_selected = len(st.session_state.custom_profile)
    mean_63 = prof_df['63calculated. met zoutcorrectie'].mean()
    mean_d50 = prof_df['d50'].mean()
    profile_length = current_dist
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Selected Boreholes</div><div class="metric-value">{num_bh_selected}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Profile Length</div><div class="metric-value">{profile_length:.1f} m</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Average Silt/Clay</div><div class="metric-value">{mean_63:.2f} %</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Average d50</div><div class="metric-value">{mean_d50:.4f} mm</div></div>', unsafe_allow_html=True)

    # 11. Plotly Subplots configuration
    fig_sub = make_subplots(
        rows=1, cols=2, 
        shared_yaxes=True, 
        horizontal_spacing=0.04,
        subplot_titles=(
            "<b>Percentage Content (%) < 0.063mm </b><br>", 
            "<b>d50 (mm)</b><br>"
        )
    )
    
    # Trace 1: Silt/Clay profile
    fig_sub.add_trace(
        go.Bar(
            x=prof_df['cum_dist'],
            y=heights,
            base=bottoms,
            width=bar_width,
            marker=dict(
                color=prof_df['63calculated. met zoutcorrectie'],
                colorscale=selected_cmap_63,
                cmin=limits_63[0],
                cmax=limits_63[1],
                colorbar=dict(
                    title="Silt/Clay (%)", 
                    x=0.47, 
                    thickness=15,
                    len=0.85,
                    y=0.45
                ),
                showscale=True,
                line=dict(color='rgba(0,0,0,0.15)' if not is_dark_plot else 'rgba(255,255,255,0.15)', width=0.4)
            ),
            text=prof_df['63calculated_text'] if show_labels else None,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white' if is_dark_plot else 'black', size=10),
            hoverinfo='text',
            hovertext=[
                f"Borehole: {row['Boornummer']}<br>"
                f"Depth: {row['Tra_van_lat']:.2f} - {row['Tra_tot_lat']:.2f} m<br>"
                f"Silt/Clay: {row['63calculated. met zoutcorrectie']:.2f}%<br>"
                f"d50: {row['d50']:.4f} mm"
                for _, row in prof_df.iterrows()
            ],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Trace 2: d50 profile
    fig_sub.add_trace(
        go.Bar(
            x=prof_df['cum_dist'],
            y=heights,
            base=bottoms,
            width=bar_width,
            marker=dict(
                color=prof_df['d50'],
                colorscale=selected_cmap_d50,
                cmin=limits_d50[0],
                cmax=limits_d50[1],
                colorbar=dict(
                    title="d50 (mm)", 
                    x=1.02, 
                    thickness=15,
                    len=0.85,
                    y=0.45
                ),
                showscale=True,
                line=dict(color='rgba(0,0,0,0.15)' if not is_dark_plot else 'rgba(255,255,255,0.15)', width=0.4)
            ),
            text=prof_df['d50_text'] if show_labels else None,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white' if is_dark_plot else 'black', size=10),
            hoverinfo='text',
            hovertext=[
                f"Borehole: {row['Boornummer']}<br>"
                f"Depth: {row['Tra_van_lat']:.2f} - {row['Tra_tot_lat']:.2f} m<br>"
                f"Silt/Clay: {row['63calculated. met zoutcorrectie']:.2f}%<br>"
                f"d50: {row['d50']:.4f} mm"
                for _, row in prof_df.iterrows()
            ],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add Topography Line (connect tops of columns) to both subplots
    for col_idx in [1, 2]:
        fig_sub.add_trace(
            go.Scatter(
                x=[p[0] for p in top_points],
                y=[p[1] for p in top_points],
                mode='lines+markers',
                line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2, dash='dash'),
                marker=dict(size=6, color='#ef4444'),
                name='Top Surface (LAT)',
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
                name='Borehole Bottom (LAT)',
                showlegend=(col_idx == 1)
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
        title_text="Depth below LAT (m)", 
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
    st.markdown('<div class="chart-wrap"><div class="chart-header"><div class="chart-title">🧩 Interpolated Cross-Section</div><div class="chart-subtitle">Fill gaps between boreholes using image inpainting. Adjust dx / dy in the sidebar.</div></div>', unsafe_allow_html=True)

    if not HAS_CV2 and interp_method.startswith("OpenCV"):
        st.warning(
            "⚠️ **OpenCV not installed** – select a different method in the sidebar before running."
        )

    st.button(
        "🧩 Interpolate Profile",
        on_click=trigger_interpolation,
        use_container_width=False,
        help="Rasterise borehole layers onto a 2-D grid and fill gaps with image inpainting.",
    )

    if st.session_state.run_interpolation:
        with st.spinner(f"Running interpolation ({interp_method})…"):
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
                    title="Depth below LAT (m)",
                    autorange="reversed",
                    gridcolor="rgba(0,0,0,0.06)" if not is_dark_plot else "rgba(255,255,255,0.06)",
                )

                # ── Silt / Clay heatmap ───────────────────────────────────────
                fig63 = go.Figure()
                fig63.add_trace(go.Heatmap(
                    z=grid63,
                    x=x63,
                    y=y63,
                    colorscale=selected_cmap_63,
                    zmin=limits_63[0],
                    zmax=limits_63[1],
                    colorbar=dict(title="Silt/Clay (%)", thickness=15),
                    connectgaps=False,
                    hovertemplate="Dist: %{x:.0f} m<br>Depth: %{y:.2f} m<br>Silt/Clay: %{z:.2f}%<extra></extra>",
                ))
                fig63.add_trace(go.Scatter(
                    x=top_x, y=top_y,
                    mode='lines+markers',
                    line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2, dash='dash'),
                    marker=dict(size=6, color='#ef4444'),
                    name='Top Surface (LAT)'
                ))
                fig63.add_trace(go.Scatter(
                    x=bot_x, y=bot_y,
                    mode='lines+markers',
                    line=dict(color='#3b82f6' if not is_dark_plot else '#93c5fd', width=2, dash='dash'),
                    marker=dict(size=6, color='#3b82f6'),
                    name='Borehole Bottom (LAT)'
                ))
                fig63.update_layout(
                    title="<b>Silt/Clay Content (%) – Interpolated</b>",
                    template="plotly_dark" if is_dark_plot else "plotly_white",
                    height=420,
                    margin=dict(l=60, r=20, t=60, b=50),
                    paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    xaxis=dict(title="Distance along profile path (m)", **axis_common),
                    yaxis=dict(**yaxis_common),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                )

                # ── d50 heatmap ───────────────────────────────────────────────
                fig_d50 = go.Figure()
                fig_d50.add_trace(go.Heatmap(
                    z=gridd50,
                    x=xd50,
                    y=yd50,
                    colorscale=selected_cmap_d50,
                    zmin=limits_d50[0],
                    zmax=limits_d50[1],
                    colorbar=dict(title="d50 (mm)", thickness=15),
                    connectgaps=False,
                    hovertemplate="Dist: %{x:.0f} m<br>Depth: %{y:.2f} m<br>d50: %{z:.4f} mm<extra></extra>",
                ))
                fig_d50.add_trace(go.Scatter(
                    x=top_x, y=top_y,
                    mode='lines+markers',
                    line=dict(color='#ef4444' if not is_dark_plot else '#fca5a5', width=2, dash='dash'),
                    marker=dict(size=6, color='#ef4444'),
                    name='Top Surface (LAT)'
                ))
                fig_d50.add_trace(go.Scatter(
                    x=bot_x, y=bot_y,
                    mode='lines+markers',
                    line=dict(color='#3b82f6' if not is_dark_plot else '#93c5fd', width=2, dash='dash'),
                    marker=dict(size=6, color='#3b82f6'),
                    name='Borehole Bottom (LAT)'
                ))
                fig_d50.update_layout(
                    title="<b>Median Grain Size d50 (mm) – Interpolated</b>",
                    template="plotly_dark" if is_dark_plot else "plotly_white",
                    height=420,
                    margin=dict(l=60, r=20, t=60, b=50),
                    paper_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    plot_bgcolor="rgba(0,0,0,0)" if is_dark_plot else "#ffffff",
                    xaxis=dict(title="Distance along profile path (m)", **axis_common),
                    yaxis=dict(**yaxis_common),
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                )

                st.plotly_chart(fig63, use_container_width=True)
                st.plotly_chart(fig_d50, use_container_width=True)

            except Exception as e:
                st.error(f"Interpolation failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # 12. Row 3: Raw Details Table
    st.markdown("### Profile Borehole Layers Data Details")
    
    # Fetch data sorted by selected sequence and depth
    tbl_rows = []
    for bh in st.session_state.custom_profile:
        bh_df = df[df['Boornummer'] == bh].sort_values('Tra_van_lat')
        for _, row in bh_df.iterrows():
            tbl_rows.append(row)
            
    if tbl_rows:
        rows_html = ""
        for row in tbl_rows:
            rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #4f46e5;">{row['Boornummer']}</td>
                <td>{row['monster nummer']}</td>
                <td>{row['Tra_van_lat']:.2f} - {row['Tra_tot_lat']:.2f} m</td>
                <td style="font-weight: 500;">{row['63calculated. met zoutcorrectie']:.2f}%</td>
                <td style="font-weight: 500;">{row['d50']:.4f} mm</td>
                <td><span class="badge badge-blue">Layer {row['nummer_diepte']}</span></td>
            </tr>
            """
        
        table_html = f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Borehole</th>
                    <th>Monster Nummer</th>
                    <th>Depth Range (LAT)</th>
                    <th>Silt/Clay Content (%)</th>
                    <th>d50 (mm)</th>
                    <th>Layer Index</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
