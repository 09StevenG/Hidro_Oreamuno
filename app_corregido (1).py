
from pathlib import Path
import io
import base64
import warnings
import tempfile
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import plotly.express as px

warnings.filterwarnings("ignore")

# ============================================================
# GEOALERTA OREAMUNO
# Visor de resultados geoespaciales e hidrológicos
# ============================================================

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    ROOT / "data",
    ROOT / "01_Datos_Originales",
    ROOT / "02_Datos_Procesados",
    ROOT / "03_Modelos_Hidrologicos",
    ROOT / "04_Productos_Cartograficos",
]

st.set_page_config(
    page_title="GeoAlerta Oreamuno",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PALETAS CARTOGRÁFICAS
# ============================================================

SUS_COLORS = {
    1: "#2c7bb6",
    2: "#abd9e9",
    3: "#ffffbf",
    4: "#fdae61",
    5: "#d7191c",
}

SUS_NAMES = {
    1: "Muy baja",
    2: "Baja",
    3: "Moderada",
    4: "Alta",
    5: "Muy alta",
}

WORLDCOVER = {
    10: ("Árboles", "#006400"),
    20: ("Matorral", "#ffbb22"),
    30: ("Pastizales", "#ffff4c"),
    40: ("Cultivos", "#f096ff"),
    50: ("Construcciones", "#fa0000"),
    60: ("Suelo desnudo", "#b4b4b4"),
    70: ("Nieve y hielo", "#f0f0f0"),
    80: ("Agua", "#0064c8"),
    90: ("Humedales herbáceos", "#0096a0"),
    95: ("Manglares", "#00cf75"),
    100: ("Musgos y líquenes", "#fae6a0"),
}

GROUP_COLORS = {
    2: ("Grupo B", "#2ca25f"),
    3: ("Grupo C", "#fec44f"),
    4: ("Grupo D", "#de2d26"),
}

CONTINUOUS_COLORS = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
]


# ============================================================
# UTILIDADES DE ARCHIVOS
# ============================================================

@st.cache_data(show_spinner=False)
def locate_file(filename):

    for folder in FOLDERS:

        if not folder.exists():
            continue

        candidate = folder / filename

        if candidate.exists():
            return str(candidate)

        matches = list(folder.rglob(filename))

        if matches:
            return str(matches[0])

    return None


def get_files():

    extensions = {
        ".tif", ".tiff", ".gpkg", ".shp",
        ".geojson", ".csv", ".png", ".jpg", ".jpeg"
    }

    files = []

    for folder in FOLDERS:

        if folder.exists():

            for path in folder.rglob("*"):

                if path.is_file() and path.suffix.lower() in extensions:
                    files.append(path)

    return sorted(set(files))


@st.cache_data(show_spinner=False)
def read_csv_file(path):

    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_vector_file(path):

    gdf = gpd.read_file(path)

    if gdf.empty or gdf.crs is None:
        return None

    return gdf.to_crs("EPSG:4326")


@st.cache_data(show_spinner=False)
def get_raster_info(path):

    with rasterio.open(path) as src:

        return {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": src.res,
            "bounds": tuple(src.bounds),
            "nodata": src.nodata,
            "count": src.count,
        }


# ============================================================
# MAPAS BASE
# ============================================================

def crear_mapa_base(location, zoom_start=11):

    m = folium.Map(
        location=location,
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        "CartoDB positron",
        name="Mapa base | CartoDB",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Satélite | Esri World Imagery",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="Calles | OpenStreetMap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    return m


# ============================================================
# FUNCIONES DE SIMBOLOGÍA
# ============================================================

def hex_to_rgb(color):

    color = color.lstrip("#")

    return tuple(
        int(color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def interpolar_color(value, low, high):

    palette = [hex_to_rgb(c) for c in CONTINUOUS_COLORS]

    if high <= low:
        position = 0.5
    else:
        position = np.clip(
            (value - low) / (high - low),
            0,
            1,
        )

    scaled = position * (len(palette) - 1)

    index = min(int(scaled), len(palette) - 2)

    fraction = scaled - index

    color1 = np.array(palette[index])
    color2 = np.array(palette[index + 1])

    rgb = (
        color1 * (1 - fraction)
        + color2 * fraction
    ).astype(int)

    return tuple(rgb)


def crear_leyenda_continua(
    title,
    minimum,
    maximum,
    units="",
):
    # Se calculan cinco valores equidistantes para representar la escala.
    values = np.linspace(minimum, maximum, 5)
    gradient = ", ".join(CONTINUOUS_COLORS)

    # Se elimina la indentación del HTML para evitar que Markdown
    # lo interprete como un bloque de código en lugar de renderizarlo.
    html = textwrap.dedent(f"""
        <div style="
            padding:14px;
            border:1px solid rgba(128,128,128,.25);
            border-radius:10px;
            margin-top:8px;
            margin-bottom:14px;
        ">
            <div style="font-weight:650; margin-bottom:10px;">
                {title}
            </div>

            <div style="
                height:17px;
                border-radius:5px;
                background:linear-gradient(to right, {gradient});
            "></div>

            <div style="
                display:flex;
                justify-content:space-between;
                gap:8px;
                margin-top:7px;
                font-size:12px;
                line-height:16px;
                color:inherit;
                font-variant-numeric:tabular-nums;
            ">
                <span style="white-space:nowrap;">{minimum:,.2f} {units}</span>
                <span style="white-space:nowrap;">{values[1]:,.2f}</span>
                <span style="white-space:nowrap;">{values[2]:,.2f}</span>
                <span style="white-space:nowrap;">{values[3]:,.2f}</span>
                <span style="white-space:nowrap;">{maximum:,.2f} {units}</span>
            </div>
        </div>
    """).strip()

    st.markdown(html, unsafe_allow_html=True)


def crear_leyenda_categorica(title, categories):

    st.markdown(f"**{title}**")

    columns = st.columns(min(len(categories), 5))

    for i, (name, color) in enumerate(categories):

        with columns[i % len(columns)]:

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:7px;
                    margin-bottom:10px;
                    font-size:13px;
                ">
                    <span style="
                        display:inline-block;
                        width:17px;
                        height:17px;
                        min-width:17px;
                        background:{color};
                        border:1px solid rgba(0,0,0,.25);
                        border-radius:3px;
                    "></span>
                    <span>{name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# CONVERSIÓN DE RÁSTER A PNG
# ============================================================

@st.cache_data(show_spinner=False)
def raster_to_png(
    path,
    categorical=False,
    category_type=None,
    opacity=0.85,
    max_dim=1600,
):

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError("El ráster no tiene CRS definido.")

        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
        )

        scale = min(
            1,
            max_dim / max(width, height),
        )

        if scale < 1:

            width = max(1, int(width * scale))
            height = max(1, int(height * scale))

            transform, _, _ = calculate_default_transform(
                src.crs,
                "EPSG:4326",
                src.width,
                src.height,
                *src.bounds,
                dst_width=width,
                dst_height=height,
            )

        destination = np.full(
            (height, width),
            np.nan,
            dtype="float32",
        )

        source = (
            src.read(1, masked=True)
            .astype("float32")
            .filled(np.nan)
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=(
                Resampling.nearest
                if categorical
                else Resampling.bilinear
            ),
        )

        valid = np.isfinite(destination)

        rgba = np.zeros(
            (height, width, 4),
            dtype=np.uint8,
        )

        legend = {}

        if not valid.any():

            raise ValueError("El ráster no contiene píxeles válidos.")

        # ----------------------------------------------------
        # SUSCEPTIBILIDAD
        # ----------------------------------------------------

        if category_type == "susceptibility":

            values = np.zeros(
                destination.shape,
                dtype=np.int32,
            )

            values[valid] = np.rint(
                destination[valid]
            ).astype(int)

            for cls, color in SUS_COLORS.items():

                mask = valid & (values == cls)

                rgba[mask, :3] = hex_to_rgb(color)
                rgba[mask, 3] = int(255 * opacity)

            legend = {
                "type": "categorical",
                "title": "Susceptibilidad a inundaciones",
                "categories": [
                    (f"{k}. {SUS_NAMES[k]}", v)
                    for k, v in SUS_COLORS.items()
                ],
            }

        # ----------------------------------------------------
        # WORLDCOVER
        # ----------------------------------------------------

        elif category_type == "worldcover":

            values = np.zeros(
                destination.shape,
                dtype=np.int32,
            )

            values[valid] = np.rint(
                destination[valid]
            ).astype(int)

            categories = []

            for code, (name, color) in WORLDCOVER.items():

                mask = valid & (values == code)

                rgba[mask, :3] = hex_to_rgb(color)
                rgba[mask, 3] = int(255 * opacity)

                if mask.any():
                    categories.append((name, color))

            legend = {
                "type": "categorical",
                "title": "Cobertura terrestre | WorldCover 2021",
                "categories": categories,
            }

        # ----------------------------------------------------
        # GRUPOS HIDROLÓGICOS
        # ----------------------------------------------------

        elif category_type == "groups":

            values = np.zeros(
                destination.shape,
                dtype=np.int32,
            )

            values[valid] = np.rint(
                destination[valid]
            ).astype(int)

            categories = []

            for code, (name, color) in GROUP_COLORS.items():

                mask = valid & (values == code)

                rgba[mask, :3] = hex_to_rgb(color)
                rgba[mask, 3] = int(255 * opacity)

                if mask.any():
                    categories.append((name, color))

            legend = {
                "type": "categorical",
                "title": "Grupos hidrológicos del suelo",
                "categories": categories,
            }

        # ----------------------------------------------------
        # VARIABLES CONTINUAS
        # ----------------------------------------------------

        else:

            low, high = np.nanpercentile(
                destination[valid],
                [2, 98],
            )

            if high <= low:
                high = low + 1

            normalized = np.zeros(
                destination.shape,
                dtype=np.float32,
            )

            normalized[valid] = np.clip(
                (destination[valid] - low)
                / (high - low),
                0,
                1,
            )

            for index in range(len(CONTINUOUS_COLORS) - 1):

                lower = index / (
                    len(CONTINUOUS_COLORS) - 1
                )

                upper = (index + 1) / (
                    len(CONTINUOUS_COLORS) - 1
                )

                mask = (
                    valid
                    & (normalized >= lower)
                    & (
                        normalized <= upper
                        if index == len(CONTINUOUS_COLORS) - 2
                        else normalized < upper
                    )
                )

                if mask.any():

                    positions = (
                        (normalized[mask] - lower)
                        / (upper - lower)
                    )

                    c1 = np.array(
                        hex_to_rgb(CONTINUOUS_COLORS[index])
                    )

                    c2 = np.array(
                        hex_to_rgb(CONTINUOUS_COLORS[index + 1])
                    )

                    colors = (
                        c1[None, :]
                        * (1 - positions[:, None])
                        + c2[None, :]
                        * positions[:, None]
                    ).astype(np.uint8)

                    rgba[mask, :3] = colors

            rgba[..., 3][valid] = int(255 * opacity)

            legend = {
                "type": "continuous",
                "minimum": float(low),
                "maximum": float(high),
            }

        image = Image.fromarray(rgba, mode="RGBA")

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )

        return (
            f"data:image/png;base64,{encoded}",
            [[south, west], [north, east]],
            legend,
        )


# ============================================================
# VECTORES Y CENTRO DEL MAPA
# ============================================================

def add_vector(
    m,
    gdf,
    name,
    color="#333333",
    weight=2,
    fill=False,
):

    if gdf is None:
        return

    fields = [
        c for c in gdf.columns
        if c != "geometry"
    ][:3]

    tooltip = None

    if fields:

        tooltip = folium.GeoJsonTooltip(
            fields=fields,
            aliases=fields,
            localize=True,
        )

    folium.GeoJson(
        gdf.__geo_interface__,
        name=name,
        style_function=lambda feature: {
            "color": color,
            "weight": weight,
            "fillColor": color,
            "fillOpacity": 0.10 if fill else 0,
        },
        tooltip=tooltip,
    ).add_to(m)


def get_center(canton):

    if canton is not None:

        try:

            try:
                union_geometry = canton.geometry.union_all()
            except AttributeError:
                union_geometry = canton.geometry.unary_union

            centroid = union_geometry.centroid

            return [
                centroid.y,
                centroid.x,
            ]

        except Exception:
            pass

    return [9.90, -83.87]


# ============================================================
# VISUALIZACIÓN DE RÁSTERES
# ============================================================

def show_raster_map(
    raster_path,
    title,
    categorical=False,
    category_type=None,
    units="",
    canton=None,
    districts=None,
    drainage=None,
):

    if raster_path is None:

        st.warning(
            f"No se encontró el archivo correspondiente a {title}."
        )

        return

    try:

        image, bounds, legend = raster_to_png(
            raster_path,
            categorical=categorical,
            category_type=category_type,
        )

        m = crear_mapa_base(
            location=get_center(canton),
            zoom_start=11,
        )

        ImageOverlay(
            image,
            bounds=bounds,
            name=title,
            opacity=1,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m)

        add_vector(
            m,
            drainage,
            "Red de drenaje",
            "#2878a5",
            2,
        )

        add_vector(
            m,
            districts,
            "Distritos",
            "#555555",
            1,
        )

        add_vector(
            m,
            canton,
            "Límite cantonal",
            "#111111",
            3,
        )

        folium.LayerControl(
            collapsed=False
        ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=570,
            returned_objects=[],
        )

        st.markdown("#### Simbología cartográfica")

        if legend["type"] == "categorical":

            crear_leyenda_categorica(
                legend["title"],
                legend["categories"],
            )

        else:

            crear_leyenda_continua(
                title=title,
                minimum=legend["minimum"],
                maximum=legend["maximum"],
                units=units,
            )

            st.caption(
                "La escala utiliza los percentiles 2 y 98 "
                "de los píxeles válidos para mejorar el contraste. "
                "Los valores extremos pueden quedar fuera del rango "
                "representado en la leyenda."
            )

    except Exception as exc:

        st.error(
            f"No fue posible representar {title}: {exc}"
        )


def show_png(filename, caption=None):

    path = locate_file(filename)

    if path is not None:

        st.image(
            str(path),
            use_container_width=True,
        )

        if caption:
            st.caption(caption)

    else:

        st.info(
            f"Producto cartográfico no encontrado: {filename}"
        )


# ============================================================
# ARCHIVOS DEL PROYECTO
# ============================================================

canton_path = locate_file("Oreamuno_Canton.gpkg")
district_path = locate_file("Oreamuno_Distritos.gpkg")

drainage_path = locate_file(
    "Oreamuno_Red_Drenaje_1000_Completa.gpkg"
)

sus_path = locate_file(
    "Oreamuno_Susceptibilidad_Inundacion_5_Clases.tif"
)

continuous_path = locate_file(
    "Oreamuno_Indice_Susceptibilidad_Continuo.tif"
)

dem_path = locate_file(
    "Oreamuno_Copernicus_GLO30_CRTM05.tif"
)

slope_path = locate_file(
    "Oreamuno_Pendiente_grados.tif"
)

flow_path = locate_file(
    "Oreamuno_Acumulacion_Flujo.tif"
)

distance_path = locate_file(
    "Oreamuno_Distancia_Drenaje.tif"
)

groups_path = locate_file(
    "Oreamuno_Grupos_Hidrologicos_CRTM05.tif"
)

cover_path = locate_file(
    "Oreamuno_WorldCover_2021_CRTM05.tif"
)

cn_path = locate_file("Oreamuno_CN_II.tif")

rainfall_files = {
    "Precipitación media anual": locate_file(
        "Oreamuno_CHIRPS_Precipitacion_media_Alineado.tif"
    ),
    "Precipitación máxima anual": locate_file(
        "Oreamuno_CHIRPS_Precipitacion_maxima_Alineado.tif"
    ),
    "Desviación estándar": locate_file(
        "Oreamuno_CHIRPS_Desviacion_estandar_Alineado.tif"
    ),
    "Coeficiente de variación": locate_file(
        "Oreamuno_CHIRPS_Coeficiente_variacion_Alineado.tif"
    ),
}

canton = (
    read_vector_file(canton_path)
    if canton_path else None
)

districts = (
    read_vector_file(district_path)
    if district_path else None
)

drainage = (
    read_vector_file(drainage_path)
    if drainage_path else None
)


# ============================================================
# ESCORRENTÍA SUPERFICIAL | MÉTODO SCS-CN
# ============================================================

def calcular_escorrentia_scs_cn(cn_raster_path, precipitacion_mm):
    """Calcula escorrentía directa por píxel a partir del ráster CN II.

    La precipitación es un escenario ingresado por el usuario (mm).
    No se interpreta como pronóstico ni como dato CHIRPS de lluvia de evento.
    """
    if not cn_raster_path:
        raise FileNotFoundError("No se encontró Oreamuno_CN_II.tif.")

    p = float(precipitacion_mm)
    if not np.isfinite(p) or p <= 0:
        raise ValueError("La precipitación debe ser un valor mayor que cero.")

    with rasterio.open(cn_raster_path) as src:
        cn_ma = src.read(1, masked=True).astype("float32")
        cn = cn_ma.filled(np.nan)
        valid = (
            np.isfinite(cn)
            & (cn > 0)
            & (cn <= 100)
            & (~np.ma.getmaskarray(cn_ma))
        )

        if not np.any(valid):
            raise ValueError("El ráster CN II no contiene valores válidos entre 1 y 100.")

        s_retencion = np.full(cn.shape, np.nan, dtype="float32")
        ia = np.full(cn.shape, np.nan, dtype="float32")
        q = np.full(cn.shape, np.nan, dtype="float32")

        s_retencion[valid] = (25400.0 / cn[valid]) - 254.0
        ia[valid] = 0.2 * s_retencion[valid]

        genera_escorrentia = valid & (p > ia)
        q[valid] = 0.0
        q[genera_escorrentia] = (
            (p - ia[genera_escorrentia]) ** 2
            / (p + 0.8 * s_retencion[genera_escorrentia])
        )

        q[~valid] = np.nan
        finite_q = q[valid & np.isfinite(q)]

        if finite_q.size == 0:
            raise ValueError("No fue posible obtener valores de escorrentía válidos.")

        # Área de píxel mediante el determinante de la transformación afín.
        pixel_area_native = abs(
            src.transform.a * src.transform.e
            - src.transform.b * src.transform.d
        )
        projected_metric = bool(
            src.crs
            and src.crs.is_projected
            and (src.crs.linear_units or "").lower()
            in {"metre", "meter", "metres", "meters"}
        )

        area_km2 = (
            float(valid.sum() * pixel_area_native / 1_000_000)
            if projected_metric else None
        )
        volume_m3 = (
            float(np.nansum(q[valid]) / 1000.0 * pixel_area_native)
            if projected_metric else None
        )

        stats = {
            "precipitacion_mm": p,
            "cn_mean": float(np.nanmean(cn[valid])),
            "q_min": float(np.nanmin(finite_q)),
            "q_max": float(np.nanmax(finite_q)),
            "q_mean": float(np.nanmean(finite_q)),
            "q_std": float(np.nanstd(finite_q)),
            "valid_cells": int(valid.sum()),
            "area_km2": area_km2,
            "volume_m3": volume_m3,
            "runoff_percent": float(100 * np.nanmean(finite_q) / p),
            "crs": str(src.crs),
            "resolution": tuple(src.res),
        }

        # Se conserva el mismo CRS, transformación, resolución y extensión del CN II.
        safe_p = str(p).replace(".", "_")
        output_path = Path(tempfile.gettempdir()) / f"GeoAlerta_Escorrentia_SCSCN_{safe_p}mm.tif"
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
        )
        output_array = np.where(np.isfinite(q), q, -9999.0).astype("float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output_array, 1)
            dst.set_band_description(1, f"Escorrentía directa SCS-CN para P={p:g} mm")
            dst.update_tags(
                metodo="SCS-CN",
                precipitacion_mm=str(p),
                ecuacion="Q=(P-0.2S)^2/(P+0.8S), si P>0.2S; de lo contrario Q=0",
                cn_source=Path(cn_raster_path).name,
                units="mm",
            )

    return str(output_path), stats


def formato_volumen(stats):
    if stats["volume_m3"] is None:
        return "No disponible (CRS no métrico)"
    return f"{stats['volume_m3'] / 1_000_000:,.3f} Mm³"


# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: rgba(120, 120, 120, 0.07);
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3 {
    letter-spacing: -0.3px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# NAVEGACIÓN
# ============================================================

with st.sidebar:

    st.title("🌧️ GeoAlerta Oreamuno")

    st.caption(
        "Visor geoespacial de susceptibilidad "
        "a inundaciones"
    )

    st.divider()

    page = st.radio(
        "Navegación",
        [
            "Inicio",
            "Susceptibilidad",
            "Precipitación",
            "Escorrentía superficial",
            "Contexto hidrológico",
            "Cobertura y grupos hidrológicos",
            "Reporte de incidentes",
            "Metodología e inventario",
        ],
    )

    st.divider()

    st.caption(
        "Cantón de Oreamuno · Cartago · Costa Rica"
    )


# ============================================================
# INICIO
# ============================================================

if page == "Inicio":

    st.title("GeoAlerta Oreamuno")

    st.subheader(
        "Visor territorial de susceptibilidad a inundaciones"
    )

    st.write(
        """
        GeoAlerta Oreamuno integra información geoespacial,
        hidrológica y territorial para facilitar la consulta
        de los resultados del análisis de susceptibilidad
        a inundaciones en el cantón de Oreamuno, Cartago.
        """
    )

    st.info(
        "Esta herramienta presenta resultados previamente "
        "procesados. No genera pronósticos meteorológicos, "
        "no monitorea eventos en tiempo real y no sustituye "
        "los sistemas oficiales de alerta."
    )

    a, b, c, d = st.columns(4)

    a.metric("Clases de susceptibilidad", "5")
    b.metric("Periodo CHIRPS", "1981–2025")
    c.metric("Modelo de elevación", "Copernicus GLO-30")
    d.metric("CRS de trabajo", "EPSG:5367")

    st.divider()

    st.subheader("Mapa general de susceptibilidad")

    show_raster_map(
        sus_path,
        "Susceptibilidad a inundaciones",
        categorical=True,
        category_type="susceptibility",
        canton=canton,
        districts=districts,
    )

    st.divider()

    st.subheader("Componentes del visor")

    x1, x2, x3 = st.columns(3)

    x1.markdown("""
    **Susceptibilidad**

    Consulta del resultado clasificado,
    índice continuo y distribución territorial.
    """)

    x2.markdown("""
    **Contexto hidrológico**

    Precipitación, drenaje, relieve,
    acumulación de flujo y áreas de aporte.
    """)

    x3.markdown("""
    **Contexto territorial**

    Cobertura del suelo y grupos
    hidrológicos del territorio.
    """)


# ============================================================
# SUSCEPTIBILIDAD
# ============================================================

elif page == "Susceptibilidad":

    st.title("Susceptibilidad a inundaciones")

    st.write(
        "Consulta de los productos cartográficos "
        "del modelo de susceptibilidad."
    )

    tab1, tab2, tab3 = st.tabs([
        "Mapa clasificado",
        "Índice continuo",
        "Distribución territorial",
    ])

    with tab1:

        show_raster_map(
            sus_path,
            "Susceptibilidad en cinco clases",
            categorical=True,
            category_type="susceptibility",
            canton=canton,
            districts=districts,
            drainage=drainage,
        )

    with tab2:

        st.write(
            "Representación espacial del índice continuo "
            "utilizado como producto complementario."
        )

        show_png(
            "Mapa_Indice_Continuo_Oreamuno.png",
            "Producto cartográfico del índice continuo.",
        )

        show_raster_map(
            continuous_path,
            "Índice continuo de susceptibilidad",
            canton=canton,
            districts=districts,
        )

    with tab3:

        csv_path = locate_file(
            "Distribucion_Susceptibilidad_Oreamuno.csv"
        )

        if csv_path:

            df = read_csv_file(csv_path)

            fig = px.bar(
                df,
                x="Nivel",
                y="Porcentaje",
                color="Nivel",
                text_auto=".2f",
                title="Distribución porcentual de la susceptibilidad",
                labels={
                    "Nivel": "Clase de susceptibilidad",
                    "Porcentaje": "Área del cantón (%)",
                },
            )

            fig.update_layout(
                showlegend=False,
                yaxis_title="Porcentaje (%)",
                xaxis_title="Categoría",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                df.rename(columns={
                    "Categoria": "Código",
                    "Nivel": "Nivel",
                    "Pixeles": "Píxeles",
                    "Area_km2": "Área (km²)",
                    "Porcentaje": "Porcentaje (%)",
                }),
                use_container_width=True,
                hide_index=True,
            )

            if "Area_km2" in df.columns:

                total_area = df["Area_km2"].sum()

                st.metric(
                    "Área representada por la clasificación",
                    f"{total_area:,.2f} km²",
                )

        else:

            st.warning(
                "No se encontró el CSV de distribución."
            )


# ============================================================
# PRECIPITACIÓN
# ============================================================

elif page == "Precipitación":

    st.title("Precipitación")

    st.write(
        """
        Consulta de los productos de precipitación
        derivados de CHIRPS para el periodo 1981–2025.
        """
    )

    csv_path = locate_file(
        "Validacion_CHIRPS_Alineados.csv"
    )

    if csv_path:

        df = read_csv_file(csv_path)

        st.subheader("Estadísticas de los productos")

        st.dataframe(
            df.rename(columns={
                "Variable": "Variable",
                "Pixeles_validos": "Píxeles válidos",
                "Minimo": "Mínimo",
                "Media": "Media",
                "Maximo": "Máximo",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Las estadísticas corresponden a los píxeles "
            "válidos reportados en el archivo de validación."
        )

    st.divider()

    st.subheader("Productos espaciales CHIRPS")

    selected = st.selectbox(
        "Seleccione el producto de precipitación",
        list(rainfall_files.keys()),
    )

    raster = rainfall_files[selected]

    units = (
        "%"
        if "variación" in selected.lower()
        else "mm"
    )

    if raster:

        show_raster_map(
            raster,
            selected,
            units=units,
            canton=canton,
            districts=districts,
        )

        info = get_raster_info(raster)

        c1, c2, c3 = st.columns(3)

        c1.metric("Ancho", f"{info['width']:,} px")
        c2.metric("Alto", f"{info['height']:,} px")
        c3.metric("CRS", info["crs"])

    else:

        st.warning(
            "No se encontró el ráster alineado seleccionado."
        )

    st.divider()

    st.subheader("Interpretación")

    st.write(
        """
        Los productos permiten consultar la distribución
        espacial de la precipitación media, los máximos
        anuales, la variabilidad interanual y la desviación
        estándar durante el periodo analizado.

        Las capas de precipitación representan características
        históricas del periodo 1981–2025 y no condiciones
        meteorológicas actuales.
        """
    )


# ============================================================
# ESCORRENTÍA SUPERFICIAL
# ============================================================

elif page == "Escorrentía superficial":

    st.title("Escorrentía superficial")
    st.write(
        "Estimación espacial de la escorrentía directa mediante el método "
        "del Número de Curva (SCS-CN), utilizando el ráster CN II elaborado "
        "para Oreamuno."
    )

    st.info(
        "Los valores de precipitación que se introducen aquí representan "
        "escenarios hipotéticos de lluvia acumulada (mm). No son pronósticos, "
        "lluvia observada en tiempo real ni periodos de retorno."
    )

    if not cn_path:
        st.error(
            "No se encontró Oreamuno_CN_II.tif. Verifique que el ráster esté "
            "en alguna de las carpetas de datos del proyecto."
        )
    else:
        with st.expander("Insumo y referencia espacial", expanded=False):
            try:
                cn_info = get_raster_info(cn_path)
                c1, c2, c3 = st.columns(3)
                c1.metric("Ráster CN II", "Disponible")
                c2.metric("CRS", cn_info["crs"])
                c3.metric(
                    "Resolución",
                    f"{cn_info['resolution'][0]:.2f} × {cn_info['resolution'][1]:.2f}"
                )
                st.caption(
                    "La malla de salida conserva exactamente la extensión, "
                    "resolución, transformación y sistema de referencia del ráster CN II."
                )
            except Exception as exc:
                st.warning(f"No fue posible leer los metadatos del CN II: {exc}")

        st.subheader("Definición de escenarios")
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.number_input(
                "Escenario 1 · Precipitación acumulada (mm)",
                min_value=0.1,
                max_value=1000.0,
                value=50.0,
                step=5.0,
                help="Lámina de lluvia hipotética para evaluar la respuesta SCS-CN."
            )
        with c2:
            p2 = st.number_input(
                "Escenario 2 · Precipitación acumulada (mm)",
                min_value=0.1,
                max_value=1000.0,
                value=100.0,
                step=5.0,
                help="Segundo escenario de lluvia para comparar la escorrentía estimada."
            )

        if abs(float(p1) - float(p2)) < 1e-9:
            st.warning("Los dos escenarios tienen la misma precipitación. Modifique uno para realizar una comparación útil.")

        try:
            with st.spinner("Calculando escorrentía espacial a partir del CN II..."):
                raster_1, stats_1 = calcular_escorrentia_scs_cn(cn_path, p1)
                if abs(float(p1) - float(p2)) < 1e-9:
                    raster_2, stats_2 = raster_1, stats_1
                else:
                    raster_2, stats_2 = calcular_escorrentia_scs_cn(cn_path, p2)

            st.success("Cálculo SCS-CN completado para los escenarios definidos.")

            st.subheader("Comparación de resultados")
            comparison = pd.DataFrame(
                [
                    {
                        "Escenario": f"{p1:g} mm",
                        "CN medio": stats_1["cn_mean"],
                        "Escorrentía media (mm)": stats_1["q_mean"],
                        "Escorrentía mínima (mm)": stats_1["q_min"],
                        "Escorrentía máxima (mm)": stats_1["q_max"],
                        "Desviación estándar (mm)": stats_1["q_std"],
                        "% medio de lluvia": stats_1["runoff_percent"],
                        "Volumen (Mm³)": (
                            stats_1["volume_m3"] / 1_000_000
                            if stats_1["volume_m3"] is not None else np.nan
                        ),
                    },
                    {
                        "Escenario": f"{p2:g} mm",
                        "CN medio": stats_2["cn_mean"],
                        "Escorrentía media (mm)": stats_2["q_mean"],
                        "Escorrentía mínima (mm)": stats_2["q_min"],
                        "Escorrentía máxima (mm)": stats_2["q_max"],
                        "Desviación estándar (mm)": stats_2["q_std"],
                        "% medio de lluvia": stats_2["runoff_percent"],
                        "Volumen (Mm³)": (
                            stats_2["volume_m3"] / 1_000_000
                            if stats_2["volume_m3"] is not None else np.nan
                        ),
                    },
                ]
            )
            st.dataframe(
                comparison.style.format(
                    {
                        "CN medio": "{:.2f}",
                        "Escorrentía media (mm)": "{:.2f}",
                        "Escorrentía mínima (mm)": "{:.2f}",
                        "Escorrentía máxima (mm)": "{:.2f}",
                        "Desviación estándar (mm)": "{:.2f}",
                        "% medio de lluvia": "{:.1f}%",
                        "Volumen (Mm³)": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.plotly_chart(
                px.bar(
                    comparison,
                    x="Escenario",
                    y="Escorrentía media (mm)",
                    color="Escenario",
                    text_auto=".2f",
                    title="Escorrentía directa media por escenario",
                    labels={"Escenario": "Precipitación acumulada", "Escorrentía media (mm)": "Escorrentía media (mm)"},
                    color_discrete_sequence=["#2c7bb6", "#d7191c"],
                ),
                use_container_width=True,
            )

            st.divider()
            st.subheader("Distribución espacial de la escorrentía")
            selected_label = st.selectbox(
                "Seleccione el escenario que desea visualizar",
                [f"Escenario 1 · {p1:g} mm", f"Escenario 2 · {p2:g} mm"],
            )
            if selected_label.startswith("Escenario 1"):
                selected_path, selected_stats, selected_p = raster_1, stats_1, p1
            else:
                selected_path, selected_stats, selected_p = raster_2, stats_2, p2

            show_raster_map(
                selected_path,
                f"Escorrentía directa SCS-CN · {selected_p:g} mm",
                units="mm",
                canton=canton,
                districts=districts,
                drainage=drainage,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Escorrentía media", f"{selected_stats['q_mean']:.2f} mm")
            m2.metric("Escorrentía máxima", f"{selected_stats['q_max']:.2f} mm")
            m3.metric("Fracción media de lluvia", f"{selected_stats['runoff_percent']:.1f}%")
            m4.metric("Volumen estimado", formato_volumen(selected_stats))

            if selected_stats["volume_m3"] is None:
                st.warning(
                    "No se calculó el volumen total porque el CRS del ráster no "
                    "está proyectado en metros. La lámina de escorrentía sí se calculó."
                )
            else:
                st.caption(
                    f"Área válida procesada: {selected_stats['area_km2']:,.2f} km² · "
                    f"Celdas válidas: {selected_stats['valid_cells']:,} · "
                    f"CRS: {selected_stats['crs']}"
                )

            with open(selected_path, "rb") as raster_file:
                st.download_button(
                    "Descargar GeoTIFF de escorrentía",
                    data=raster_file.read(),
                    file_name=f"GeoAlerta_Escorrentia_SCSCN_{selected_p:g}mm.tif",
                    mime="image/tiff",
                    use_container_width=False,
                )

            st.divider()
            st.subheader("Fundamento metodológico")
            st.markdown(
                r"""
                Se aplica el método del **Soil Conservation Service - Curve Number (SCS-CN)**.
                Para cada celda válida del ráster CN II se calcula la retención potencial máxima:

                \[S = \frac{25400}{CN} - 254\]

                La abstracción inicial se estima como:

                \[I_a = 0.2S\]

                La escorrentía directa se obtiene mediante:

                \[Q = \frac{(P-I_a)^2}{P+0.8S},\quad P>I_a\]

                Cuando \(P\leq I_a\), se asigna \(Q=0\). Los valores de \(P\), \(S\), \(I_a\) y \(Q\) se expresan en milímetros.
                """
            )
            st.warning(
                "La salida representa una lámina estimada de escorrentía directa (mm) "
                "y un volumen agregado sobre el área válida. No representa caudal "
                "instantáneo (m³/s), profundidad de inundación, velocidad del flujo "
                "ni una simulación hidráulica. La validez depende de la calidad del "
                "CN II, de la precipitación seleccionada y de los supuestos del método."
            )

        except Exception as exc:
            st.error(f"No fue posible calcular la escorrentía: {exc}")


# ============================================================
# CONTEXTO HIDROLÓGICO
# ============================================================

elif page == "Contexto hidrológico":

    st.title("Contexto hidrológico")

    st.write(
        "Exploración de las características del relieve, "
        "el drenaje y la organización del flujo superficial."
    )

    layers = {}

    if dem_path:
        layers["Modelo digital de elevación"] = dem_path

    if slope_path:
        layers["Pendiente del terreno"] = slope_path

    if flow_path:
        layers["Acumulación de flujo"] = flow_path

    if distance_path:
        layers["Distancia al drenaje"] = distance_path

    if drainage is not None:
        layers["Red de drenaje"] = drainage

    if not layers:

        st.warning(
            "No se encontraron capas hidrológicas disponibles."
        )

    else:

        selected = st.multiselect(
            "Capas de referencia",
            list(layers.keys()),
            default=[
                name for name in layers
                if name in [
                    "Red de drenaje",
                    "Modelo digital de elevación",
                ]
            ],
        )

        m = crear_mapa_base(
            location=get_center(canton),
            zoom_start=11,
        )

        add_vector(
            m,
            canton,
            "Límite cantonal",
            "#111111",
            3,
        )

        add_vector(
            m,
            districts,
            "Distritos",
            "#555555",
            1,
        )

        raster_legends = []

        for name in selected:

            obj = layers[name]

            if isinstance(obj, gpd.GeoDataFrame):

                add_vector(
                    m,
                    obj,
                    name,
                    "#2878a5",
                    2,
                )

            else:

                try:

                    image, bounds, legend = raster_to_png(
                        obj,
                        categorical=False,
                        opacity=0.75,
                    )

                    ImageOverlay(
                        image,
                        bounds=bounds,
                        name=name,
                        opacity=1,
                        interactive=True,
                        cross_origin=False,
                    ).add_to(m)

                    raster_legends.append((name, legend))

                except Exception as exc:

                    st.warning(
                        f"No se pudo representar {name}: {exc}"
                    )

        folium.LayerControl(
            collapsed=False
        ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=600,
            returned_objects=[],
        )

        if raster_legends:

            st.markdown("#### Leyendas de las capas")

            for name, legend in raster_legends:

                crear_leyenda_continua(
                    name,
                    legend["minimum"],
                    legend["maximum"],
                    units="",
                )

    if drainage is not None:

        st.metric(
            "Entidades de drenaje",
            f"{len(drainage):,}",
        )

    st.divider()

    st.subheader("Áreas de aporte")

    show_png(
        "Mapa_Grupos_Hidrologicos_Oreamuno.png",
        "Mapa de grupos hidrológicos del suelo.",
    )


# ============================================================
# COBERTURA Y GRUPOS HIDROLÓGICOS
# ============================================================

elif page == "Cobertura y grupos hidrológicos":

    st.title("Cobertura y grupos hidrológicos")

    st.write(
        """
        Consulta de la distribución de las coberturas
        terrestres y los grupos hidrológicos utilizados
        en la caracterización del territorio.
        """
    )

    tab1, tab2, tab3 = st.tabs([
        "Cobertura terrestre",
        "Grupos hidrológicos",
        "Parámetros CN II",
    ])

    with tab1:

        show_raster_map(
            cover_path,
            "Cobertura terrestre WorldCover 2021",
            categorical=True,
            category_type="worldcover",
            canton=canton,
            districts=districts,
        )

        st.caption(
            "Producto derivado de ESA WorldCover 2021."
        )

        show_png(
            "Mapa_Exposicion_Edificaciones_Oreamuno.png",
            "Producto cartográfico de exposición de edificaciones. "
            "Se presenta como referencia y no como estimación "
            "de daños o pérdidas.",
        )

    with tab2:

        show_raster_map(
            groups_path,
            "Grupos hidrológicos",
            categorical=True,
            category_type="groups",
            canton=canton,
            districts=districts,
        )

        stats_path = locate_file(
            "Estadisticas_Grupos_Hidrologicos.csv"
        )

        if stats_path:

            df = read_csv_file(stats_path)

            st.subheader("Distribución de grupos hidrológicos")

            fig = px.bar(
                df,
                x="Codigo",
                y="Porcentaje",
                color="Codigo",
                text_auto=".2f",
                title="Distribución territorial por grupo",
                labels={
                    "Codigo": "Código del grupo",
                    "Porcentaje": "Área (%)",
                },
                color_discrete_sequence=[
                    "#2ca25f",
                    "#fec44f",
                    "#de2d26",
                ],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

        cross_path = locate_file(
            "Cruce_WorldCover_Grupos_Hidrologicos.csv"
        )

        if cross_path:

            cross = read_csv_file(cross_path)

            st.subheader(
                "Cruce entre cobertura y grupos hidrológicos"
            )

            fig = px.bar(
                cross,
                x="Tipo_Cobertura",
                y="Area_km2",
                color="Grupo",
                barmode="stack",
                title="Área por cobertura y grupo hidrológico",
                labels={
                    "Tipo_Cobertura": "Cobertura",
                    "Area_km2": "Área (km²)",
                    "Grupo": "Grupo hidrológico",
                },
                color_discrete_sequence=[
                    "#2ca25f",
                    "#fec44f",
                    "#de2d26",
                ],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with tab3:

        cn_file = locate_file(
            "Parametros_CN_II.csv"
        )

        if cn_file:

            df = read_csv_file(cn_file)

            st.subheader(
                "Parámetros de número de curva"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Los valores corresponden a los parámetros "
                "documentados en el archivo del proyecto."
            )

        if cn_path:

            show_raster_map(
                cn_path,
                "Número de curva CN II",
                units="CN",
                canton=canton,
                districts=districts,
            )

        else:

            st.info(
                "No se encontró un ráster CN II en el repositorio."
            )


# ============================================================
# REPORTE DE INCIDENTES
# ============================================================

elif page == "Reporte de incidentes":

    from incidentes import render_incidentes

    render_incidentes(districts)


# ============================================================
# METODOLOGÍA E INVENTARIO
# ============================================================

elif page == "Metodología e inventario":

    st.title("Metodología e inventario")

    tab1, tab2 = st.tabs([
        "Metodología y alcance",
        "Inventario de datos",
    ])

    with tab1:

        st.subheader("Ruta metodológica")

        st.markdown("""
        **1. Preparación de información geoespacial**

        Organización de los límites cantonales y distritales,
        el modelo digital de elevación y las capas temáticas.

        **2. Caracterización física e hidrológica**

        Procesamiento de variables de relieve, pendiente,
        red de drenaje, acumulación de flujo y distancia
        a los cauces.

        **3. Caracterización de la precipitación**

        Consulta de productos CHIRPS correspondientes
        al periodo 1981–2025.

        **4. Caracterización territorial e hidrológica**

        Integración de información de cobertura terrestre,
        grupos hidrológicos del suelo y Número de Curva CN II.

        **5. Estimación de escorrentía directa**

        Aplicación espacial del método SCS-CN sobre el ráster
        CN II de Oreamuno para escenarios definidos de precipitación.
        Se obtienen láminas de escorrentía (mm) y, cuando el
        sistema de referencia está proyectado en metros, volúmenes
        agregados estimados (m³).

        **6. Modelo de susceptibilidad**

        Integración de los factores considerados en el
        modelo y generación de un índice espacial continuo.

        **7. Clasificación y comunicación**

        Representación del resultado mediante cinco clases
        y publicación de los productos en un visor web.
        """)

        st.divider()

        st.subheader("Alcances y limitaciones")

        st.warning(
            """
            La susceptibilidad representa una condición
            espacial relativa según los factores y criterios
            del modelo. No equivale a probabilidad temporal
            de ocurrencia, profundidad de inundación,
            velocidad del flujo ni pronóstico de eventos.

            La estimación SCS-CN es un componente hidrológico
            independiente: expresa escorrentía directa en mm
            para escenarios de precipitación definidos por el
            usuario. No constituye un modelo hidráulico ni
            produce caudales instantáneos.
            """
        )

        st.write(
            """
            La herramienta tiene fines de consulta,
            exploración territorial y comunicación de
            resultados. No sustituye estudios hidráulicos
            de detalle, sistemas oficiales de monitoreo
            ni protocolos institucionales de emergencia.
            """
        )

    with tab2:

        st.subheader("Archivos disponibles")

        files = get_files()

        if not files:

            st.warning(
                "No se encontraron archivos en las carpetas del proyecto."
            )

        else:

            rows = []

            for path in files:

                row = {
                    "Archivo": path.name,
                    "Carpeta": str(
                        path.parent.relative_to(ROOT)
                    ),
                    "Formato": path.suffix.upper(),
                    "Tamaño (MB)": round(
                        path.stat().st_size / (1024 * 1024),
                        3,
                    ),
                }

                if path.suffix.lower() in [
                    ".tif", ".tiff"
                ]:

                    try:

                        info = get_raster_info(str(path))

                        row["CRS"] = info["crs"]

                        row["Dimensiones"] = (
                            f"{info['width']} × {info['height']}"
                        )

                        row["Resolución"] = (
                            f"{info['resolution'][0]:.3f} × "
                            f"{info['resolution'][1]:.3f}"
                        )

                    except Exception:

                        row["CRS"] = "No leído"
                        row["Dimensiones"] = "—"
                        row["Resolución"] = "—"

                elif path.suffix.lower() in [
                    ".gpkg", ".shp", ".geojson"
                ]:

                    try:

                        gdf = gpd.read_file(path)

                        row["CRS"] = str(gdf.crs)
                        row["Entidades"] = len(gdf)

                    except Exception:

                        row["CRS"] = "No leído"
                        row["Entidades"] = "—"

                rows.append(row)

            inventory = pd.DataFrame(rows)

            c1, c2 = st.columns(2)

            c1.metric(
                "Archivos detectados",
                len(inventory),
            )

            c2.metric(
                "Tamaño total",
                f"{inventory['Tamaño (MB)'].sum():,.2f} MB",
            )

            st.dataframe(
                inventory,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Descargar inventario CSV",
                inventory.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "inventario_geoalerta_oreamuno.csv",
                "text/csv",
            )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "GeoAlerta Oreamuno · Visor de resultados geoespaciales "
    "e hidrológicos · Cartago, Costa Rica"
)
