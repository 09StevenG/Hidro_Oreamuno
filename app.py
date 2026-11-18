
from pathlib import Path
import io
import base64
import warnings

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

    values = np.linspace(minimum, maximum, 5)

    gradient = ", ".join(CONTINUOUS_COLORS)

    st.markdown(
        f"""
        <div style="
            padding:14px;
            border:1px solid rgba(128,128,128,.25);
            border-radius:10px;
            margin-top:8px;
            margin-bottom:14px;
        ">
            <div style="
                font-weight:650;
                margin-bottom:10px;
            ">
                {title}
            </div>

            <div style="
                height:17px;
                border-radius:5px;
                background:linear-gradient(
                    to right, {gradient}
                );
            "></div>

            <div style="
                display:flex;
                justify-content:space-between;
                margin-top:7px;
                font-size:12px;
                color:inherit;
            ">
                <span>{minimum:,.2f} {units}</span>
                <span>{values[1]:,.2f}</span>
                <span>{values[2]:,.2f}</span>
                <span>{values[3]:,.2f}</span>
                <span>{maximum:,.2f} {units}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

        **4. Caracterización territorial**

        Integración de información de cobertura terrestre
        y grupos hidrológicos del suelo.

        **5. Modelo de susceptibilidad**

        Integración de los factores considerados en el
        modelo y generación de un índice espacial continuo.

        **6. Clasificación y comunicación**

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
