
from pathlib import Path
import warnings

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import folium
import plotly.express as px
from rasterio.enums import Resampling
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="GeoAlerta Oreamuno",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ GeoAlerta Oreamuno")
st.subheader("Sistema de consulta territorial y análisis de inundaciones")
st.caption("Oreamuno, Cartago, Costa Rica")

# --------------------------------------------------
# IDENTIFICACIÓN DE DATOS
# --------------------------------------------------

def discover_files():
    vectors = []
    rasters = []

    if DATA_DIR.exists():
        for file in DATA_DIR.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() in [".gpkg", ".geojson", ".shp"]:
                vectors.append(file)

            elif file.suffix.lower() in [".tif", ".tiff"]:
                rasters.append(file)

    return sorted(vectors), sorted(rasters)


vectors, rasters = discover_files()


def find_layer(keywords):
    for file in vectors + rasters:
        name = file.stem.lower()
        if any(word in name for word in keywords):
            return file
    return None


@st.cache_data
def load_vector(path, modified):
    layer = gpd.read_file(path)

    if layer.crs is not None:
        layer = layer.to_crs(epsg=4326)

    return layer


@st.cache_data
def read_raster(path, modified):
    with rasterio.open(path) as src:
        image = src.read(
            1,
            out_shape=(
                min(src.height, 800),
                min(src.width, 800)
            ),
            resampling=Resampling.nearest,
            masked=True
        )

        values = image.compressed()
        values = values[np.isfinite(values)]

        return {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "values": values[::max(1, len(values)//30000)].tolist()
        }


# --------------------------------------------------
# MENÚ PRINCIPAL
# --------------------------------------------------

with st.sidebar:
    st.header("GeoAlerta Oreamuno")

    section = st.radio(
        "Navegación",
        [
            "Inicio",
            "Visor geográfico",
            "Precipitación",
            "Susceptibilidad",
            "Exposición territorial",
            "Inventario de datos"
        ]
    )

    st.divider()
    st.caption("Prototipo académico de análisis territorial.")


# --------------------------------------------------
# INICIO
# --------------------------------------------------

if section == "Inicio":

    st.header("Panel general")

    col1, col2, col3 = st.columns(3)

    col1.metric("Capas vectoriales", len(vectors))
    col2.metric("Capas ráster", len(rasters))

    buildings = find_layer(["edific", "constru"])

    col3.metric(
        "Edificaciones",
        "Disponible" if buildings else "Pendiente"
    )

    st.divider()

    st.markdown("""
    ### Objetivo de la herramienta

    Facilitar la consulta y visualización de información
    territorial relacionada con las condiciones de
    susceptibilidad a inundaciones en el cantón de Oreamuno.

    ### Componentes

    - Información de precipitación.
    - Susceptibilidad a inundaciones.
    - Red de drenaje.
    - Exposición territorial mediante edificaciones.
    - Visualización de información geográfica.
    """)

    if not vectors and not rasters:
        st.info(
            "La aplicación está funcionando. "
            "Pendiente incorporar las capas geográficas del proyecto."
        )


# --------------------------------------------------
# VISOR GEOGRÁFICO
# --------------------------------------------------

elif section == "Visor geográfico":

    st.header("Visor geográfico")

    if vectors:

        selected = st.selectbox(
            "Seleccionar capa",
            [file.name for file in vectors]
        )

        path = next(file for file in vectors if file.name == selected)

        try:
            layer = load_vector(
                str(path),
                path.stat().st_mtime
            )

            if not layer.empty:

                center = [
                    layer.geometry.centroid.y.mean(),
                    layer.geometry.centroid.x.mean()
                ]

                map_object = folium.Map(
                    location=center,
                    zoom_start=12,
                    tiles="CartoDB positron"
                )

                folium.GeoJson(
                    layer,
                    name=path.stem,
                    tooltip=folium.GeoJsonTooltip(
                        fields=[
                            col for col in layer.columns
                            if col != "geometry"
                        ][:5]
                    )
                ).add_to(map_object)

                folium.LayerControl().add_to(map_object)

                st_folium(
                    map_object,
                    use_container_width=True,
                    height=600
                )

                st.caption(
                    f"Entidades: {len(layer):,}"
                )

        except Exception as error:
            st.error(f"Error al cargar la capa: {error}")

    else:
        st.warning("No hay capas vectoriales disponibles.")


# --------------------------------------------------
# PRECIPITACIÓN
# --------------------------------------------------

elif section == "Precipitación":

    st.header("Análisis de precipitación")

    path = find_layer([
        "chirps",
        "precipit",
        "lluvia"
    ])

    if path:

        try:
            result = read_raster(
                str(path),
                path.stat().st_mtime
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Mínimo", round(result["minimum"], 2))
            c2.metric("Promedio", round(result["mean"], 2))
            c3.metric("Máximo", round(result["maximum"], 2))

            figure = px.histogram(
                pd.DataFrame({"Valor": result["values"]}),
                x="Valor",
                nbins=35,
                title="Distribución de precipitación"
            )

            st.plotly_chart(
                figure,
                use_container_width=True
            )

            st.caption(f"Fuente: {path.name}")

        except Exception as error:
            st.error(f"No se pudo leer el ráster: {error}")

    else:
        st.info("Pendiente incorporar el ráster de precipitación.")


# --------------------------------------------------
# SUSCEPTIBILIDAD
# --------------------------------------------------

elif section == "Susceptibilidad":

    st.header("Susceptibilidad a inundaciones")

    path = find_layer([
        "suscept",
        "inund"
    ])

    if path:

        try:
            result = read_raster(
                str(path),
                path.stat().st_mtime
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Valor mínimo", round(result["minimum"], 3))
            c2.metric("Valor medio", round(result["mean"], 3))
            c3.metric("Valor máximo", round(result["maximum"], 3))

            figure = px.histogram(
                pd.DataFrame({"Índice": result["values"]}),
                x="Índice",
                nbins=35,
                title="Distribución del índice de susceptibilidad"
            )

            st.plotly_chart(
                figure,
                use_container_width=True
            )

            st.caption(f"Fuente: {path.name}")

        except Exception as error:
            st.error(f"No se pudo leer el ráster: {error}")

    else:
        st.info("Pendiente incorporar el resultado de susceptibilidad.")


# --------------------------------------------------
# EXPOSICIÓN TERRITORIAL
# --------------------------------------------------

elif section == "Exposición territorial":

    st.header("Exposición territorial")

    buildings = find_layer([
        "edific",
        "constru"
    ])

    drainage = find_layer([
        "dren",
        "rios",
        "quebrad"
    ])

    c1, c2 = st.columns(2)

    if buildings:
        try:
            layer = load_vector(
                str(buildings),
                buildings.stat().st_mtime
            )

            c1.metric(
                "Edificaciones identificadas",
                f"{len(layer):,}"
            )

        except Exception as error:
            c1.error(str(error))

    else:
        c1.info("Capa de edificaciones pendiente.")

    if drainage:
        try:
            layer = load_vector(
                str(drainage),
                drainage.stat().st_mtime
            )

            c2.metric(
                "Elementos de drenaje",
                f"{len(layer):,}"
            )

        except Exception as error:
            c2.error(str(error))

    else:
        c2.info("Capa de drenaje pendiente.")

    st.caption(
        "La exposición territorial considera edificaciones. "
        "No se utiliza población como variable."
    )


# --------------------------------------------------
# INVENTARIO
# --------------------------------------------------

elif section == "Inventario de datos":

    st.header("Inventario de información geoespacial")

    records = []

    for file in vectors + rasters:

        records.append({
            "Archivo": file.name,
            "Tipo": (
                "Vectorial"
                if file in vectors
                else "Ráster"
            ),
            "Tamaño (MB)": round(
                file.stat().st_size / (1024**2),
                2
            )
        })

    if records:

        st.dataframe(
            pd.DataFrame(records),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No hay archivos geográficos incorporados.")


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.divider()

st.caption(
    "GeoAlerta Oreamuno | Prototipo académico | "
    "Oreamuno, Cartago, Costa Rica"
)
