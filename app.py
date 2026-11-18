from pathlib import Path
import io
import base64
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.enums import Resampling as RioResampling
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# GEOALERTA OREAMUNO | VISOR DE RESULTADOS DE SUSCEPTIBILIDAD
# La aplicación consulta y comunica resultados ya procesados.
# No ejecuta nuevos modelos ni incorpora exposición territorial.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
st.set_page_config(page_title="GeoAlerta Oreamuno", page_icon="🌧️", layout="wide")

COLORS = {
    1: "#2c7bb6",
    2: "#abd9e9",
    3: "#ffffbf",
    4: "#fdae61",
    5: "#d7191c",
}


def all_data_files():
    if not DATA_DIR.exists():
        return []
    allowed = {".gpkg", ".geojson", ".json", ".shp", ".tif", ".tiff", ".csv", ".gpkg-wal", ".gpkg-shm"}
    return sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def find_file(*terms, suffixes=None):
    files = all_data_files()
    terms = [t.lower() for t in terms]
    if suffixes:
        suffixes = {s.lower() for s in suffixes}
    matches = []
    for p in files:
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        name = p.name.lower()
        score = sum(1 for term in terms if term in name)
        if score:
            matches.append((score, len(p.name), p))
    return sorted(matches, key=lambda x: (-x[0], x[1]))[0][2] if matches else None


def read_vector(path):
    if not path:
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return None
        if gdf.crs is None:
            st.warning(f"La capa {path.name} no tiene CRS definido y no se dibujó.")
            return None
        return gdf.to_crs(4326)
    except Exception as exc:
        st.warning(f"No se pudo leer {path.name}: {exc}")
        return None


def raster_info(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        vals = arr.compressed()
        if vals.size:
            return {
                "crs": str(src.crs), "width": src.width, "height": src.height,
                "resolution": src.res, "nodata": src.nodata,
                "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
                "valid": int(vals.size), "bounds": src.bounds,
            }
        return {"crs": str(src.crs), "width": src.width, "height": src.height,
                "resolution": src.res, "nodata": src.nodata, "min": None, "max": None,
                "valid": 0, "bounds": src.bounds}


def raster_to_overlay(path, categorical=False, opacity=0.72, max_dim=1800):
    """Reproyecta un ráster a EPSG:4326 y devuelve PNG transparente y bounds Folium."""
    with rasterio.open(path) as src:
        src_arr = src.read(1, masked=True)
        if src.crs is None:
            raise ValueError("El ráster no tiene sistema de referencia definido.")
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
            dst_width=min(src.width, max_dim), dst_height=min(src.height, max_dim)
        )
        # Limitar el tamaño de salida conservando la proporción espacial.
        scale = min(1.0, max_dim / max(dst_width, dst_height))
        if scale < 1:
            dst_width = max(1, int(dst_width * scale))
            dst_height = max(1, int(dst_height * scale))
            dst_transform, _, _ = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
                dst_width=dst_width, dst_height=dst_height
            )
        dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
        reproject(
            source=src_arr.astype("float32").filled(np.nan), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
        rgba = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
        valid = np.isfinite(dst)
        if categorical:
            for cls, hex_color in COLORS.items():
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
                mask = valid & (np.rint(dst).astype("int32") == cls)
                rgba[mask, :3] = rgb
                rgba[mask, 3] = int(255 * opacity)
        else:
            if valid.any():
                lo, hi = np.nanpercentile(dst[valid], [2, 98])
                if hi <= lo:
                    hi = lo + 1
                scaled = np.clip((dst - lo) / (hi - lo), 0, 1)
                # Rampa monocromática neutra para capas de contexto.
                intensity = (scaled * 210 + 35).astype("uint8")
                rgba[..., 0] = intensity
                rgba[..., 1] = intensity
                rgba[..., 2] = intensity
                rgba[..., 3][valid] = int(255 * opacity)
        image = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        return f"data:image/png;base64,{encoded}", [[south, west], [north, east]]


def add_vector(map_obj, gdf, name, color="#333333", weight=2, fill=False):
    if gdf is None:
        return
    folium.GeoJson(
        gdf.__geo_interface__, name=name,
        style_function=lambda _x: {
            "color": color, "weight": weight,
            "fillColor": color, "fillOpacity": 0.08 if fill else 0,
        },
        tooltip=folium.GeoJsonTooltip(fields=[c for c in gdf.columns if c != "geometry"][:2])
        if any(c != "geometry" for c in gdf.columns) else None,
    ).add_to(map_obj)


def center_from_gdf(gdf, fallback=(9.90, -83.87)):
    if gdf is None:
        return fallback
    try:
        c = gdf.geometry.union_all().centroid
        return [c.y, c.x]
    except Exception:
        return fallback


# Archivos identificados por nombre; el inventario completo se muestra en su módulo.
files = all_data_files()
canton_path = find_file("oreamuno", "canton", suffixes={".gpkg", ".geojson", ".json", ".shp"})
district_path = find_file("oreamuno", "distrito", suffixes={".gpkg", ".geojson", ".json", ".shp"})
drainage_path = find_file("red_drenaje", suffixes={".gpkg", ".geojson", ".json", ".shp"}) or find_file("drenaje", suffixes={".gpkg", ".geojson", ".json", ".shp"})
sus_path = find_file("susceptibilidad", "5_clases", suffixes={".tif", ".tiff"}) or find_file("susceptibilidad", suffixes={".tif", ".tiff"})
dem_path = find_file("copernicus", suffixes={".tif", ".tiff"})
flow_path = find_file("acumulacion", "flujo", suffixes={".tif", ".tiff"})
distance_path = find_file("distancia", "drenaje", suffixes={".tif", ".tiff"})

canton = read_vector(canton_path)
districts = read_vector(district_path)
drainage = read_vector(drainage_path)

with st.sidebar:
    st.title("🌧️ GeoAlerta Oreamuno")
    st.caption("Visor territorial de susceptibilidad a inundaciones")
    page = st.radio("Navegación", [
        "Inicio", "Susceptibilidad a inundaciones", "Contexto hidrológico",
        "Metodología", "Inventario de datos"
    ])
    st.divider()
    st.caption("Área de estudio: cantón de Oreamuno, Cartago, Costa Rica")


if page == "Inicio":
    st.title("GeoAlerta Oreamuno")
    st.subheader("Visor de resultados de susceptibilidad a inundaciones")
    st.write(
        "Herramienta de consulta geoespacial que reúne los principales resultados "
        "del análisis territorial e hidrológico desarrollado para el cantón de Oreamuno. "
        "Su producto central es el mapa de susceptibilidad a inundaciones clasificado en cinco categorías."
    )
    st.info("Esta aplicación presenta resultados previamente procesados. No constituye un sistema de pronóstico en tiempo real ni sustituye los avisos oficiales de emergencia.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clases del modelo", "5")
    c2.metric("Sistema de referencia", "CRTM05 · EPSG:5367")
    c3.metric("Resolución de trabajo", "≈ 30,65 m")
    st.divider()
    st.subheader("Mapa principal")
    if sus_path:
        try:
            img, bounds = raster_to_overlay(sus_path, categorical=True, opacity=0.78)
            m = folium.Map(location=center_from_gdf(canton), zoom_start=11, tiles="CartoDB positron", control_scale=True)
            ImageOverlay(img, bounds=bounds, opacity=1, name="Susceptibilidad a inundaciones", interactive=True, cross_origin=False, zindex=2).add_to(m)
            add_vector(m, districts, "Distritos", "#555555", 1)
            add_vector(m, canton, "Límite cantonal", "#111111", 3)
            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, use_container_width=True, height=570, returned_objects=[])
            st.caption("La simbología identifica las clases 1 a 5. Consulte el módulo de susceptibilidad para ver la leyenda.")
        except Exception as exc:
            st.error(f"No fue posible representar el ráster de susceptibilidad: {exc}")
    else:
        st.warning("No se encontró el ráster de susceptibilidad. Coloque el GeoTIFF correspondiente dentro de la carpeta data/ del repositorio.")
    st.subheader("¿Qué ofrece GeoAlerta?")
    a, b, c = st.columns(3)
    a.markdown("**Consulta del resultado**\n\nVisualización del mapa de susceptibilidad y sus cinco clases.")
    b.markdown("**Contexto territorial**\n\nConsulta de drenajes y capas de referencia disponibles.")
    c.markdown("**Trazabilidad**\n\nDescripción metodológica e inventario de datos utilizados.")


elif page == "Susceptibilidad a inundaciones":
    st.title("Susceptibilidad a inundaciones")
    st.write("Producto cartográfico principal del proyecto: clasificación territorial en cinco clases de susceptibilidad.")
    if not sus_path:
        st.error("No se encontró el ráster de susceptibilidad en data/. Verifique el nombre y la ubicación del archivo.")
    else:
        try:
            info = raster_info(sus_path)
            img, bounds = raster_to_overlay(sus_path, categorical=True, opacity=0.82)
            m = folium.Map(location=center_from_gdf(canton), zoom_start=11, tiles="CartoDB positron", control_scale=True)
            ImageOverlay(img, bounds=bounds, opacity=1, name="Susceptibilidad", interactive=True, cross_origin=False, zindex=2).add_to(m)
            add_vector(m, districts, "Distritos", "#555555", 1)
            add_vector(m, canton, "Límite cantonal", "#111111", 3)
            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, use_container_width=True, height=620, returned_objects=[])
            st.markdown("**Leyenda de clases**")
            legend_cols = st.columns(5)
            for i, col in enumerate(legend_cols, start=1):
                col.markdown(f"<div style='background:{COLORS[i]};height:17px;border-radius:3px'></div><div style='text-align:center;font-size:0.9rem'>Clase {i}</div>", unsafe_allow_html=True)
            st.caption("La leyenda se presenta como Clase 1–5 para no atribuir niveles cualitativos que no hayan sido confirmados en la clasificación final del modelo.")
            x1, x2, x3 = st.columns(3)
            x1.metric("Dimensiones", f"{info['width']} × {info['height']} px")
            x2.metric("Celdas válidas", f"{info['valid']:,}")
            x3.metric("CRS", info["crs"] or "No definido")
        except Exception as exc:
            st.error(f"Error al cargar el mapa: {exc}")


elif page == "Contexto hidrológico":
    st.title("Contexto hidrológico y territorial")
    st.write("Capas de referencia disponibles para interpretar el contexto físico e hidrológico del cantón. Seleccione las capas que desea consultar.")
    available = {}
    if drainage_path and drainage is not None:
        available["Red de drenaje"] = ("vector", drainage)
    if dem_path:
        available["Modelo digital de elevación (Copernicus GLO-30)"] = ("raster", dem_path)
    if flow_path:
        available["Acumulación de flujo"] = ("raster", flow_path)
    if distance_path:
        available["Distancia al drenaje"] = ("raster", distance_path)
    if districts is not None:
        available["Distritos"] = ("vector", districts)
    if not available:
        st.warning("No se encontraron capas de contexto en data/.")
    else:
        selected = st.multiselect("Capas para visualizar", list(available.keys()), default=[k for k in available if k in ("Red de drenaje", "Distritos")])
        m = folium.Map(location=center_from_gdf(canton), zoom_start=11, tiles="CartoDB positron", control_scale=True)
        if canton is not None:
            add_vector(m, canton, "Límite cantonal", "#111111", 3)
        for name in selected:
            kind, obj = available[name]
            try:
                if kind == "vector":
                    add_vector(m, obj, name, "#2878a5" if name == "Red de drenaje" else "#666666", 2 if name == "Red de drenaje" else 1)
                else:
                    img, bounds = raster_to_overlay(obj, categorical=False, opacity=0.48)
                    ImageOverlay(img, bounds=bounds, opacity=1, name=name, interactive=True, cross_origin=False).add_to(m)
            except Exception as exc:
                st.warning(f"No se pudo agregar {name}: {exc}")
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, use_container_width=True, height=600, returned_objects=[])
    if drainage is not None:
        st.caption(f"Red de drenaje: {len(drainage):,} entidades vectoriales.")


elif page == "Metodología":
    st.title("Metodología")
    st.write("Síntesis de la ruta de trabajo representada en GeoAlerta Oreamuno. La aplicación funciona como visor y no vuelve a ejecutar los procesos analíticos.")
    st.markdown("""
    1. **Preparación de información geoespacial.** Organización de los límites cantonales y distritales, el modelo digital de elevación y las capas hidrológicas y territoriales disponibles.
    2. **Procesamiento del contexto físico e hidrológico.** Preparación de la información de relieve, drenaje, acumulación de flujo y distancia al drenaje para caracterizar el territorio.
    3. **Construcción del modelo de susceptibilidad.** Integración de los criterios y variables definidos en el procedimiento analítico del proyecto.
    4. **Clasificación cartográfica.** Representación del resultado final mediante cinco clases de susceptibilidad a inundaciones.
    5. **Publicación y consulta.** Integración del ráster final y de las capas de referencia en una aplicación web de consulta.
    """)
    st.info("La descripción es deliberadamente general: los parámetros, ponderaciones y reglas específicas deben consultarse en el documento metodológico del proyecto, y no se infieren desde el visor.")
    st.subheader("Características espaciales del producto")
    if sus_path:
        info = raster_info(sus_path)
        left, right = st.columns(2)
        left.metric("Sistema de referencia", info["crs"] or "No definido")
        right.metric("Tamaño del ráster", f"{info['width']} × {info['height']} celdas")
        st.write(f"Resolución reportada: {info['resolution'][0]:.3f} × {info['resolution'][1]:.3f} unidades del sistema de referencia.")
    st.subheader("Alcance")
    st.write("El resultado expresa susceptibilidad espacial según el modelo desarrollado. No equivale a una predicción de cuándo ocurrirá una inundación, no representa condiciones meteorológicas en tiempo real y no sustituye los sistemas oficiales de monitoreo y alerta.")


elif page == "Inventario de datos":
    st.title("Inventario de datos")
    st.write("Archivos detectados automáticamente dentro de la carpeta data/ del repositorio.")
    if not files:
        st.warning("La carpeta data/ está vacía o no contiene formatos geoespaciales reconocidos.")
    else:
        rows = []
        for p in files:
            row = {"Archivo": p.name, "Formato": p.suffix.upper(), "Tamaño (MB)": round(p.stat().st_size / (1024 * 1024), 3), "Ubicación": str(p.relative_to(DATA_DIR))}
            if p.suffix.lower() in {".tif", ".tiff"}:
                try:
                    inf = raster_info(p)
                    row.update({"CRS": inf["crs"], "Dimensiones": f"{inf['width']} × {inf['height']}", "Resolución": f"{inf['resolution'][0]:.3f} × {inf['resolution'][1]:.3f}"})
                except Exception:
                    row.update({"CRS": "No leído", "Dimensiones": "—", "Resolución": "—"})
            elif p.suffix.lower() in {".gpkg", ".geojson", ".json", ".shp"}:
                try:
                    g = gpd.read_file(p)
                    row.update({"CRS": str(g.crs), "Entidades": len(g)})
                except Exception:
                    row.update({"CRS": "No leído", "Entidades": "—"})
            rows.append(row)
        df = pd.DataFrame(rows)
        st.metric("Archivos geoespaciales detectados", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Descargar inventario CSV", csv, "inventario_geoalerta_oreamuno.csv", "text/csv")
    st.caption("El inventario solo examina los archivos que se encuentran dentro de data/. No modifica los datos originales.")

st.divider()
st.caption("GeoAlerta Oreamuno · Prototipo de consulta geoespacial · Cartago, Costa Rica")
