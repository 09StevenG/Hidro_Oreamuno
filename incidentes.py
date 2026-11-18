
# ============================================================
# GEOALERTA OREAMUNO
# MÓDULO DE REPORTE DE INCIDENTES
# ============================================================

import streamlit as st
import pandas as pd
import folium
import uuid
import os

from datetime import datetime
from supabase import create_client
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN
# ============================================================

BUCKET = "evidencias-incidentes"

TIPOS_INCIDENTES = [
    "Río desbordado",
    "Deslizamiento",
    "Calle cerrada",
    "Tubería rota",
    "Material en carretera",
    "Inundación",
    "Otro"
]

ESTADOS = [
    "Reportado",
    "En revisión",
    "Atendido"
]


# ============================================================
# CONEXIÓN CON SUPABASE
# ============================================================

@st.cache_resource
def conectar_supabase():
    """
    Conecta con el proyecto Supabase mediante
    las credenciales configuradas en Streamlit Secrets.
    """

    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]

        return create_client(url, key)

    except Exception as e:
        st.error(
            "No fue posible conectar con Supabase. "
            "Verifique la configuración de Secrets."
        )
        st.caption(str(e))
        return None


# ============================================================
# PREPARACIÓN DE DISTRITOS
# ============================================================

def preparar_distritos(districts):

    if districts is None:
        return None, []

    try:
        capa = districts.copy()

        if capa.empty:
            return None, []

        if capa.crs is None:
            st.warning(
                "La capa de distritos no tiene un sistema "
                "de referencia espacial definido."
            )
            return None, []

        capa = capa.to_crs(epsg=4326)

        campos_posibles = [
            "DISTRITO",
            "Distrito",
            "distrito",
            "NOM_DIST",
            "NOMBRE",
            "Nombre",
            "NOM_DISTR",
            "DSD_NOM",
            "NOM_DISTRI"
        ]

        campo = next(
            (c for c in campos_posibles if c in capa.columns),
            None
        )

        if campo is None:
            campos_texto = capa.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

            if campos_texto:
                campo = campos_texto[0]

        if campo is None:
            st.warning(
                "No se identificó un campo con el nombre "
                "del distrito."
            )
            return None, []

        capa = capa[
            capa.geometry.notna()
            & ~capa.geometry.is_empty
        ].copy()

        capa["_nombre_distrito"] = (
            capa[campo].astype(str).str.strip()
        )

        nombres = sorted(
            capa["_nombre_distrito"]
            .dropna()
            .unique()
            .tolist()
        )

        return capa, nombres

    except Exception as e:
        st.warning(
            f"No fue posible preparar la capa de distritos: {e}"
        )
        return None, []


# ============================================================
# CONSULTA DE INCIDENTES
# ============================================================

def obtener_incidentes(cliente):

    if cliente is None:
        return pd.DataFrame()

    try:
        respuesta = (
            cliente.table("incidentes")
            .select("*")
            .order("fecha_reporte", desc=True)
            .execute()
        )

        datos = respuesta.data or []

        return pd.DataFrame(datos)

    except Exception as e:
        st.error(
            f"No fue posible consultar los incidentes: {e}"
        )
        return pd.DataFrame()


# ============================================================
# CENTRO DEL MAPA
# ============================================================

def centro_mapa(capa):

    if capa is not None and not capa.empty:

        try:
            centro = capa.geometry.union_all().centroid

            return [
                centro.y,
                centro.x
            ]

        except Exception:
            pass

    # Centro aproximado del cantón de Oreamuno
    return [9.98, -83.87]


# ============================================================
# MAPA PARA SELECCIONAR UBICACIÓN
# ============================================================

def mapa_seleccion(capa, lat_inicial=None, lon_inicial=None):

    centro = centro_mapa(capa)

    if lat_inicial is not None and lon_inicial is not None:
        centro = [lat_inicial, lon_inicial]

    mapa = folium.Map(
        location=centro,
        zoom_start=12,
        control_scale=True,
        tiles=None
    )

    # Mapas base
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(mapa)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Imagen satelital",
        control=True
    ).add_to(mapa)

    # Distritos
    if capa is not None:

        folium.GeoJson(
            capa,
            name="Distritos",
            style_function=lambda feature: {
                "fillColor": "#3388ff",
                "color": "#1d3557",
                "weight": 1.5,
                "fillOpacity": 0.08
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["_nombre_distrito"],
                aliases=["Distrito:"]
            )
        ).add_to(mapa)

    # Marcador de ubicación seleccionada
    if lat_inicial is not None and lon_inicial is not None:

        folium.Marker(
            [lat_inicial, lon_inicial],
            tooltip="Ubicación seleccionada",
            icon=folium.Icon(
                color="red",
                icon="exclamation-triangle",
                prefix="fa"
            )
        ).add_to(mapa)

    folium.LayerControl().add_to(mapa)

    return mapa


# ============================================================
# IDENTIFICAR DISTRITO SEGÚN COORDENADAS
# ============================================================

def identificar_distrito(lat, lon, capa):

    if capa is None:
        return None

    try:
        from shapely.geometry import Point

        punto = Point(lon, lat)

        coincidencias = capa[
            capa.geometry.intersects(punto)
        ]

        if not coincidencias.empty:
            return coincidencias.iloc[0]["_nombre_distrito"]

        return None

    except Exception:
        return None


# ============================================================
# SUBIR FOTOGRAFÍA A SUPABASE STORAGE
# ============================================================

def subir_fotografia(cliente, archivo):

    if archivo is None:
        return None

    try:

        extensiones = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp"
        }

        contenido = archivo.getvalue()

        if len(contenido) > 8 * 1024 * 1024:
            st.error(
                "La fotografía supera el límite permitido de 8 MB."
            )
            return None

        extension = extensiones.get(archivo.type)

        if extension is None:
            st.error(
                "Formato no permitido. Utilice JPG, PNG o WEBP."
            )
            return None

        nombre_archivo = (
            f"reportes/{uuid.uuid4().hex}{extension}"
        )

        cliente.storage.from_(BUCKET).upload(
            path=nombre_archivo,
            file=contenido,
            file_options={
                "content-type": archivo.type,
                "upsert": "false"
            }
        )

        resultado = cliente.storage.from_(BUCKET).get_public_url(
            nombre_archivo
        )

        return resultado

    except Exception as e:
        st.error(
            f"No fue posible subir la fotografía: {e}"
        )
        return None


# ============================================================
# REGISTRAR INCIDENTE
# ============================================================

def registrar_incidente(
    cliente,
    capa_distritos,
    nombres_distritos
):

    st.subheader("Registrar nuevo incidente")

    st.write(
        "Complete la información del evento y seleccione "
        "su ubicación directamente en el mapa."
    )

    if "incidente_lat" not in st.session_state:
        st.session_state.incidente_lat = None

    if "incidente_lon" not in st.session_state:
        st.session_state.incidente_lon = None

    # --------------------------------------------------------
    # MAPA
    # --------------------------------------------------------

    st.markdown("**1. Ubicación del incidente**")

    mapa = mapa_seleccion(
        capa_distritos,
        st.session_state.incidente_lat,
        st.session_state.incidente_lon
    )

    resultado_mapa = st_folium(
        mapa,
        height=440,
        width=None,
        key="mapa_seleccion_incidente",
        returned_objects=["last_clicked"]
    )

    if resultado_mapa and resultado_mapa.get("last_clicked"):

        punto = resultado_mapa["last_clicked"]

        st.session_state.incidente_lat = punto["lat"]
        st.session_state.incidente_lon = punto["lng"]

    lat = st.session_state.incidente_lat
    lon = st.session_state.incidente_lon

    if lat is not None and lon is not None:

        st.success(
            f"Ubicación seleccionada: "
            f"{lat:.6f}, {lon:.6f}"
        )

        distrito_detectado = identificar_distrito(
            lat,
            lon,
            capa_distritos
        )

        if distrito_detectado:
            st.info(
                f"Distrito identificado automáticamente: "
                f"{distrito_detectado}"
            )

    else:
        distrito_detectado = None

        st.info(
            "Haga clic sobre el mapa para indicar "
            "la ubicación del incidente."
        )

    # --------------------------------------------------------
    # FORMULARIO
    # --------------------------------------------------------

    st.divider()
    st.markdown("**2. Información del incidente**")

    with st.form("formulario_incidente", clear_on_submit=False):

        tipo = st.selectbox(
            "Tipo de incidente",
            TIPOS_INCIDENTES
        )

        if nombres_distritos:

            indice_distrito = 0

            if distrito_detectado in nombres_distritos:
                indice_distrito = nombres_distritos.index(
                    distrito_detectado
                )

            distrito = st.selectbox(
                "Distrito",
                nombres_distritos,
                index=indice_distrito
            )

        else:

            distrito = st.text_input(
                "Distrito"
            )

        descripcion = st.text_area(
            "Descripción del incidente",
            placeholder=(
                "Describa qué ocurrió, las condiciones "
                "observadas y posibles afectaciones..."
            ),
            height=120,
            max_chars=1500
        )

        fotografia = st.file_uploader(
            "Fotografía de evidencia (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Tamaño máximo: 8 MB."
        )

        if fotografia is not None:
            st.image(
                fotografia,
                caption="Vista previa de la fotografía",
                use_container_width=True
            )

        confirmar = st.checkbox(
            "Confirmo que la información suministrada "
            "corresponde a un incidente real."
        )

        enviar = st.form_submit_button(
            "Enviar reporte de incidente",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PROCESAMIENTO
    # --------------------------------------------------------

    if enviar:

        if cliente is None:
            st.error("No hay conexión con Supabase.")
            return

        if lat is None or lon is None:
            st.error(
                "Debe seleccionar una ubicación en el mapa."
            )
            return

        if not descripcion.strip():
            st.error(
                "Debe ingresar una descripción del incidente."
            )
            return

        if not distrito or not str(distrito).strip():
            st.error("Debe indicar el distrito.")
            return

        if not confirmar:
            st.error(
                "Debe confirmar la veracidad del reporte."
            )
            return

        with st.spinner("Registrando incidente..."):

            foto_url = None

            if fotografia is not None:

                foto_url = subir_fotografia(
                    cliente,
                    fotografia
                )

                if foto_url is None:
                    st.error(
                        "No se pudo cargar la fotografía. "
                        "El reporte no fue registrado."
                    )
                    return

            registro = {
                "tipo_incidente": tipo,
                "distrito": distrito,
                "descripcion": descripcion.strip(),
                "latitud": float(lat),
                "longitud": float(lon),
                "estado": "Reportado",
                "foto_url": foto_url
            }

            try:

                cliente.table("incidentes").insert(
                    registro
                ).execute()

                st.success(
                    "¡Incidente registrado correctamente!"
                )

                st.session_state.incidente_lat = None
                st.session_state.incidente_lon = None

                st.balloons()

                st.rerun()

            except Exception as e:

                st.error(
                    f"No fue posible registrar el incidente: {e}"
                )


# ============================================================
# MAPA GENERAL DE INCIDENTES
# ============================================================

def mostrar_mapa_incidentes(df, capa_distritos=None):

    st.subheader("Mapa de incidentes reportados")

    if df.empty:
        st.info("No existen incidentes registrados.")
        return

    datos = df.copy()

    datos["latitud"] = pd.to_numeric(
        datos["latitud"],
        errors="coerce"
    )

    datos["longitud"] = pd.to_numeric(
        datos["longitud"],
        errors="coerce"
    )

    datos = datos.dropna(
        subset=["latitud", "longitud"]
    )

    if datos.empty:
        st.warning(
            "No hay coordenadas válidas para mostrar."
        )
        return

    centro = [
        datos["latitud"].mean(),
        datos["longitud"].mean()
    ]

    mapa = folium.Map(
        location=centro,
        zoom_start=12,
        control_scale=True,
        tiles=None
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap"
    ).add_to(mapa)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Imagen satelital"
    ).add_to(mapa)

    colores = {
        "Río desbordado": "blue",
        "Deslizamiento": "orange",
        "Calle cerrada": "red",
        "Tubería rota": "purple",
        "Material en carretera": "darkred",
        "Inundación": "cadetblue",
        "Otro": "gray"
    }

    for _, incidente in datos.iterrows():

        tipo = incidente.get(
            "tipo_incidente",
            "Otro"
        )

        distrito = incidente.get(
            "distrito",
            "No especificado"
        )

        descripcion = incidente.get(
            "descripcion",
            ""
        )

        estado = incidente.get(
            "estado",
            "Reportado"
        )

        fecha = incidente.get(
            "fecha_reporte",
            ""
        )

        foto = incidente.get(
            "foto_url",
            None
        )

        contenido = f"""
        <div style="width:250px">
            <h4>{tipo}</h4>
            <b>Distrito:</b> {distrito}<br>
            <b>Estado:</b> {estado}<br>
            <b>Fecha:</b> {fecha}<br>
            <hr>
            <b>Descripción:</b><br>
            {descripcion}
        """

        if pd.notna(foto) and foto:
            contenido += (
                f'<br><br><a href="{foto}" target="_blank">'
                "Ver fotografía de evidencia</a>"
            )

        contenido += "</div>"

        folium.Marker(
            location=[
                incidente["latitud"],
                incidente["longitud"]
            ],
            tooltip=str(tipo),
            popup=folium.Popup(
                contenido,
                max_width=300
            ),
            icon=folium.Icon(
                color=colores.get(tipo, "gray"),
                icon="exclamation-sign"
            )
        ).add_to(mapa)

    folium.LayerControl().add_to(mapa)

    st_folium(
        mapa,
        height=520,
        width=None,
        key="mapa_general_incidentes"
    )


# ============================================================
# HISTORIAL DE INCIDENTES
# ============================================================

def mostrar_historial(df):

    st.subheader("Historial de reportes")

    if df.empty:
        st.info(
            "Todavía no existen incidentes registrados."
        )
        return

    datos = df.copy()

    # Filtros
    col1, col2 = st.columns(2)

    with col1:

        tipos = sorted(
            datos["tipo_incidente"]
            .dropna()
            .unique()
            .tolist()
        )

        filtro_tipo = st.multiselect(
            "Filtrar por tipo",
            tipos,
            default=tipos
        )

    with col2:

        distritos = sorted(
            datos["distrito"]
            .dropna()
            .unique()
            .tolist()
        )

        filtro_distrito = st.multiselect(
            "Filtrar por distrito",
            distritos,
            default=distritos
        )

    filtrados = datos[
        datos["tipo_incidente"].isin(filtro_tipo)
        & datos["distrito"].isin(filtro_distrito)
    ].copy()

    columnas = [
        "tipo_incidente",
        "distrito",
        "descripcion",
        "fecha_reporte",
        "estado",
        "latitud",
        "longitud"
    ]

    columnas = [
        c for c in columnas
        if c in filtrados.columns
    ]

    st.dataframe(
        filtrados[columnas],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Se muestran {len(filtrados)} de "
        f"{len(datos)} incidentes registrados."
    )

    # Descarga CSV
    csv = filtrados.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "Descargar reportes en CSV",
        data=csv,
        file_name="GeoAlerta_Oreamuno_Incidentes.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# PANEL DE INDICADORES
# ============================================================

def mostrar_indicadores(df):

    if df.empty:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total de incidentes", 0)
        col2.metric("Reportados", 0)
        col3.metric("Atendidos", 0)

        return

    total = len(df)

    reportados = (
        df["estado"] == "Reportado"
    ).sum() if "estado" in df.columns else 0

    atendidos = (
        df["estado"] == "Atendido"
    ).sum() if "estado" in df.columns else 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total de incidentes",
        total
    )

    col2.metric(
        "Reportados",
        int(reportados)
    )

    col3.metric(
        "Atendidos",
        int(atendidos)
    )


# ============================================================
# PANEL PRINCIPAL DEL MÓDULO
# ============================================================

def render_incidentes(districts):

    st.title("🚨 Reporte de incidentes")

    st.markdown(
        """
        Este módulo permite registrar y consultar eventos
        que pueden representar amenazas o afectaciones
        en el territorio del cantón de Oreamuno.

        Los reportes pueden incluir ubicación geográfica,
        descripción del evento y fotografía de evidencia.
        """
    )

    st.divider()

    # Conexión
    cliente = conectar_supabase()

    if cliente is None:

        st.error(
            "El módulo de incidentes no está disponible "
            "hasta restablecer la conexión con Supabase."
        )
        return

    # Preparar distritos
    capa_distritos, nombres_distritos = preparar_distritos(
        districts
    )

    # Consultar datos
    df = obtener_incidentes(cliente)

    # Indicadores
    mostrar_indicadores(df)

    st.divider()

    # Pestañas
    tab1, tab2, tab3 = st.tabs(
        [
            "📍 Registrar incidente",
            "🗺️ Mapa de incidentes",
            "📋 Historial de reportes"
        ]
    )

    with tab1:

        registrar_incidente(
            cliente,
            capa_distritos,
            nombres_distritos
        )

    with tab2:

        mostrar_mapa_incidentes(
            df,
            capa_distritos
        )

    with tab3:

        mostrar_historial(df)

    st.divider()

    st.caption(
        "GeoAlerta Oreamuno | Sistema de información "
        "y monitoreo territorial."
    )
