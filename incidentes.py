
# ============================================================
# GEOALERTA OREAMUNO
# MÓDULO DE REPORTE DE INCIDENTES
# ============================================================

from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from html import escape

import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium
from supabase import create_client


# ============================================================
# CONFIGURACIÓN
# ============================================================

BUCKET = "evidencias-incidentes"

TIPOS_INCIDENTE = [
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

COLORES = {
    "Río desbordado": "blue",
    "Deslizamiento": "orange",
    "Calle cerrada": "red",
    "Tubería rota": "cadetblue",
    "Material en carretera": "darkred",
    "Inundación": "purple",
    "Otro": "gray"
}


# ============================================================
# CONEXIÓN A SUPABASE
# ============================================================

@st.cache_resource
def conectar_supabase():

    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["anon_key"]

    return create_client(url, key)


# ============================================================
# CONSULTA DE INCIDENTES
# ============================================================

def obtener_incidentes(cliente):

    try:

        respuesta = (
            cliente.table("incidentes")
            .select("*")
            .order("fecha_reporte", desc=True)
            .execute()
        )

        return pd.DataFrame(respuesta.data or [])

    except Exception as error:

        st.error(
            f"No fue posible consultar los incidentes: {error}"
        )

        return pd.DataFrame()


# ============================================================
# PREPARACIÓN DE DISTRITOS
# ============================================================

def preparar_distritos(districts):

    if districts is None or districts.empty:

        return None, [], None

    capa = districts.copy()

    if capa.crs is None:

        st.error(
            "La capa de distritos no tiene un sistema "
            "de coordenadas definido."
        )

        return None, [], None

    capa = capa.to_crs("EPSG:4326")

    posibles_campos = [
        "distrito",
        "DISTRITO",
        "nom_distr",
        "NOM_DISTR",
        "nombre",
        "NOMBRE",
        "name",
        "NAME"
    ]

    campo_nombre = next(
        (
            campo
            for campo in posibles_campos
            if campo in capa.columns
        ),
        None
    )

    if campo_nombre:

        nombres = sorted(
            capa[campo_nombre]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        nombres = [
            f"Distrito {i + 1}"
            for i in range(len(capa))
        ]

    return capa, nombres, campo_nombre


# ============================================================
# CENTRO DEL MAPA
# ============================================================

def centro_mapa(capa):

    if capa is not None and not capa.empty:

        centro = capa.geometry.unary_union.centroid

        return [
            centro.y,
            centro.x
        ]

    return [9.90, -83.90]


# ============================================================
# MAPA DE SELECCIÓN
# ============================================================

def mapa_seleccion(centro, capa=None):

    mapa = folium.Map(
        location=centro,
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True
    )

    if capa is not None:

        tooltip_fields = [
            campo
            for campo in capa.columns
            if campo != "geometry"
        ][:1]

        opciones = {
            "name": "Distritos",
            "style_function": lambda feature: {
                "fillColor": "#3388ff",
                "color": "#315b82",
                "weight": 1.2,
                "fillOpacity": 0.08
            }
        }

        if tooltip_fields:

            opciones["tooltip"] = folium.GeoJsonTooltip(
                fields=tooltip_fields
            )

        folium.GeoJson(
            capa,
            **opciones
        ).add_to(mapa)

    folium.LatLngPopup().add_to(mapa)

    return mapa


# ============================================================
# REGISTRO DE INCIDENTES
# ============================================================

def registrar_incidente(
    cliente,
    distritos_capa,
    distritos_nombres
):

    st.subheader("Registrar un incidente")

    st.write(
        "Selecciona en el mapa el punto donde ocurrió "
        "el incidente. Después completa la información."
    )

    centro = centro_mapa(distritos_capa)

    mapa = mapa_seleccion(
        centro,
        distritos_capa
    )

    resultado = st_folium(
        mapa,
        width=None,
        height=420,
        returned_objects=["last_clicked"],
        key="mapa_nuevo_incidente"
    )

    punto = (
        resultado.get("last_clicked")
        if resultado
        else None
    )

    latitud = None
    longitud = None

    if punto:

        latitud = float(punto["lat"])
        longitud = float(punto["lng"])

        st.success(
            f"Ubicación seleccionada: "
            f"{latitud:.6f}, {longitud:.6f}"
        )

    else:

        st.info(
            "Haz clic sobre el mapa para seleccionar "
            "la ubicación del incidente."
        )

    st.divider()

    with st.form("formulario_incidente"):

        tipo = st.selectbox(
            "Tipo de incidente",
            TIPOS_INCIDENTE
        )

        distrito = st.selectbox(
            "Distrito",
            distritos_nombres
        )

        descripcion = st.text_area(
            "Descripción del incidente",
            placeholder=(
                "Describe qué ocurrió, las condiciones "
                "observadas y posibles afectaciones..."
            ),
            max_chars=1000
        )

        fotografia = st.file_uploader(
            "Fotografía como evidencia (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Tamaño máximo permitido: 8 MB."
        )

        enviar = st.form_submit_button(
            "Enviar reporte",
            type="primary",
            use_container_width=True
        )

    if enviar:

        if latitud is None or longitud is None:

            st.error(
                "Debes seleccionar una ubicación en el mapa."
            )

            return

        if not descripcion.strip():

            st.error(
                "Debes escribir una descripción del incidente."
            )

            return

        if fotografia and fotografia.size > 8 * 1024 * 1024:

            st.error(
                "La fotografía supera el límite de 8 MB."
            )

            return

        foto_url = None
        ruta_foto = None

        try:

            # ----------------------------------------
            # SUBIR FOTOGRAFÍA
            # ----------------------------------------

            if fotografia:

                extension = (
                    fotografia.name.rsplit(".", 1)[-1].lower()
                )

                ruta_foto = (
                    f"reportes/"
                    f"{datetime.now().strftime('%Y/%m')}/"
                    f"{uuid4().hex}.{extension}"
                )

                tipos_mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp"
                }

                cliente.storage.from_(BUCKET).upload(
                    ruta_foto,
                    fotografia.getvalue(),
                    file_options={
                        "content-type": tipos_mime[extension],
                        "upsert": "false"
                    }
                )

                foto_url = (
                    cliente.storage
                    .from_(BUCKET)
                    .get_public_url(ruta_foto)
                )

            # ----------------------------------------
            # REGISTRAR INFORMACIÓN
            # ----------------------------------------

            fecha = datetime.now(
                ZoneInfo("America/Costa_Rica")
            ).isoformat()

            registro = {

                "tipo_incidente": tipo,

                "distrito": distrito,

                "descripcion": descripcion.strip(),

                "latitud": latitud,

                "longitud": longitud,

                "fecha_reporte": fecha,

                "estado": "Reportado",

                "foto_url": foto_url

            }

            cliente.table("incidentes").insert(
                registro
            ).execute()

            st.success(
                "¡Incidente registrado correctamente!"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"No fue posible guardar el reporte: {error}"
            )

            if ruta_foto:

                st.warning(
                    "Si la fotografía llegó a almacenarse, "
                    "puede haber quedado sin un registro asociado."
                )


# ============================================================
# MAPA GENERAL DE INCIDENTES
# ============================================================

def mostrar_mapa_incidentes(
    df,
    distritos_capa
):

    st.subheader(
        "Distribución territorial de incidentes"
    )

    if df.empty:

        st.info(
            "Todavía no existen incidentes registrados."
        )

        return

    centro = centro_mapa(distritos_capa)

    mapa = folium.Map(
        location=centro,
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True
    )

    if distritos_capa is not None:

        folium.GeoJson(
            distritos_capa,
            name="Distritos",
            style_function=lambda feature: {
                "fillColor": "#3388ff",
                "color": "#315b82",
                "weight": 1,
                "fillOpacity": 0.06
            }
        ).add_to(mapa)

    for _, incidente in df.iterrows():

        try:

            lat = float(incidente["latitud"])
            lon = float(incidente["longitud"])

            tipo = str(
                incidente.get(
                    "tipo_incidente",
                    "Otro"
                )
            )

            distrito = str(
                incidente.get(
                    "distrito",
                    "No especificado"
                )
            )

            descripcion = str(
                incidente.get(
                    "descripcion",
                    ""
                )
            )

            fecha = str(
                incidente.get(
                    "fecha_reporte",
                    ""
                )
            )

            estado = str(
                incidente.get(
                    "estado",
                    "Reportado"
                )
            )

            foto_url = incidente.get("foto_url")

            popup_html = f"""
            <div style="width:260px;font-family:Arial">

                <h4>{escape(tipo)}</h4>

                <b>Distrito:</b>
                {escape(distrito)}
                <br>

                <b>Estado:</b>
                {escape(estado)}
                <br>

                <b>Fecha:</b>
                {escape(fecha)}
                <br><br>

                <b>Descripción:</b>

                <p>{escape(descripcion)}</p>

            </div>
            """

            if isinstance(foto_url, str) and foto_url.startswith("https://"):

                popup_html = popup_html.replace(
                    "</div>",
                    f"""
                    <img
                        src="{escape(foto_url, quote=True)}"
                        style="width:100%;margin-top:8px;border-radius:6px"
                    />
                    </div>
                    """
                )

            folium.Marker(

                [lat, lon],

                tooltip=tipo,

                popup=folium.Popup(
                    popup_html,
                    max_width=320
                ),

                icon=folium.Icon(
                    color=COLORES.get(tipo, "gray"),
                    icon="info-sign"
                )

            ).add_to(mapa)

        except Exception:

            continue

    folium.LayerControl().add_to(mapa)

    st_folium(
        mapa,
        width=None,
        height=550,
        key="mapa_general_incidentes"
    )


# ============================================================
# HISTORIAL DE INCIDENTES
# ============================================================

def mostrar_historial(df):

    st.subheader(
        "Historial de incidentes"
    )

    if df.empty:

        st.info(
            "No hay reportes para mostrar."
        )

        return

    datos = df.copy()

    col1, col2, col3 = st.columns(3)

    tipos = sorted(
        datos["tipo_incidente"]
        .dropna()
        .unique()
        .tolist()
    )

    distritos = sorted(
        datos["distrito"]
        .dropna()
        .unique()
        .tolist()
    )

    estados = sorted(
        datos["estado"]
        .dropna()
        .unique()
        .tolist()
    )

    with col1:

        filtro_tipo = st.selectbox(
            "Tipo",
            ["Todos"] + tipos,
            key="filtro_tipo_incidentes"
        )

    with col2:

        filtro_distrito = st.selectbox(
            "Distrito",
            ["Todos"] + distritos,
            key="filtro_distrito_incidentes"
        )

    with col3:

        filtro_estado = st.selectbox(
            "Estado",
            ["Todos"] + estados,
            key="filtro_estado_incidentes"
        )

    if filtro_tipo != "Todos":

        datos = datos[
            datos["tipo_incidente"] == filtro_tipo
        ]

    if filtro_distrito != "Todos":

        datos = datos[
            datos["distrito"] == filtro_distrito
        ]

    if filtro_estado != "Todos":

        datos = datos[
            datos["estado"] == filtro_estado
        ]

    columnas = [

        "fecha_reporte",

        "tipo_incidente",

        "distrito",

        "descripcion",

        "estado",

        "latitud",

        "longitud"

    ]

    columnas = [
        c for c in columnas
        if c in datos.columns
    ]

    st.dataframe(
        datos[columnas],
        use_container_width=True,
        hide_index=True
    )

    csv = datos.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Descargar reportes en CSV",
        data=csv,
        file_name="GeoAlerta_Incidentes.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        f"Se muestran {len(datos)} reportes."
    )


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

def render_incidentes(districts):

    st.title(
        "Reporte de incidentes"
    )

    st.markdown(
        """
        Registra y consulta eventos que puedan afectar
        la seguridad territorial y la infraestructura
        del cantón de Oreamuno.
        """
    )

    # ----------------------------------------
    # CONEXIÓN
    # ----------------------------------------

    try:

        cliente = conectar_supabase()

    except Exception as error:

        st.error(
            "No se pudo conectar con Supabase. "
            "Verifica las credenciales en Streamlit Secrets."
        )

        st.exception(error)

        return

    # ----------------------------------------
    # DISTRITOS
    # ----------------------------------------

    (
        capa_distritos,
        distritos_nombres,
        campo_nombre
    ) = preparar_distritos(districts)

    if not distritos_nombres:

        st.error(
            "No se pudo obtener la lista de distritos. "
            "Revisa la capa districts."
        )

        return

    # ----------------------------------------
    # CONSULTA
    # ----------------------------------------

    df = obtener_incidentes(cliente)

    total = len(df)

    reportados = 0
    en_revision = 0
    atendidos = 0

    if not df.empty and "estado" in df.columns:

        reportados = int(
            (df["estado"] == "Reportado").sum()
        )

        en_revision = int(
            (df["estado"] == "En revisión").sum()
        )

        atendidos = int(
            (df["estado"] == "Atendido").sum()
        )

    # ----------------------------------------
    # INDICADORES
    # ----------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total de incidentes",
        total
    )

    c2.metric(
        "Reportados",
        reportados
    )

    c3.metric(
        "En revisión",
        en_revision
    )

    c4.metric(
        "Atendidos",
        atendidos
    )

    st.divider()

    # ----------------------------------------
    # PESTAÑAS
    # ----------------------------------------

    tab1, tab2, tab3 = st.tabs([

        "Registrar incidente",

        "Mapa de incidentes",

        "Historial"

    ])

    with tab1:

        registrar_incidente(
            cliente,
            capa_distritos,
            distritos_nombres
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
        "GeoAlerta Oreamuno | Sistema de reporte territorial. "
        "Los incidentes registrados son reportes ciudadanos "
        "y no representan necesariamente eventos verificados "
        "por las autoridades competentes."
    )
