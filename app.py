import io
import csv
from turtle import lt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MRP Shortage Analyzer", layout="wide")
st.sidebar.title("MRP Shortage Analyzer")
st.sidebar.caption("Demo version with synthetic dataset")
st.title("MRP Shortage Analyzer (Unrestricted < 0)")
st.caption("Analiza exportaciones MRP y detecta faltantes por componente usando el campo 'Unrestricted'.")

# ---------------------------
# Helpers
# ---------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    # Match case-insensitive exact column name
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    return None

def to_number(series: pd.Series) -> pd.Series:
    """
    Convierte números que pueden venir como:
    - 1.234,56 (EU) o 1234.56 (US)
    - con espacios
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False)
    # Si viene formato europeo (1.234,56) => quitar miles "." y convertir "," a "."
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def parse_datetime(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    # Fecha tipo "13.02.2026" y hora "16:43:20"
    date_str = df[date_col].astype(str).str.strip()
    time_str = df[time_col].astype(str).str.strip()
    dt = pd.to_datetime(date_str + " " + time_str, errors="coerce", dayfirst=True)
    return dt

def robust_read_table(raw_bytes: bytes) -> pd.DataFrame | None:
    """
    Lector robusto para exports industriales:
    - Encodings típicos: utf-16 (muy común), utf-8-sig, cp1252/latin1
    - Separadores: autodetect, y fallback a \t ; , |
    """
    encodings = ["utf-16", "utf-8-sig", "cp1252", "latin1", "utf-16le", "utf-16be"]

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
        except Exception:
            continue

        # 1) Autodetect separador con pandas
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            pass

        # 2) Sniffer csv para , ; \t |
        try:
            sample = text[:8000]
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            df = pd.read_csv(io.StringIO(text), sep=dialect.delimiter, engine="python")
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            pass

        # 3) Fallback explícito (incluye TAB primero, típico en tu caso)
        for sep in ["\t", ";", ",", "|"]:
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
                if df is not None and df.shape[1] > 1:
                    return df
            except Exception:
                pass

    return None

# ---------------------------
# Upload
# ---------------------------
uploaded = st.file_uploader(
    "Sube tu CSV de MRP",
    type=["csv", "txt"]
)

if uploaded:
    raw_bytes = uploaded.getvalue()
    df = robust_read_table(raw_bytes)
else:
    st.info("Usando dataset de ejemplo.")
    df = pd.read_csv("data/sample.csv", sep=";")

# ---------------------------
# Column mapping (flexible)
# ---------------------------
col_component = find_col(df, ["Componente", "Component", "Material"])
col_desc      = find_col(df, ["Descripción del componente", "Descripción", "Description"])
col_date      = find_col(df, ["Fecha de necesidad", "Need date", "Fecha"])
col_time      = find_col(df, ["Hora mas temprana", "Hora más temprana", "Earliest time", "Hora"])
col_need_open = find_col(df, ["Cantidad Necesaria Abierta", "Open required qty", "Cantidad necesaria abierta"])
col_need_acc  = find_col(df, ["Necesidad Total Acumulada", "Total accumulated requirement", "Necesidad acumulada"])
col_unres     = find_col(df, ["Unrestricted", "Stock projected", "Libre utilización"])

required = {
    "Componente": col_component,
    "Fecha de necesidad": col_date,
    "Hora mas temprana": col_time,
    "Necesidad Total Acumulada": col_need_acc,
    "Unrestricted": col_unres,
}
missing = [k for k, v in required.items() if v is None]

if missing:
    st.error(
        "Faltan columnas necesarias: "
        + ", ".join(missing)
        + "\n\nColumnas detectadas:\n- "
        + "\n- ".join(list(df.columns))
    )
    st.stop()

# ---------------------------
# Cleaning / types
# ---------------------------
work = df.copy()

work["component"] = work[col_component].astype(str).str.strip()
work["desc"] = work[col_desc].astype(str).str.strip() if col_desc else ""

work["need_open"] = to_number(work[col_need_open]) if col_need_open else pd.NA
work["need_acc"] = to_number(work[col_need_acc])
work["unrestricted"] = to_number(work[col_unres])

work["dt"] = parse_datetime(work, col_date, col_time)

# Drop invalid
work = work.dropna(subset=["component", "need_acc", "unrestricted", "dt"]).copy()
work = work.sort_values(["component", "dt"])

# Stock initial estimate if relation holds:
# Unrestricted = Stock - NeedAcc  => Stock ≈ Unrestricted + NeedAcc
work["stock_est"] = work["unrestricted"] + work["need_acc"]

# ---------------------------
# Shortage summary per component
# ---------------------------
def shortage_summary(df_comp: pd.DataFrame) -> dict:
    df_comp = df_comp.sort_values("dt")
    start_dt = df_comp["dt"].min()

    neg = df_comp[df_comp["unrestricted"] < 0]
    if neg.empty:
        return {
            "has_shortage": False,
            "critical_time": pd.NaT,
            "max_deficit": 0.0,
            "min_unrestricted": float(df_comp["unrestricted"].min()),
            "stock_est_median": float(df_comp["stock_est"].median()),
            "hours_to_shortage": None,
        }

    critical_dt = neg.iloc[0]["dt"]
    hours_to = (critical_dt - start_dt).total_seconds() / 3600.0

    return {
        "has_shortage": True,
        "critical_time": critical_dt,
        "max_deficit": float((-neg["unrestricted"]).max()),
        "min_unrestricted": float(df_comp["unrestricted"].min()),
        "stock_est_median": float(df_comp["stock_est"].median()),
        "hours_to_shortage": float(hours_to),
    }

rows = []
for comp, g in work.groupby("component"):
    info = shortage_summary(g)
    rows.append({
        "Componente": comp,
        "Descripción": (g["desc"].iloc[0] if "desc" in g.columns else ""),
        "Shortage": "Sí" if info["has_shortage"] else "No",
        "Hora crítica (primera vez <0)": info["critical_time"],
        "Déficit máximo (uds)": info["max_deficit"],
        "Unrestricted mínimo": info["min_unrestricted"],
        "Stock inicial estimado (mediana)": info["stock_est_median"],
        "Horas hasta shortage": info["hours_to_shortage"],
    })

summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(
    by=["Shortage", "Déficit máximo (uds)"],
    ascending=[False, False]
)

# Orden: primero los que tienen shortage y más deficit arriba
summary_df["Shortage_sort"] = (summary_df["Shortage"] == "Sí").astype(int)
summary_df = summary_df.sort_values(
    by=["Shortage_sort", "Déficit máximo (uds)"],
    ascending=[False, False]
).drop(columns=["Shortage_sort"])

# ---------------------------
# UI
# ---------------------------
st.sidebar.header("Filtros")
components = work["component"].unique().tolist()
selected = st.sidebar.selectbox("Selecciona componente", ["(Todos)"] + components)
only_shortages = st.sidebar.checkbox("Mostrar solo componentes con shortage", value=True)

view_df = summary_df.copy()
if only_shortages:
    view_df = view_df[view_df["Shortage"] == "Sí"].copy()

colA, colB = st.columns([1.35, 1])


with colA:
    st.subheader("Resumen de shortages por componente")

    def highlight_shortage(val):
        if val == "Sí":
            return "background-color: #7a1f1f; color: white;"
        if val == "No":
            return "background-color: #1f7a3a; color: white;"
        return ""

    # OJO: view_df debe existir
    styled = view_df.style.applymap(highlight_shortage, subset=["Shortage"])
    st.dataframe(styled, use_container_width=True)

    # Descarga CSV del resumen (en el MISMO bloque colA)
    csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar resumen (CSV)",
        data=csv_bytes,
        file_name="mrp_shortage_summary.csv",
        mime="text/csv"
    )

with colB:
    st.subheader("Stats generales")
    st.metric("Componentes totales", len(work["component"].unique()))
    st.metric("Componentes con shortage", int((summary_df["Shortage"] == "Sí").sum()))
    st.caption("Tip: ordena por 'Déficit máximo' para ver lo más crítico.")

def highlight_shortage(val):
    if val == "Sí":
        return "background-color: #7a1f1f; color: white;"
    if val == "No":
        return "background-color: #1f7a3a; color: white;"
    return ""

styled = view_df.style.applymap(highlight_shortage, subset=["Shortage"])
st.dataframe(styled, use_container_width=True)
    

    # Descarga CSV del resumen
csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar resumen (CSV)",
    data=csv_bytes,
    file_name="mrp_shortage_summary.csv",
    mime="text/csv",
    key="download_summary_csv"
)

with colB:
    st.subheader("Stats generales")
    st.metric("Componentes totales", len(work["component"].unique()))
    st.metric("Componentes con shortage", int((summary_df["Shortage"] == "Sí").sum()))
    st.caption("Tip: ordena por 'Déficit máximo' para ver lo más crítico.")

st.divider()

st.subheader("Shortages por hora (visión global)")

shortage_rows = work[work["unrestricted"] < 0].copy()
if shortage_rows.empty:
    st.info("No hay filas con shortage (Unrestricted < 0) en el dataset.")
else:
    shortage_rows["hour"] = shortage_rows["dt"].dt.hour
    counts = shortage_rows.groupby("hour").size().sort_index()

    fig = plt.figure()
    plt.bar(counts.index, counts.values)
    plt.grid(axis="y", alpha=0.3)
    plt.title("Distribución de shortages por hora del día")
    plt.xlabel("Hora del día")
    plt.ylabel("Cantidad de eventos shortage")
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Detail view
# ---------------------------
if selected != "(Todos)":
    dfc = work[work["component"] == selected].copy()
    st.subheader(f"Detalle: {selected}")
    if dfc["desc"].iloc[0]:
        st.caption(dfc["desc"].iloc[0])

    neg = dfc[dfc["unrestricted"] < 0]
    if neg.empty:
        st.success("Este componente no presenta shortage (Unrestricted nunca < 0).")
    else:
        critical_time = neg.iloc[0]["dt"]
        max_def = float((-neg["unrestricted"]).max())
        st.error(f"Shortage detectado. Hora crítica: {critical_time} | Déficit máximo: {max_def:.0f} uds")

    show_cols = ["dt", "need_open", "need_acc", "unrestricted", "stock_est"]
    label_map = {
        "dt": "FechaHora",
        "need_open": "Cantidad Necesaria Abierta",
        "need_acc": "Necesidad Total Acumulada",
        "unrestricted": "Unrestricted",
        "stock_est": "Stock inicial estimado",
    }
    view = dfc[show_cols].rename(columns=label_map)
    st.dataframe(view, use_container_width=True)

    st.subheader("Gráfico: Unrestricted vs tiempo")
    fig = plt.figure()
    plt.plot(dfc["dt"], dfc["unrestricted"])
    plt.axhline(0)
    plt.xlabel("Tiempo")
    plt.ylabel("Unrestricted")
    plt.xticks(rotation=30)
    st.pyplot(fig, clear_figure=True)

else:
    st.subheader("Vista rápida (Top 20 más críticos por déficit)")
    top = view_df.sort_values("Déficit máximo (uds)", ascending=False).head(20)
    st.dataframe(top, use_container_width=True)

st.divider()
st.caption(
    "Interpretación: si 'Unrestricted' se vuelve negativo, faltan unidades para cubrir la necesidad acumulada a esa hora "
    "(riesgo de falta de material / paro de línea)."
)

