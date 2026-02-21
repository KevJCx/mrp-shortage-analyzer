import csv
import io
import unicodedata
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="MRP Shortage Analyzer", layout="wide")
st.sidebar.title("MRP Shortage Analyzer")
st.sidebar.caption("Demo version with synthetic dataset")
st.title("MRP Shortage Analyzer (Unrestricted < 0)")
st.caption(
    "Analiza exportaciones MRP y detecta faltantes por componente usando el campo 'Unrestricted'."
)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(c).strip().replace("\ufeff", "") for c in clean.columns]
    return clean


def normalize_key(text: object) -> str:
    value = str(text).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    return value


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {normalize_key(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_key(cand)
        if key in cols_lower:
            return cols_lower[key]
    return None


def _parse_number_value(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return ""

    text = text.replace(" ", "").replace("\u00a0", "").replace("'", "")
    last_dot = text.rfind(".")
    last_comma = text.rfind(",")

    if last_dot >= 0 and last_comma >= 0:
        # Decimal separator is the last seen char; the other one is thousands separator.
        if last_dot > last_comma:
            return text.replace(",", "")
        return text.replace(".", "").replace(",", ".")

    if last_comma >= 0:
        if text.count(",") > 1:
            return text.replace(",", "")
        left, right = text.split(",", 1)
        if left.isdigit() and right.isdigit() and len(right) == 3:
            return left + right
        return text.replace(",", ".")

    if last_dot >= 0:
        if text.count(".") > 1:
            return text.replace(".", "")
        left, right = text.split(".", 1)
        if left.isdigit() and right.isdigit() and len(right) == 3:
            return left + right

    return text


def to_number(series: pd.Series) -> pd.Series:
    normalized = series.apply(_parse_number_value)
    return pd.to_numeric(normalized, errors="coerce")


def parse_datetime(df: pd.DataFrame, date_col: str, time_col: str | None) -> pd.Series:
    date_str = df[date_col].astype(str).str.strip()
    if time_col:
        time_str = df[time_col].astype(str).str.strip()
        return pd.to_datetime(date_str + " " + time_str, errors="coerce", dayfirst=True)
    return pd.to_datetime(date_str, errors="coerce", dayfirst=True)


@st.cache_data(show_spinner=False)
def robust_read_table(raw_bytes: bytes) -> pd.DataFrame | None:
    encodings = ["utf-16", "utf-8-sig", "cp1252", "latin1", "utf-16le", "utf-16be"]

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
        except Exception:
            continue

        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if df is not None and df.shape[1] > 1:
                return normalize_cols(df)
        except Exception:
            pass

        try:
            sample = text[:8000]
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            df = pd.read_csv(io.StringIO(text), sep=dialect.delimiter, engine="python")
            if df is not None and df.shape[1] > 1:
                return normalize_cols(df)
        except Exception:
            pass

        for sep in ["\t", ";", ",", "|"]:
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
                if df is not None and df.shape[1] > 1:
                    return normalize_cols(df)
            except Exception:
                pass

    return None


def shortage_summary(df_comp: pd.DataFrame) -> dict[str, object]:
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


def highlight_shortage(val: str) -> str:
    if val == "Si":
        return "background-color: #7a1f1f; color: white;"
    if val == "No":
        return "background-color: #1f7a3a; color: white;"
    return ""


uploaded = st.file_uploader("Sube tu CSV de MRP", type=["csv", "txt"])
if uploaded:
    df = robust_read_table(uploaded.getvalue())
    if df is None:
        st.error(
            "No se pudo leer el archivo. Revisa encoding/separador y confirma que tenga cabeceras."
        )
        st.stop()
else:
    st.info("Usando dataset de ejemplo.")
    df = normalize_cols(pd.read_csv("data/sample.csv", sep=";"))


col_component = find_col(df, ["Componente", "Component", "Material"])
col_desc = find_col(df, ["Descripcion del componente", "Descripcion", "Description"])
col_date = find_col(df, ["Fecha de necesidad", "Need date", "Fecha"])
col_time = find_col(df, ["Hora mas temprana", "Hora", "Earliest time"])
col_need_open = find_col(
    df, ["Cantidad Necesaria Abierta", "Open required qty", "Cantidad necesaria abierta"]
)
col_need_acc = find_col(
    df,
    [
        "Necesidad Total Acumulada",
        "Total accumulated requirement",
        "Necesidad acumulada",
    ],
)
col_unres = find_col(df, ["Unrestricted", "Stock projected", "Libre utilizacion"])

required = {
    "Componente": col_component,
    "Fecha de necesidad": col_date,
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


work = df.copy()
work["component"] = work[col_component].astype(str).str.strip()
work["desc"] = work[col_desc].astype(str).str.strip() if col_desc else ""
work["need_open"] = (
    to_number(work[col_need_open]) if col_need_open else pd.Series([pd.NA] * len(work))
)
work["need_acc"] = to_number(work[col_need_acc])
work["unrestricted"] = to_number(work[col_unres])
work["dt"] = parse_datetime(work, col_date, col_time)

work = work.dropna(subset=["need_acc", "unrestricted", "dt"]).copy()
work = work[work["component"].ne("")].copy()
work = work.sort_values(["component", "dt"])
work["stock_est"] = work["unrestricted"] + work["need_acc"]

if work.empty:
    st.warning("No quedaron filas validas despues de limpiar datos.")
    st.stop()


rows = []
for comp, group in work.groupby("component", sort=True):
    info = shortage_summary(group)
    rows.append(
        {
            "Componente": comp,
            "Descripcion": group["desc"].iloc[0],
            "Shortage": "Si" if info["has_shortage"] else "No",
            "Hora critica (primera vez <0)": info["critical_time"],
            "Deficit maximo (uds)": info["max_deficit"],
            "Unrestricted minimo": info["min_unrestricted"],
            "Stock inicial estimado (mediana)": info["stock_est_median"],
            "Horas hasta shortage": info["hours_to_shortage"],
        }
    )

summary_df = pd.DataFrame(rows)
summary_df["Shortage_sort"] = (summary_df["Shortage"] == "Si").astype(int)
summary_df = summary_df.sort_values(
    by=["Shortage_sort", "Deficit maximo (uds)"], ascending=[False, False]
).drop(columns=["Shortage_sort"])


st.sidebar.header("Filtros")
components = sorted(work["component"].dropna().unique().tolist())
selected = st.sidebar.selectbox("Selecciona componente", ["(Todos)"] + components)
only_shortages = st.sidebar.checkbox("Mostrar solo componentes con shortage", value=True)

view_df = summary_df.copy()
if only_shortages:
    view_df = view_df[view_df["Shortage"] == "Si"].copy()

col_a, col_b = st.columns([1.35, 1])
with col_a:
    st.subheader("Resumen de shortages por componente")
    styled = view_df.style.applymap(highlight_shortage, subset=["Shortage"])
    st.dataframe(styled, use_container_width=True)
    st.download_button(
        "Descargar resumen (CSV)",
        data=view_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="mrp_shortage_summary.csv",
        mime="text/csv",
        key="download_summary_csv",
    )

with col_b:
    st.subheader("Stats generales")
    st.metric("Componentes totales", len(components))
    st.metric("Componentes con shortage", int((summary_df["Shortage"] == "Si").sum()))
    st.caption("Tip: ordena por 'Deficit maximo' para ver lo mas critico.")

st.divider()
st.subheader("Shortages por hora (vision global)")

shortage_rows = work[work["unrestricted"] < 0].copy()
if shortage_rows.empty:
    st.info("No hay filas con shortage (Unrestricted < 0) en el dataset.")
else:
    shortage_rows["hour"] = shortage_rows["dt"].dt.hour
    counts = shortage_rows.groupby("hour").size().sort_index()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Distribucion de shortages por hora del dia")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Cantidad de eventos shortage")
    st.pyplot(fig, clear_figure=True)

if selected != "(Todos)":
    dfc = work[work["component"] == selected].copy()
    st.subheader(f"Detalle: {selected}")
    if not dfc.empty and dfc["desc"].iloc[0]:
        st.caption(dfc["desc"].iloc[0])

    neg = dfc[dfc["unrestricted"] < 0]
    if neg.empty:
        st.success("Este componente no presenta shortage (Unrestricted nunca < 0).")
    else:
        critical_time = neg.iloc[0]["dt"]
        max_def = float((-neg["unrestricted"]).max())
        st.error(
            f"Shortage detectado. Hora critica: {critical_time} | Deficit maximo: {max_def:.0f} uds"
        )

    detail = dfc[["dt", "need_open", "need_acc", "unrestricted", "stock_est"]].rename(
        columns={
            "dt": "FechaHora",
            "need_open": "Cantidad Necesaria Abierta",
            "need_acc": "Necesidad Total Acumulada",
            "unrestricted": "Unrestricted",
            "stock_est": "Stock inicial estimado",
        }
    )
    st.dataframe(detail, use_container_width=True)

    st.subheader("Grafico: Unrestricted vs tiempo")
    fig, ax = plt.subplots()
    ax.plot(dfc["dt"], dfc["unrestricted"])
    ax.axhline(0)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Unrestricted")
    plt.xticks(rotation=30)
    st.pyplot(fig, clear_figure=True)
else:
    st.subheader("Vista rapida (Top 20 mas criticos por deficit)")
    top = view_df.sort_values("Deficit maximo (uds)", ascending=False).head(20)
    st.dataframe(top, use_container_width=True)

st.divider()
st.caption(
    "Interpretacion: si 'Unrestricted' se vuelve negativo, faltan unidades para cubrir la necesidad acumulada a esa hora "
    "(riesgo de falta de material / paro de linea)."
)
