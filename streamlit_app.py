"""
BASILISCO — Aplicación Streamlit
Traducción fiel de BASILISCO_MINI_VF.ipynb.
3 corridas: C1 (1–7), C2 (11–17), C3 (21–32).
Sin C4 fuzzy — no existe en el notebook VF.
"""

import io
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BASILISCO",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0f1117; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.basilisco-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #2d3748;
    border-left: 4px solid #38b2ac;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.basilisco-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    color: #38b2ac;
    font-size: 2rem;
    margin: 0;
    letter-spacing: 0.05em;
}
.basilisco-header p { color: #718096; margin: 0.3rem 0 0; font-size: 0.9rem; }

.metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #38b2ac;
}
.metric-card .label {
    color: #718096;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.corrida-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
.c1 { background: #1a365d; color: #90cdf4; border: 1px solid #2b6cb0; }
.c2 { background: #1a3a2a; color: #9ae6b4; border: 1px solid #276749; }
.c3 { background: #3d2b00; color: #f6ad55; border: 1px solid #c05621; }

.step-header {
    font-family: 'IBM Plex Mono', monospace;
    color: #38b2ac;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.stProgress > div > div > div { background-color: #38b2ac; }

.stButton > button {
    background: #38b2ac;
    color: #0f1117;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover { background: #4fd1c5; transform: translateY(-1px); }
.stDataFrame { border-radius: 8px; overflow: hidden; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #2d3748; }

.info-box {
    background: #1a2744; border: 1px solid #2b6cb0; border-radius: 6px;
    padding: 0.8rem 1rem; color: #90cdf4; font-size: 0.85rem; margin-bottom: 1rem;
}
.success-box {
    background: #1a3a2a; border: 1px solid #276749; border-radius: 6px;
    padding: 0.8rem 1rem; color: #9ae6b4; font-size: 0.85rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MOTOR BASILISCO — traducción fiel de BASILISCO_MINI_VF.ipynb
# ═════════════════════════════════════════════════════════════════════════════

# ── Palabras a eliminar ───────────────────────────────────────────────────────

PALABRAS_C1 = [
    "AA", "ALM", "AMPL", "BLOCK", "BQ", "CHALET", "CL", "CM", "CMDTE",
    "COOP", "CT", "CTE", "DA", "DIST", "DPTO", "EN", "ERA", "LOS", "DEL",
    "HH", "INT", "KM", "NN", "NRO", "NUCLEO POBLACIONAL", "PISO",
    "PREDIO", "RA", "SIN NUMERO", "SN", "SR", "SS", "TA", "VV", "MZ", "MZG"
]
PALABRAS_C2 = [
    "AA", "ALM", "AMPL", "BLOCK", "BQ", "CHALET", "CL", "CM", "CMDTE",
    "COOP", "CT", "CTE", "DA", "DEL", "DIST", "DPTO", "EN", "ERA",
    "HH", "INT", "KM", "LOS", "NN", "NRO", "NUCLEO POBLACIONAL", "PISO",
    "PREDIO", "RA", "SIN NUMERO", "SN", "SR", "SS", "TA", "VV", "MZ", "MZG"
]

_PAT_C1 = r"\b(" + "|".join(re.escape(p) for p in PALABRAS_C1) + r")\b"
_PAT_C2 = r"\b(" + "|".join(re.escape(p) for p in PALABRAS_C2) + r")\b"

# ── Reemplazos (idénticos al notebook VF) ─────────────────────────────────────

_R0 = {
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
    "Ñ": "N", "W": "HU", "S/T": "SECTOR",
    "AV. SN": "AVENIDA SIN NOMBRE", "VIVIENDAS": "VIVIENDA"
}

_R2_BASE = {
    " DE ": " DEL ", " 1ERO ": " 01 ", " 1RO ": " 01 ",
    " SR ": " SENOR ",
    " AGR ": " AGRUPACION ", " AGRP ": " AGRUPACION ", " AGRUP ": " AGRUPACION ",
    " AAH ": " ASENTAMIENTO ", " AAHH ": " ASENTAMIENTO ", " AH ": " ASENTAMIENTO ",
    " AA H ": " ASENTAMIENTO ", " AA HH ": " ASENTAMIENTO ", " A H ": " ASENTAMIENTO ",
    " ASENT H ": " ASENTAMIENTO ", " ASENTH ": " ASENTAMIENTO ",
    " AASENT ": " ASENTAMIENTO ", " ASENT ": " ASENTAMIENTO ",
    " AV ": " AVENIDA ",
    " AAVV ": " ASOCIACION ", " AAVVV ": " ASOCIACION ",
    " AA VV ": " ASOCIACION ", " AA VVV ": " ASOCIACION ", " AA VVS ": " ASOCIACION ",
    " ASOC PRO VIVIENDA ": " ASOCIACION ", " ASOC DE VIVIENDA ": " ASOCIACION ",
    " ASOC VIVIENDA ": " ASOCIACION ", " ASOC VIV ": " ASOCIACION ",
    " APV ": " ASOCIACION ", " ASOCIACION PRO VIVIENDA ": " ASOCIACION ",
    " PRO VIVIENDA ": " ASOCIACION ", " ASOC ": " ASOCIACION ",
    " ANRXO ": " ANEXO ", " ANAEXO ": " ANEXO ",
    " BARR ": " BARRIO ", " BR ": " BARRIO ", " BRRIO ": " BARRIO ",
    " CAP ": " CAPITAN ",
    " COM NATIVA ": " COMUNIDAD ", " COM NAT ": " COMUNIDAD ", " COM ": " COMUNIDAD ",
    " COMUNID CAMPESINA ": " COMUNIDAD ", " COMUN ": " COMUNIDAD ",
    " COMUNID NAT ": " COMUNIDAD ",
    " CC ": " COMUNIDAD ", " CMUNIDAD CAMPESINA ": " COMUNIDAD ",
    " C CAM ": " COMUNIDAD ", " C CAMPESINA ": " COMUNIDAD ",
    " COMUNID CAMP ": " COMUNIDAD ", " COMUNID ": " COMUNIDAD ",
    " COMUNIDCAMPESINA ": " COMUNIDAD ", " COMUNNIDAD ": " COMUNIDAD ",
    " COMUNIDA ": " COMUNIDAD ",
    " CC NN ": " COMUNIDAD NATIVA ", " CN ": " COMUNIDAD NATIVA ",
    " CCNN ": " COMUNIDAD NATIVA ", " C N ": " COMUNIDAD NATIVA ",
    " COMUNIDAD NATIVA ": " COMUNIDAD ",
    " CENTRO POBLADO MAYOR ": " POBLADO ", " CENTRO POBLADO MENOR ": " POBLADO ",
    " CENTRO POBLADO ME ": " POBLADO ", " CENTRO POBLADO MEN ": " POBLADO ",
    " CENTRO POBLADO ": " POBLADO ",
    " C POBLADO ": " POBLADO ", " C.POBLADO ": " POBLADO ", " C P ": " POBLADO ",
    " CP ": " POBLADO ", " CP DE ": " POBLADO ", " CPDE ": " POBLADO ",
    " CPM ": " POBLADO ", " CPME ": " POBLADO ", " CPMEN ": " POBLADO ",
    " CA ": " CALLE ", " CALL ": " CALLE ",
    " CAMP ": " CAMPAMENTO ", " CAMP MINERO ": " CAMPAMENTO ",
    " CARR ": " CARRETERA ", " CAS ": " CASERIO ",
    " CDRA ": " CUADRA ", " CMTE ": " COMITE ", " CMT ": " COMITE ",
    " ET ": " ETAPA ", " ETP ": " ETAPA ", " JR ": " JIRON ",
    " LOT ": " LOTE ", " LT ": " LOTE ", " LTE ": " LOTE ",
    " LA ": " LOS ", " LAS ": " LOS ", " LO ": " LOS ", " EL ": " LOS ",
    " ME ": " MENOR ",
    " PBLO ": " PUEBLO ", " PUEBO ": " PUEBLO ",
    " PJ ": " PUEBLO ", " PPJJ ": " PUEBLO ", " PP JJ ": " PUEBLO ",
    " PORRES ": " PORRAS ",
    " PSJ ": " PASAJE ", " PJE ": " PASAJE ", " PSJE ": " PASAJE ",
    " STA ": " SANTA ",
    " SE ": " SECTOR ", " SEC ": " SECTOR ", " SECT ": " SECTOR ",
    " URB ": " URBANIZACION ", " LOCALIDA ": " LOCALIDAD ",
    " LOCALIDAD ": " CASERIO ", " FUNDO ": " CASERIO ",
    " ANEXO ": " CASERIO ", " POBLADO ": " CASERIO ",
    " CAMPAMENTO ": " CASERIO ", " PUEBLO ": " CASERIO ",
    " COMUNIDAD ": " CASERIO ", " ASENTAMIENTO ": " CASERIO ",
    " VILLA ": " CASERIO ",
}

# C2/C3 agrega BARRIO y SECTOR → CASERIO
_R2_C2_EXTRA = {" BARRIO ": " CASERIO ", " SECTOR ": " CASERIO "}

_R3 = {
    " JIRON ": " AVENIDA ", " PASAJE ": " AVENIDA ", " CALLE": " AVENIDA ",
    " LOTE ": " ASOCIACION ",
}


def _normalizar(texto, patron: str, extra: dict = None):
    """normalizar_texto del notebook VF, parametrizada."""
    if pd.isna(texto):
        return texto
    texto = str(texto).upper()
    for k, v in _R0.items():
        texto = texto.replace(k, v)
    texto = " " + texto
    texto = re.sub(r"\b\d\b", lambda m: "0" + m.group(), texto)
    texto = " ".join(texto.split())
    texto = re.sub(r"[^A-Z0-9]", " ", texto)
    texto = " ".join(texto.split())
    texto = " " + texto + " "
    for k, v in _R2_BASE.items():
        texto = texto.replace(k, v)
    if extra:
        for k, v in extra.items():
            texto = texto.replace(k, v)
    for k, v in _R3.items():
        texto = texto.replace(k, v)
    texto = re.sub(r"\b\w\b", "", texto)
    texto = re.sub(patron, "", texto)
    texto = " ".join(texto.split()).strip()
    return texto if texto else None


def norm_c1(t): return _normalizar(t, _PAT_C1, extra=None)
def norm_c2(t): return _normalizar(t, _PAT_C2, extra=_R2_C2_EXTRA)


# ── Levenshtein (sin dependencia externa) ─────────────────────────────────────

def lev(s1: str, s2: str) -> int:
    if s1 == s2: return 0
    l1, l2 = len(s1), len(s2)
    if l1 == 0: return l2
    if l2 == 0: return l1
    if l1 < l2: s1, s2, l1, l2 = s2, s1, l2, l1
    prev = list(range(l2 + 1))
    for i in range(1, l1 + 1):
        curr = [i] + [0] * l2
        for j in range(1, l2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
        prev = curr
    return prev[l2]


# ── Preparar catálogo MCP ─────────────────────────────────────────────────────

def preparar_mcp(mcp_raw: pd.DataFrame, norm_fn,
                 quitar_caserio=False, vias_sinteticas=False) -> dict:
    m = mcp_raw.copy()
    m["UBI"] = m["UBI"].astype(str).str.strip().str.zfill(6)
    m["POB"] = pd.to_numeric(m.get("POB", 0), errors="coerce").fillna(0)
    if quitar_caserio:
        m["CP"] = m["CP"].astype(str).str.replace("CASERIO", "", regex=False).str.strip()
    if vias_sinteticas:
        m["VIAS"] = "AVENIDA " + m["CP"].astype(str)

    m["MCP_normalizado"]     = m["MCP"].apply(norm_fn)
    m["MCP_normalizado_se"]  = m["MCP_normalizado"].str.replace(" ", "", regex=False)
    m["CP_normalizado"]      = m["CP"].apply(norm_fn)
    m["CP_normalizado"]      = m["CP_normalizado"].str.replace(
        r"^CASERIO\s+CASERIO", "CASERIO", regex=True)
    m["CP_normalizado_se"]   = m["CP_normalizado"].str.replace(" ", "", regex=False)
    m["VIAS_normalizado"]    = m["VIAS"].apply(norm_fn)
    m["VIAS_normalizado"]    = m["VIAS_normalizado"].apply(
        lambda x: " ".join(dict.fromkeys(x.split())) if pd.notna(x) else x)
    m["VIAS_normalizado_se"] = m["VIAS_normalizado"].str.replace(" ", "", regex=False)

    m.loc[m["CP_normalizado"] == "",   ["CP_normalizado", "CP_normalizado_se"]]     = None
    m["CP_normalizado"]    = m["CP_normalizado"].fillna("CPQUEDOVACIO")
    m["CP_normalizado_se"] = m["CP_normalizado_se"].fillna("CPQUEDOVACIO")
    m.loc[m["VIAS_normalizado"] == "", ["VIAS_normalizado", "VIAS_normalizado_se"]] = None
    m["VIAS_normalizado"]   = m["VIAS_normalizado"].fillna("NOTIENEVIASCONOCIDAS")
    m["VIAS_normalizado_se"] = m["VIAS_normalizado_se"].fillna("NOTIENEVIASCONOCIDAS")

    return {k: v.reset_index(drop=True) for k, v in m.groupby("UBI")}


# ── Lógica de ambigüedad ──────────────────────────────────────────────────────

def _res_exacta(matches, cod):
    if matches.empty: return None
    if len(matches) == 1:
        r = matches.iloc[0]; return (str(r["CP"]), str(r["MCP"]), cod)
    mu = matches["MCP"].unique()
    if matches["COD"].nunique() == 1 and len(mu) == 1:
        r = matches.iloc[0]; return (str(r["CP"]), str(r["MCP"]), cod)
    if len(mu) == 1:
        r = matches.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{cod}_DOBLE_CP")
    r = matches.iloc[0]; return (str(r["CP"]), "INDETERMINADA", f"{cod}_DOBLE_MCP")


def _res_lev(m, cod):
    if m.empty: return None
    if len(m) == 1:
        return (str(m.iloc[0]["CP"]), str(m.iloc[0]["MCP"]), cod)
    if m["CP"].nunique() == 1:
        if m["COD"].nunique() == 1:
            return (str(m.iloc[0]["CP"]), str(m.iloc[0]["MCP"]), cod)
        if m["MCP"].nunique() == 1:
            return ("DOBLE_CP", str(m["MCP"].unique()[0]), f"{cod}_DOBLE_CP")
        return (str(m.iloc[0]["CP"]), "INDETERMINADA", f"{cod}_DOBLE_CP")
    if m["MCP"].nunique() == 1:
        return ("DOBLE_CP", str(m["MCP"].unique()[0]), f"{cod}_DOBLE_CP")
    return ("DOBLE_CP", "INDETERMINADA", f"{cod}_DOBLE_MCP")


def _res_interna(ci, cod):
    if ci.empty: return None
    if len(ci) == 1:
        r = ci.iloc[0]; return (str(r["CP"]), str(r["MCP"]), cod)
    if ci["CP"].nunique() == 1:
        mu = ci["MCP"].unique()
        if len(mu) == 1:
            r = ci.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{cod}_DOBLE_CP")
        r = ci.iloc[0]; return (str(r["CP"]), "INDETERMINADA", f"{cod}_DOBLE_MCP")
    sel = ci.loc[[ci["POB"].idxmin()]]
    if sel["CP"].nunique() == 1:
        r = sel.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{cod}_MENOR_POB")
    return ("DOBLE_CP", "INDETERMINADA", f"{cod}_DOBLE_MCP")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 1 — métodos 1–7
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c1(dn, dnse, ubi, md):
    sub = md.get(ubi)
    if sub is None: return (None, None, "")
    dn   = dn   if pd.notna(dn)   and dn   != "" else ""
    dnse = dnse if pd.notna(dnse) and dnse != "" else ""

    if dn:
        r = _res_exacta(sub[sub["CP_normalizado"] == dn], "1")
        if r: return r
    if dnse:
        r = _res_exacta(sub[sub["CP_normalizado_se"] == dnse], "2")
        if r: return r
    for mn, dl, cd in [(16,1,"3"),(12,1,"4"),(20,2,"5"),(16,2,"6")]:
        c = sub[sub["CP_normalizado"].str.len().fillna(0) >= mn].copy()
        if len(c):
            c["_d"] = c["CP_normalizado_se"].apply(lambda x: lev(dnse,x) if pd.notna(x) else 9999)
            r = _res_lev(c[c["_d"] <= dl], cd)
            if r: return r
    ci = sub[(sub["CP_normalizado_se"].str.len().fillna(0) > 9) &
             sub["CP_normalizado_se"].apply(lambda x: x in dnse if pd.notna(x) else False)]
    r = _res_interna(ci, "7")
    if r: return r
    return (None, None, "")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 2 — métodos 11–17
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c2(dn, dnse, ubi, md):
    sub = md.get(ubi)
    if sub is None: return (None, None, "")
    dn   = dn   if pd.notna(dn)   and dn   != "" else ""
    dnse = dnse if pd.notna(dnse) and dnse != "" else ""

    if dn:
        r = _res_exacta(sub[sub["CP_normalizado"] == dn], "11")
        if r: return r
    if dnse:
        r = _res_exacta(sub[sub["CP_normalizado_se"] == dnse], "12")
        if r: return r
    for mn, dl, cd in [(16,1,"13"),(12,1,"14"),(20,2,"15"),(16,2,"16")]:
        c = sub[sub["CP_normalizado"].str.len().fillna(0) >= mn].copy()
        if len(c):
            c["_d"] = c["CP_normalizado_se"].apply(lambda x: lev(dnse,x) if pd.notna(x) else 9999)
            r = _res_lev(c[c["_d"] <= dl], cd)
            if r: return r
    ci = sub[(sub["CP_normalizado_se"].str.len().fillna(0) > 8) &
             sub["CP_normalizado_se"].apply(lambda x: x in dnse if pd.notna(x) else False)]
    r = _res_interna(ci, "17")
    if r: return r
    return (None, None, "")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 3 — métodos 21–32
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c3(dn, dnse, ubi, md):
    sub = md.get(ubi)
    if sub is None: return (None, None, "")
    dn   = dn   if pd.notna(dn)   and dn   != "" else ""
    dnse = dnse if pd.notna(dnse) and dnse != "" else ""

    if dn:
        r = _res_exacta(sub[sub["CP_normalizado"] == dn], "21")
        if r: return r
    if dnse:
        r = _res_exacta(sub[sub["CP_normalizado_se"] == dnse], "22")
        if r: return r
    for mn, dl, cd in [(9,1,"23"),(7,1,"24"),(11,2,"25"),(9,2,"26"),(9,2,"27"),(5,1,"28")]:
        c = sub[sub["CP_normalizado"].str.len().fillna(0) >= mn].copy()
        if len(c):
            c["_d"] = c["CP_normalizado_se"].apply(lambda x: lev(dnse,x) if pd.notna(x) else 9999)
            r = _res_lev(c[c["_d"] <= dl], cd)
            if r: return r
    # 29 VIAS con espacios
    if dn:
        ci = sub[(sub["VIAS_normalizado"].str.len().fillna(0) > 0) &
                 sub["VIAS_normalizado"].apply(
                     lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dn if pd.notna(x) else False)]
        r = _res_interna(ci, "29")
        if r: return r
    # 30 VIAS sin espacios
    if dnse:
        ci = sub[(sub["VIAS_normalizado_se"].str.len().fillna(0) > 0) &
                 sub["VIAS_normalizado_se"].apply(
                     lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dnse if pd.notna(x) else False)]
        r = _res_interna(ci, "30")
        if r: return r
    # 31 Levenshtein VIAS con espacios, len>=8, DL<=2
    if dn:
        c = sub[sub["VIAS_normalizado"].str.len().fillna(0) >= 8].copy()
        if len(c):
            c["_d"] = c["VIAS_normalizado"].apply(lambda x: lev(dn,x) if pd.notna(x) else 9999)
            r = _res_lev(c[c["_d"] <= 2], "31")
            if r: return r
    # 32 Substring CP_se >5 chars
    ci = sub[(sub["CP_normalizado_se"].str.len().fillna(0) > 5) &
             sub["CP_normalizado_se"].apply(lambda x: x in dnse if pd.notna(x) else False)]
    r = _res_interna(ci, "32")
    if r: return r
    return (None, None, "")


# ── Ejecutar corrida ──────────────────────────────────────────────────────────

def ejecutar_corrida(data, mcp_dict, motor_fn, num, pb=None, st_txt=None):
    resultados = []
    total = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        cp, mcp, met = motor_fn(
            row.get("DOM2024_normalizado", ""),
            row.get("DOM2024_normalizado_se", ""),
            row["UBI"], mcp_dict)
        resultados.append({"CP": cp, "MCP": mcp, "metodo_CP": met})
        if pb and i % max(1, total // 100) == 0:
            pb.progress(min(i / total, 1.0))
            if st_txt: st_txt.text(f"Corrida {num}: {i:,} / {total:,}…")
    res = pd.DataFrame(resultados)
    out = data.copy()
    out["CP"] = res["CP"].values
    out["MCP"] = res["MCP"].values
    out["metodo_CP"] = res["metodo_CP"].values
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def fmt_int(x, n):
    """Formatea como entero de n dígitos — igual que el notebook VF."""
    if pd.isna(x) or str(x).strip() == "": return None
    try: return f"{int(float(x)):0{n}d}"
    except (ValueError, TypeError): return None


def leer_padron(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, dtype=str)
    try:
        return pd.read_csv(file, sep="|", dtype=str, encoding="latin-1")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=",", dtype=str, encoding="latin-1")


def detectar_columnas(df):
    cu = {c.upper().strip(): c for c in df.columns}
    ubi_c = ["UBIGEO_R","UBIGEO_DOM","UBIGEO","UBI","COD_UBIGEO","COD_UBI",
             "UBIGEOR","COD_DOM","UBIG","CODUBI"]
    dom_c = ["DIRECC_DOM","DIRECCION_DOM","DIRECC","DIRECCION","DOM2024",
             "DOMICILIO","DOM","DIRECCIÓN","DIR_DOM","DIR"]
    def buscar(cands):
        for c in cands:
            if c in cu: return cu[c]
        for c in cands:
            for ck, co in cu.items():
                if c in ck: return co
        return None
    ubi = buscar(ubi_c)
    dom = buscar(dom_c)
    if ubi is None:
        for col in df.columns:
            if df[col].dropna().astype(str).str.strip().head(20).str.match(r"^\d{4,6}$").mean() > 0.7:
                ubi = col; break
    if dom is None:
        tc = [(c, df[c].dropna().astype(str).str.len().mean()) for c in df.columns if c != ubi]
        if tc: dom = max(tc, key=lambda x: x[1])[0]
    return ubi, dom


def preparar_padron(df, col_ubi, col_dom, ubis_filtro=None):
    """
    Limpieza fiel al notebook VF + filtro isin si se indicaron ubigeos.
    """
    out = df[[col_ubi, col_dom]].copy()
    out = out.rename(columns={col_ubi: "UBI", col_dom: "DOM2024"})
    out["UBI"]     = out["UBI"].apply(lambda x: fmt_int(x, 6))
    out["DOM2024"] = out["DOM2024"].astype(str).str.strip()
    out = out.dropna(subset=["UBI", "DOM2024"])
    out = out[out["DOM2024"].str.strip() != ""]
    # Filtro UBI — equivalente a DATA_B = DATA_B[DATA_B["UBI"].isin([...])]
    if ubis_filtro:
        norms = [fmt_int(u, 6) for u in ubis_filtro]
        norms = [u for u in norms if u]
        if norms:
            out = out[out["UBI"].isin(norms)]
    out.insert(0, "ID", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def enriquecer_con_cod(data_final, mcp_raw):
    """
    Fiel al notebook: merge MCPm (UBI+MCP+CP → COD),
    luego merge MINI (UBI+MCP → MCP_COD).
    """
    mcpm = (mcp_raw[["UBI","COD","MCP","CP"]]
            .drop_duplicates()
            .groupby(["UBI","MCP","CP"], as_index=False).first())
    mcpm["UBI"] = mcpm["UBI"].astype(str).str.strip().str.zfill(6)
    out = data_final.merge(mcpm, on=["UBI","MCP","CP"], how="left")

    if "MCP_COD" in mcp_raw.columns:
        mini = (mcp_raw[["UBI","MCP_COD","MCP"]]
                .dropna(subset=["MCP_COD"]).drop_duplicates())
        mini["UBI"] = mini["UBI"].astype(str).str.strip().str.zfill(6)
        out = out.merge(mini, on=["UBI","MCP"], how="left")

    out = out.rename(columns={"CP":"CP_A","MCP":"MCP_A","COD":"COD_A"})
    out["CP_A"] = out["CP_A"].astype(str).str.replace("CASERIO ", "", regex=False)
    return out


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Resultados")
    return buf.getvalue()


def calcular_exactitud(df, col_auto, col_manual):
    """clasificar_exactitud del notebook VF."""
    def _cl(row):
        a, m = row[col_auto], row[col_manual]
        if pd.notna(a) and pd.notna(m) and a == m:   return "c"
        if a == "INDETERMINADA" and pd.notna(m):      return "fD"
        if pd.notna(a) and pd.notna(m) and a != m:   return "f"
        if pd.isna(a)  and pd.notna(m):               return "d"
        if a == "INDETERMINADA" and pd.isna(m):       return "eD"
        if pd.notna(a) and pd.isna(m):                return "e"
        if pd.isna(a)  and pd.isna(m):                return "v"
        return None
    out = df.copy(); out["EXACTITUD"] = out.apply(_cl, axis=1)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="basilisco-header">
  <h1>🦎 BASILISCO</h1>
  <p>Motor de asignación de Centros Poblados · Padrón Electoral Peruano</p>
</div>
""", unsafe_allow_html=True)

# ── Cargar MCP fijo ───────────────────────────────────────────────────────────

MCP_PATH = Path(__file__).parent / "MCP_ORIGINAL.xlsx"

@st.cache_resource(show_spinner="Cargando catálogo MCP…")
def cargar_mcp(path):
    df = pd.read_excel(path, dtype={"UBI": str})
    df["UBI"] = df["UBI"].astype(str).str.strip().str.zfill(6)
    df["POB"] = pd.to_numeric(df.get("POB", 0), errors="coerce").fillna(0)
    for col in ["CP","MCP","VIAS"]:
        if col not in df.columns: df[col] = ""
    if "COD" not in df.columns: df["COD"] = np.nan
    return df

if not MCP_PATH.exists():
    st.error(f"No se encontró `{MCP_PATH}`. Coloca **MCP_ORIGINAL.xlsx** junto a `app.py`.")
    st.stop()

mcp_raw = cargar_mcp(MCP_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Opciones")
    st.markdown("---")

    st.markdown("**Filtro por UBIGEO** *(opcional)*")
    st.caption(
        "Uno o más ubigeos de 6 dígitos separados por coma. "
        "Equivale a `DATA_B[DATA_B['UBI'].isin([...])]`. "
        "Vacío = procesa **todos** los ubigeos del archivo."
    )
    ubi_input = st.text_input(
        "Ubigeos a procesar",
        value="",
        placeholder="Ej: 200110, 200111, 201202"
    )

    st.markdown("---")
    st.markdown("**Módulo de validación** *(opcional)*")
    val_file = st.file_uploader(
        "Revisión manual (.xlsx)", type=["xlsx","xls"],
        help="Excel con MCP asignada manualmente para calcular indicadores de exactitud."
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="color:#4a5568;font-size:0.75rem;line-height:1.8;">
    <b>Catálogo activo:</b><br>
    <span style="color:#38b2ac;font-family:monospace;">MCP_ORIGINAL.xlsx</span><br>
    <span style="color:#718096;">{len(mcp_raw):,} entradas · {mcp_raw['UBI'].nunique():,} distritos</span><br><br>
    <b>Corridas del motor:</b><br>
    🔵 C1 · métodos 1–7<br>
    🟢 C2 · métodos 11–17<br>
    🟠 C3 · métodos 21–32
    </div>
    """, unsafe_allow_html=True)

# ── Paso 1: subir padrón ──────────────────────────────────────────────────────

st.markdown('<div class="step-header">① Subir sección del padrón</div>', unsafe_allow_html=True)

padron_file = st.file_uploader(
    "Selecciona el archivo de padrón (.xlsx, .xls, .csv)",
    type=["xlsx","xls","csv"]
)

if padron_file is None:
    st.markdown("""
    <div class="info-box">
    📂 &nbsp; Sube el archivo del padrón para continuar.
    Formatos: <b>.xlsx</b>, <b>.xls</b>, <b>.csv</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data(show_spinner="Leyendo padrón…")
def leer_y_cachear(fb, fn):
    buf = io.BytesIO(fb); buf.name = fn
    return leer_padron(buf)

try:
    padron_raw = leer_y_cachear(padron_file.read(), padron_file.name)
except Exception as e:
    st.error(f"Error leyendo el padrón: {e}"); st.stop()

# ── Paso 2: confirmar columnas ────────────────────────────────────────────────

st.markdown('<div class="step-header">② Confirmar columnas de ubigeo y dirección</div>',
            unsafe_allow_html=True)

cua, cud = detectar_columnas(padron_raw)
cols = padron_raw.columns.tolist()

cs1, cs2 = st.columns(2)
with cs1:
    col_ubi = st.selectbox("Columna de **UBIGEO**", cols,
                           index=cols.index(cua) if cua in cols else 0)
with cs2:
    col_dom = st.selectbox("Columna de **DIRECCIÓN**", cols,
                           index=cols.index(cud) if cud in cols else min(1, len(cols)-1))

if col_ubi == col_dom:
    st.warning("⚠️ Las columnas de ubigeo y dirección son iguales. Corrígelas."); st.stop()

with st.expander("👁️  Vista previa — primeras 10 filas", expanded=True):
    prev = padron_raw[[col_ubi, col_dom]].head(10).copy()
    prev.columns = [f"{col_ubi} → UBIGEO", f"{col_dom} → DIRECCIÓN"]
    st.dataframe(prev, use_container_width=True, hide_index=True)
    nt = len(padron_raw)
    ns = padron_raw[col_dom].isna().sum() + (padron_raw[col_dom].astype(str).str.strip() == "").sum()
    a, b, c_ = st.columns(3)
    a.metric("Total filas", f"{nt:,}")
    b.metric("Con dirección", f"{nt-ns:,}")
    c_.metric("Sin dirección (se descartan)", f"{ns:,}")

# ── Paso 3: ejecutar ──────────────────────────────────────────────────────────

st.markdown("---")
run_btn = st.button("🚀  EJECUTAR BASILISCO", use_container_width=True)
if not run_btn: st.stop()

# ── Procesamiento ─────────────────────────────────────────────────────────────

t0 = time.time()

ubis_filtro = [u.strip() for u in ubi_input.split(",") if u.strip()] if ubi_input.strip() else []

DATA_B = preparar_padron(padron_raw, col_ubi, col_dom, ubis_filtro=ubis_filtro)

if DATA_B.empty:
    st.error("No quedaron registros tras aplicar el filtro de ubigeo."); st.stop()

# Filtrar MCP con los mismos ubigeos — igual que notebook:
# MCP_B = MCP_B[MCP_B["UBI"].isin([...])]
if ubis_filtro:
    unorms = [fmt_int(u, 6) for u in ubis_filtro]
    unorms = [u for u in unorms if u]
    mcp_filt = mcp_raw[mcp_raw["UBI"].isin(unorms)].copy()
    if mcp_filt.empty:
        st.error(f"Sin entradas en MCP para: {', '.join(unorms)}"); st.stop()
else:
    mcp_filt = mcp_raw.copy()

# Métricas de partida
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.markdown(f'<div class="metric-card"><div class="value">{len(DATA_B):,}</div><div class="label">Registros a procesar</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="value">{len(mcp_filt):,}</div><div class="label">Entradas en el MCP</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div class="value">{DATA_B["UBI"].nunique()}</div><div class="label">Distritos únicos</div></div>', unsafe_allow_html=True)
st.markdown("---")

# ─── CORRIDA 1 ────────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c1">CORRIDA 1</span> &nbsp; Exacta + Levenshtein CP · métodos 1–7', unsafe_allow_html=True)
pb1 = st.progress(0); st1 = st.empty()

DATA_B["DOM2024_normalizado"]    = DATA_B["DOM2024"].apply(norm_c1)
DATA_B["DOM2024_normalizado_se"] = DATA_B["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
DATA_B.loc[DATA_B["DOM2024_normalizado"].isna() | (DATA_B["DOM2024_normalizado"] == ""),
           ["DOM2024_normalizado", "DOM2024_normalizado_se"]] = None
DATA_B = DATA_B[DATA_B["DOM2024_normalizado"].notna()].reset_index(drop=True)

md1    = preparar_mcp(mcp_filt, norm_c1)
DATA_C1 = ejecutar_corrida(DATA_B, md1, extraer_c1, 1, pb1, st1)
pb1.progress(1.0)

DATA1   = DATA_C1[DATA_C1["CP"].notna()].copy()
DATA2_in = DATA_C1[DATA_C1["CP"].isna()].drop(
    columns=["CP","MCP","metodo_CP","DOM2024_normalizado","DOM2024_normalizado_se"],
    errors="ignore")
st1.markdown(f'<div class="success-box">✅ C1 &nbsp;·&nbsp; <b>{len(DATA1):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA2_in):,}</b> pendientes</div>', unsafe_allow_html=True)

# ─── CORRIDA 2 ────────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c2">CORRIDA 2</span> &nbsp; +BARRIO/SECTOR → CASERIO · métodos 11–17', unsafe_allow_html=True)
pb2 = st.progress(0); st2 = st.empty()

if not DATA2_in.empty:
    DATA2_in["DOM2024_normalizado"]    = DATA2_in["DOM2024"].apply(norm_c2)
    DATA2_in["DOM2024_normalizado_se"] = DATA2_in["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
    DATA2_in.loc[DATA2_in["DOM2024_normalizado"].isna() | (DATA2_in["DOM2024_normalizado"] == ""),
                 ["DOM2024_normalizado","DOM2024_normalizado_se"]] = None
    DATA2_in = DATA2_in[DATA2_in["DOM2024_normalizado"].notna()].reset_index(drop=True)
    md2     = preparar_mcp(mcp_filt, norm_c2)
    DATA_C2  = ejecutar_corrida(DATA2_in, md2, extraer_c2, 2, pb2, st2)
    pb2.progress(1.0)
    DATA2    = DATA_C2[DATA_C2["CP"].notna()].copy()
    DATA3_in = DATA_C2[DATA_C2["CP"].isna()].drop(
        columns=["CP","MCP","metodo_CP","DOM2024_normalizado","DOM2024_normalizado_se"],
        errors="ignore")
    st2.markdown(f'<div class="success-box">✅ C2 &nbsp;·&nbsp; <b>{len(DATA2):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA3_in):,}</b> pendientes</div>', unsafe_allow_html=True)
else:
    DATA2 = DATA3_in = pd.DataFrame()
    pb2.progress(1.0)
    st2.markdown('<div class="info-box">⏭️ Sin pendientes para C2.</div>', unsafe_allow_html=True)

# ─── CORRIDA 3 ────────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c3">CORRIDA 3</span> &nbsp; CP sin "CASERIO" + VIAS sintéticas · métodos 21–32', unsafe_allow_html=True)
pb3 = st.progress(0); st3 = st.empty()

if not DATA3_in.empty:
    DATA3_in["DOM2024_normalizado"]    = DATA3_in["DOM2024"].apply(norm_c2)
    DATA3_in["DOM2024_normalizado_se"] = DATA3_in["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
    DATA3_in.loc[DATA3_in["DOM2024_normalizado"].isna() | (DATA3_in["DOM2024_normalizado"] == ""),
                 ["DOM2024_normalizado","DOM2024_normalizado_se"]] = None
    DATA3_in = DATA3_in[DATA3_in["DOM2024_normalizado"].notna()].reset_index(drop=True)
    md3      = preparar_mcp(mcp_filt, norm_c2, quitar_caserio=True, vias_sinteticas=True)
    DATA_C3  = ejecutar_corrida(DATA3_in, md3, extraer_c3, 3, pb3, st3)
    pb3.progress(1.0)
    DATA3_ok   = DATA_C3[DATA_C3["CP"].notna()].copy()
    DATA3_nulo = DATA_C3[DATA_C3["CP"].isna()].copy()
    st3.markdown(f'<div class="success-box">✅ C3 &nbsp;·&nbsp; <b>{len(DATA3_ok):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA3_nulo):,}</b> sin asignar</div>', unsafe_allow_html=True)
else:
    DATA3_ok = DATA3_nulo = pd.DataFrame()
    pb3.progress(1.0)
    st3.markdown('<div class="info-box">⏭️ Sin pendientes para C3.</div>', unsafe_allow_html=True)

# ─── Consolidar — concat([DATA1, DATA2, DATA3]) fiel al notebook ──────────────

DATA3 = pd.concat([df for df in [DATA3_ok, DATA3_nulo] if not df.empty],
                  ignore_index=True) if (not DATA3_ok.empty or not DATA3_nulo.empty) else pd.DataFrame()

partes = [df for df in [DATA1, DATA2, DATA3] if not df.empty]
DATA_FINAL = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

for col in ["CP","MCP","metodo_CP"]:
    if col in DATA_FINAL.columns:
        DATA_FINAL[col] = DATA_FINAL[col].replace("", np.nan)

DATA_FINAL = enriquecer_con_cod(DATA_FINAL, mcp_filt)
t_total = (time.time() - t0) / 60

# ── Módulo de validación ──────────────────────────────────────────────────────

if val_file:
    st.markdown("---"); st.markdown("### 🔍 Módulo de Validación")
    try:
        val_df = pd.read_excel(val_file, dtype=str)
        vc1, vc2, vc3 = st.columns(3)
        with vc1: col_manual  = st.selectbox("Columna MCP manual:", val_df.columns.tolist())
        with vc2: col_key_val = st.selectbox("Columna clave en revisión:", val_df.columns.tolist())
        with vc3: col_key_res = st.selectbox(
            "Columna clave en resultado:", DATA_FINAL.columns.tolist(),
            index=list(DATA_FINAL.columns).index("ID") if "ID" in DATA_FINAL.columns else 0)

        DATA_FINAL = DATA_FINAL.merge(
            val_df[[col_key_val, col_manual]].rename(
                columns={col_key_val: col_key_res, col_manual: "MCP_MANUAL"}),
            on=col_key_res, how="left")
        col_autom = "MCP_A" if "MCP_A" in DATA_FINAL.columns else "MCP"
        DATA_FINAL = calcular_exactitud(DATA_FINAL, col_autom, "MCP_MANUAL")

        i1 = DATA_FINAL["MCP_MANUAL"].notna().mean() * 100
        i2 = DATA_FINAL[col_autom].notna().mean() * 100
        vi1, vi2 = st.columns(2)
        vi1.metric("Clasificado manualmente", f"{i1:.1f}%")
        vi2.metric("Clasificado automáticamente", f"{i2:.1f}%")

        cnt = DATA_FINAL["EXACTITUD"].value_counts()
        pct = DATA_FINAL["EXACTITUD"].value_counts(normalize=True) * 100
        labs = {"c":"✅ Correctos","f":"❌ Fallos","fD":"⚠️ Fallos indet.",
                "e":"➕ Excesos","eD":"➕ Excesos indet.","d":"➖ Déficits","v":"⬜ Vacíos correctos"}
        rows_v = [{"Indicador": lbl, "N": int(cnt[k]), "%": f"{pct[k]:.2f}%"}
                  for k, lbl in labs.items() if k in cnt]
        st.dataframe(pd.DataFrame(rows_v), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error en módulo de validación: {e}")

# ── Resultados ────────────────────────────────────────────────────────────────

st.markdown("---"); st.markdown("### 📊 Resultados")

total     = len(DATA_FINAL)
asignados = DATA_FINAL["CP_A"].notna().sum() if "CP_A" in DATA_FINAL.columns else 0
sin_asig  = total - asignados
tasa      = asignados / total * 100 if total else 0

r1,r2,r3,r4,r5 = st.columns(5)
for col_st, val, lbl in [
    (r1, f"{total:,}",    "Total"),
    (r2, f"{asignados:,}","Asignados"),
    (r3, f"{sin_asig:,}", "Sin asignar"),
    (r4, f"{tasa:.1f}%",  "Tasa asignación"),
    (r5, f"{t_total:.1f} min","Tiempo total"),
]:
    col_st.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("---")
rows_c = []
for nom, df in [("🔵 C1 (1–7)", DATA1), ("🟢 C2 (11–17)", DATA2), ("🟠 C3 (21–32)", DATA3_ok)]:
    n = len(df) if not df.empty else 0
    rows_c.append({"Corrida": nom, "Asignados": f"{n:,}",
                   "% del total": f"{n/total*100:.1f}%" if total else "0%"})
n_nulo = len(DATA3_nulo) if not DATA3_nulo.empty else 0
rows_c.append({"Corrida": "⚫ Sin asignar", "Asignados": f"{n_nulo:,}",
               "% del total": f"{n_nulo/total*100:.1f}%" if total else "0%"})
st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)

# ── Explorar resultados ───────────────────────────────────────────────────────

st.markdown("---"); st.markdown("### 🔎 Explorar resultados")

f1,f2,f3,f4 = st.columns([2,3,2,2])
with f1: fu = st.text_input("Filtrar por UBI", placeholder="Ej: 200110")
with f2: fd = st.text_input("Filtrar por dirección", placeholder="Texto libre")
with f3:
    mets = (["Todos"] + sorted(DATA_FINAL["metodo_CP"].dropna().unique().tolist())
            if "metodo_CP" in DATA_FINAL.columns else ["Todos"])
    fm = st.selectbox("Método", mets)
with f4: fe = st.selectbox("Estado", ["Todos","Asignados","Sin asignar"])

dv = DATA_FINAL.copy()
if fu.strip(): dv = dv[dv["UBI"].str.contains(fu.strip(), na=False)]
if fd.strip(): dv = dv[dv["DOM2024"].str.upper().str.contains(fd.strip().upper(), na=False)]
if fm != "Todos" and "metodo_CP" in dv.columns: dv = dv[dv["metodo_CP"] == fm]
if fe == "Asignados"   and "CP_A" in dv.columns: dv = dv[dv["CP_A"].notna()]
if fe == "Sin asignar" and "CP_A" in dv.columns: dv = dv[dv["CP_A"].isna()]

COLS = ["ID","UBI","DOM2024","CP_A","MCP_A","COD_A","metodo_CP"]
if "MCP_COD"    in dv.columns: COLS.append("MCP_COD")
if "EXACTITUD"  in dv.columns: COLS.append("EXACTITUD")
if "MCP_MANUAL" in dv.columns: COLS.append("MCP_MANUAL")
cols_ok = [c for c in COLS if c in dv.columns]

st.caption(f"Mostrando **{len(dv):,}** registros de **{total:,}**")
st.dataframe(dv[cols_ok].reset_index(drop=True), use_container_width=True, height=420,
    column_config={
        "ID":        st.column_config.NumberColumn("ID", width="small"),
        "UBI":       st.column_config.TextColumn("UBIGEO", width="small"),
        "DOM2024":   st.column_config.TextColumn("DIRECCIÓN", width="large"),
        "CP_A":      st.column_config.TextColumn("CP asignado"),
        "MCP_A":     st.column_config.TextColumn("MCP asignada"),
        "COD_A":     st.column_config.TextColumn("COD", width="small"),
        "metodo_CP": st.column_config.TextColumn("Método", width="small"),
        "EXACTITUD": st.column_config.TextColumn("Exactitud", width="small"),
    })

# ── Descargas ─────────────────────────────────────────────────────────────────

st.markdown("---"); st.markdown("### ⬇️ Descargar")
base = padron_file.name.split(".")[0]
d1, d2, d3 = st.columns(3)

with d1:
    st.download_button("📥 Resultado completo (.xlsx)",
        data=to_excel_bytes(DATA_FINAL[cols_ok] if cols_ok else DATA_FINAL),
        file_name=f"BASILISCO_{base}_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
with d2:
    df_nulos = DATA_FINAL[DATA_FINAL["CP_A"].isna()][cols_ok] \
               if "CP_A" in DATA_FINAL.columns else pd.DataFrame()
    if not df_nulos.empty:
        st.download_button("📥 Sin asignar (.xlsx)",
            data=to_excel_bytes(df_nulos),
            file_name=f"BASILISCO_{base}_sin_asignar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    else:
        st.success("🎉 ¡Todos los registros fueron asignados!")
with d3:
    if not dv.empty:
        st.download_button("📥 Vista filtrada (.xlsx)",
            data=to_excel_bytes(dv[cols_ok].reset_index(drop=True)),
            file_name=f"BASILISCO_{base}_filtrado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
