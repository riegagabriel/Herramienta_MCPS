"""
BASILISCO — Aplicación Streamlit
Versión alineada con BASILISCO_MINI_VF.ipynb (versión final de referencia).
Motor de asignación de Centros Poblados y MCP para el Padrón Electoral Peruano.
4 corridas de búsqueda progresiva (métodos 1–7, 11–17, 21–32, 41–46).
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
.c4 { background: #3d1a2e; color: #fbb6ce; border: 1px solid #97266d; }

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

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] .stMarkdown { color: #a0aec0; }

.info-box {
    background: #1a2744;
    border: 1px solid #2b6cb0;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #90cdf4;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.success-box {
    background: #1a3a2a;
    border: 1px solid #276749;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #9ae6b4;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.warn-box {
    background: #3d2b00;
    border: 1px solid #c05621;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #f6ad55;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MOTOR BASILISCO
# Traducción fiel de BASILISCO_MINI_VF.ipynb
# ═════════════════════════════════════════════════════════════════════════════

# ── Listas de palabras a eliminar ────────────────────────────────────────────

PALABRAS_ELIMINAR_C1 = [
    "AA", "ALM", "AMPL", "BLOCK", "BQ", "CHALET", "CL", "CM", "CMDTE",
    "COOP", "CT", "CTE", "DA", "DIST", "DPTO", "EN", "ERA", "LOS", "DEL",
    "HH", "INT", "KM", "NN", "NRO", "NUCLEO POBLACIONAL", "PISO",
    "PREDIO", "RA", "SIN NUMERO", "SN", "SR", "SS", "TA", "VV", "MZ", "MZG"
]

PALABRAS_ELIMINAR_C2 = [
    "AA", "ALM", "AMPL", "BLOCK", "BQ", "CHALET", "CL", "CM", "CMDTE",
    "COOP", "CT", "CTE", "DA", "DEL", "DIST", "DPTO", "EN", "ERA",
    "HH", "INT", "KM", "LOS", "NN", "NRO", "NUCLEO POBLACIONAL", "PISO",
    "PREDIO", "RA", "SIN NUMERO", "SN", "SR", "SS", "TA", "VV", "MZ", "MZG"
]

# ── Diccionarios de reemplazos (idénticos al notebook VF) ─────────────────────

REEMPLAZOS_BASE = {
    " DE ": " DEL ",
    " 1ERO ": " 01 ", " 1RO ": " 01 ",
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
    " CARR ": " CARRETERA ",
    " CAS ": " CASERIO ",
    " CDRA ": " CUADRA ",
    " CMTE ": " COMITE ", " CMT ": " COMITE ",
    " ET ": " ETAPA ", " ETP ": " ETAPA ",
    " JR ": " JIRON ",
    " LOT ": " LOTE ", " LT ": " LOTE ", " LTE ": " LOTE ",
    " LA ": " LOS ", " LAS ": " LOS ", " LO ": " LOS ", " EL ": " LOS ",
    " ME ": " MENOR ",
    " PBLO ": " PUEBLO ", " PUEBO ": " PUEBLO ",
    " PJ ": " PUEBLO ", " PPJJ ": " PUEBLO ", " PP JJ ": " PUEBLO ",
    " PORRES ": " PORRAS ",
    " PSJ ": " PASAJE ", " PJE ": " PASAJE ", " PSJE ": " PASAJE ",
    " STA ": " SANTA ",
    " SE ": " SECTOR ", " SEC ": " SECTOR ", " SECT ": " SECTOR ",
    " URB ": " URBANIZACION ",
    " LOCALIDA ": " LOCALIDAD ",
    " LOCALIDAD ": " CASERIO ", " FUNDO ": " CASERIO ",
    " ANEXO ": " CASERIO ", " POBLADO ": " CASERIO ",
    " CAMPAMENTO ": " CASERIO ", " PUEBLO ": " CASERIO ",
    " COMUNIDAD ": " CASERIO ", " ASENTAMIENTO ": " CASERIO ",
    " VILLA ": " CASERIO ",
}

# C2/C3/C4: agrega BARRIO y SECTOR → CASERIO
REEMPLAZOS_C2_EXTRA = {
    " BARRIO ": " CASERIO ",
    " SECTOR ": " CASERIO ",
}

REEMPLAZOS_SIMILARES = {
    " JIRON ": " AVENIDA ", " PASAJE ": " AVENIDA ", " CALLE": " AVENIDA ",
    " LOTE ": " ASOCIACION ",
}

TILDES = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N", "W": "HU"}


def _normalizar_texto(texto: str, patron_str: str, extra: dict = None) -> str:
    """
    Normaliza un texto de domicilio o CP siguiendo exactamente la lógica
    del notebook BASILISCO_MINI_VF.ipynb (función normalizar_texto).
    """
    if pd.isna(texto):
        return texto
    texto = str(texto).upper()
    # Tildes, abreviaturas tempranas
    reemplazos_iniciales = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
                            "Ñ": "N", "W": "HU", "S/T": "SECTOR",
                            "AV. SN": "AVENIDA SIN NOMBRE", "VIVIENDAS": "VIVIENDA"}
    for k, v in reemplazos_iniciales.items():
        texto = texto.replace(k, v)
    texto = " " + texto
    texto = re.sub(r"\b\d\b", lambda m: "0" + m.group(), texto)
    texto = " ".join(texto.split())
    texto = re.sub(r"[^A-Z0-9]", " ", texto)
    texto = " ".join(texto.split())
    texto = " " + texto + " "
    # Reemplazos base
    for k, v in REEMPLAZOS_BASE.items():
        texto = texto.replace(k, v)
    # Reemplazos extra (C2/C3/C4)
    if extra:
        for k, v in extra.items():
            texto = texto.replace(k, v)
    # Reemplazos de similares
    for k, v in REEMPLAZOS_SIMILARES.items():
        texto = texto.replace(k, v)
    # Eliminar letras sueltas
    texto = re.sub(r"\b\w\b", "", texto)
    # Eliminar palabras ruido
    texto = re.sub(patron_str, "", texto)
    texto = " ".join(texto.split()).strip()
    return texto if texto else None


def _build_patron(palabras):
    return r"\b(" + "|".join(re.escape(p) for p in palabras) + r")\b"


_PATRON_C1 = _build_patron(PALABRAS_ELIMINAR_C1)
_PATRON_C2 = _build_patron(PALABRAS_ELIMINAR_C2)


def normalizar_c1(texto):
    return _normalizar_texto(texto, _PATRON_C1, extra=None)


def normalizar_c2(texto):
    return _normalizar_texto(texto, _PATRON_C2, extra=REEMPLAZOS_C2_EXTRA)


def normalizar_serie_c1(serie: pd.Series) -> pd.Series:
    return serie.apply(normalizar_c1)


def normalizar_serie_c2(serie: pd.Series) -> pd.Series:
    return serie.apply(normalizar_c2)


# ── Levenshtein (sin dependencia externa) ─────────────────────────────────────

def lev_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if len1 < len2:
        s1, s2, len1, len2 = s2, s1, len2, len1
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len2]


# ── Token Sort Ratio y Partial Ratio (C4) ─────────────────────────────────────

def _token_sort_ratio(a: str, b: str) -> float:
    a_tok = " ".join(sorted(a.split()))
    b_tok = " ".join(sorted(b.split()))
    longer = max(len(a_tok), len(b_tok))
    if longer == 0:
        return 100.0
    return (1 - lev_distance(a_tok, b_tok) / longer) * 100


def _partial_ratio(query: str, target: str) -> float:
    if len(query) == 0 or len(target) == 0:
        return 0.0
    if len(query) > len(target):
        query, target = target, query
    best = 0.0
    for i in range(len(target) - len(query) + 1):
        sub = target[i:i + len(query)]
        score = (1 - lev_distance(query, sub) / len(query)) * 100
        if score > best:
            best = score
    return best


# ── Preparar catálogo MCP (con normalización) ────────────────────────────────

def preparar_mcp(mcp_raw: pd.DataFrame, normalizar_fn,
                 quitar_caserio: bool = False,
                 vias_sinteticas: bool = False) -> dict:
    """
    Normaliza el catálogo MCP y devuelve un dict {UBI: DataFrame}.
    Fiel al notebook: CP sin 'CASERIO' en C3/C4, VIAS sintéticas = 'AVENIDA ' + CP.
    """
    mcp = mcp_raw.copy()
    mcp["UBI"] = mcp["UBI"].astype(str).str.strip().str.zfill(6)
    mcp["POB"] = pd.to_numeric(mcp.get("POB", 0), errors="coerce").fillna(0)

    # C3/C4: quitar "CASERIO" del campo CP antes de normalizar
    if quitar_caserio:
        mcp["CP"] = mcp["CP"].astype(str).str.replace("CASERIO", "", regex=False).str.strip()

    # C3/C4: VIAS sintéticas = "AVENIDA " + CP (ya sin "CASERIO")
    if vias_sinteticas:
        mcp["VIAS"] = "AVENIDA " + mcp["CP"].astype(str)

    mcp["MCP_normalizado"]     = mcp["MCP"].apply(normalizar_fn)
    mcp["MCP_normalizado_se"]  = mcp["MCP_normalizado"].str.replace(" ", "", regex=False)
    mcp["CP_normalizado"]      = mcp["CP"].apply(normalizar_fn)
    # Corregir doble CASERIO
    mcp["CP_normalizado"]      = mcp["CP_normalizado"].str.replace(
        r"^CASERIO\s+CASERIO", "CASERIO", regex=True)
    mcp["CP_normalizado_se"]   = mcp["CP_normalizado"].str.replace(" ", "", regex=False)
    mcp["VIAS_normalizado"]    = mcp["VIAS"].apply(normalizar_fn)
    # Deduplicar palabras en VIAS (igual que notebook: dict.fromkeys)
    mcp["VIAS_normalizado"]    = mcp["VIAS_normalizado"].apply(
        lambda x: " ".join(dict.fromkeys(x.split())) if pd.notna(x) else x)
    mcp["VIAS_normalizado_se"] = mcp["VIAS_normalizado"].str.replace(" ", "", regex=False)

    # Limpiar vacíos (igual que notebook)
    mcp.loc[mcp["CP_normalizado"] == "",   ["CP_normalizado", "CP_normalizado_se"]]     = None
    mcp["CP_normalizado"]    = mcp["CP_normalizado"].fillna("CPQUEDOVACIO")
    mcp["CP_normalizado_se"] = mcp["CP_normalizado_se"].fillna("CPQUEDOVACIO")
    mcp.loc[mcp["VIAS_normalizado"] == "", ["VIAS_normalizado", "VIAS_normalizado_se"]] = None
    mcp["VIAS_normalizado"]   = mcp["VIAS_normalizado"].fillna("NOTIENEVIASCONOCIDAS")
    mcp["VIAS_normalizado_se"] = mcp["VIAS_normalizado_se"].fillna("NOTIENEVIASCONOCIDAS")

    return {k: v.reset_index(drop=True) for k, v in mcp.groupby("UBI")}


# ── Lógica de ambigüedad (fiel al notebook VF) ────────────────────────────────

def _resolver(matches: pd.DataFrame, cod: str, campo_cp: str = "CP"):
    """
    Resuelve ambigüedad exactamente como el notebook VF.
    Retorna (CP, MCP, metodo) o None si no hay matches.
    """
    if matches.empty:
        return None

    if len(matches) == 1:
        r = matches.iloc[0]
        return (str(r["CP"]), str(r["MCP"]), cod)

    # Múltiples matches
    mcp_unique = matches["MCP"].unique()
    cod_unique  = matches["COD"].unique() if "COD" in matches.columns else ["X"]

    # Mismo COD y misma MCP → retorna sin sufijo
    if matches["COD"].nunique() == 1 and len(mcp_unique) == 1:
        r = matches.iloc[0]
        return (str(r["CP"]), str(r["MCP"]), cod)
    # Misma MCP diferente COD → _DOBLE_CP
    if len(mcp_unique) == 1:
        r = matches.iloc[0]
        return (str(r["CP"]), str(r["MCP"]), f"{cod}_DOBLE_CP")
    # Diferente MCP → INDETERMINADA / _DOBLE_MCP
    return ("DOBLE_CP", "INDETERMINADA", f"{cod}_DOBLE_MCP")


def _resolver_lev(cp_match: pd.DataFrame, cod: str):
    """Lógica de ambigüedad para resultados Levenshtein (fiel al notebook)."""
    if cp_match.empty:
        return None

    if len(cp_match) == 1:
        return (str(cp_match.iloc[0]["CP"]), str(cp_match.iloc[0]["MCP"]), cod)

    if cp_match["CP"].nunique() == 1:
        if cp_match["COD"].nunique() == 1:
            return (str(cp_match.iloc[0]["CP"]), str(cp_match.iloc[0]["MCP"]), cod)
        else:
            if cp_match["MCP"].nunique() == 1:
                return ("DOBLE_CP", str(cp_match["MCP"].unique()[0]), f"{cod}_DOBLE_CP")
            else:
                return (str(cp_match.iloc[0]["CP"]), "INDETERMINADA", f"{cod}_DOBLE_CP")
    else:
        if cp_match["MCP"].nunique() == 1:
            return ("DOBLE_CP", str(cp_match["MCP"].unique()[0]), f"{cod}_DOBLE_CP")
        else:
            return ("DOBLE_CP", "INDETERMINADA", f"{cod}_DOBLE_MCP")


def _resolver_interna(cp_interna: pd.DataFrame, cod: str):
    """
    Lógica de ambigüedad para búsquedas internas (substring/VIAS).
    Incluye desempate por POB mínima (fiel al notebook).
    """
    if cp_interna.empty:
        return None

    if len(cp_interna) == 1:
        r = cp_interna.iloc[0]
        return (str(r["CP"]), str(r["MCP"]), cod)

    if cp_interna["CP"].nunique() == 1:
        mcp_unique = cp_interna["MCP"].unique()
        if len(mcp_unique) == 1:
            r = cp_interna.iloc[0]
            return (str(r["CP"]), str(r["MCP"]), f"{cod}_DOBLE_CP")
        else:
            r = cp_interna.iloc[0]
            return (str(r["CP"]), "INDETERMINADA", f"{cod}_DOBLE_MCP")
    else:
        # Desempate por POB mínima
        cp_sel = cp_interna.loc[[cp_interna["POB"].idxmin()]]
        if cp_sel["CP"].nunique() == 1:
            r = cp_sel.iloc[0]
            return (str(r["CP"]), str(r["MCP"]), f"{cod}_MENOR_POB")
        else:
            return ("DOBLE_CP", "INDETERMINADA", f"{cod}_DOBLE_MCP")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 1 — Métodos 1–7 (normalización C1)
# Exacta con/sin espacios + Levenshtein CP + substring CP (>9 chars)
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c1(dom_n: str, dom_nse: str, ubi: str, mcp_dict: dict):
    sub = mcp_dict.get(ubi)
    if sub is None:
        return (None, None, "")

    dom_n   = dom_n   if pd.notna(dom_n)   and dom_n   != "" else ""
    dom_nse = dom_nse if pd.notna(dom_nse) and dom_nse != "" else ""

    # 1 Exacta CP (con espacios)
    if dom_n:
        m = sub[sub["CP_normalizado"] == dom_n]
        if len(m) > 0:
            r = _resolver(m, "1")
            if r: return r

    # 2 Exacta CP (sin espacios)
    if dom_nse:
        m = sub[sub["CP_normalizado_se"] == dom_nse]
        if len(m) > 0:
            r = _resolver(m, "2")
            if r: return r

    # 3 Levenshtein CP_se, len>=16, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 16].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "3")
        if r: return r

    # 4 Levenshtein CP_se, len>=12, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 12].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "4")
        if r: return r

    # 5 Levenshtein CP_se, len>=20, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 20].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "5")
        if r: return r

    # 6 Levenshtein CP_se, len>=16, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 16].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "6")
        if r: return r

    # 7 Substring CP_se (>9 chars) en DOM
    cp_interna = sub[
        (sub["CP_normalizado_se"].str.len().fillna(0) > 9) &
        sub["CP_normalizado_se"].apply(lambda x: x in dom_nse if pd.notna(x) else False)
    ]
    r = _resolver_interna(cp_interna, "7")
    if r: return r

    return (None, None, "")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 2 — Métodos 11–17 (normalización C2: +BARRIO/SECTOR → CASERIO)
# Idénticos a C1 pero con normalización ampliada y método 17 usa >8 chars
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c2(dom_n: str, dom_nse: str, ubi: str, mcp_dict: dict):
    sub = mcp_dict.get(ubi)
    if sub is None:
        return (None, None, "")

    dom_n   = dom_n   if pd.notna(dom_n)   and dom_n   != "" else ""
    dom_nse = dom_nse if pd.notna(dom_nse) and dom_nse != "" else ""

    # 11 Exacta CP (con espacios)
    if dom_n:
        m = sub[sub["CP_normalizado"] == dom_n]
        if len(m) > 0:
            r = _resolver(m, "11")
            if r: return r

    # 12 Exacta CP (sin espacios)
    if dom_nse:
        m = sub[sub["CP_normalizado_se"] == dom_nse]
        if len(m) > 0:
            r = _resolver(m, "12")
            if r: return r

    # 13 Levenshtein CP_se, len>=16, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 16].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "13")
        if r: return r

    # 14 Levenshtein CP_se, len>=12, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 12].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "14")
        if r: return r

    # 15 Levenshtein CP_se, len>=20, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 20].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "15")
        if r: return r

    # 16 Levenshtein CP_se, len>=16, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 16].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "16")
        if r: return r

    # 17 Substring CP_se (>8 chars) en DOM
    cp_interna = sub[
        (sub["CP_normalizado_se"].str.len().fillna(0) > 8) &
        sub["CP_normalizado_se"].apply(lambda x: x in dom_nse if pd.notna(x) else False)
    ]
    r = _resolver_interna(cp_interna, "17")
    if r: return r

    return (None, None, "")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 3 — Métodos 21–32 (normalización C2, CP sin "CASERIO", VIAS sintéticas)
# Levenshtein con umbrales más permisivos + búsqueda por VIAS
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c3(dom_n: str, dom_nse: str, ubi: str, mcp_dict: dict):
    sub = mcp_dict.get(ubi)
    if sub is None:
        return (None, None, "")

    dom_n   = dom_n   if pd.notna(dom_n)   and dom_n   != "" else ""
    dom_nse = dom_nse if pd.notna(dom_nse) and dom_nse != "" else ""

    # 21 Exacta CP (con espacios)
    if dom_n:
        m = sub[sub["CP_normalizado"] == dom_n]
        if len(m) > 0:
            r = _resolver(m, "21")
            if r: return r

    # 22 Exacta CP (sin espacios)
    if dom_nse:
        m = sub[sub["CP_normalizado_se"] == dom_nse]
        if len(m) > 0:
            r = _resolver(m, "22")
            if r: return r

    # 23 Levenshtein CP_se, len>=9, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 9].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "23")
        if r: return r

    # 24 Levenshtein CP_se, len>=7, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 7].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "24")
        if r: return r

    # 25 Levenshtein CP_se, len>=11, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 11].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "25")
        if r: return r

    # 26 Levenshtein CP_se, len>=9, DL<=2
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 9].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "26")
        if r: return r

    # 27 Levenshtein CP_se, len>=9, DL<=2 (igual que 26 según notebook — umbral idéntico)
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 9].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 2]
        r = _resolver_lev(m, "27")
        if r: return r

    # 28 Levenshtein CP_se, len>=5, DL<=1
    cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= 5].copy()
    if len(cands) > 0:
        cands["_d"] = cands["CP_normalizado_se"].apply(
            lambda x: lev_distance(dom_nse, x) if pd.notna(x) else 9999)
        m = cands[cands["_d"] <= 1]
        r = _resolver_lev(m, "28")
        if r: return r

    # 29 Substring VIAS (con espacios) en DOM
    if dom_n:
        cp_interna = sub[
            (sub["VIAS_normalizado"].str.len().fillna(0) > 0) &
            sub["VIAS_normalizado"].apply(
                lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dom_n
                if pd.notna(x) else False)
        ]
        r = _resolver_interna(cp_interna, "29")
        if r: return r

    # 30 Substring VIAS (sin espacios) en DOM
    if dom_nse:
        cp_interna = sub[
            (sub["VIAS_normalizado_se"].str.len().fillna(0) > 0) &
            sub["VIAS_normalizado_se"].apply(
                lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dom_nse
                if pd.notna(x) else False)
        ]
        r = _resolver_interna(cp_interna, "30")
        if r: return r

    # 31 Levenshtein VIAS (con espacios), len>=8, DL<=2
    if dom_n:
        cands = sub[sub["VIAS_normalizado"].str.len().fillna(0) >= 8].copy()
        if len(cands) > 0:
            cands["_d"] = cands["VIAS_normalizado"].apply(
                lambda x: lev_distance(dom_n, x) if pd.notna(x) else 9999)
            m = cands[cands["_d"] <= 2]
            r = _resolver_lev(m, "31")
            if r: return r

    # 32 Substring CP_se (>5 chars) en DOM
    cp_interna = sub[
        (sub["CP_normalizado_se"].str.len().fillna(0) > 5) &
        sub["CP_normalizado_se"].apply(lambda x: x in dom_nse if pd.notna(x) else False)
    ]
    r = _resolver_interna(cp_interna, "32")
    if r: return r

    return (None, None, "")


# ═════════════════════════════════════════════════════════════════════════════
# CORRIDA 4 — Métodos 41–46 (Fuzzy: Token Sort Ratio, Partial Ratio)
# Usa el mismo mcp_dict que C3
# ═════════════════════════════════════════════════════════════════════════════

def extraer_c4(dom_n: str, dom_nse: str, ubi: str, mcp_dict: dict):
    sub = mcp_dict.get(ubi)
    if sub is None:
        return (None, None, "")

    dom_n = dom_n if pd.notna(dom_n) and dom_n != "" else ""
    if not dom_n:
        return (None, None, "")

    cands_cp = sub[sub["CP_normalizado"] != "CPQUEDOVACIO"].copy()

    if not cands_cp.empty:
        # 41 Token Sort Ratio >= 85 (CP)
        cands_cp["_s"] = cands_cp["CP_normalizado"].apply(
            lambda r: _token_sort_ratio(dom_n, r))
        m = cands_cp[cands_cp["_s"] >= 85].sort_values("_s", ascending=False)
        res = _resolver_lev(m, "41_TSR85")
        if res: return res

        # 42 Partial Ratio >= 85 (CP) — "Token Set" simulado
        cands_cp["_s"] = cands_cp["CP_normalizado"].apply(
            lambda r: _partial_ratio(r, dom_n))
        m = cands_cp[cands_cp["_s"] >= 85].sort_values("_s", ascending=False)
        res = _resolver_lev(m, "42_TSE85")
        if res: return res

        # 43 Partial Ratio >= 88, CP >= 6 chars
        cands_cp["_s"] = cands_cp["CP_normalizado"].apply(
            lambda r: _partial_ratio(r, dom_n) if len(r) >= 6 else 0)
        m = cands_cp[cands_cp["_s"] >= 88].sort_values("_s", ascending=False)
        res = _resolver_lev(m, "43_PR88")
        if res: return res

        # 44 Partial Ratio >= 82, CP >= 8 chars
        cands_cp["_s"] = cands_cp["CP_normalizado"].apply(
            lambda r: _partial_ratio(r, dom_n) if len(r) >= 8 else 0)
        m = cands_cp[cands_cp["_s"] >= 82].sort_values("_s", ascending=False)
        res = _resolver_lev(m, "44_PR82")
        if res: return res

    # 45 Token Sort Ratio >= 85 (MCP)
    cands_mcp = sub.copy()
    cands_mcp["_s"] = cands_mcp["MCP_normalizado"].apply(
        lambda r: _token_sort_ratio(dom_n, r))
    m = cands_mcp[cands_mcp["_s"] >= 85].sort_values("_s", ascending=False)
    res = _resolver_lev(m, "45_MCP_TSR85")
    if res: return res

    # 46 Partial Ratio >= 88, MCP >= 6 chars
    cands_mcp["_s"] = cands_mcp["MCP_normalizado"].apply(
        lambda r: _partial_ratio(r, dom_n) if len(r) >= 6 else 0)
    m = cands_mcp[cands_mcp["_s"] >= 88].sort_values("_s", ascending=False)
    res = _resolver_lev(m, "46_MCP_PR88")
    if res: return res

    return (None, None, "")


# ── Ejecutar una corrida sobre un DataFrame ───────────────────────────────────

def ejecutar_corrida(data: pd.DataFrame, mcp_dict: dict, motor_fn,
                     corrida: int, progress_bar=None, status_text=None) -> pd.DataFrame:
    resultados = []
    total = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        cp, mcp, metodo = motor_fn(
            row.get("DOM2024_normalizado", ""),
            row.get("DOM2024_normalizado_se", ""),
            row["UBI"],
            mcp_dict
        )
        resultados.append({"CP": cp, "MCP": mcp, "metodo_CP": metodo})
        if progress_bar and i % max(1, total // 100) == 0:
            progress_bar.progress(min(i / total, 1.0))
            if status_text:
                status_text.text(f"Corrida {corrida}: {i:,} / {total:,}…")

    res = pd.DataFrame(resultados)
    out = data.copy()
    out["CP"]        = res["CP"].values
    out["MCP"]       = res["MCP"].values
    out["metodo_CP"] = res["metodo_CP"].values
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES DE LA APP
# ═════════════════════════════════════════════════════════════════════════════

def leer_padron(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, dtype=str)
    else:
        try:
            df = pd.read_csv(file, sep="|", dtype=str, encoding="latin-1")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=",", dtype=str, encoding="latin-1")
    return df


def detectar_columnas(df: pd.DataFrame):
    cols_upper = {c.upper().strip(): c for c in df.columns}

    ubi_candidatos = [
        "UBIGEO_R", "UBIGEO_DOM", "UBIGEO", "UBI",
        "COD_UBIGEO", "COD_UBI", "CODIGO_UBIGEO", "CODIGO_UBI",
        "UBIGEOR", "COD_DOM", "UBIG", "CODUBI",
    ]
    dom_candidatos = [
        "DIRECC_DOM", "DIRECCION_DOM", "DIRECC", "DIRECCION",
        "DOM2024", "DOMICILIO", "DOM", "DIRECCIÓN",
        "DIR_DOM", "DIR", "DOMICILIO_DOM", "DIRECC_R",
    ]

    def buscar(candidatos):
        for c in candidatos:
            if c in cols_upper:
                return cols_upper[c]
        for c in candidatos:
            for col_up, col_orig in cols_upper.items():
                if c in col_up:
                    return col_orig
        return None

    ubi = buscar(ubi_candidatos)
    dom = buscar(dom_candidatos)

    if ubi is None:
        for col in df.columns:
            sample = df[col].dropna().astype(str).str.strip().head(20)
            if sample.str.match(r"^\d{4,6}$").mean() > 0.7:
                ubi = col
                break

    if dom is None:
        text_cols = [(col, df[col].dropna().astype(str).str.len().mean())
                     for col in df.columns if col != ubi]
        if text_cols:
            dom = max(text_cols, key=lambda x: x[1])[0]

    return ubi, dom


def preparar_padron(df: pd.DataFrame, col_ubi: str, col_dom: str) -> pd.DataFrame:
    out = df[[col_ubi, col_dom]].copy()
    out = out.rename(columns={col_ubi: "UBI", col_dom: "DOM2024"})
    out["UBI"]    = out["UBI"].astype(str).str.strip().str.zfill(6)
    out["DOM2024"] = out["DOM2024"].astype(str).str.strip()
    out = out.dropna(subset=["DOM2024"])
    out = out[out["DOM2024"] != ""]
    out.insert(0, "ID", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def enriquecer_con_cod(data_final: pd.DataFrame, mcp_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega COD_A y MCP_COD al resultado final.
    Fiel al notebook: merge por UBI+MCP+CP para COD, luego merge por UBI+MCP para MCP_COD.
    """
    # Merge COD
    mcp_ref = (mcp_raw[["UBI", "COD", "MCP", "CP"]]
               .drop_duplicates()
               .groupby(["UBI", "MCP", "CP"], as_index=False)
               .first())
    mcp_ref["UBI"] = mcp_ref["UBI"].astype(str).str.strip().str.zfill(6)

    out = data_final.merge(mcp_ref, on=["UBI", "MCP", "CP"], how="left")

    # Merge MCP_COD (primera columna del catálogo junto con UBI y MCP)
    if "MCP_COD" in mcp_raw.columns:
        mini = (mcp_raw[["UBI", "MCP_COD", "MCP"]]
                .dropna(subset=["MCP_COD"])
                .drop_duplicates())
        mini["UBI"] = mini["UBI"].astype(str).str.strip().str.zfill(6)
        out = out.merge(mini, on=["UBI", "MCP"], how="left")

    out = out.rename(columns={"CP": "CP_A", "MCP": "MCP_A", "COD": "COD_A"})
    out["CP_A"] = out["CP_A"].astype(str).str.replace("CASERIO ", "", regex=False)
    return out


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return buf.getvalue()


def calcular_exactitud(df: pd.DataFrame, col_auto: str, col_manual: str) -> pd.DataFrame:
    """
    Módulo de validación fiel al notebook: clasificar_exactitud row-by-row.
    """
    def _clasif(row):
        autom  = row[col_auto]
        manual = row[col_manual]
        if pd.notna(autom) and pd.notna(manual) and autom == manual:
            return "c"
        elif autom == "INDETERMINADA" and pd.notna(manual):
            return "fD"
        elif pd.notna(autom) and pd.notna(manual) and autom != manual:
            return "f"
        elif pd.isna(autom) and pd.notna(manual):
            return "d"
        elif autom == "INDETERMINADA" and pd.isna(manual):
            return "eD"
        elif pd.notna(autom) and pd.isna(manual):
            return "e"
        elif pd.isna(autom) and pd.isna(manual):
            return "v"
        return None

    out = df.copy()
    out["EXACTITUD"] = out.apply(_clasif, axis=1)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="basilisco-header">
  <h1>🦎 BASILISCO</h1>
  <p>Motor de asignación de Centros Poblados · Padrón Electoral Peruano · v3.0 (VF)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MCP FIJO
# El archivo MCP_ORIGINAL.xlsx debe estar en la misma carpeta que app.py
# ─────────────────────────────────────────────────────────────────────────────

MCP_PATH = Path(__file__).parent / "MCP_ORIGINAL.xlsx"


@st.cache_resource(show_spinner="Cargando catálogo MCP…")
def cargar_mcp(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"UBI": str})
    df["UBI"] = df["UBI"].astype(str).str.strip().str.zfill(6)
    df["POB"] = pd.to_numeric(df.get("POB", 0), errors="coerce").fillna(0)
    for col in ["CP", "MCP", "VIAS"]:
        if col not in df.columns:
            df[col] = ""
    if "COD" not in df.columns:
        df["COD"] = np.nan
    return df


if not MCP_PATH.exists():
    st.error(
        f"No se encontró el catálogo MCP en `{MCP_PATH}`. "
        "Coloca el archivo **MCP_ORIGINAL.xlsx** en la misma carpeta que `app.py` y reinicia la app."
    )
    st.stop()

mcp_raw = cargar_mcp(MCP_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Opciones")
    st.markdown("---")

    st.markdown("**Filtro geográfico** *(opcional)*")
    prov_filtro = st.text_input(
        "Código de provincia (4 dígitos)",
        value="",
        placeholder="Ej: 2001",
        help="Deja vacío para procesar todos los ubigeos del archivo."
    )

    st.markdown("---")
    st.markdown("**Módulo de validación** *(opcional)*")
    val_file = st.file_uploader(
        "Revisión manual (.xlsx)",
        type=["xlsx", "xls"],
        help="Excel con columna de MCP asignada manualmente, para calcular indicadores de exactitud."
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="color:#4a5568; font-size:0.75rem; line-height:1.8;">
    <b>Catálogo activo:</b><br>
    <span style="color:#38b2ac; font-family:monospace;">MCP_ORIGINAL.xlsx</span><br>
    <span style="color:#718096;">{len(mcp_raw):,} entradas · {mcp_raw['UBI'].nunique():,} distritos</span><br><br>
    <b>Corridas del motor:</b><br>
    🔵 C1 · métodos 1–7 (normalización base)<br>
    🟢 C2 · métodos 11–17 (+ BARRIO/SECTOR)<br>
    🟠 C3 · métodos 21–32 (VIAS sintéticas)<br>
    🔴 C4 · métodos 41–46 (fuzzy TSR/PR)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — SUBIR EL PADRÓN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="step-header">① Subir sección del padrón</div>', unsafe_allow_html=True)

padron_file = st.file_uploader(
    "Selecciona el archivo de padrón (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"],
    help="Puede ser separado por | o , si es CSV."
)

if padron_file is None:
    st.markdown("""
    <div class="info-box">
    📂 &nbsp; Sube el archivo del padrón para continuar. Formatos: <b>.xlsx</b>, <b>.xls</b>, <b>.csv</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


@st.cache_data(show_spinner="Leyendo padrón…")
def leer_y_cachear(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    return leer_padron(buf)


try:
    padron_raw = leer_y_cachear(padron_file.read(), padron_file.name)
except Exception as e:
    st.error(f"Error leyendo el padrón: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — SELECCIÓN DE COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="step-header">② Confirmar columnas de ubigeo y dirección</div>',
            unsafe_allow_html=True)

col_ubi_auto, col_dom_auto = detectar_columnas(padron_raw)
todas_cols = padron_raw.columns.tolist()

col_sel_a, col_sel_b = st.columns(2)
with col_sel_a:
    idx_ubi = todas_cols.index(col_ubi_auto) if col_ubi_auto in todas_cols else 0
    col_ubi = st.selectbox("Columna de **UBIGEO**", todas_cols, index=idx_ubi)
with col_sel_b:
    idx_dom = todas_cols.index(col_dom_auto) if col_dom_auto in todas_cols else min(1, len(todas_cols) - 1)
    col_dom = st.selectbox("Columna de **DIRECCIÓN**", todas_cols, index=idx_dom)

if col_ubi == col_dom:
    st.warning("⚠️ Las columnas de ubigeo y dirección son iguales. Por favor corrígelas.")
    st.stop()

with st.expander("👁️  Vista previa del padrón — primeras 10 filas", expanded=True):
    prev = padron_raw[[col_ubi, col_dom]].head(10).copy()
    prev.columns = [f"{col_ubi}  →  UBIGEO", f"{col_dom}  →  DIRECCIÓN"]
    st.dataframe(prev, use_container_width=True, hide_index=True)

    n_total   = len(padron_raw)
    n_sin_dom = (padron_raw[col_dom].isna().sum() +
                 (padron_raw[col_dom].astype(str).str.strip() == "").sum())
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total filas", f"{n_total:,}")
    mc2.metric("Con dirección", f"{n_total - n_sin_dom:,}")
    mc3.metric("Sin dirección (se descartan)", f"{n_sin_dom:,}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — EJECUTAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
run_btn = st.button("🚀  EJECUTAR BASILISCO", use_container_width=True)

if not run_btn:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
t_inicio = time.time()

DATA_B = preparar_padron(padron_raw, col_ubi, col_dom)

# Filtro geográfico
if prov_filtro.strip():
    prov = prov_filtro.strip().zfill(4)
    mcp_raw_filt = mcp_raw[mcp_raw["UBI"].str[:4] == prov].copy()
    DATA_B = DATA_B[DATA_B["UBI"].str[:4] == prov].copy().reset_index(drop=True)
    if DATA_B.empty:
        st.error(f"No hay registros con provincia `{prov}` en el padrón cargado.")
        st.stop()
    if mcp_raw_filt.empty:
        st.error(f"No hay entradas en el MCP para la provincia `{prov}`.")
        st.stop()
else:
    mcp_raw_filt = mcp_raw.copy()

# Métricas de partida
st.markdown("---")
mi1, mi2, mi3 = st.columns(3)
mi1.markdown(f'<div class="metric-card"><div class="value">{len(DATA_B):,}</div><div class="label">Registros a procesar</div></div>', unsafe_allow_html=True)
mi2.markdown(f'<div class="metric-card"><div class="value">{len(mcp_raw_filt):,}</div><div class="label">Entradas en el MCP</div></div>', unsafe_allow_html=True)
mi3.markdown(f'<div class="metric-card"><div class="value">{DATA_B["UBI"].nunique()}</div><div class="label">Distritos únicos</div></div>', unsafe_allow_html=True)
st.markdown("---")

# ── CORRIDA 1 ──────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c1">CORRIDA 1</span> &nbsp; Exacta + Levenshtein CP (métodos 1–7) · Normalización base', unsafe_allow_html=True)
pb1 = st.progress(0)
st1 = st.empty()

DATA_B["DOM2024_normalizado"]    = normalizar_serie_c1(DATA_B["DOM2024"])
DATA_B["DOM2024_normalizado_se"] = DATA_B["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
DATA_B.loc[DATA_B["DOM2024_normalizado"] == "", ["DOM2024_normalizado", "DOM2024_normalizado_se"]] = None
DATA_B = DATA_B[DATA_B["DOM2024_normalizado"].notna()].reset_index(drop=True)

mcp_dict1 = preparar_mcp(mcp_raw_filt, normalizar_c1)
DATA_C1   = ejecutar_corrida(DATA_B, mcp_dict1, extraer_c1, 1, pb1, st1)

pb1.progress(1.0)
DATA_ok1   = DATA_C1[DATA_C1["CP"].notna()].copy()
DATA_pend1 = DATA_C1[DATA_C1["CP"].isna()].drop(
    columns=["CP", "MCP", "metodo_CP", "DOM2024_normalizado", "DOM2024_normalizado_se"],
    errors="ignore")
st1.markdown(f'<div class="success-box">✅ C1 terminada &nbsp;·&nbsp; <b>{len(DATA_ok1):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA_pend1):,}</b> pendientes</div>', unsafe_allow_html=True)

# ── CORRIDA 2 ──────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c2">CORRIDA 2</span> &nbsp; Normalización ampliada +BARRIO/SECTOR (métodos 11–17)', unsafe_allow_html=True)
pb2 = st.progress(0)
st2 = st.empty()

if not DATA_pend1.empty:
    DATA_pend1["DOM2024_normalizado"]    = normalizar_serie_c2(DATA_pend1["DOM2024"])
    DATA_pend1["DOM2024_normalizado_se"] = DATA_pend1["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
    DATA_pend1.loc[DATA_pend1["DOM2024_normalizado"] == "",
                   ["DOM2024_normalizado", "DOM2024_normalizado_se"]] = None
    DATA_pend1 = DATA_pend1[DATA_pend1["DOM2024_normalizado"].notna()].reset_index(drop=True)
    mcp_dict2  = preparar_mcp(mcp_raw_filt, normalizar_c2)
    DATA_C2    = ejecutar_corrida(DATA_pend1, mcp_dict2, extraer_c2, 2, pb2, st2)
    pb2.progress(1.0)
    DATA_ok2   = DATA_C2[DATA_C2["CP"].notna()].copy()
    DATA_pend2 = DATA_C2[DATA_C2["CP"].isna()].drop(
        columns=["CP", "MCP", "metodo_CP", "DOM2024_normalizado", "DOM2024_normalizado_se"],
        errors="ignore")
    st2.markdown(f'<div class="success-box">✅ C2 terminada &nbsp;·&nbsp; <b>{len(DATA_ok2):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA_pend2):,}</b> pendientes</div>', unsafe_allow_html=True)
else:
    DATA_ok2 = DATA_pend2 = pd.DataFrame()
    pb2.progress(1.0)
    st2.markdown('<div class="info-box">⏭️ Sin pendientes para C2.</div>', unsafe_allow_html=True)

# ── CORRIDA 3 ──────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c3">CORRIDA 3</span> &nbsp; CP sin "CASERIO" + VIAS sintéticas (métodos 21–32)', unsafe_allow_html=True)
pb3 = st.progress(0)
st3 = st.empty()

if not DATA_pend2.empty:
    DATA_pend2["DOM2024_normalizado"]    = normalizar_serie_c2(DATA_pend2["DOM2024"])
    DATA_pend2["DOM2024_normalizado_se"] = DATA_pend2["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
    DATA_pend2.loc[DATA_pend2["DOM2024_normalizado"] == "",
                   ["DOM2024_normalizado", "DOM2024_normalizado_se"]] = None
    DATA_pend2 = DATA_pend2[DATA_pend2["DOM2024_normalizado"].notna()].reset_index(drop=True)
    mcp_dict3  = preparar_mcp(mcp_raw_filt, normalizar_c2, quitar_caserio=True, vias_sinteticas=True)
    DATA_C3    = ejecutar_corrida(DATA_pend2, mcp_dict3, extraer_c3, 3, pb3, st3)
    pb3.progress(1.0)
    DATA_ok3  = DATA_C3[DATA_C3["CP"].notna()].copy()
    DATA_nulo = DATA_C3[DATA_C3["CP"].isna()].copy()
    st3.markdown(f'<div class="success-box">✅ C3 terminada &nbsp;·&nbsp; <b>{len(DATA_ok3):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA_nulo):,}</b> pendientes</div>', unsafe_allow_html=True)
else:
    DATA_ok3 = DATA_nulo = pd.DataFrame()
    pb3.progress(1.0)
    st3.markdown('<div class="info-box">⏭️ Sin pendientes para C3.</div>', unsafe_allow_html=True)

# ── CORRIDA 4 ──────────────────────────────────────────────────────────────

st.markdown('<span class="corrida-badge c4">CORRIDA 4</span> &nbsp; Fuzzy — Token Sort Ratio, Partial Ratio (métodos 41–46)', unsafe_allow_html=True)
pb4 = st.progress(0)
st4 = st.empty()

if not DATA_nulo.empty:
    DATA_pend3 = DATA_nulo.drop(
        columns=["CP", "MCP", "metodo_CP", "DOM2024_normalizado", "DOM2024_normalizado_se"],
        errors="ignore").copy()
    DATA_pend3["DOM2024_normalizado"]    = normalizar_serie_c2(DATA_pend3["DOM2024"])
    DATA_pend3["DOM2024_normalizado_se"] = DATA_pend3["DOM2024_normalizado"].fillna("").str.replace(" ", "", regex=False)
    DATA_pend3.loc[DATA_pend3["DOM2024_normalizado"] == "",
                   ["DOM2024_normalizado", "DOM2024_normalizado_se"]] = None
    DATA_pend3 = DATA_pend3[DATA_pend3["DOM2024_normalizado"].notna()].reset_index(drop=True)
    # C4 usa el mismo mcp_dict3 (quitar_caserio=True, vias_sinteticas=True)
    DATA_C4   = ejecutar_corrida(DATA_pend3, mcp_dict3, extraer_c4, 4, pb4, st4)
    pb4.progress(1.0)
    DATA_ok4   = DATA_C4[DATA_C4["CP"].notna()].copy()
    DATA_nulo4 = DATA_C4[DATA_C4["CP"].isna()].copy()
    st4.markdown(f'<div class="success-box">✅ C4 terminada &nbsp;·&nbsp; <b>{len(DATA_ok4):,}</b> asignados &nbsp;·&nbsp; <b>{len(DATA_nulo4):,}</b> sin asignar</div>', unsafe_allow_html=True)
else:
    DATA_ok4   = pd.DataFrame()
    DATA_nulo4 = DATA_nulo.copy() if not DATA_nulo.empty else pd.DataFrame()
    pb4.progress(1.0)
    st4.markdown('<div class="info-box">⏭️ Sin pendientes para C4.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDAR Y ENRIQUECER
# Fiel al notebook: concat [DATA1, DATA2, DATA3_asignados + DATA4_asignados + nulos]
# ─────────────────────────────────────────────────────────────────────────────
partes = [df for df in [DATA_ok1, DATA_ok2, DATA_ok3, DATA_ok4, DATA_nulo4] if not df.empty]
DATA_FINAL = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

for col in ["CP", "MCP", "metodo_CP"]:
    if col in DATA_FINAL.columns:
        DATA_FINAL[col] = DATA_FINAL[col].replace("", np.nan)

DATA_FINAL = enriquecer_con_cod(DATA_FINAL, mcp_raw_filt)

t_total = (time.time() - t_inicio) / 60

# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO DE VALIDACIÓN OPCIONAL
# ─────────────────────────────────────────────────────────────────────────────
if val_file:
    st.markdown("---")
    st.markdown("### 🔍 Módulo de Validación")
    try:
        val_df = pd.read_excel(val_file, dtype=str)
        vcol_a, vcol_b, vcol_c = st.columns(3)
        with vcol_a:
            col_manual = st.selectbox("Columna con MCP manual:", val_df.columns.tolist())
        with vcol_b:
            col_key_val = st.selectbox("Columna clave en revisión:", val_df.columns.tolist())
        with vcol_c:
            col_key_res = st.selectbox(
                "Columna clave en resultado BASILISCO:",
                DATA_FINAL.columns.tolist(),
                index=list(DATA_FINAL.columns).index("ID") if "ID" in DATA_FINAL.columns else 0
            )

        DATA_FINAL = DATA_FINAL.merge(
            val_df[[col_key_val, col_manual]].rename(
                columns={col_key_val: col_key_res, col_manual: "MCP_MANUAL"}),
            on=col_key_res, how="left"
        )
        col_autom = "MCP_A" if "MCP_A" in DATA_FINAL.columns else "MCP"
        DATA_FINAL = calcular_exactitud(DATA_FINAL, col_autom, "MCP_MANUAL")

        ind1 = DATA_FINAL["MCP_MANUAL"].notna().mean() * 100
        ind2 = DATA_FINAL[col_autom].notna().mean() * 100
        vi1, vi2 = st.columns(2)
        vi1.metric("Clasificado manualmente", f"{ind1:.1f}%")
        vi2.metric("Clasificado automáticamente", f"{ind2:.1f}%")

        conteos = DATA_FINAL["EXACTITUD"].value_counts()
        porcs   = DATA_FINAL["EXACTITUD"].value_counts(normalize=True) * 100
        etiquetas = {
            "c":  "✅ Correctos",     "f":  "❌ Fallos",
            "fD": "⚠️ Fallos indet.", "e":  "➕ Excesos",
            "eD": "➕ Excesos indet.","d":  "➖ Déficits",
            "v":  "⬜ Vacíos correctos"
        }
        rows_v = [{"Indicador": lbl, "N": int(conteos[k]), "%": f"{porcs[k]:.2f}%"}
                  for k, lbl in etiquetas.items() if k in conteos]
        st.dataframe(pd.DataFrame(rows_v), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error en módulo de validación: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Resultados")

total     = len(DATA_FINAL)
asignados = DATA_FINAL["CP_A"].notna().sum() if "CP_A" in DATA_FINAL.columns else 0
sin_asig  = total - asignados
tasa      = asignados / total * 100 if total else 0

rm1, rm2, rm3, rm4, rm5 = st.columns(5)
rm1.markdown(f'<div class="metric-card"><div class="value">{total:,}</div><div class="label">Total</div></div>', unsafe_allow_html=True)
rm2.markdown(f'<div class="metric-card"><div class="value">{asignados:,}</div><div class="label">Asignados</div></div>', unsafe_allow_html=True)
rm3.markdown(f'<div class="metric-card"><div class="value">{sin_asig:,}</div><div class="label">Sin asignar</div></div>', unsafe_allow_html=True)
rm4.markdown(f'<div class="metric-card"><div class="value">{tasa:.1f}%</div><div class="label">Tasa asignación</div></div>', unsafe_allow_html=True)
rm5.markdown(f'<div class="metric-card"><div class="value">{t_total:.1f} min</div><div class="label">Tiempo total</div></div>', unsafe_allow_html=True)

# Distribución por corrida
st.markdown("---")
corridas_info = [
    ("🔵 C1 (1–7)",   DATA_ok1),
    ("🟢 C2 (11–17)", DATA_ok2),
    ("🟠 C3 (21–32)", DATA_ok3),
    ("🔴 C4 (fuzzy)", DATA_ok4),
]
rows_c = []
for nombre, df in corridas_info:
    n = len(df) if not df.empty else 0
    rows_c.append({
        "Corrida": nombre,
        "Asignados": f"{n:,}",
        "% del total": f"{n/total*100:.1f}%" if total else "0%"
    })
n_nulo = len(DATA_nulo4) if not DATA_nulo4.empty else 0
rows_c.append({"Corrida": "⚫ Sin asignar", "Asignados": f"{n_nulo:,}",
               "% del total": f"{n_nulo/total*100:.1f}%" if total else "0%"})
st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABLA INTERACTIVA CON FILTROS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔎 Explorar resultados")

f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
with f1:
    f_ubi = st.text_input("Filtrar por UBI", placeholder="Ej: 200110")
with f2:
    f_dom = st.text_input("Filtrar por dirección", placeholder="Texto libre en DOM2024")
with f3:
    metodos_disp = (["Todos"] + sorted(DATA_FINAL["metodo_CP"].dropna().unique().tolist())
                    if "metodo_CP" in DATA_FINAL.columns else ["Todos"])
    f_met = st.selectbox("Método", metodos_disp)
with f4:
    f_estado = st.selectbox("Estado", ["Todos", "Asignados", "Sin asignar"])

df_view = DATA_FINAL.copy()
if f_ubi.strip():
    df_view = df_view[df_view["UBI"].str.contains(f_ubi.strip(), na=False)]
if f_dom.strip():
    df_view = df_view[df_view["DOM2024"].str.upper().str.contains(f_dom.strip().upper(), na=False)]
if f_met != "Todos" and "metodo_CP" in df_view.columns:
    df_view = df_view[df_view["metodo_CP"] == f_met]
if f_estado == "Asignados" and "CP_A" in df_view.columns:
    df_view = df_view[df_view["CP_A"].notna()]
elif f_estado == "Sin asignar" and "CP_A" in df_view.columns:
    df_view = df_view[df_view["CP_A"].isna()]

COLS_SHOW = ["ID", "UBI", "DOM2024", "CP_A", "MCP_A", "COD_A", "metodo_CP"]
if "MCP_COD" in df_view.columns:    COLS_SHOW.append("MCP_COD")
if "EXACTITUD" in df_view.columns:  COLS_SHOW.append("EXACTITUD")
if "MCP_MANUAL" in df_view.columns: COLS_SHOW.append("MCP_MANUAL")
cols_ok = [c for c in COLS_SHOW if c in df_view.columns]

st.caption(f"Mostrando **{len(df_view):,}** registros de **{total:,}**")
st.dataframe(
    df_view[cols_ok].reset_index(drop=True),
    use_container_width=True,
    height=420,
    column_config={
        "ID":        st.column_config.NumberColumn("ID", width="small"),
        "UBI":       st.column_config.TextColumn("UBIGEO", width="small"),
        "DOM2024":   st.column_config.TextColumn("DIRECCIÓN", width="large"),
        "CP_A":      st.column_config.TextColumn("CP asignado"),
        "MCP_A":     st.column_config.TextColumn("MCP asignada"),
        "COD_A":     st.column_config.TextColumn("COD", width="small"),
        "metodo_CP": st.column_config.TextColumn("Método", width="small"),
        "EXACTITUD": st.column_config.TextColumn("Exactitud", width="small"),
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⬇️ Descargar")

d1, d2, d3 = st.columns(3)

with d1:
    st.download_button(
        label="📥 Resultado completo (.xlsx)",
        data=to_excel_bytes(DATA_FINAL[cols_ok] if cols_ok else DATA_FINAL),
        file_name=f"BASILISCO_{padron_file.name.split('.')[0]}_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with d2:
    df_nulos = (DATA_FINAL[DATA_FINAL["CP_A"].isna()][cols_ok]
                if "CP_A" in DATA_FINAL.columns else pd.DataFrame())
    if not df_nulos.empty:
        st.download_button(
            label="📥 Sin asignar (.xlsx)",
            data=to_excel_bytes(df_nulos),
            file_name=f"BASILISCO_{padron_file.name.split('.')[0]}_sin_asignar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.success("🎉 ¡Todos los registros fueron asignados!")

with d3:
    if not df_view.empty:
        st.download_button(
            label="📥 Vista filtrada (.xlsx)",
            data=to_excel_bytes(df_view[cols_ok].reset_index(drop=True)),
            file_name=f"BASILISCO_{padron_file.name.split('.')[0]}_filtrado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
