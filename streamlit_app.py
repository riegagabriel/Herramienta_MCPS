"""
BASILISCO — Aplicación Streamlit
Traducción fiel de BASILISCO_MINI_VF.ipynb
3 corridas: C1 (1–7), C2 (11–17), C3 (21–32)
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
# PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BASILISCO",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0f1117; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.bheader {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #2d3748; border-left: 4px solid #38b2ac;
    border-radius: 8px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.bheader h1 { font-family: 'IBM Plex Mono', monospace; color: #38b2ac;
    font-size: 2rem; margin: 0; letter-spacing: 0.05em; }
.bheader p { color: #718096; margin: 0.3rem 0 0; font-size: 0.9rem; }

.mcard { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px;
    padding: 1rem 1.2rem; text-align: center; }
.mcard .val { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    font-weight: 600; color: #38b2ac; }
.mcard .lbl { color: #718096; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 0.2rem; }

.cbadge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; }
.c1 { background:#1a365d; color:#90cdf4; border:1px solid #2b6cb0; }
.c2 { background:#1a3a2a; color:#9ae6b4; border:1px solid #276749; }
.c3 { background:#3d2b00; color:#f6ad55; border:1px solid #c05621; }

.step-hdr { font-family: 'IBM Plex Mono', monospace; color: #38b2ac;
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #2d3748; padding-bottom: 0.5rem; margin-bottom: 1rem; }

.ibox { background:#1a2744; border:1px solid #2b6cb0; border-radius:6px;
    padding:0.8rem 1rem; color:#90cdf4; font-size:0.85rem; margin-bottom:1rem; }
.sbox { background:#1a3a2a; border:1px solid #276749; border-radius:6px;
    padding:0.8rem 1rem; color:#9ae6b4; font-size:0.85rem; margin-bottom:1rem; }

.stProgress > div > div > div { background-color: #38b2ac; }
.stButton > button { background:#38b2ac; color:#0f1117; font-weight:700;
    border:none; border-radius:6px; padding:0.6rem 2rem;
    font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
    letter-spacing:0.05em; transition:all 0.2s; }
.stButton > button:hover { background:#4fd1c5; transform:translateY(-1px); }
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #2d3748; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MOTOR — traducción exacta del notebook VF
# ═════════════════════════════════════════════════════════════════════════════

def fmt_int(x, n):
    if pd.isna(x) or str(x).strip() == "": return None
    try:
        v = int(float(x))
        return f"{v:0{n}d}" if n > 0 else str(v)
    except Exception:
        return None


# ── Levenshtein puro Python (sin rapidfuzz) ───────────────────────────────────
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


# ── Normalización ─────────────────────────────────────────────────────────────

_R0 = {
    "Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U",
    "Ñ":"N","W":"HU","S/T":"SECTOR",
    "AV. SN":"AVENIDA SIN NOMBRE","VIVIENDAS":"VIVIENDA"
}
_R2_BASE = {
    " DE ":" DEL "," 1ERO ":" 01 "," 1RO ":" 01 "," SR ":" SENOR ",
    " AGR ":" AGRUPACION "," AGRP ":" AGRUPACION "," AGRUP ":" AGRUPACION ",
    " AAH ":" ASENTAMIENTO "," AAHH ":" ASENTAMIENTO "," AH ":" ASENTAMIENTO ",
    " AA H ":" ASENTAMIENTO "," AA HH ":" ASENTAMIENTO "," A H ":" ASENTAMIENTO ",
    " ASENT H ":" ASENTAMIENTO "," ASENTH ":" ASENTAMIENTO ",
    " AASENT ":" ASENTAMIENTO "," ASENT ":" ASENTAMIENTO ",
    " AV ":" AVENIDA ",
    " AAVV ":" ASOCIACION "," AAVVV ":" ASOCIACION ",
    " AA VV ":" ASOCIACION "," AA VVV ":" ASOCIACION "," AA VVS ":" ASOCIACION ",
    " ASOC PRO VIVIENDA ":" ASOCIACION "," ASOC DE VIVIENDA ":" ASOCIACION ",
    " ASOC VIVIENDA ":" ASOCIACION "," ASOC VIV ":" ASOCIACION ",
    " APV ":" ASOCIACION "," ASOCIACION PRO VIVIENDA ":" ASOCIACION ",
    " PRO VIVIENDA ":" ASOCIACION "," ASOC ":" ASOCIACION ",
    " ANRXO ":" ANEXO "," ANAEXO ":" ANEXO ",
    " BARR ":" BARRIO "," BR ":" BARRIO "," BRRIO ":" BARRIO ",
    " CAP ":" CAPITAN ",
    " COM NATIVA ":" COMUNIDAD "," COM NAT ":" COMUNIDAD "," COM ":" COMUNIDAD ",
    " COMUNID CAMPESINA ":" COMUNIDAD "," COMUN ":" COMUNIDAD ",
    " COMUNID NAT ":" COMUNIDAD "," CC ":" COMUNIDAD ",
    " CMUNIDAD CAMPESINA ":" COMUNIDAD "," C CAM ":" COMUNIDAD ",
    " C CAMPESINA ":" COMUNIDAD "," COMUNID CAMP ":" COMUNIDAD ",
    " COMUNID ":" COMUNIDAD "," COMUNIDCAMPESINA ":" COMUNIDAD ",
    " COMUNNIDAD ":" COMUNIDAD "," COMUNIDA ":" COMUNIDAD ",
    " CC NN ":" COMUNIDAD NATIVA "," CN ":" COMUNIDAD NATIVA ",
    " CCNN ":" COMUNIDAD NATIVA "," C N ":" COMUNIDAD NATIVA ",
    " COMUNIDAD NATIVA ":" COMUNIDAD ",
    " CENTRO POBLADO MAYOR ":" POBLADO "," CENTRO POBLADO MENOR ":" POBLADO ",
    " CENTRO POBLADO ME ":" POBLADO "," CENTRO POBLADO MEN ":" POBLADO ",
    " CENTRO POBLADO ":" POBLADO "," C POBLADO ":" POBLADO ",
    " C.POBLADO ":" POBLADO "," C P ":" POBLADO "," CP ":" POBLADO ",
    " CP DE ":" POBLADO "," CPDE ":" POBLADO "," CPM ":" POBLADO ",
    " CPME ":" POBLADO "," CPMEN ":" POBLADO ",
    " CA ":" CALLE "," CALL ":" CALLE ",
    " CAMP ":" CAMPAMENTO "," CAMP MINERO ":" CAMPAMENTO ",
    " CARR ":" CARRETERA "," CAS ":" CASERIO ",
    " CDRA ":" CUADRA "," CMTE ":" COMITE "," CMT ":" COMITE ",
    " ET ":" ETAPA "," ETP ":" ETAPA "," JR ":" JIRON ",
    " LOT ":" LOTE "," LT ":" LOTE "," LTE ":" LOTE ",
    " LA ":" LOS "," LAS ":" LOS "," LO ":" LOS "," EL ":" LOS ",
    " ME ":" MENOR ",
    " PBLO ":" PUEBLO "," PUEBO ":" PUEBLO ",
    " PJ ":" PUEBLO "," PPJJ ":" PUEBLO "," PP JJ ":" PUEBLO ",
    " PORRES ":" PORRAS ",
    " PSJ ":" PASAJE "," PJE ":" PASAJE "," PSJE ":" PASAJE ",
    " STA ":" SANTA ",
    " SE ":" SECTOR "," SEC ":" SECTOR "," SECT ":" SECTOR ",
    " URB ":" URBANIZACION "," LOCALIDA ":" LOCALIDAD ",
    " LOCALIDAD ":" CASERIO "," FUNDO ":" CASERIO ",
    " ANEXO ":" CASERIO "," POBLADO ":" CASERIO ",
    " CAMPAMENTO ":" CASERIO "," PUEBLO ":" CASERIO ",
    " COMUNIDAD ":" CASERIO "," ASENTAMIENTO ":" CASERIO ",
    " VILLA ":" CASERIO ",
}
_R2_C2_EXTRA = {" BARRIO ":" CASERIO "," SECTOR ":" CASERIO "}
_R3 = {" JIRON ":" AVENIDA "," PASAJE ":" AVENIDA "," CALLE":" AVENIDA "," LOTE ":" ASOCIACION "}

_PAL_C1 = ["AA","ALM","AMPL","BLOCK","BQ","CHALET","CL","CM","CMDTE","COOP","CT","CTE",
           "DA","DIST","DPTO","EN","ERA","LOS","DEL","HH","INT","KM","NN","NRO",
           "NUCLEO POBLACIONAL","PISO","PREDIO","RA","SIN NUMERO","SN","SR","SS","TA","VV","MZ","MZG"]
_PAL_C2 = ["AA","ALM","AMPL","BLOCK","BQ","CHALET","CL","CM","CMDTE","COOP","CT","CTE",
           "DA","DEL","DIST","DPTO","EN","ERA","HH","INT","KM","LOS","NN","NRO",
           "NUCLEO POBLACIONAL","PISO","PREDIO","RA","SIN NUMERO","SN","SR","SS","TA","VV","MZ","MZG"]

_PAT_C1 = r"\b(" + "|".join(re.escape(p) for p in _PAL_C1) + r")\b"
_PAT_C2 = r"\b(" + "|".join(re.escape(p) for p in _PAL_C2) + r")\b"


def _norm(texto, patron, extra=None):
    if pd.isna(texto): return texto
    texto = str(texto).upper()
    for k, v in _R0.items(): texto = texto.replace(k, v)
    texto = " " + texto
    texto = re.sub(r"\b\d\b", lambda m: "0"+m.group(), texto)
    texto = " ".join(texto.split())
    texto = re.sub(r"[^A-Z0-9]", " ", texto)
    texto = " ".join(texto.split())
    texto = " " + texto + " "
    for k, v in _R2_BASE.items(): texto = texto.replace(k, v)
    if extra:
        for k, v in extra.items(): texto = texto.replace(k, v)
    for k, v in _R3.items(): texto = texto.replace(k, v)
    texto = re.sub(r"\b\w\b", "", texto)
    texto = re.sub(patron, "", texto)
    texto = " ".join(texto.split()).strip()
    return texto if texto else None

def norm_c1(t): return _norm(t, _PAT_C1)
def norm_c2(t): return _norm(t, _PAT_C2, extra=_R2_C2_EXTRA)


# ── Preparar MCP (fiel al notebook) ──────────────────────────────────────────

def preparar_mcp(mcp_raw, norm_fn, quitar_caserio=False, vias_sinteticas=False):
    m = mcp_raw.copy()
    m["UBI"] = m["UBI"].astype(str).str.strip().apply(lambda x: fmt_int(x, 6))
    m["POB"] = pd.to_numeric(m["POB"] if "POB" in m.columns else 0, errors="coerce").fillna(0)

    if "VIAS" not in m.columns: m["VIAS"] = ""
    m["VIAS"] = m["VIAS"].fillna("")

    if "MCP_COD" in m.columns:
        m["MCP_COD"] = m["MCP_COD"].astype(str).str.replace(r"\.0$","",regex=True).apply(
            lambda x: fmt_int(x, 9))
    if "COD" in m.columns:
        m["COD"] = m["COD"].astype(str).str.replace(r"\.0$","",regex=True)

    if quitar_caserio:
        m["CP"] = m["CP"].astype(str).str.replace("CASERIO","",regex=False).str.strip()
    if vias_sinteticas:
        m["VIAS"] = "AVENIDA " + m["CP"].astype(str)

    m["MCP_normalizado"]     = m["MCP"].apply(norm_fn)
    m["MCP_normalizado_se"]  = m["MCP_normalizado"].str.replace(" ","",regex=False)
    m["CP_normalizado"]      = m["CP"].apply(norm_fn)
    m["CP_normalizado"]      = m["CP_normalizado"].str.replace(
        r"^CASERIO\s+CASERIO","CASERIO",regex=True)
    m["CP_normalizado_se"]   = m["CP_normalizado"].str.replace(" ","",regex=False)
    m["VIAS_normalizado"]    = m["VIAS"].apply(norm_fn)
    m["VIAS_normalizado"]    = m["VIAS_normalizado"].apply(
        lambda x: " ".join(dict.fromkeys(x.split())) if pd.notna(x) else x)
    m["VIAS_normalizado_se"] = m["VIAS_normalizado"].str.replace(" ","",regex=False)

    m.loc[m["CP_normalizado"]=="",   ["CP_normalizado","CP_normalizado_se"]]    = None
    m["CP_normalizado"]    = m["CP_normalizado"].fillna("CPQUEDOVACIO")
    m["CP_normalizado_se"] = m["CP_normalizado_se"].fillna("CPQUEDOVACIO")
    m.loc[m["VIAS_normalizado"]=="", ["VIAS_normalizado","VIAS_normalizado_se"]] = None
    m["VIAS_normalizado"]    = m["VIAS_normalizado"].fillna("NOTIENEVIASCONOCIDAS")
    m["VIAS_normalizado_se"] = m["VIAS_normalizado_se"].fillna("NOTIENEVIASCONOCIDAS")

    return m, {k: v.reset_index(drop=True) for k, v in m.groupby("UBI")}


# ── Función de extracción genérica ────────────────────────────────────────────

def _resolver_interna(ci, num):
    if len(ci) == 0: return None
    if len(ci) == 1:
        r = ci.iloc[0]; return (str(r["CP"]), str(r["MCP"]), num)
    if ci["CP"].nunique() == 1:
        mu = ci["MCP"].unique()
        if len(mu) == 1:
            r = ci.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{num}_DOBLE_CP")
        r = ci.iloc[0]; return (str(r["CP"]), "INDETERMINADA", f"{num}_DOBLE_MCP")
    sel = ci.loc[[ci["POB"].idxmin()]]
    if sel["CP"].nunique() == 1:
        r = sel.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{num}_MENOR_POB")
    return ("DOBLE_CP","INDETERMINADA",f"{num}_DOBLE_MCP")


def extraer_cp_mcp(dom_n, dom_nse, ubi, mcp_dict, metodos):
    sub = mcp_dict.get(ubi)
    if sub is None: return (None, None, "")
    dom_n   = dom_n   if pd.notna(dom_n)   and dom_n   != "" else ""
    dom_nse = dom_nse if pd.notna(dom_nse) and dom_nse != "" else ""

    for m in metodos:
        t = m[0]

        if t == "exacta_c":
            num = m[1]
            if not dom_n: continue
            hits = sub[sub["CP_normalizado"] == dom_n]
            if len(hits) == 0: continue
            if len(hits) == 1:
                r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), num)
            mu = hits["MCP"].unique()
            if hits["COD"].nunique() == 1 and len(mu) == 1:
                r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), num)
            if len(mu) == 1:
                r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{num}_DOBLE_CP")
            r = hits.iloc[0]; return (str(r["CP"]), "INDETERMINADA", f"{num}_DOBLE_MCP")

        elif t == "exacta_se":
            num = m[1]
            if not dom_nse: continue
            hits = sub[sub["CP_normalizado_se"] == dom_nse]
            if len(hits) == 0: continue
            if len(hits) == 1:
                r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), num)
            if hits["CP"].nunique() == 1:
                mu = hits["MCP"].unique()
                if hits["COD"].nunique() == 1 and len(mu) == 1:
                    r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), num)
                if len(mu) == 1:
                    r = hits.iloc[0]; return (str(r["CP"]), str(r["MCP"]), f"{num}_DOBLE_CP")
                r = hits.iloc[0]; return (str(r["CP"]), "INDETERMINADA", f"{num}_DOBLE_MCP")

        elif t == "lev_cp":
            _, min_len, dl, num = m
            cands = sub[sub["CP_normalizado"].str.len().fillna(0) >= min_len].copy()
            if len(cands) == 0: continue
            cands["_d"] = cands["CP_normalizado_se"].apply(
                lambda x: lev(dom_nse, x) if pd.notna(x) else 9999)
            hits = cands[cands["_d"] <= dl]
            if len(hits) == 0: continue
            if len(hits) == 1:
                return (str(hits.iloc[0]["CP"]), str(hits.iloc[0]["MCP"]), num)
            if hits["CP"].nunique() == 1:
                if hits["COD"].nunique() == 1:
                    return (str(hits.iloc[0]["CP"]), str(hits.iloc[0]["MCP"]), num)
                if hits["MCP"].nunique() == 1:
                    return ("DOBLE_CP", str(hits["MCP"].unique()[0]), f"{num}_DOBLE_CP")
                return (str(hits.iloc[0]["CP"]), "INDETERMINADA", f"{num}_DOBLE_CP")
            if hits["MCP"].nunique() == 1:
                return ("DOBLE_CP", str(hits["MCP"].unique()[0]), f"{num}_DOBLE_CP")
            return ("DOBLE_CP","INDETERMINADA",f"{num}_DOBLE_MCP")

        elif t == "sub_cp":
            _, min_len_cp, num = m
            ci = sub[
                (sub["CP_normalizado_se"].str.len().fillna(0) > min_len_cp) &
                sub["CP_normalizado_se"].apply(lambda x: x in dom_nse if pd.notna(x) else False)
            ]
            r = _resolver_interna(ci, num)
            if r: return r

        elif t == "sub_vias_c":
            num = m[1]
            if not dom_n: continue
            ci = sub[
                (sub["VIAS_normalizado"].str.len().fillna(0) > 0) &
                sub["VIAS_normalizado"].apply(
                    lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dom_n if pd.notna(x) else False)
            ]
            r = _resolver_interna(ci, num)
            if r: return r

        elif t == "sub_vias_se":
            num = m[1]
            if not dom_nse: continue
            ci = sub[
                (sub["VIAS_normalizado_se"].str.len().fillna(0) > 0) &
                sub["VIAS_normalizado_se"].apply(
                    lambda x: x != "NOTIENEVIASCONOCIDAS" and x in dom_nse if pd.notna(x) else False)
            ]
            r = _resolver_interna(ci, num)
            if r: return r

        elif t == "lev_vias_c":
            _, min_len, dl, num = m
            if not dom_n: continue
            cands = sub[sub["VIAS_normalizado"].str.len().fillna(0) >= min_len].copy()
            if len(cands) == 0: continue
            cands["_d"] = cands["VIAS_normalizado"].apply(
                lambda x: lev(dom_n, x) if pd.notna(x) else 9999)
            hits = cands[cands["_d"] <= dl]
            if len(hits) == 0: continue
            if len(hits) == 1:
                return (str(hits.iloc[0]["CP"]), str(hits.iloc[0]["MCP"]), num)
            if hits["CP"].nunique() == 1:
                if hits["COD"].nunique() == 1:
                    return (str(hits.iloc[0]["CP"]), str(hits.iloc[0]["MCP"]), num)
                if hits["MCP"].nunique() == 1:
                    return ("DOBLE_CP", str(hits["MCP"].unique()[0]), f"{num}_DOBLE_CP")
                return (str(hits.iloc[0]["CP"]), "INDETERMINADA", f"{num}_DOBLE_CP")
            if hits["MCP"].nunique() == 1:
                return ("DOBLE_CP", str(hits["MCP"].unique()[0]), f"{num}_DOBLE_CP")
            return ("DOBLE_CP","INDETERMINADA",f"{num}_DOBLE_MCP")

    return (None, None, "")


# ── Definición de métodos por corrida ─────────────────────────────────────────

METODOS_C1 = [
    ("exacta_c",  "1"),
    ("exacta_se", "2"),
    ("lev_cp", 16, 1, "3"),
    ("lev_cp", 12, 1, "4"),
    ("lev_cp", 20, 2, "5"),
    ("lev_cp", 16, 2, "6"),
    ("sub_cp",  9, "7"),
]

METODOS_C2 = [
    ("exacta_c",  "11"),
    ("exacta_se", "12"),
    ("lev_cp", 16, 1, "13"),
    ("lev_cp", 12, 1, "14"),
    ("lev_cp", 20, 2, "15"),
    ("lev_cp", 16, 2, "16"),
    ("sub_cp",  8, "17"),
]

METODOS_C3 = [
    ("exacta_c",  "21"),
    ("exacta_se", "22"),
    ("lev_cp",  9, 1, "23"),
    ("lev_cp",  7, 1, "24"),
    ("lev_cp", 11, 2, "25"),
    ("lev_cp",  9, 2, "26"),
    ("lev_cp",  9, 2, "27"),
    ("lev_cp",  5, 1, "28"),
    ("sub_vias_c",  "29"),
    ("sub_vias_se", "30"),
    ("lev_vias_c", 8, 2, "31"),
    ("sub_cp",   5, "32"),
]


def ejecutar_corrida(data, mcp_dict, metodos, num, pb=None, txt=None):
    total = len(data)
    res = []
    for i, (_, row) in enumerate(data.iterrows()):
        cp, mcp, met = extraer_cp_mcp(
            row.get("DOM2024_normalizado",""),
            row.get("DOM2024_normalizado_se",""),
            row["UBI"], mcp_dict, metodos)
        res.append({"CP": cp, "MCP": mcp, "metodo_CP": met})
        if pb and i % max(1, total//100) == 0:
            pb.progress(min(i/total, 1.0))
            if txt: txt.text(f"Corrida {num}: {i:,} / {total:,}…")
    rdf = pd.DataFrame(res)
    out = data.copy()
    out["CP"]        = rdf["CP"].values
    out["MCP"]       = rdf["MCP"].values
    out["metodo_CP"] = rdf["metodo_CP"].values
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def leer_archivo(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, dtype=str)
    try:
        return pd.read_csv(file, sep="|", dtype=str, encoding="latin-1")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=",", dtype=str, encoding="latin-1")


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Resultados")
    return buf.getvalue()


def detectar_columnas(df):
    cu = {c.upper().strip(): c for c in df.columns}
    def buscar(cands):
        for c in cands:
            if c in cu: return cu[c]
        for c in cands:
            for ck, co in cu.items():
                if c in ck: return co
        return None
    ubi   = buscar(["UBIGEO_R","UBIGEO_DOM","UBIGEO","UBI","COD_UBIGEO","COD_UBI","UBIGEOR","COD_DOM","UBIG","CODUBI"])
    dom   = buscar(["DIRECC_DOM","DIRECCION_DOM","DIRECC","DIRECCION","DOM2024","DOMICILIO","DOM","DIRECCIÓN","DIR_DOM","DIR"])
    comcp = buscar(["CO_MCP","COMCP","COD_MCP","CODIGO_MCP"])
    dni   = buscar(["DNI","DOCUMENTO","DOC","NUM_DOC"])
    if ubi is None:
        for col in df.columns:
            if df[col].dropna().astype(str).str.strip().head(20).str.match(r"^\d{4,6}$").mean() > 0.7:
                ubi = col; break
    if dom is None:
        tc = [(c, df[c].dropna().astype(str).str.len().mean()) for c in df.columns if c != ubi]
        if tc: dom = max(tc, key=lambda x: x[1])[0]
    return ubi, dom, comcp, dni


# ── Exactitud (fiel al notebook) ──────────────────────────────────────────────

def clasificar_exactitud(row):
    a, m = row["MCP_AUTOM"], row["MCP_MANUAL"]
    if pd.notna(a) and pd.notna(m) and a == m:    return "c"
    elif a == "INDETERMINADA" and pd.notna(m):     return "fD"
    elif pd.notna(a) and pd.notna(m) and a != m:  return "f"
    elif pd.isna(a)  and pd.notna(m):              return "d"
    elif a == "INDETERMINADA" and pd.isna(m):      return "eD"
    elif pd.notna(a) and pd.isna(m):               return "e"
    elif pd.isna(a)  and pd.isna(m):               return "v"
    return None


# ═════════════════════════════════════════════════════════════════════════════
# INTERFAZ
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="bheader">
  <h1>🦎 BASILISCO</h1>
  <p>Motor de asignación de Centros Poblados · Padrón Electoral Peruano</p>
</div>
""", unsafe_allow_html=True)

# ── MCP fijo ──────────────────────────────────────────────────────────────────

MCP_PATH = Path(__file__).parent / "MCP_ORIGINAL.xlsx"

@st.cache_resource(show_spinner="Cargando catálogo MCP…")
def cargar_mcp(path):
    df = pd.read_excel(path, dtype=str)
    df["UBI"] = df["UBI"].astype(str).str.strip().apply(lambda x: fmt_int(x, 6))
    df["POB"] = pd.to_numeric(df["POB"] if "POB" in df.columns else pd.Series(dtype=float),
                              errors="coerce").fillna(0)
    if "MCP_COD" in df.columns:
        df["MCP_COD"] = df["MCP_COD"].astype(str).str.replace(r"\.0$","",regex=True).apply(
            lambda x: fmt_int(x, 9))
    if "COD" in df.columns:
        df["COD"] = df["COD"].astype(str).str.replace(r"\.0$","",regex=True)
    for col in ["CP","MCP","VIAS"]:
        if col not in df.columns: df[col] = ""
    return df

if not MCP_PATH.exists():
    st.error("No se encontró **MCP_ORIGINAL.xlsx** junto a `app.py`.")
    st.stop()

mcp_raw = cargar_mcp(MCP_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Opciones")
    st.markdown("---")
    st.markdown("**Filtro por UBIGEO** *(opcional)*")
    st.caption("Ubigeos de 6 dígitos separados por coma. Vacío = todos.")
    ubi_input = st.text_input("Ubigeos", value="", placeholder="Ej: 200106, 200110")
    st.markdown("---")
    st.markdown(f"""
    <div style="color:#4a5568;font-size:0.75rem;line-height:1.8;">
    <b>Catálogo:</b> <span style="color:#38b2ac;font-family:monospace;">MCP_ORIGINAL.xlsx</span><br>
    <span style="color:#718096;">{len(mcp_raw):,} filas · {mcp_raw['UBI'].nunique():,} distritos</span><br><br>
    <b>Corridas:</b><br>
    🔵 C1 · métodos 1–7<br>
    🟢 C2 · métodos 11–17<br>
    🟠 C3 · métodos 21–32
    </div>""", unsafe_allow_html=True)

# ── Paso 1: subir padrón ──────────────────────────────────────────────────────

st.markdown('<div class="step-hdr">① Subir padrón</div>', unsafe_allow_html=True)

padron_file = st.file_uploader(
    "Archivo del padrón (.xlsx, .xls, .csv)",
    type=["xlsx","xls","csv"]
)

if padron_file is None:
    st.markdown('<div class="ibox">📂 Sube el archivo de padrón para continuar.</div>',
                unsafe_allow_html=True)
    st.stop()

@st.cache_data(show_spinner="Leyendo padrón…")
def leer_y_cachear(fb, fn):
    buf = io.BytesIO(fb); buf.name = fn
    return leer_archivo(buf)

try:
    padron_raw = leer_y_cachear(padron_file.read(), padron_file.name)
except Exception as e:
    st.error(f"Error leyendo el padrón: {e}"); st.stop()

# ── Paso 2: confirmar columnas ────────────────────────────────────────────────

st.markdown('<div class="step-hdr">② Confirmar columnas</div>', unsafe_allow_html=True)

det_ubi, det_dom, det_comcp, det_dni = detectar_columnas(padron_raw)
cols = padron_raw.columns.tolist()

ca, cb, cc, cd = st.columns(4)
with ca:
    col_ubi = st.selectbox("UBIGEO *", cols,
                           index=cols.index(det_ubi) if det_ubi in cols else 0)
with cb:
    col_dom = st.selectbox("DIRECCIÓN *", cols,
                           index=cols.index(det_dom) if det_dom in cols else min(1,len(cols)-1))
with cc:
    opts_cc = ["(ninguna)"] + cols
    col_comcp = st.selectbox("CO_MCP (manual, opcional)", opts_cc,
                             index=opts_cc.index(det_comcp) if det_comcp in opts_cc else 0)
with cd:
    opts_dni = ["(ninguna)"] + cols
    col_dni = st.selectbox("DNI (opcional)", opts_dni,
                           index=opts_dni.index(det_dni) if det_dni in opts_dni else 0)

if col_ubi == col_dom:
    st.warning("⚠️ Ubigeo y dirección apuntan a la misma columna."); st.stop()

with st.expander("👁️ Vista previa — primeras 10 filas", expanded=False):
    st.dataframe(padron_raw.head(10), use_container_width=True, hide_index=True)

# ── Paso 3: ejecutar ──────────────────────────────────────────────────────────

st.markdown("---")
run_btn = st.button("🚀  EJECUTAR BASILISCO", use_container_width=True)
if not run_btn: st.stop()

t0 = time.time()

# ── Preparar DATA_B (fiel al notebook) ───────────────────────────────────────

ubis_filtro = [u.strip() for u in ubi_input.split(",") if u.strip()] if ubi_input.strip() else []

extra_cols = {}
if col_comcp != "(ninguna)": extra_cols[col_comcp] = "CO_MCP"
if col_dni   != "(ninguna)": extra_cols[col_dni]   = "DNI"

keep = [col_ubi, col_dom] + [c for c in extra_cols if c in padron_raw.columns]
DATA_B = padron_raw[[c for c in keep if c in padron_raw.columns]].copy()
DATA_B = DATA_B.rename(columns={col_ubi: "UBI", col_dom: "DOM2024"})
for old, new in extra_cols.items():
    if old in DATA_B.columns: DATA_B = DATA_B.rename(columns={old: new})

DATA_B["UBI"] = DATA_B["UBI"].astype(str).str.strip().apply(lambda x: fmt_int(x, 6))
if "CO_MCP" in DATA_B.columns:
    DATA_B["CO_MCP"] = DATA_B["CO_MCP"].astype(str).str.strip().apply(lambda x: fmt_int(x, 9))
if "DNI" in DATA_B.columns:
    DATA_B["DNI"] = DATA_B["DNI"].astype(str).str.strip().apply(lambda x: fmt_int(x, 8))
DATA_B["DOM2024"] = DATA_B["DOM2024"].astype(str).str.strip()
DATA_B = DATA_B.dropna(subset=["UBI","DOM2024"])
DATA_B = DATA_B[DATA_B["DOM2024"].str.strip() != ""]

DATA_B["depto"] = DATA_B["UBI"].str[:2]
DATA_B["PROV"]  = DATA_B["UBI"].str[:4]

if ubis_filtro:
    unorms = [fmt_int(u,6) for u in ubis_filtro if fmt_int(u,6)]
    if unorms: DATA_B = DATA_B[DATA_B["UBI"].isin(unorms)]

if DATA_B.empty:
    st.error("Sin registros tras aplicar filtros."); st.stop()

if ubis_filtro:
    unorms = [fmt_int(u,6) for u in ubis_filtro if fmt_int(u,6)]
    mcp_filt = mcp_raw[mcp_raw["UBI"].isin(unorms)].copy() if unorms else mcp_raw.copy()
    if mcp_filt.empty:
        st.error(f"Sin entradas MCP para: {', '.join(unorms)}"); st.stop()
else:
    mcp_filt = mcp_raw.copy()

DATA_B = DATA_B.reset_index(drop=True)

# Métricas
st.markdown("---")
mc1, mc2, mc3 = st.columns(3)
mc1.markdown(f'<div class="mcard"><div class="val">{len(DATA_B):,}</div><div class="lbl">Registros</div></div>', unsafe_allow_html=True)
mc2.markdown(f'<div class="mcard"><div class="val">{len(mcp_filt):,}</div><div class="lbl">Entradas MCP</div></div>', unsafe_allow_html=True)
mc3.markdown(f'<div class="mcard"><div class="val">{DATA_B["UBI"].nunique()}</div><div class="lbl">Distritos</div></div>', unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════ #
# CORRIDA 1                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown('<span class="cbadge c1">CORRIDA 1</span> &nbsp; Exacta + Levenshtein CP · métodos 1–7',
            unsafe_allow_html=True)
pb1 = st.progress(0); t1 = st.empty()

DATA_B["DOM2024_normalizado"]    = DATA_B["DOM2024"].apply(norm_c1)
DATA_B["DOM2024_normalizado_se"] = DATA_B["DOM2024_normalizado"].fillna("").str.replace(" ","",regex=False)
DATA_B.loc[DATA_B["DOM2024_normalizado"].isna()|(DATA_B["DOM2024_normalizado"]==""),
           ["DOM2024_normalizado","DOM2024_normalizado_se"]] = None
DATA_B = DATA_B[DATA_B["DOM2024_normalizado"].notna()].reset_index(drop=True)

_, md1   = preparar_mcp(mcp_filt, norm_c1)
DATA_C1  = ejecutar_corrida(DATA_B, md1, METODOS_C1, 1, pb1, t1)
pb1.progress(1.0)

DATA1    = DATA_C1[DATA_C1["CP"].notna()].copy().reset_index(drop=True)
DATA2_in = DATA_C1[DATA_C1["CP"].isna()].drop(
    columns=["CP","MCP","metodo_CP","DOM2024_normalizado","DOM2024_normalizado_se"],
    errors="ignore").reset_index(drop=True)
t1.markdown(f'<div class="sbox">✅ C1 · <b>{len(DATA1):,}</b> asignados · <b>{len(DATA2_in):,}</b> pendientes</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════ #
# CORRIDA 2                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown('<span class="cbadge c2">CORRIDA 2</span> &nbsp; +BARRIO/SECTOR→CASERIO · métodos 11–17',
            unsafe_allow_html=True)
pb2 = st.progress(0); t2 = st.empty()

if not DATA2_in.empty:
    DATA2_in["DOM2024_normalizado"]    = DATA2_in["DOM2024"].apply(norm_c2)
    DATA2_in["DOM2024_normalizado_se"] = DATA2_in["DOM2024_normalizado"].fillna("").str.replace(" ","",regex=False)
    DATA2_in.loc[DATA2_in["DOM2024_normalizado"].isna()|(DATA2_in["DOM2024_normalizado"]==""),
                 ["DOM2024_normalizado","DOM2024_normalizado_se"]] = None
    DATA2_in = DATA2_in[DATA2_in["DOM2024_normalizado"].notna()].reset_index(drop=True)
    _, md2   = preparar_mcp(mcp_filt, norm_c2)
    DATA_C2  = ejecutar_corrida(DATA2_in, md2, METODOS_C2, 2, pb2, t2)
    pb2.progress(1.0)
    DATA2    = DATA_C2[DATA_C2["CP"].notna()].copy().reset_index(drop=True)
    DATA3_in = DATA_C2[DATA_C2["CP"].isna()].drop(
        columns=["CP","MCP","metodo_CP","DOM2024_normalizado","DOM2024_normalizado_se"],
        errors="ignore").reset_index(drop=True)
    t2.markdown(f'<div class="sbox">✅ C2 · <b>{len(DATA2):,}</b> asignados · <b>{len(DATA3_in):,}</b> pendientes</div>',
                unsafe_allow_html=True)
else:
    DATA2 = DATA3_in = pd.DataFrame()
    pb2.progress(1.0)
    t2.markdown('<div class="ibox">⏭️ Sin pendientes para C2.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════ #
# CORRIDA 3                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown('<span class="cbadge c3">CORRIDA 3</span> &nbsp; CP sin CASERIO + VIAS sintéticas · métodos 21–32',
            unsafe_allow_html=True)
pb3 = st.progress(0); t3 = st.empty()

if not DATA3_in.empty:
    DATA3_in["DOM2024_normalizado"]    = DATA3_in["DOM2024"].apply(norm_c2)
    DATA3_in["DOM2024_normalizado_se"] = DATA3_in["DOM2024_normalizado"].fillna("").str.replace(" ","",regex=False)
    DATA3_in.loc[DATA3_in["DOM2024_normalizado"].isna()|(DATA3_in["DOM2024_normalizado"]==""),
                 ["DOM2024_normalizado","DOM2024_normalizado_se"]] = None
    DATA3_in = DATA3_in[DATA3_in["DOM2024_normalizado"].notna()].reset_index(drop=True)
    MCP_C3, md3 = preparar_mcp(mcp_filt, norm_c2, quitar_caserio=True, vias_sinteticas=True)
    DATA_C3  = ejecutar_corrida(DATA3_in, md3, METODOS_C3, 3, pb3, t3)
    pb3.progress(1.0)
    DATA3_ok   = DATA_C3[DATA_C3["CP"].notna()].copy().reset_index(drop=True)
    DATA3_nulo = DATA_C3[DATA_C3["CP"].isna()].copy().reset_index(drop=True)
    t3.markdown(f'<div class="sbox">✅ C3 · <b>{len(DATA3_ok):,}</b> asignados · <b>{len(DATA3_nulo):,}</b> sin asignar</div>',
                unsafe_allow_html=True)
else:
    MCP_C3 = pd.DataFrame()
    DATA3_ok = DATA3_nulo = pd.DataFrame()
    pb3.progress(1.0)
    t3.markdown('<div class="ibox">⏭️ Sin pendientes para C3.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════ #
# CONSOLIDAR — fiel al notebook: concat([DATA1, DATA2, DATA3])               #
# ═══════════════════════════════════════════════════════════════════════════ #

DATA3 = pd.concat([df for df in [DATA3_ok, DATA3_nulo] if not df.empty], ignore_index=True) \
        if (not DATA3_ok.empty or not DATA3_nulo.empty) else pd.DataFrame()

partes = [df for df in [DATA1, DATA2, DATA3] if not df.empty]
DATA = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

for col in ["CP","MCP","metodo_CP"]:
    if col in DATA.columns: DATA[col] = DATA[col].replace("", np.nan)

# ── Merge COD — fiel al notebook (usa MCP de la 3ª corrida = MCP_C3) ─────────
# El notebook: MCPm = MCP[["UBI","COD","MCP","CP"]].drop_duplicates()...
# MCP en ese punto es el MCP de la corrida 3 (con quitar_caserio y vias sinteticas)
mcp_for_cod = MCP_C3 if not MCP_C3.empty else mcp_filt

MCPm = (mcp_for_cod[["UBI","COD","MCP","CP"]].drop_duplicates()
        .groupby(["UBI","MCP","CP"], as_index=False).first())
MCPm["UBI"] = MCPm["UBI"].astype(str).str.strip().apply(lambda x: fmt_int(x, 6))
DATA = DATA.merge(MCPm, on=["UBI","MCP","CP"], how="left")

DATA = DATA.rename(columns={"CP":"CP_A","MCP":"MCP_A","COD":"COD_A"})
DATA["CP_A"] = DATA["CP_A"].astype(str).str.replace("CASERIO ","",regex=False)

# COD_A: float → str entero (50403.0 → "50403")
DATA["COD_A"] = DATA["COD_A"].apply(
    lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() not in ("","nan","None") else None)

# ── Merge MINI — copia exacta del notebook ───────────────────────────────────
# MINI = MCP[[MCP.columns[0], MCP.columns[1], MCP.columns[2]]].dropna().drop_duplicates()
# DATA = DATA.merge(MINI, left_on=["UBI", "MCP_A"], right_on=["UBI", "MCP"], how="left")
# Funciona porque en este punto DATA ya no tiene columna "MCP" (renombrada a MCP_A).
MCP = mcp_for_cod.copy()
MINI = MCP[[MCP.columns[0], MCP.columns[1], MCP.columns[2]]].dropna().drop_duplicates()
DATA = DATA.merge(MINI, left_on=["UBI", "MCP_A"], right_on=["UBI", "MCP"], how="left")

if "MCP_COD" in DATA.columns:
    DATA["MCP_COD"] = DATA["MCP_COD"].astype(str).str.replace(r"\.0$","",regex=True).apply(
        lambda x: fmt_int(x, 9) if x not in ("None","nan","") else None)

t_total = (time.time() - t0) / 60

# ═══════════════════════════════════════════════════════════════════════════ #
# MCPS_DETECT — fiel al notebook                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

MCPS_DETECT = DATA.copy()

rename_map = {}
if "CO_MCP"  in MCPS_DETECT.columns: rename_map["CO_MCP"]  = "MCP_MANUAL"
if "MCP_COD" in MCPS_DETECT.columns: rename_map["MCP_COD"] = "MCP_AUTOM"
MCPS_DETECT = MCPS_DETECT.rename(columns=rename_map)

if "MCP_MANUAL" in MCPS_DETECT.columns:
    MCPS_DETECT["MCP_MANUAL"] = MCPS_DETECT["MCP_MANUAL"].replace("", None)
if "MCP_AUTOM"  in MCPS_DETECT.columns:
    MCPS_DETECT["MCP_AUTOM"]  = MCPS_DETECT["MCP_AUTOM"].replace("", None)

MCPS_DETECT["UBI"] = MCPS_DETECT["UBI"].astype(str).str.strip().apply(
    lambda x: f"{int(float(x)):06d}" if x not in ("","nan") else x)

tiene_manual = "MCP_MANUAL" in MCPS_DETECT.columns and MCPS_DETECT["MCP_MANUAL"].notna().any()
tiene_autom  = "MCP_AUTOM"  in MCPS_DETECT.columns and MCPS_DETECT["MCP_AUTOM"].notna().any()
if tiene_manual and tiene_autom:
    MCPS_DETECT["EXACTITUD"] = MCPS_DETECT.apply(clasificar_exactitud, axis=1)

# ═══════════════════════════════════════════════════════════════════════════ #
# RESULTADOS                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown("---")
st.markdown("### 📊 Resultados")

total     = len(MCPS_DETECT)
asignados = MCPS_DETECT["CP_A"].notna().sum() if "CP_A" in MCPS_DETECT.columns else 0
sin_asig  = total - asignados
tasa      = asignados / total * 100 if total else 0

r1,r2,r3,r4,r5 = st.columns(5)
for col_st, val, lbl in [
    (r1, f"{total:,}",        "Total"),
    (r2, f"{asignados:,}",    "Asignados"),
    (r3, f"{sin_asig:,}",     "Sin asignar"),
    (r4, f"{tasa:.1f}%",      "Tasa asignación"),
    (r5, f"{t_total:.1f} min","Tiempo"),
]:
    col_st.markdown(f'<div class="mcard"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True)

st.markdown("---")
rows_c = []
for nom, df_c in [("🔵 C1 (1–7)",DATA1),("🟢 C2 (11–17)",DATA2),("🟠 C3 (21–32)",DATA3_ok)]:
    n = len(df_c) if not df_c.empty else 0
    rows_c.append({"Corrida":nom,"Asignados":f"{n:,}","% del total":f"{n/total*100:.1f}%" if total else "0%"})
n_nulo = len(DATA3_nulo) if not DATA3_nulo.empty else 0
rows_c.append({"Corrida":"⚫ Sin asignar","Asignados":f"{n_nulo:,}",
               "% del total":f"{n_nulo/total*100:.1f}%" if total else "0%"})
st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)

# ── Módulo de validación ──────────────────────────────────────────────────────
if "EXACTITUD" in MCPS_DETECT.columns:
    st.markdown("---")
    st.markdown("### 🔍 Módulo de Validación")

    n_manual = int(MCPS_DETECT["MCP_MANUAL"].notna().sum())
    n_autom  = int(MCPS_DETECT["MCP_AUTOM"].notna().sum())
    vi1, vi2 = st.columns(2)
    vi1.metric("Clasificado manualmente",     f"{n_manual/total*100:.2f}%", f"{n_manual:,} registros")
    vi2.metric("Clasificado automáticamente", f"{n_autom/total*100:.2f}%",  f"{n_autom:,} registros")

    conteos = MCPS_DETECT["EXACTITUD"].value_counts()
    pcts    = MCPS_DETECT["EXACTITUD"].value_counts(normalize=True) * 100
    LABS = {"c":"✅ Correctas llenas","f":"❌ Fallos","fD":"⚠️ Fallos indet.",
            "e":"➕ Excesos","eD":"➕ Excesos indet.","d":"➖ Déficits","v":"⬜ Correctas vacías"}
    rows_ex = [{"Indicador":lbl,"N":int(conteos.get(k,0)),"%":f"{pcts.get(k,0):.2f}%"}
               for k,lbl in LABS.items()]
    st.dataframe(pd.DataFrame(rows_ex), use_container_width=True, hide_index=True)

    tab_d, tab_m, tab_met = st.tabs(["📍 Por distrito","🏘️ Por MCP","⚙️ Por método"])

    with tab_d:
        rd = (MCPS_DETECT.groupby("UBI").agg(
            TOTAL=("UBI","count"),
            MANUAL=("MCP_MANUAL", lambda x: x.notna().sum()),
            AUTOM=("MCP_AUTOM",   lambda x: x.notna().sum()),
            CORRECTOS=("EXACTITUD", lambda x: (x=="c").sum())
        ).reset_index())
        rd["DETECTADO"]      = (rd["CORRECTOS"]/rd["MANUAL"]*100).round(2)
        rd["CORRETAMENTE_D"] = (rd["CORRECTOS"]/rd["AUTOM"]*100).round(2)
        st.dataframe(rd, use_container_width=True, hide_index=True)
        st.download_button("📥 RES_DIST.xlsx", data=to_excel_bytes(rd), file_name="RES_DIST.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_m:
        rm = (MCPS_DETECT.groupby("MCP_MANUAL").agg(
            TOTAL=("MCP_MANUAL","count"),
            MANUAL=("MCP_MANUAL", lambda x: x.notna().sum()),
            AUTOM=("MCP_AUTOM",   lambda x: x.notna().sum()),
            CORRECTOS=("EXACTITUD", lambda x: (x=="c").sum())
        ).reset_index())
        rm["DETECTADO"]      = (rm["CORRECTOS"]/rm["MANUAL"]*100).round(2)
        rm["CORRETAMENTE_D"] = (rm["CORRECTOS"]/rm["AUTOM"]*100).round(2)
        st.dataframe(rm, use_container_width=True, hide_index=True)
        st.download_button("📥 RES_MCP.xlsx", data=to_excel_bytes(rm), file_name="RES_MCP.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_met:
        if "metodo_CP" in MCPS_DETECT.columns:
            rmet = (MCPS_DETECT.groupby(["metodo_CP","EXACTITUD"]).size()
                    .reset_index(name="n")
                    .pivot(index="metodo_CP", columns="EXACTITUD", values="n")
                    .fillna(0).reset_index())
            rmet.columns.name = None
            st.dataframe(rmet, use_container_width=True, hide_index=True)
            st.download_button("📥 RES_METODO.xlsx", data=to_excel_bytes(rmet), file_name="RES_METODO.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════════════════════════ #
# EXPLORADOR MCPS_DETECT — todas las columnas, filtro por columna            #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown("---")
st.markdown("### 🔎 Explorador MCPS_DETECT")
st.caption("Resultado completo — todas las columnas. Filtra por cualquier columna.")

mcps_cols = MCPS_DETECT.columns.tolist()
ef1, ef2, ef3 = st.columns([2, 3, 2])
with ef1:
    filtro_col = st.selectbox("Columna a filtrar", ["(sin filtro)"] + mcps_cols, key="fcol_detect")
with ef2:
    filtro_val = st.text_input("Valor (insensible a mayúsculas)", key="fval_detect")
with ef3:
    estado_det = st.selectbox("Estado asignación", ["Todos","Asignados","Sin asignar"], key="fest_detect")

dv = MCPS_DETECT.copy()
if filtro_col != "(sin filtro)" and filtro_val.strip():
    dv = dv[dv[filtro_col].astype(str).str.upper().str.contains(filtro_val.strip().upper(), na=False)]
if estado_det == "Asignados"   and "CP_A" in dv.columns: dv = dv[dv["CP_A"].notna()]
if estado_det == "Sin asignar" and "CP_A" in dv.columns: dv = dv[dv["CP_A"].isna()]

st.caption(f"Mostrando **{len(dv):,}** de **{total:,}** registros · {len(mcps_cols)} columnas")
st.dataframe(dv.reset_index(drop=True), use_container_width=True, height=480)

# ═══════════════════════════════════════════════════════════════════════════ #
# EXPLORADOR CATÁLOGO MCP — todas las columnas, filtro por columna           #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown("---")
st.markdown("### 🗂️ Explorador Catálogo MCP")
st.caption("Catálogo MCP_ORIGINAL.xlsx cargado. Útil para cruzar y auditar resultados.")

mcp_cols_all = mcp_filt.columns.tolist()
mf1, mf2 = st.columns([2, 3])
with mf1:
    mcp_filtro_col = st.selectbox("Columna a filtrar", ["(sin filtro)"] + mcp_cols_all, key="fcol_mcp")
with mf2:
    mcp_filtro_val = st.text_input("Valor", key="fval_mcp")

mv = mcp_filt.copy()
if mcp_filtro_col != "(sin filtro)" and mcp_filtro_val.strip():
    mv = mv[mv[mcp_filtro_col].astype(str).str.upper().str.contains(
        mcp_filtro_val.strip().upper(), na=False)]

st.caption(f"Mostrando **{len(mv):,}** de **{len(mcp_filt):,}** entradas MCP · {len(mcp_cols_all)} columnas")
st.dataframe(mv.reset_index(drop=True), use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════════════════════════ #
# DESCARGAS                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

st.markdown("---")
st.markdown("### ⬇️ Descargar")

base = padron_file.name.rsplit(".", 1)[0]
d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "📥 MCPS_DETECT completo (.xlsx)",
        data=to_excel_bytes(MCPS_DETECT),
        file_name=f"MCPS_DETECT_{base}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
with d2:
    df_nulos = MCPS_DETECT[MCPS_DETECT["CP_A"].isna()] if "CP_A" in MCPS_DETECT.columns else pd.DataFrame()
    if not df_nulos.empty:
        st.download_button(
            "📥 Sin asignar (.xlsx)",
            data=to_excel_bytes(df_nulos),
            file_name=f"SIN_ASIGNAR_{base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    else:
        st.success("🎉 ¡Todos los registros fueron asignados!")
with d3:
    if not dv.empty:
        st.download_button(
            "📥 Vista filtrada MCPS_DETECT (.xlsx)",
            data=to_excel_bytes(dv.reset_index(drop=True)),
            file_name=f"FILTRADO_DETECT_{base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
with d4:
    if not mv.empty:
        st.download_button(
            "📥 Vista filtrada MCP (.xlsx)",
            data=to_excel_bytes(mv.reset_index(drop=True)),
            file_name=f"FILTRADO_MCP_{base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
