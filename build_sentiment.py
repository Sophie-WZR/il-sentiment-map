# build_sentiment.py
import pandas as pd
df = pd.read_csv("Ameren Muni Public Sentiment Research - Sentiment Research.csv")

# =========================
# AHJ public sentiment (dataset-aware)
#  - concern score from counts (citizen:official = 4:6)
#  - robust vote parsing: 7-0-2, 7-0, "All ayes (unanimous)", "13-1-1", NA-2, Yea/Nay strings
#  - includes abstentions in denominator when present
#  - safe MW parsing: requires MW/kW units; ignores acres/panels; handles "2*2 MW" & "X MW each"
#  - 5–20 MW weight boost
#  - AG/Agriculture zoning bonus (supports "Zoning" or "Zoning District")
#  - recency weighting; prefer "Approval/Denial Year" (mid-year proxy)
#  - extra swing: approvals in last 2y +points; denials in last 2y −points
# =========================

import re
import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# -------------------------
# Vote parsing
# -------------------------
ALL_UNANIMOUS_PHRASES = (
    "all ayes", "all aye", "all yes", "unanimous", "approved unanimously",
)

# plain patterns
VOTE_3NUM_RE = re.compile(r'\b(\d+)\s*[-–]\s*(\d+)\s*[-–]\s*(\d+)\b')
VOTE_2NUM_RE = re.compile(r'\b(\d+)\s*[-–]\s*(\d+)\b')
NA_MINUS_N_RE = re.compile(r'^\s*NA\s*[-–]\s*(\d+)(?:\s*[-–]\s*\d+)?\s*$', re.IGNORECASE)

# "Yea: 7; Nay: 2", "Ayes 13; Nays 1", etc.
VOTE_YEA_NAY_RE = re.compile(
    r'(?:yea|aye|yes|yeas|ayes)\s*[:\-]?\s*(\d+)[^0-9]+(?:nay|nays|no|nos)\s*[:\-]?\s*(\d+)',
    re.IGNORECASE | re.DOTALL
)

# Labeled sections (we prefer Council when present)
LABELS = ("City Council vote", "Planning Commission", "ZBA Recommendation")

def _ratio_from_counts(y: int, n: int, a: int = 0) -> Optional[float]:
    denom = y + n + a
    return (y / denom) if denom > 0 else None

def _parse_triplet(s: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'Y-N-A' or 'Y-N'. Returns (Y, N, A) with A defaulting to 0."""
    m3 = VOTE_3NUM_RE.search(s)
    if m3:
        y, n, a = map(int, m3.groups())
        return y, n, a
    m2 = VOTE_2NUM_RE.search(s)
    if m2:
        y, n = map(int, m2.groups())
        return y, n, 0
    return None

def _parse_yea_nay_strings(s: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'Yea: N; Nay: M' style."""
    m = VOTE_YEA_NAY_RE.search(s)
    if m:
        y, n = map(int, m.groups())
        return y, n, 0
    return None

def _extract_label_chunk(text: str, label: str) -> Optional[str]:
    """Return the text immediately following '<label>:' up to line/end."""
    m = re.search(re.escape(label) + r'\s*:\s*([^\n\r;]+)', text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def parse_votes(cell: Optional[str]) -> Optional[float]:
    """
    Returns approval ratio in [0,1] or None.
    Priority:
      1) Unanimous phrases → 1.0
      2) Prefer labeled 'City Council vote: ...'
      3) If Council is explicitly 'N/A', do not treat other bodies as Council
      4) Otherwise parse first numeric or Yea/Nay pattern
      5) 'NA-2' (unknown yeas) and 'N/A' → None
    """
    if not isinstance(cell, str) or not cell.strip():
        return None
    t = cell.strip()
    low = t.lower()

    # 1) unanimous
    if any(p in low for p in ALL_UNANIMOUS_PHRASES):
        return 1.0

    # 2) labeled Council chunk if present
    council = _extract_label_chunk(t, "City Council vote")
    if council:
        if re.search(r'\bN/?A\b', council, re.IGNORECASE):
            return None
        if NA_MINUS_N_RE.match(council):
            return None
        tri = _parse_triplet(council) or _parse_yea_nay_strings(council)
        if tri:
            y, n, a = tri
            return _ratio_from_counts(y, n, a)

    # 3) if the whole cell is 'N/A' or an NA-* form → None
    if re.fullmatch(r'\s*N/?A\s*', t, flags=re.IGNORECASE):
        return None
    if NA_MINUS_N_RE.match(t):
        return None

    # 4) otherwise, any numeric triplet/pair or Yea/Nay in the text
    tri = _parse_triplet(t) or _parse_yea_nay_strings(t)
    if tri:
        y, n, a = tri
        return _ratio_from_counts(y, n, a)

    return None

# --- recommendation ratio (PC/ZBA), used only as a weak proxy when Council missing & recent ----
REC_PC_RE = re.compile(
    r'(?:planning\s*commission|pc)\s*[:\-]?\s*(?:recommended|recommendation)?[^0-9\-]*?(\d+)\s*[-–]\s*(\d+)(?:\s*[-–]\s*\d+)?',
    re.IGNORECASE | re.DOTALL
)
REC_ZBA_RE = re.compile(
    r'(?:zba|zoning\s*board(?:\s*of)?\s*appeals?)\s*[:\-]?\s*(?:recommended|recommendation)?[^0-9\-]*?(\d+)\s*[-–]\s*(\d+)(?:\s*[-–]\s*\d+)?',
    re.IGNORECASE | re.DOTALL
)
REC_ANY_RE = re.compile(
    r'recommend(?:ed|ation)[^0-9\-]*?(\d+)\s*[-–]\s*(\d+)(?:\s*[-–]\s*\d+)?',
    re.IGNORECASE | re.DOTALL
)
REC_ALL_AYES_RE = re.compile(r'(planning\s*commission|pc|zba)[^\.:\n]*?(all\s+ayes|all\s+yeas|unanimous)', re.IGNORECASE)

def extract_recommendation_ratio(vtext: Optional[str]) -> Optional[float]:
    """Read PC/ZBA recommendation ratio Y/(Y+N). Includes abstentions only if present in the pair as zeros."""
    if not isinstance(vtext, str) or not vtext.strip():
        return None
    t = vtext.strip()

    if REC_ALL_AYES_RE.search(t):
        return 1.0
    for rx in (REC_PC_RE, REC_ZBA_RE, REC_ANY_RE):
        m = rx.search(t)
        if m:
            y, n = int(m.group(1)), int(m.group(2))
            return (y / (y + n)) if (y + n) > 0 else None
    return None

def map_status(s: Optional[str]) -> float:
    """Approved→1.0, Denied→0.0, else 0.5."""
    if not isinstance(s, str):
        return 0.5
    t = s.lower()
    if 'approved' in t or 'approval' in t:
        return 1.0
    if 'denied' in t or 'rejected' in t or 'denial' in t:
        return 0.0
    return 0.5

# -------------------------
# MW parsing (unit-aware)
# -------------------------
ACRES_PER_MW_APPROX = 5.0  # 5 acres ≈ 1 MW

_SPELLED = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}

def parse_mw(s: Optional[str]) -> Optional[float]:
    """
    Extract total MW. Only counts numbers attached to MW/megawatt/kW (converts kW→MW).
    Handles "2*2 MW", "two ... of 2 MW each", "2 MW each", "Greater than 2 MW".
    If no MW/kW found but acreage present, approximates as (acres / ACRES_PER_MW_APPROX).
    Examples:
      "Proposed on a 36-acre parcel." -> 36 / 5 = 7.2
      "Site acreage: 50 acres"        -> 10.0
    """
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.lower()

    # 2 * 2 MW
    m = re.search(r'(\d+)\s*[*xX]\s*(\d+)\s*mw', t)
    if m:
        a, b = map(float, m.groups())
        return a * b

    # "<N> MW each" with a preceding count nearby
    m = re.search(r'(\d+(?:\.\d+)?)\s*mw\s*each', t)
    if m:
        per = float(m.group(1))
        m2 = re.search(r'(\d+)\s+(?:solar|farms?|projects?)\s+of\s+\d+(?:\.\d+)?\s*mw\s*each', t)
        if m2:
            return float(m2.group(1)) * per
        m3 = re.search(r'\b(' + '|'.join(_SPELLED.keys()) + r')\b\s+(?:solar|farms?|projects?)\s+of\s+\d+(?:\.\d+)?\s*mw\s*each', t)
        if m3:
            return float(_SPELLED[m3.group(1)]) * per
        return per

    # "Two solar farms of 2 MW each"
    m = re.search(r'\b(' + '|'.join(_SPELLED.keys()) + r'|\d+)\b\s+(?:solar|farms?|projects?)\s+of\s+(\d+(?:\.\d+)?)\s*mw', t)
    if m:
        raw_cnt, per = m.groups()
        cnt = float(_SPELLED.get(raw_cnt, raw_cnt))
        return cnt * float(per)

    # "Greater than 2 MW" -> lower bound
    m = re.search(r'(?:greater\s+than|>\s*)\s*(\d+(?:\.\d+)?)\s*mw', t)
    if m:
        return float(m.group(1))

    # sum plain MW and converted kW
    mw_nums = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*mw\b', t)]
    kw_nums = [float(x)/1000.0 for x in re.findall(r'(\d+(?:\.\d+)?)\s*kw\b', t)]
    vals = mw_nums + kw_nums
    if vals:
        return sum(vals)

    # ---- Fallback: acreage -> MW approximation (acres / ACRES_PER_MW_APPROX) ----
    acres = []

    # numeric forms: "36-acre", "36 acres", "acreage: 36", "36 ac"
    for rx in (
        r'(\d+(?:\.\d+)?)\s*-\s*acre',           # 36-acre
        r'(\d+(?:\.\d+)?)\s*acres?\b',           # 36 acres / 36 acre
        r'acreage[^0-9]{0,10}(\d+(?:\.\d+)?)',   # acreage: 36
        r'(\d+(?:\.\d+)?)\s*ac\b\.?'             # 36 ac
    ):
        acres += [float(x) for x in re.findall(rx, t)]

    # spelled-number forms: "ten-acre", "ten acres"
    spelled_alt = '|'.join(_SPELLED.keys())
    for rx in (
        rf'\b({spelled_alt})\b\s*-\s*acre',
        rf'\b({spelled_alt})\b\s*acres?\b',
    ):
        for word in re.findall(rx, t):
            acres.append(float(_SPELLED[word]))

    if acres:
        # Use the largest acreage mentioned to avoid picking counts or minor sub-areas
        ac = max(acres)
        return ac / ACRES_PER_MW_APPROX

    return None


# -------------------------
# Date helpers (use Approval/Denial Year as primary/only source)
# -------------------------
YEAR_COL_CANDIDATES = ["Approval/Denial Year"]

def parse_date(row) -> Optional[pd.Timestamp]:
    """
    Build a date from 'Approval/Denial Year' (mid-year proxy).
    Accepts ints, floats like 2024.0, strings like '2024' or '2024.0'.
    """
    for c in YEAR_COL_CANDIDATES:
        if c in row and pd.notna(row[c]):
            val = row[c]
            try:
                if isinstance(val, (int, np.integer)):
                    y = int(val)
                elif isinstance(val, (float, np.floating)):
                    y = int(val)  # 2024.0 -> 2024
                else:
                    s = str(val).strip()
                    try:
                        y = int(float(s))  # "2024.0" -> 2024
                    except Exception:
                        m = re.search(r'(19|20)\d{2}', s)
                        y = int(m.group()) if m else None
                if y is not None and 1900 <= y <= 2100:
                    return pd.Timestamp(year=y, month=6, day=30)
            except Exception:
                pass
    return None

def recency_weight(
    dt: Optional[pd.Timestamp],
    now: Optional[pd.Timestamp] = None,
    half_life_years: float = 3.0,
    recent_years: float = 2.0,
    recent_boost: float = 1.25
) -> float:
    """
    Half-life decay for overall weighting; optional ≤2y boost.
    NOTE: Bonus/penalty in the aggregator is *separately* limited to this & last year.
    """
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None)
    if dt is None:
        return 1.0
    age_years = max(0.0, (now - dt).days / 365.25)
    base = float(0.5 ** (age_years / half_life_years))
    if age_years <= recent_years:
        base *= recent_boost
    return base
# -------------------------
# Concern counting (no zero-shot)
# -------------------------
_BULLET_RE = re.compile(r'(?:^|\n)\s*(?:[-–•\u2022]|\d+\.)\s*([^\n]+)')

def count_concerns(text: Optional[str]) -> int:
    """Count bullet/semicolon/newline-delimited concerns. 'N/A' => 0."""
    if not isinstance(text, str) or not text.strip():
        return 0
    t = text.strip()
    if t.lower() in {"n/a", "na", "none"}:
        return 0
    bullets = _BULLET_RE.findall(t)
    if bullets:
        return len(bullets)
    parts = [p.strip() for p in re.split(r'[;\n]+', t) if p.strip()]
    return len(parts)

def concern_score_from_counts(cit_n: int, off_n: int, k: float = 0.30) -> float:
    """
    Map counts -> [0,1] via exp decay; combine as 0.4*cit + 0.6*off.
    More concerns => lower score.
    """
    cit = math.exp(-k * max(0, cit_n))
    off = math.exp(-k * max(0, off_n))
    return 0.4 * cit + 0.6 * off

# -------------------------
# Zoning bonus helper
# -------------------------
_AG_RE = re.compile(r'\bag\b', re.IGNORECASE)

def has_ag_zoning(row) -> bool:
    z = None
    if 'Zoning' in row and isinstance(row['Zoning'], str):
        z = row['Zoning']
    elif 'Zoning District' in row and isinstance(row['Zoning District'], str):
        z = row['Zoning District']
    if not z:
        return False
    t = z.lower()
    return ('agricultur' in t) or bool(_AG_RE.search(t))

# -------------------------
# Per-project score & weight
# -------------------------
def compute_project_score(
    row,
    *,
    half_life_years: float = 3.0,
    size_focus_low: float = 5.0,
    size_focus_high: float = 20.0,
    size_focus_boost: float = 1.6,     # peak size weight inside 5–20 MW
    zoning_bonus_points: float = 3.0,  # add to per-project score if AG/Agriculture
    concern_decay_k: float = 0.30,
    # --- Scheme-B knobs ---
    recency_for_reco_years: float = 2.0,  # only use recommendation as weak proxy if within last 2y
    reco_beta: float = 0.50               # shrinkage toward 0.5: V_proxy = 0.5 + beta*(rec_ratio-0.5)
):
    # --- Council vote / status -> V_effective ---
    raw_votes = row.get('Council Votes')
    V_direct  = parse_votes(raw_votes)
    S         = map_status(row.get('Status'))

    dt  = parse_date(row)
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_years = None if dt is None else max(0.0, (now - dt).days / 365.25)

    V_effective = V_direct
    if V_effective is None:
        rec_ratio = extract_recommendation_ratio(raw_votes)
        if rec_ratio is not None:
            V_effective = 0.5 + reco_beta * (rec_ratio - 0.5)
        else:
            V_effective = None
    if V_effective is None:
        V_effective = S

    # --- Approval affinity & tri-level approved flag ---
    A = 0.5 * V_effective + 0.5 * S  # in [0,1]
    if S > 0.5:
        approved = 1.0
    elif S == 0.5:
        approved = 0.5
    else:
        approved = 0.0

    # --- Concern score (higher is better) ---
    cit_n = count_concerns(row.get('Citizen Concerns'))
    off_n = count_concerns(row.get('Official Concerns'))
    H = concern_score_from_counts(cit_n, off_n, k=concern_decay_k)

    # --- Base project score ---
    score_p = 100.0 * (0.7 * A + 0.3 * H)

    # --- Zoning bonus ---
    if has_ag_zoning(row):
        score_p += zoning_bonus_points

    # --- Size band (5–20 MW), but peak applies only to approved projects ---
    mw = parse_mw(row.get('Project Size (MW)'))

    def _size_band_weight(mw_val: Optional[float],
                          low: float, high: float,
                          peak: float, outside: float = 1.0) -> float:
        if mw_val is None:
            return outside
        x = max(0.0, float(mw_val))
        if x < low:
            u = x / max(low, 1e-9)
            ramp = 0.5 * (1 - math.cos(math.pi * u))      # 0->1
            return outside + (peak - outside) * ramp
        if x <= high:
            return peak
        span = max(high, 1e-9)
        u = min(1.0, (x - high) / span)                   # 0 at high, 1 at ~2*high
        fall = 0.5 * (1 + math.cos(math.pi * u))          # 1->0
        return outside + (peak - outside) * fall

    size_w_raw = _size_band_weight(mw, size_focus_low, size_focus_high, size_focus_boost, outside=1.0)
    # Gate the boost by approval level (1.0 -> full boost, 0.5 -> half boost, 0.0 -> no boost)
    size_w_eff = 1.0 + (size_w_raw - 1.0) * approved

    # --- Recency & approval tilt ---
    r_w       = recency_weight(dt, half_life_years=half_life_years)
    approve_w = 0.8 + 0.4 * A

    total_w = size_w_eff * r_w * approve_w

    return float(score_p), float(total_w), float(r_w), float(approved), dt


# -------------------------
# Aggregate to AHJ
# -------------------------
def compute_ahj_scores(
    df: pd.DataFrame,
    *,
    half_life_years: float = 3.0,
    count_bonus_k: float = 0.45,
    count_bonus_max: float = 4.0,
    recent_bonus_k: float = 2.0,
    recent_bonus_max: float = 4.0,
    past_any_bonus: float = 2.0,
    recent_denial_k: float = 2.0,
    recent_denial_max: float = 4.0
) -> pd.DataFrame:

    rows = []
    for _, r in df.iterrows():
        s, w, r_w, approved, dt = compute_project_score(r, half_life_years=half_life_years)
        rows.append({
            'AHJ Name': r.get('AHJ Name'),
            'project_score': s,
            'weight': w,
            'recency_weight': r_w,
            'approved': float(approved),
            'date': dt,
            'past_flag': 1 if str(r.get('Past Solar Project?')).strip().lower() == 'yes' else 0
        })

    tmp = pd.DataFrame(rows).dropna(subset=['AHJ Name'])
    if tmp.empty:
        return pd.DataFrame(columns=['AHJ Name', 'public_sentiment', 'n_projects'])

    def _agg(g: pd.DataFrame):
        base = float(np.average(g['project_score'], weights=g['weight']))
        n = int(len(g))
        past_any = int(g['past_flag'].max())

        # project-count bonus (saturating)
        count_bonus = count_bonus_max * math.tanh(count_bonus_k * max(n - 1, 0))

        now = pd.Timestamp.utcnow().tz_localize(None)
        years = pd.to_datetime(g['date']).dt.year
        recent_mask = years.isin([now.year, now.year - 1]).astype(float)

        approved_f = g['approved'].astype(float)

        # Count only clear approvals/denials
        recent_approvals = float((recent_mask * (approved_f == 1.0)).sum())
        recent_denials   = float((recent_mask * (approved_f == 0.0)).sum())

        # Denominator: include ties as neutral (dilutes both bonus & penalty)
        denom = max(int(recent_mask.sum()), 1)

        recent_signal        = recent_approvals / denom
        recent_denial_signal = recent_denials   / denom

        recent_bonus = recent_bonus_max * (1.0 - math.exp(-recent_bonus_k * recent_signal))
        recent_denial_penalty = recent_denial_max * (1.0 - math.exp(-recent_denial_k * recent_denial_signal))

        final = np.clip(base + count_bonus + recent_bonus - recent_denial_penalty + past_any_bonus * past_any, 0, 100)

        return pd.Series({
            'public_sentiment': final,
            'score_raw': base,
            'n_projects': n,
            'recent_signal': recent_signal,
            'recent_bonus': recent_bonus,
            'recent_denial_signal': recent_denial_signal,
            'recent_denial_penalty': recent_denial_penalty,
            'count_bonus': count_bonus,
            'has_past': past_any,
            'latest_date': pd.to_datetime(g['date']).max()
        })

    out = tmp.groupby('AHJ Name', as_index=False).apply(_agg)
    out = out.sort_values('public_sentiment', ascending=False).reset_index(drop=True)

    # df = your final AHJ scores DataFrame
    mask = (out['has_past'] == 0)

    # set all numeric columns to 0
    num_cols = out.select_dtypes(include='number').columns
    out.loc[mask, num_cols] = 0
    # set public_sentiment to 0 for those with no projects
    out.loc[mask, 'public_sentiment'] = 50.0
    # sort by public_sentiment
    out = out.sort_values('public_sentiment', ascending=False).reset_index(drop=True)
    return out


# -------------------------
# Usage
# -------------------------
ahj_scores = compute_ahj_scores(df)
import pandas as pd
import numpy as np

# Ensure numeric
ahj_scores['public_sentiment'] = pd.to_numeric(ahj_scores['public_sentiment'], errors='coerce')

# Segment with bins
bins = [-0.1, 40, 70, 101]
labels = ['Negative', 'Mixed', 'Positive']
ahj_scores['segment'] = pd.cut(ahj_scores['public_sentiment'], bins=bins, labels=labels, right=False)

# fill in county for ahj_scores by using data in df
# deduplicate df to ensure no duplicates in 'AHJ Name'
df = df[['AHJ Name', 'County', 'AHJ Type']].drop_duplicates()
# strip AHJ Type
df['AHJ Type'] = df['AHJ Type'].str.strip()
sentiment = ahj_scores.merge(df[['AHJ Name', 'County','AHJ Type']], on='AHJ Name', how='left')

# save ahj_scores to sentiment.csv
sentiment.to_csv('sentiment.csv', index=False)
