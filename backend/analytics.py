import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR, STRUCTURED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


STRUCTURED_CACHE_DIR = Path(CACHE_DIR) / "structured"
MANIFEST_PATH = Path(CACHE_DIR) / "ingest_manifest.json"

DATAFRAMES: Dict[str, pd.DataFrame] = {}
DATAFRAME_META: Dict[str, Dict[str, Any]] = {}

def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _try_parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_probably_date_column(series: pd.Series) -> bool:
    if series.empty:
        return False
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
    return parsed.notna().mean() >= 0.6


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load manifest: %s", exc)
    return {"files": {}}


def _structured_files_from_manifest() -> List[Tuple[str, Dict[str, Any]]]:
    manifest = _load_manifest()
    items = []
    for file_path, meta in manifest.get("files", {}).items():
        if meta.get("kind") == "structured":
            items.append((file_path, meta))
    return items


def _cached_dataframe_path(meta: Dict[str, Any]) -> Optional[str]:
    path = meta.get("cached_dataframe_path")
    if path and os.path.exists(path):
        return path
    return None


def load_cached_dataframes(force_reload: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load all structured files cached by ingest.py into memory.
    """
    global DATAFRAMES, DATAFRAME_META

    if DATAFRAMES and not force_reload:
        return DATAFRAMES

    DATAFRAMES = {}
    DATAFRAME_META = {}

    for file_path, meta in _structured_files_from_manifest():
        cache_path = _cached_dataframe_path(meta)
        if not cache_path:
            logger.warning("Missing cached dataframe for %s", file_path)
            continue

        try:
            df = pd.read_pickle(cache_path)
            key = file_path
            DATAFRAMES[key] = df
            DATAFRAME_META[key] = meta
            logger.info("Loaded structured dataset: %s (%d rows)", file_path, len(df))
        except Exception as exc:
            logger.warning("Failed to load dataframe %s: %s", file_path, exc)

    return DATAFRAMES


def get_available_datasets() -> List[Dict[str, Any]]:
    load_cached_dataframes()
    datasets = []

    for file_path, df in DATAFRAMES.items():
        meta = DATAFRAME_META.get(file_path, {})
        datasets.append(
            {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
                "metadata": meta,
            }
        )

    return datasets


def infer_best_dataset(query: str) -> Optional[Tuple[str, pd.DataFrame]]:
    """
    Very simple dataset selector.
    If there are multiple structured files, pick the one whose
    name or column names match the query best.
    """
    load_cached_dataframes()

    if not DATAFRAMES:
        return None

    q = _normalize_text(query)
    best_score = -1
    best_item = None

    for file_path, df in DATAFRAMES.items():
        score = 0
        file_name = _normalize_text(Path(file_path).stem)

        if file_name and file_name in q:
            score += 5

        for col in df.columns:
            col_text = _normalize_text(col)
            if col_text and col_text in q:
                score += 2

        if score > best_score:
            best_score = score
            best_item = (file_path, df)

    return best_item or next(iter(DATAFRAMES.items()))

DATE_PATTERNS = [
    r"\bfrom\s+([a-zA-Z0-9\-/ ]+)\s+to\s+([a-zA-Z0-9\-/ ]+)",
    r"\bbetween\s+([a-zA-Z0-9\-/ ]+)\s+and\s+([a-zA-Z0-9\-/ ]+)",
]


@dataclass
class QueryFilters:
    raw_query: str
    include_terms: List[str]
    exclude_terms: List[str]
    date_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    limit: Optional[int] = None


def extract_filters(query: str) -> QueryFilters:
    q = query.strip()

    exclude_terms = re.findall(r"\bexcluding\s+([a-zA-Z0-9_ \-]+)", q, flags=re.I)
    exclude_terms += re.findall(r"\bexcept\s+([a-zA-Z0-9_ \-]+)", q, flags=re.I)

    include_terms = []
    cleaned = re.sub(r"\bexcluding\s+[a-zA-Z0-9_ \-]+", "", q, flags=re.I)
    cleaned = re.sub(r"\bexcept\s+[a-zA-Z0-9_ \-]+", "", cleaned, flags=re.I)

    date_start = None
    date_end = None

    for pattern in DATE_PATTERNS:
        match = re.search(pattern, q, flags=re.I)
        if match:
            d1 = _try_parse_date(match.group(1))
            d2 = _try_parse_date(match.group(2))
            if d1 is not None and d2 is not None:
                date_start, date_end = d1, d2
                break

    limit = None
    m = re.search(r"\btop\s+(\d+)\b", q, flags=re.I)
    if m:
        limit = int(m.group(1))

    return QueryFilters(
        raw_query=query,
        include_terms=include_terms,
        exclude_terms=[t.strip().lower() for t in exclude_terms if t.strip()],
        date_start=date_start,
        date_end=date_end,
        limit=limit,
    )

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    normalized_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        candidate_norm = candidate.strip().lower()
        if candidate_norm in normalized_map:
            return normalized_map[candidate_norm]

    for col in df.columns:
        col_norm = str(col).strip().lower()
        for candidate in candidates:
            if candidate.strip().lower() in col_norm:
                return col

    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "date", "order_date", "invoice_date", "created_at",
        "created", "sale_date", "transaction_date", "timestamp"
    ]
    col = find_column(df, candidates)
    if col:
        return col

    for col in df.columns:
        if _is_probably_date_column(df[col]):
            return col

    return None


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        series = _safe_numeric(df[col])
        if series.notna().sum() > 0:
            numeric_cols.append(col)
    return numeric_cols


def detect_quantity_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["quantity", "qty", "units", "unit", "count", "pieces"])


def detect_price_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["price", "unit_price", "sale_price", "amount", "value", "revenue", "total", "sales"])

def apply_text_filters(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.lower()
    out = df.copy()

    # Very lightweight text matching against object columns
    text_terms = []
    for term in re.findall(r'"([^"]+)"', query):
        text_terms.append(term)

    if not text_terms:
        words = [
            w for w in re.findall(r"[a-zA-Z0-9\u00C0-\u017F]+", query)
            if len(w) >= 3
        ]
        stop = {
            "what", "were", "show", "total", "sales", "sale", "the", "and", "for",
            "with", "from", "this", "that", "are", "was", "all", "top", "best",
            "during", "between", "excluding", "except", "customer", "customers"
        }
        text_terms = [w for w in words if w.lower() not in stop][:5]

    if not text_terms:
        return out

    object_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    if not object_cols:
        return out

    mask = pd.Series(False, index=out.index)
    for col in object_cols:
        col_values = out[col].astype(str).str.lower()
        for term in text_terms:
            mask = mask | col_values.str.contains(re.escape(term.lower()), na=False)

    if mask.any():
        return out[mask]
    return out


def apply_date_filter(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if start is None and end is None:
        return df

    date_col = detect_date_column(df)
    if not date_col:
        return df

    out = df.copy()
    parsed = pd.to_datetime(out[date_col], errors="coerce", infer_datetime_format=True)
    valid = parsed.notna()

    if start is not None:
        valid = valid & (parsed >= start)
    if end is not None:
        valid = valid & (parsed <= end)

    return out[valid]


def apply_exclusions(df: pd.DataFrame, exclude_terms: List[str]) -> pd.DataFrame:
    if not exclude_terms:
        return df

    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    if not object_cols:
        return out

    mask = pd.Series(True, index=out.index)
    for term in exclude_terms:
        term_mask = pd.Series(False, index=out.index)
        for col in object_cols:
            col_values = out[col].astype(str).str.lower()
            term_mask = term_mask | col_values.str.contains(re.escape(term), na=False)
        mask = mask & (~term_mask)

    return out[mask]

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = detect_numeric_columns(df)
    date_col = detect_date_column(df)

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "date_column": date_col,
    }

    if numeric_cols:
        col = numeric_cols[0]
        s = _safe_numeric(df[col])
        summary["sample_numeric_column"] = col
        summary["sum"] = float(s.sum(skipna=True))
        summary["mean"] = float(s.mean(skipna=True))
        summary["min"] = float(s.min(skipna=True))
        summary["max"] = float(s.max(skipna=True))

    return summary


def total_sum(df: pd.DataFrame, column: Optional[str] = None) -> Tuple[float, str]:
    if column and column in df.columns:
        s = _safe_numeric(df[column])
        return float(s.sum(skipna=True)), column

    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        return 0.0, ""
    col = numeric_cols[0]
    s = _safe_numeric(df[col])
    return float(s.sum(skipna=True)), col


def average_value(df: pd.DataFrame, column: Optional[str] = None) -> Tuple[float, str]:
    if column and column in df.columns:
        s = _safe_numeric(df[column])
        return float(s.mean(skipna=True)), column

    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        return 0.0, ""
    col = numeric_cols[0]
    s = _safe_numeric(df[col])
    return float(s.mean(skipna=True)), col


def top_n_by_column(
    df: pd.DataFrame,
    value_column: Optional[str] = None,
    n: int = 10,
) -> pd.DataFrame:
    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        return df.head(n)

    if value_column and value_column in df.columns:
        col = value_column
    else:
        col = numeric_cols[0]

    out = df.copy()
    out[col] = _safe_numeric(out[col])
    out = out.sort_values(by=col, ascending=False)
    return out.head(n)


def group_and_aggregate(
    df: pd.DataFrame,
    group_by: Optional[str] = None,
    value_column: Optional[str] = None,
    agg: str = "sum",
    top_n: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        return df.head(top_n)

    if value_column is None or value_column not in df.columns:
        value_column = numeric_cols[0]

    if group_by is None or group_by not in df.columns:
        # pick a likely categorical column
        cat_cols = [
            c for c in df.columns
            if c != value_column and df[c].dtype in ["object", "string", "category"]
        ]
        group_by = cat_cols[0] if cat_cols else None

    if group_by is None:
        return top_n_by_column(df, value_column, top_n)

    out = df.copy()
    out[value_column] = _safe_numeric(out[value_column])

    if agg == "mean":
        grouped = out.groupby(group_by, dropna=False)[value_column].mean()
    elif agg == "count":
        grouped = out.groupby(group_by, dropna=False)[value_column].count()
    else:
        grouped = out.groupby(group_by, dropna=False)[value_column].sum()

    result = grouped.sort_values(ascending=False).head(top_n).reset_index()
    result.columns = [group_by, f"{agg}_{value_column}"]
    return result


def percentage_change(current: float, previous: float) -> float:
    if previous in (0, 0.0, None) or pd.isna(previous):
        return np.nan
    return ((current - previous) / previous) * 100.0

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Main entry for structured-data questions.
    Returns a machine-readable result plus a natural language summary.
    """
    load_cached_dataframes()

    if not DATAFRAMES:
        return {
            "ok": False,
            "error": "No structured datasets have been loaded.",
            "answer": "I could not find any structured data files to analyze.",
        }

    selected = infer_best_dataset(query)
    if not selected:
        return {
            "ok": False,
            "error": "No dataset matched the query.",
            "answer": "I could not identify a suitable dataset for this question.",
        }

    file_path, df = selected
    filters = extract_filters(query)

    working = df.copy()
    working = apply_text_filters(working, query)
    working = apply_exclusions(working, filters.exclude_terms)
    working = apply_date_filter(working, filters.date_start, filters.date_end)

    if working.empty:
        return {
            "ok": True,
            "dataset": Path(file_path).name,
            "rows": 0,
            "columns": list(df.columns),
            "answer": (
                f"I found the dataset '{Path(file_path).name}', but no rows matched "
                f"the filters in your question."
            ),
            "data": [],
            "summary": {
                "rows": 0,
                "note": "No rows matched the filters.",
            },
        }

    q = query.lower()

    # Decide query intent
    if any(term in q for term in ["average", "avg", "mean"]):
        value, col = average_value(working)
        answer = (
            f"The average value for '{col}' in '{Path(file_path).name}' is {value:.2f}."
            if col else
            f"The average value in '{Path(file_path).name}' is {value:.2f}."
        )
        return {
            "ok": True,
            "dataset": Path(file_path).name,
            "metric": "average",
            "column": col,
            "value": value,
            "answer": answer,
            "summary": summarize_dataframe(working),
            "data": working.head(20).to_dict(orient="records"),
        }

    if any(term in q for term in ["top", "highest", "largest", "best", "most"]):
        value_col = detect_price_column(working) or detect_numeric_columns(working)[0] if detect_numeric_columns(working) else None
        group_by_col = None

        # heuristics for common business columns
        for candidate in [
            "product", "product_name", "item", "category",
            "customer", "customer_name", "region", "country", "salesperson"
        ]:
            found = find_column(working, [candidate])
            if found:
                group_by_col = found
                break

        result_df = group_and_aggregate(
            working,
            group_by=group_by_col,
            value_column=value_col,
            agg="sum",
            top_n=filters.limit or 10,
        )

        answer = (
            f"Here are the top results from '{Path(file_path).name}'."
            if not result_df.empty else
            f"I could not compute a ranking from '{Path(file_path).name}'."
        )
        return {
            "ok": True,
            "dataset": Path(file_path).name,
            "metric": "top",
            "group_by": group_by_col,
            "value_column": value_col,
            "answer": answer,
            "summary": summarize_dataframe(working),
            "data": result_df.to_dict(orient="records"),
        }

    if any(term in q for term in ["sum", "total", "revenue", "sales", "amount", "profit"]):
        value_col = detect_price_column(working)
        if value_col is None:
            numeric_cols = detect_numeric_columns(working)
            value_col = numeric_cols[0] if numeric_cols else None

        value, used_col = total_sum(working, value_col)
        answer = (
            f"The total for '{used_col}' in '{Path(file_path).name}' is {value:.2f}."
            if used_col else
            f"The total value in '{Path(file_path).name}' is {value:.2f}."
        )
        return {
            "ok": True,
            "dataset": Path(file_path).name,
            "metric": "sum",
            "column": used_col,
            "value": value,
            "answer": answer,
            "summary": summarize_dataframe(working),
            "data": working.head(20).to_dict(orient="records"),
        }

    # Default: return a useful summary and a small sample
    summary = summarize_dataframe(working)
    sample = working.head(10).fillna("").to_dict(orient="records")

    answer = (
        f"I found '{Path(file_path).name}' and matched {len(working)} rows. "
        f"It contains {len(working.columns)} columns."
    )

    return {
        "ok": True,
        "dataset": Path(file_path).name,
        "answer": answer,
        "summary": summary,
        "data": sample,
    }

def explain_analytics_result(result: Dict[str, Any]) -> str:
    """
    Optional helper to convert a structured result into a short readable text.
    """
    if not result.get("ok"):
        return result.get("answer", "Could not analyze the query.")

    if "metric" in result and result["metric"] == "sum":
        value = result.get("value")
        col = result.get("column")
        dataset = result.get("dataset", "dataset")
        if col:
            return f"Total {col} in {dataset}: {value:.2f}"
        return f"Total in {dataset}: {value:.2f}"

    if "metric" in result and result["metric"] == "average":
        value = result.get("value")
        col = result.get("column")
        dataset = result.get("dataset", "dataset")
        if col:
            return f"Average {col} in {dataset}: {value:.2f}"
        return f"Average in {dataset}: {value:.2f}"

    return result.get("answer", "")

if __name__ == "__main__":
    load_cached_dataframes(force_reload=True)
    print("Loaded datasets:")
    for ds in get_available_datasets():
        print(ds["file_name"], ds["rows"], ds["columns"])

    test_query = "What are the top 5 products in March excluding cancelled orders?"
    out = analyze_query(test_query)
    print("\nRESULT:")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))