import logging

logger = logging.getLogger(__name__)

import ast
import json
import time
import textwrap
import warnings
import builtins
from pathlib import Path
from datetime import date, datetime
from typing import Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import CACHE_DIR
from models import llm_client


class PandasCodeOutput(BaseModel):
    explanation: str = Field(description="ONE short sentence in French (max ~25 words) describing ONLY HOW the result was computed: which dataframe/columns were used and the operation applied (e.g. 'regroupement par client puis somme du chiffre d'affaires', 'idxmax sur le score'). Do NOT restate the question and do NOT describe the result values - only the calculation method, so the user can verify the logic if the answer looks wrong.")
    code: str = Field(description="Pure Python/Pandas code. You must load the dataframe from dfs, e.g., df = dfs['chroma_dataset'], perform operations, and assign the final output to a variable named `result`. Do not wrap this in markdown fence blocks.")
    target_dataframe: str = Field(description="The name of the dataframe being queried.")


DATAFRAMES = {}
_CACHE_DIR = Path(CACHE_DIR)

# Column-name hints used to decide which columns *might* be dates.
_DATE_HINTS = ("date", "time", "period", "month", "year")


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y%m%d",
)


def _maybe_parse_dates(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:

    for col in df.columns:
        if not any(kw in col for kw in _DATE_HINTS):
            continue
        # Never coerce numeric columns (e.g. a bare integer year).
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        sample = non_null.astype(str).head(200)
        chosen_fmt = None
        for fmt in _DATE_FORMATS:
            if pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() >= threshold:
                chosen_fmt = fmt
                break
        if chosen_fmt is not None:
            parsed = pd.to_datetime(df[col].astype(str), format=chosen_fmt, errors="coerce")
        else:
            parsed = pd.to_datetime(df[col].astype(str), format="ISO8601", errors="coerce")
            if parsed.loc[non_null.index].notna().mean() < threshold:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col].astype(str), errors="coerce")
        # Fraction of originally-present values that parsed successfully.
        valid_ratio = parsed.loc[non_null.index].notna().mean()
        if valid_ratio >= threshold:
            df[col] = parsed
    return df


def load_dataframes():
    global DATAFRAMES
    DATAFRAMES = {}

    if not _CACHE_DIR.exists():
        return DATAFRAMES

    total_start = time.perf_counter()
    for file_path in _CACHE_DIR.glob("*.pkl"):
        try:
            file_start = time.perf_counter()
            df = pd.read_pickle(file_path)

            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

            df = _maybe_parse_dates(df)

            DATAFRAMES[file_path.stem] = df
            elapsed = time.perf_counter() - file_start
            logger.info(
                f"[INFO] Loaded & Normalized {file_path.stem} "
                f"({len(df):,} rows x {len(df.columns)} cols) in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.error(f"[ERROR] {file_path.name}: {e}")

    logger.info(f"[STARTUP] All dataframes ready in {time.perf_counter() - total_start:.2f}s")
    return DATAFRAMES


def get_available_datasets():
    if not DATAFRAMES:
        load_dataframes()
    return list(DATAFRAMES.keys())


def get_schema_description() -> str:
    if not DATAFRAMES:
        load_dataframes()
    lines = []
    for name, df in DATAFRAMES.items():
        lines.append(f'DATAFRAME KEY: "{name}"')
        lines.append('AVAILABLE COLUMNS (with up to 2 sample values each):')
        for col in df.columns:
            samples = df[col].dropna().unique()[:2].tolist()
            samples = [str(s)[:50] + '...' if len(str(s)) > 50 else s for s in samples]
            lines.append(f'  - `{col}` (Samples: {samples})')
        lines.append('')
    return "\n".join(lines)


_RULES = """Rules for writing the code:
1. Extract the correct dataframe from the `dfs` dictionary using the exact DATAFRAME KEY shown above (e.g., if the key is "my_data", use `df = dfs["my_data"]`).
2. Use the provided column sample values to infer formatting (like delimiters). However, DO NOT assume the samples contain all possible values or keywords.
3. When searching for features, concepts, or keywords (e.g., "embeddings", "enterprise"), you MUST smartly search across ALL text/string columns that could logically contain that information. Combine conditions using `|` (OR), for example: `(df['tags'].str.contains('keyword', case=False, na=False)) | (df['description'].str.contains('keyword', case=False, na=False))`.
4. You MUST store the final result of your computation/retrieval in the variable `result`.
5. For date operations:
   - Convert date columns using `pd.to_datetime(df[col], errors='coerce', format='mixed')` before comparing them.
   - For relative dates, today's date is {today}.
6. Output size: If the user asks for a count, total, or a full list (e.g., "how many", "list all", "show all"), return ALL matching rows - do NOT add `.head()`. Only use `.head(50)` if the user is browsing or exploring without specifying they want the full result.
7. Keep it safe: do not import anything, do not use dunder attributes (like __class__), and do not call eval/exec/open/getattr. Only use pandas (pd) and numpy (np) and basic builtins.
8. If the user asks for specific columns (e.g., "show me the user's country... and the score"), select only those columns in the final result (e.g. `result = df[['country', 'score']]`).
9. Do not include markdown code block syntax (like triple backticks) in your response property for `code`. Provide raw Python code lines.
10. Your generated explanation (the `explanation` property) must ALWAYS be written in French.
11. When filtering for exact states or categories (like status="active"), use exact equality (`==`) rather than exclusionary logic (like `!= 'inactive'`) unless the user explicitly asks to exclude something.
12. Only when the user EXPLICITLY provides a specific text format/template to match should your Python code build a single formatted string matching that template exactly (using \\n for newlines and f-strings) and assign it to the `result` variable.
13. When finding entities (like customers, products, or reps) whose total/revenue exceeds a certain amount, you MUST group by the entity and sum the values FIRST, and then filter the aggregated result. Do not just filter individual rows.
14. CRITICAL: If the user's query requires data or metrics that DO NOT exist in the available columns (e.g., asking for "profit margin" when there is no cost column), you MUST NOT fabricate or invent a calculation from unrelated columns. Instead, explain which specific columns or data would be needed and that the current dataset does not contain them.
15. PREFER STRUCTURED, COLUMNAR OUTPUT - DO NOT concatenate multiple values into one big string. When the answer contains several named metrics (e.g. best representative, top customer, top product, total revenue), assign `result` to a SINGLE-ROW pandas DataFrame with one clearly-named column per metric (e.g. a one-row DataFrame built from a list containing a single dict, whose columns are `top_representative`, `top_customer`, `top_product` and `total_revenue`, one value per column). Use a multi-row DataFrame for lists/rankings. This lets the UI display each value in its own column instead of one long sentence.
16. If the user asks about concentration, risk, or market share, ALSO compute and include the supporting figures needed to quantify it as extra named columns - e.g. the total revenue, the top entity's revenue, and its share of the total (top_revenue / total_revenue) - so the risk can be reported with real numbers rather than vaguely."""

_SYSTEM_TEMPLATE = (
    "You are an expert business intelligence and data analyst assistant.\n"
    "Today's date is {today}.\n\n"
    "You have access to the following DataFrames in the dictionary `dfs`.\n"
    "YOU MUST ONLY USE THE DATAFRAME KEYS AND COLUMNS LISTED BELOW. DO NOT INVENT COLUMNS.\n\n"
    "{schema_desc}\n\n"
    "Your task is to write Pandas Python code to answer the user's analytical query.\n"
    "{rules}\n"
)

_CORRECTION_SYSTEM_TEMPLATE = (
    "You are an expert business intelligence and data analyst assistant.\n"
    "Today's date is {today}.\n\n"
    "Available DataFrames in the dictionary `dfs` and their columns with sample values:\n"
    "{schema_desc}\n\n"
    "Your previously generated Pandas code failed with an execution error.\n"
    "Original Query: {original_query}\n"
    "Generated Code:\n{original_code}\n\n"
    "Execution Error:\n{error_msg}\n\n"
    "Your task is to fix the bug in the code and return the corrected version. "
    "Follow the same rules:\n"
    "{rules}\n"
)


def generate_pandas_query(query: str) -> PandasCodeOutput:
    today_str = date.today().isoformat()
    schema_desc = get_schema_description()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE),
        ("user", "{query}"),
    ])

    structured_llm = llm_client.with_structured_output(PandasCodeOutput)
    chain = prompt | structured_llm
    return chain.invoke({
        "query": query,
        "schema_desc": schema_desc,
        "rules": _RULES.format(today=today_str),
        "today": today_str,
    })


def generate_pandas_query_correction(query: str, original_code: str, error_msg: str) -> PandasCodeOutput:
    today_str = date.today().isoformat()
    schema_desc = get_schema_description()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _CORRECTION_SYSTEM_TEMPLATE),
        ("user", "Please fix the code above to resolve the execution error."),
    ])

    structured_llm = llm_client.with_structured_output(PandasCodeOutput)
    chain = prompt | structured_llm
    return chain.invoke({
        "today": today_str,
        "schema_desc": schema_desc,
        "original_query": query,
        "original_code": original_code,
        "error_msg": error_msg,
        "rules": _RULES.format(today=today_str),
    })

#safe execution
_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "reversed", "round",
    "set", "sorted", "str", "sum", "tuple", "zip", "print",
}

_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "exit", "quit",
    "breakpoint", "memoryview", "object", "type", "super",
}


def _validate_code(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Security check failed: imports are not allowed.")
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError(f"Security check failed: dunder attribute '{node.attr}' is not allowed.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"Security check failed: name '{node.id}' is not allowed.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise ValueError(f"Security check failed: call to '{node.func.id}' is not allowed.")


def execute_pandas_code(code: str, dfs: dict) -> Any:
    code = textwrap.dedent(code).strip()
    _validate_code(code)

    safe_builtins = {
        name: getattr(builtins, name)
        for name in _ALLOWED_BUILTINS
        if hasattr(builtins, name)
    }
    safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    local_vars = {"dfs": dfs, "result": None}

    exec(compile(code, "<analytics>", "exec"), safe_globals, local_vars)
    return local_vars.get("result")

def _df_to_records(df) -> list:
    """Convert a DataFrame to JSON-friendly records: datetimes -> ISO strings,
    NaN/NaT -> None. Avoids fillna('') dtype clashes on datetime columns."""
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d").where(safe[col].notna(), None)
    safe = safe.astype(object).where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def convert_numpy(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [convert_numpy(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    if pd.api.types.is_scalar(obj) and pd.isna(obj):
        return None
    return obj


def format_result(result: Any) -> dict:
    if result is None:
        return {"answer": "No result was returned.", "data": []}

    if isinstance(result, pd.DataFrame):
        ret = {
            "answer": f"Found {len(result)} records matching the query.",
            "data": _df_to_records(result),
        }
    elif isinstance(result, pd.Series):
        ret = {
            "answer": "Calculated series result.",
            "data": _df_to_records(result.reset_index()),
        }
    elif isinstance(result, dict):
        ret = {"answer": f"Computed {len(result)} value(s).", "data": [result]}
    elif isinstance(result, list):
        if all(isinstance(item, dict) for item in result):
            data = result
        else:
            data = [{"value": item} for item in result]
        ret = {"answer": f"Found {len(result)} record(s).", "data": data}
    else:
        ret = {"answer": f"Result: {result}", "data": [{"value": result}]}

    return convert_numpy(ret)



_COMPOSER_SYSTEM_TEMPLATE = (
    "You are a business intelligence analyst. Write the FINAL answer for the user, in FRENCH.\n\n"
    "You are given the user's question, a short note on how the result was computed, "
    "and the COMPUTED RESULT as JSON (this is the ground truth).\n\n"
    "Rules:\n"
    "- Use ONLY the figures present in the computed result. NEVER invent, estimate, or label a value 'hypothetical'.\n"
    "- Address EVERY part of the question. Give each requested item its own short bold title "
    "(e.g. **Representant le plus performant**) followed by one or more detailed bullet points with a simple explanation "
    "containing the concrete figures (names, amounts, shares) taken from the result.\n"
    "- If something the user asked for is not present in the computed result, add a short bullet "
    "stating what is missing; if the result still allows a quantified observation (e.g. a top "
    "entity's share of the total), give it.\n"
    "- Be concise: bullet points only, no long introduction or conclusion paragraphs.\n"
    "- Keep amounts exactly as they appear in the data."
)

_composer_prompt = ChatPromptTemplate.from_messages([
    ("system", _COMPOSER_SYSTEM_TEMPLATE),
    ("user", "Question:\n{query}\n\nHow it was computed:\n{explanation}\n\nComputed result (JSON):\n{data}"),
])
_composer_chain = _composer_prompt | llm_client | StrOutputParser()


def compose_analytics_answer(query: str, explanation: str, data: Any) -> str:
    """Phrase the computed result as a detailed, bullet-point French answer.

    Falls back to the concise explanation if the composer call fails, so a
    formatting hiccup never hides an otherwise-correct calculation.
    """
    try:
        data_json = json.dumps(convert_numpy(data), ensure_ascii=False, default=str)
        # Guard against a huge table blowing the context budget.
        if len(data_json) > 8000:
            data_json = data_json[:8000] + " ...(truncated)"
        return _composer_chain.invoke({
            "query": query,
            "explanation": explanation,
            "data": data_json,
        }).strip()
    except Exception as e:
        logger.warning(f"[ANALYTICS] Answer composer failed, using concise explanation: {e}")
        return explanation


def analyze_query(query: str) -> dict:
    if not DATAFRAMES:
        load_dataframes()
    if not DATAFRAMES:
        return {"answer": "No structured datasets loaded.", "data": []}

    code = None
    explanation = ""
    error_msg = ""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                pandas_output = generate_pandas_query(query)
            else:
                logger.info(f"[ANALYTICS] Attempt {attempt + 1}: Correcting previous failure...")
                pandas_output = generate_pandas_query_correction(query, code, error_msg)

            code = pandas_output.code
            explanation = pandas_output.explanation
            logger.info(f"[ANALYTICS] Explanation: {explanation}")
            logger.info(f"[ANALYTICS] Code generated:\n{code}")

            result = execute_pandas_code(code, DATAFRAMES)

            formatted = format_result(result)
            formatted["explanation"] = explanation
            formatted["answer"] = compose_analytics_answer(
                query, explanation, formatted.get("data")
            )
            return formatted

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[ANALYTICS ERROR] Attempt {attempt + 1} failed: {error_msg}")
            if attempt == max_retries - 1:
                logger.exception(
                    "Analytics calculation failed after %s attempts", max_retries
                )
                return {
                    "answer": f"Failed to execute calculation after {max_retries} attempts: {str(e)}",
                    "data": [],
                }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Testing Analytics Code Generator...")
    load_dataframes()
    if DATAFRAMES:
        q = "show me the user's country with the signup date before 2022 and show the score"
        res = analyze_query(q)
        logger.info(json.dumps(res, indent=2, default=str))
    else:
        logger.info("No cache dataframes found. Make sure to run ingestion first.")
