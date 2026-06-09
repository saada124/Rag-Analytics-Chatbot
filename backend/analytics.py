import json
from pathlib import Path
from datetime import date
from typing import Optional, List, Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import STRUCTURED_DIR
from models import llm_client

class PandasCodeOutput(BaseModel):
    explanation: str = Field(description="A brief description of what the analysis does and its logic. MUST be written in French.")
    code: str = Field(description="Pure Python/Pandas code. You must load the dataframe from dfs, e.g., df = dfs['chroma_dataset'], perform operations, and assign the final output to a variable named `result`. Do not wrap this in markdown fence blocks.")
    target_dataframe: str = Field(description="The name of the dataframe being queried.")

DATAFRAMES = {}
CACHE_DIR = Path("cache") / "dataframes"

def load_dataframes():
    global DATAFRAMES
    DATAFRAMES = {}

    if not CACHE_DIR.exists():
        return DATAFRAMES

    for file_path in CACHE_DIR.glob("*.pkl"):
        try:
            df = pd.read_pickle(file_path)
            
            #column names to lowercase with underscores
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str).str.strip()
                
            #parsing date columns
            for col in df.columns:
                if any(kw in col for kw in ["date", "time", "period", "month", "year"]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            DATAFRAMES[file_path.stem] = df
            print(f"[INFO] Loaded & Normalized {file_path.stem}")
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

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
        lines.append('AVAILABLE COLUMNS (with up to 3 sample values each):')
        for col in df.columns:
            dtype = df[col].dtype
            samples = df[col].dropna().unique()[:3].tolist()
            samples = [str(s)[:100] + '...' if len(str(s)) > 100 else s for s in samples]
            lines.append(f'  - `{col}` (Samples: {samples})')
        lines.append('')
    return "\n".join(lines)

#langchain
def generate_pandas_query(query: str) -> PandasCodeOutput:
    today_str = date.today().isoformat()
    schema_desc = get_schema_description()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert business intelligence and data analyst assistant.
Today's date is {today_str}.

You have access to the following DataFrames in the dictionary `dfs`.
YOU MUST ONLY USE THE DATAFRAME KEYS AND COLUMNS LISTED BELOW. DO NOT INVENT COLUMNS.

{schema_desc}

Your task is to write Pandas Python code to answer the user's analytical query.
Rules for writing the code:
1. Extract the correct dataframe from the `dfs` dictionary using the exact DATAFRAME KEY shown above (e.g., if the key is "my_data", use `df = dfs["my_data"]`).
2. Use the provided column sample values to infer formatting (like delimiters). However, DO NOT assume the samples contain all possible values or keywords.
3. When searching for features, concepts, or keywords (e.g., "embeddings", "enterprise"), you MUST smartly search across ALL text/string columns that could logically contain that information (e.g., descriptions, tags, reviews, product names). Combine conditions using `|` (OR), for example: `(df['tags'].str.contains('keyword', case=False, na=False)) | (df['description'].str.contains('keyword', case=False, na=False))`.
4. You MUST store the final result of your computation/retrieval in the variable `result`.
5. For date operations:
   - Convert date columns using `pd.to_datetime(df[col], errors='coerce')` before comparing them.
   - For relative dates, today's date is {today_str}.
6. Output size: If the user asks for a count, total, or a full list (e.g., "how many", "list all", "show all"), return ALL matching rows — do NOT add `.head()`. Only use `.head(50)` if the user is browsing or exploring without specifying they want the full result.
7. Keep it safe: do not use external library imports, system calls, or built-in file writing/reading functions. Only use pandas and numpy.
8. If the user asks for specific columns (e.g., "show me the user's country... and the score"), select only those columns in the final result (e.g. `result = df[['country', 'score']]`).
9. Do not include markdown code block syntax (like ```python) in your response property for `code`. Provide raw Python code lines.
10. Your generated explanation (the `explanation` property) must ALWAYS be written in French.
11. When filtering for exact states or categories (like status="active"), use exact equality (`==`) rather than exclusionary logic (like `!= 'inactive'`) unless the user explicitly asks to exclude something.
"""),
        ("user", "{query}")
    ])

    structured_llm = llm_client.with_structured_output(PandasCodeOutput)
    chain = prompt | structured_llm
    return chain.invoke({"query": query})

def generate_pandas_query_correction(query: str, original_code: str, error_msg: str) -> PandasCodeOutput:
    today_str = date.today().isoformat()
    schema_desc = get_schema_description()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert business intelligence and data analyst assistant.
Today's date is {today_str}.

Available DataFrames in the dictionary `dfs` and their columns with data types:
{schema_desc}

Your previously generated Pandas code failed with an execution error.
Original Query: {query}
Generated Code:
{original_code}

Execution Error:
{error_msg}

Your task is to fix the bug in the code and return the corrected version. Follow the same rules:
1. Always start by loading the correct dataframe from `dfs` dictionary, e.g., `df = dfs['chroma_dataset']`.
2. Perform all necessary filtering, projections, groupings, aggregations, or sorting. When filtering for keywords, smartly search across ALL relevant text/string columns using `|` (OR) conditions.
3. You MUST store the final result of your computation/retrieval in the variable `result`.
4. Ensure string matching handles cases/spaces cleanly (e.g., use `.str.lower()` and `.str.strip()` to make comparisons case-insensitive).
5. For date operations:
   - Convert date columns using `pd.to_datetime(df[col], errors='coerce')` before comparing them.
   - For relative dates, today's date is {today_str}.
6. Output size: If the user asks for a count, total, or a full list, return ALL matching rows. Only use `.head(50)` if the user is browsing or exploring without specifying they want the full result.
7. Keep it safe: do not use external library imports, system calls, or built-in file writing/reading functions. Only use pandas and numpy.
8. If the user asks for specific columns, select only those columns in the final result.
9. Do not include markdown code block syntax (like ```python) in your response property for `code`. Provide raw Python code lines.
10. Your generated explanation (the `explanation` property) must ALWAYS be written in French.
11. When filtering for exact states or categories (like status="active"), use exact equality (`==`) rather than exclusionary logic (like `!= 'inactive'`) unless the user explicitly asks to exclude something.
"""),
        ("user", "Please fix the code above to resolve the execution error.")
    ])

    structured_llm = llm_client.with_structured_output(PandasCodeOutput)
    chain = prompt | structured_llm
    return chain.invoke({})

def execute_pandas_code(code: str, dfs: dict) -> Any:
    #security filter
    dangerous = ["import os", "import sys", "subprocess", "eval(", "exec(", "open(", "write(", "builtins", "__import__", "shutil"]
    for word in dangerous:
        if word in code:
            raise ValueError(f"Security check failed: code contains potentially unsafe keyword '{word}'")
    
    local_vars = {
        "dfs": dfs,
        "pd": pd,
        "np": np,
        "result": None
    }
    exec(code, {}, local_vars)
    return local_vars.get("result")

#result formatter
def format_result(result: Any) -> dict:
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_numpy(v) for v in obj.tolist()]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj
        elif pd.api.types.is_scalar(obj) and pd.isna(obj):
            return None
        return obj

    if result is None:
        return {"answer": "No result was returned.", "data": []}
    
    if isinstance(result, pd.DataFrame):
        records = result.fillna("").to_dict(orient="records")
        ret = {
            "answer": f"Found {len(result)} records matching the query.",
            "data": records
        }
    elif isinstance(result, pd.Series):
        records = result.reset_index().fillna("").to_dict(orient="records")
        ret = {
            "answer": "Calculated series result.",
            "data": records
        }
    elif isinstance(result, (dict, list)):
        ret = {
            "answer": "Analysis query succeeded.",
            "data": result
        }
    else:
        ret = {
            "answer": f"Result: {result}",
            "data": [{"value": result}]
        }
    
    return convert_numpy(ret)

def analyze_query(query: str) -> dict:
    if not DATAFRAMES:
        load_dataframes()

    if not DATAFRAMES:
        return {"answer": "No structured datasets loaded.", "data": []}

    code = None
    explanation = ""
    max_retries = 3
    error_msg = ""

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                pandas_output = generate_pandas_query(query)
            else:
                print(f"[ANALYTICS] Attempt {attempt + 1}: Correcting previous failure...")
                pandas_output = generate_pandas_query_correction(query, code, error_msg)
            
            code = pandas_output.code
            explanation = pandas_output.explanation
            print(f"[ANALYTICS] Explanation: {explanation}")
            print(f"[ANALYTICS] Code generated:\n{code}")

            result = execute_pandas_code(code, DATAFRAMES)
            
            # Format result
            formatted = format_result(result)
            formatted["answer"] = f"{explanation}\n\n{formatted['answer']}"
            return formatted

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[ANALYTICS ERROR] Attempt {attempt + 1} failed: {error_msg}")
            if attempt == max_retries - 1:
                import traceback
                traceback.print_exc()
                return {"answer": f"Failed to execute calculation after {max_retries} attempts: {str(e)}", "data": []}


if __name__ == "__main__":
    import os
    print("Testing Analytics Code Generator...")
    load_dataframes()
    if DATAFRAMES:
        q = "show me the user's country with the signup date before 2022 and show the score"
        res = analyze_query(q)
        print(json.dumps(res, indent=2))
    else:
        print("No cache dataframes found. Make sure to run ingestion first.")