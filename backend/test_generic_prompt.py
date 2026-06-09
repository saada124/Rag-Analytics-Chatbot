import sys, os, json
sys.path.append(os.path.abspath('../backend'))
import analytics
from analytics import execute_pandas_code, format_result
from models import llm_client
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
from pydantic import BaseModel, Field

class PandasCodeOutput(BaseModel):
    explanation: str = Field(description='A brief description of what the analysis does and its logic. MUST be written in French.')
    code: str = Field(description='Pure Python/Pandas code. You must load the dataframe from dfs, e.g., df = dfs["chroma_dataset"], perform operations, and assign the final output to a variable named `result`.')
    target_dataframe: str = Field(description='The name of the dataframe being queried.')

analytics.load_dataframes()

def get_schema_description_new():
    lines = []
    for name, df in analytics.DATAFRAMES.items():
        lines.append(f'DATAFRAME KEY: "{name}"')
        lines.append('AVAILABLE COLUMNS (with up to 3 sample values each):')
        for col in df.columns:
            dtype = df[col].dtype
            samples = df[col].dropna().unique()[:3].tolist()
            samples = [str(s)[:100] + '...' if len(str(s)) > 100 else s for s in samples]
            lines.append(f'  - `{col}` (Samples: {samples})')
        lines.append('')
    return '\n'.join(lines)

def test_query(query):
    today_str = date.today().isoformat()
    schema_desc = get_schema_description_new()

    prompt = ChatPromptTemplate.from_messages([
        ('system', f"""You are an expert business intelligence and data analyst assistant.
Today's date is {today_str}.

You have access to the following DataFrames in the dictionary `dfs`.
YOU MUST ONLY USE THE DATAFRAME KEYS AND COLUMNS LISTED BELOW. DO NOT INVENT COLUMNS.

{schema_desc}

Your task is to write Pandas Python code to answer the user's analytical query.
Rules for writing the code:
1. Extract the correct dataframe from `dfs` dictionary using the exact key shown above (e.g., `df = dfs['chroma_dataset']`).
2. YOU MUST use the column sample values to infer how to search (e.g. if tags use '|' as a delimiter, use `.str.contains(..., case=False, na=False)` to search inside them).
3. Search multiple text columns if necessary (e.g. description, tags, embedding_hint) if the user asks for related concepts.
4. Store the final result in the variable `result`.
5. Keep it safe: no external library imports. Use pandas and numpy only.
6. If returning multiple rows, limit to head(20) to avoid flooding the system.
"""),
        ('user', '{query}')
    ])
    
    chain = prompt | llm_client.with_structured_output(PandasCodeOutput)
    res = chain.invoke({'query': query})
    print(f'\n====================================')
    print(f'QUERY: {query}')
    print(f'CODE GENERATED:\n{res.code}')
    try:
        data_res = execute_pandas_code(res.code, analytics.DATAFRAMES)
        if isinstance(data_res, (int, float, str)):
            print(f'EXECUTION RESULT (Scalar): {data_res}')
        else:
            print(f'EXECUTION RESULT (Records/Length): {len(data_res) if hasattr(data_res, "__len__") else data_res}')
    except Exception as e:
        print(f'EXECUTION ERROR: {e}')

queries = [
    'Find users associated with both rag and embedding',
    'Find users related to vector similarity search',
    'Tell me everything about USR-00145',
    'How many active users are in the dataset?',
    'Find enterprise customers using products related to embeddings, with NPS above 50 and fewer than 10 support tickets.'
]

for q in queries:
    test_query(q)
