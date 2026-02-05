import streamlit as st
import pandas as pd
import boto3
import json
import traceback

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Data Analyst (AWS Bedrock)",
    layout="wide"
)

st.title("📊 AI Data Analyst (NL → Insights)")
st.caption("Powered by AWS Bedrock")

# -----------------------------
# AWS Bedrock Client
# -----------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-southeast-1"  # change if needed
)

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# -----------------------------
# Helper Functions
# -----------------------------

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def get_schema(df):
    return {
        "columns": list(df.columns),
        "types": df.dtypes.astype(str).to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records")
    }


def build_prompt(schema, question):
    return f"""
You are a senior data analyst.

Dataframe schema:
Columns: {schema['columns']}
Types: {schema['types']}
Sample rows: {schema['sample_rows']}

User question:
"{question}"

Rules:
- Use pandas only
- The dataframe variable is named df
- Do NOT modify df
- Do NOT import anything
- Return ONLY a single pandas expression
"""


def call_bedrock(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"].strip()


def safe_execute(code, df):
    banned = ["import", "os.", "open(", "exec", "eval", "__", "sys."]
    if any(b in code for b in banned):
        raise ValueError("Unsafe code detected")

    return eval(code, {"df": df})


def auto_visualize(result):
    if isinstance(result, pd.Series):
        st.bar_chart(result)
    elif isinstance(result, pd.DataFrame):
        if len(result.columns) == 2:
            st.bar_chart(result.set_index(result.columns[0]))
        else:
            st.dataframe(result)
    else:
        st.write(result)


def explain_result(question, result):
    prompt = f"""
Explain the following analysis result in simple business language.

Question:
{question}

Result:
{str(result)[:1000]}
"""
    return call_bedrock(prompt)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if not uploaded_file:
    st.info("Upload a dataset to begin.")
    st.stop()

df = load_file(uploaded_file)
schema = get_schema(df)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([3, 1])

with left:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df, use_container_width=True)

with right:
    st.subheader("🤖 AI Assistant")
    question = st.text_area(
        "Ask a question about this data",
        placeholder="e.g. Which region has the highest total revenue?"
    )
    run_btn = st.button("Run Analysis")

# -----------------------------
# Run AI Analysis
# -----------------------------
if run_btn and question:
    st.divider()
    st.subheader("🔍 Analysis")

    prompt = build_prompt(schema, question)

    with st.spinner("Thinking with AWS Bedrock..."):
        try:
            ai_code = call_bedrock(prompt)

            with st.expander("🧠 AI-Generated Pandas Code"):
                st.code(ai_code, language="python")

            result = safe_execute(ai_code, df)

            st.success("Analysis completed!")
            st.write("### Result")
            st.write(result)

            st.write("### Visualization")
            auto_visualize(result)

            st.write("### 🗣 Explanation")
            explanation = explain_result(question, result)
            st.info(explanation)

        except Exception:
            st.error("Something went wrong.")
            st.text(traceback.format_exc())

# -----------------------------
# Debug
# -----------------------------
with st.expander("⚙️ Debug Prompt"):
    st.code(build_prompt(schema, question if question else ""))
