# app.py
"""
Streamlit app: Taxonomy-based transcript classifier (Gemini)
- Taxonomy is loaded from Streamlit Secrets: st.secrets["TAXONOMY"] (JSON string)
- User inputs Gemini API key in UI (kept only in session)
- User uploads 1+ transcript .txt files
- App outputs a Data_ex-style table + downloadable Excel

requirements.txt:
  streamlit
  google-genai
  pandas
  openpyxl

Streamlit Secrets (Settings -> Secrets):
TAXONOMY = """
{
  "Category": ["..."],
  "Intent": ["..."],
  "Maneuver": ["..."],
  "Risk Level": ["..."]
}
"""
"""

import re
import json
import time
from io import BytesIO
from collections import Counter
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


# =========================
# App / Model Config
# =========================
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512

# Chunking for long transcripts
CHUNK_CHARS = 3500
CHUNK_OVERLAP = 250


# =========================
# Load taxonomy from secrets
# =========================
def load_taxonomy_from_secrets() -> Dict[str, List[str]]:
    if "TAXONOMY" not in st.secrets:
        raise KeyError('Missing Streamlit secret "TAXONOMY".')

    taxonomy = json.loads(st.secrets["TAXONOMY"])

    required_keys = {"Category", "Intent", "Maneuver", "Risk Level"}
    if not isinstance(taxonomy, dict) or not required_keys.issubset(set(taxonomy.keys())):
        raise ValueError(f'TAXONOMY secret must be a JSON object with keys: {sorted(required_keys)}')

    # Normalize: ensure lists, strings trimmed, dedupe while preserving order
    def norm_list(x: Any) -> List[str]:
        if not isinstance(x, list):
            raise ValueError("Each taxonomy value must be a list.")
        out = []
        seen = set()
        for v in x:
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    taxonomy_norm = {k: norm_list(taxonomy[k]) for k in required_keys}

    for k, v in taxonomy_norm.items():
        if not v:
            raise ValueError(f'Taxonomy "{k}" list is empty.')

    return taxonomy_norm


# =========================
# Cleaning + Chunking
# =========================
_TS_LINE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*$")  # 00:12:34 line only
_TS_INLINE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")    # 00:12:34 inline


def clean_transcript(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _TS_LINE.match(line):
            continue
        lines.append(line)

    t = "\n".join(lines)
    t = _TS_INLINE.sub("", t)

    # normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


# =========================
# Gemini Classification
# =========================
SYSTEM_INSTRUCTIONS = """
You are a strict classification engine.
The transcript may be Japanese.
Choose EXACTLY ONE label for each field from the allowed lists.
Return ONLY valid JSON. No markdown. No extra keys.
If ambiguous, choose best-fit label (do not invent labels).
""".strip()


def build_prompt(segment: str, taxonomy: Dict[str, List[str]]) -> str:
    return f"""
Allowed labels:
Category = {taxonomy["Category"]}
Intent = {taxonomy["Intent"]}
Maneuver = {taxonomy["Maneuver"]}
Risk Level = {taxonomy["Risk Level"]}

Classify this transcript segment:

{segment}

Return JSON with this exact schema:
{{
  "Category": "<one of Category labels>",
  "Intent": "<one of Intent labels>",
  "Maneuver": "<one of Maneuver labels>",
  "Risk Level": "<one of Risk Level labels>"
}}
""".strip()


def _parse_json_strict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # try extracting first JSON object if extra text leaked
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _validate_labels(obj: Dict[str, Any], taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    for k in ["Category", "Intent", "Maneuver", "Risk Level"]:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    c = str(obj["Category"]).strip()
    i = str(obj["Intent"]).strip()
    m = str(obj["Maneuver"]).strip()
    r = str(obj["Risk Level"]).strip()

    if c not in taxonomy["Category"]:
        raise ValueError(f"Invalid Category: {c}")
    if i not in taxonomy["Intent"]:
        raise ValueError(f"Invalid Intent: {i}")
    if m not in taxonomy["Maneuver"]:
        raise ValueError(f"Invalid Maneuver: {m}")
    if r not in taxonomy["Risk Level"]:
        raise ValueError(f"Invalid Risk Level: {r}")

    return {"Category": c, "Intent": i, "Maneuver": m, "Risk Level": r}


def classify_segment(
    segment: str,
    client: genai.Client,
    taxonomy: Dict[str, List[str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    retries: int = 3,
) -> Dict[str, str]:
    prompt = build_prompt(segment, taxonomy)
    last_err = None

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            raw = (resp.text or "").strip()
            obj = _parse_json_strict(raw)
            return _validate_labels(obj, taxonomy)

        except Exception as e:
            last_err = e
            time.sleep(0.7 * (attempt + 1))

    raise RuntimeError(f"Failed to classify segment after {retries} attempts. Last error: {last_err}")


def risk_rank(r: str, allowed_risks: List[str]) -> int:
    # Prefer "Tier N" numeric ordering if present; else fall back to list order
    m = re.search(r"Tier\s*(\d+)", r, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return allowed_risks.index(r) + 1


def aggregate_labels(chunk_labels: List[Dict[str, str]], taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    # majority vote for Category/Intent/Maneuver
    category = Counter([x["Category"] for x in chunk_labels]).most_common(1)[0][0]
    intent = Counter([x["Intent"] for x in chunk_labels]).most_common(1)[0][0]
    maneuver = Counter([x["Maneuver"] for x in chunk_labels]).most_common(1)[0][0]

    # take highest risk across chunks
    risk = max([x["Risk Level"] for x in chunk_labels], key=lambda x: risk_rank(x, taxonomy["Risk Level"]))

    return {"Category": category, "Intent": intent, "Maneuver": maneuver, "Risk Level": risk}


def classify_transcript(
    text: str,
    client: genai.Client,
    taxonomy: Dict[str, List[str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, str]:
    cleaned = clean_transcript(text)
    chunks = chunk_text(cleaned)
    chunk_labels = [
        classify_segment(
            c, client, taxonomy, model=model, temperature=temperature, max_output_tokens=max_output_tokens
        )
        for c in chunks
    ]
    return aggregate_labels(chunk_labels, taxonomy)


# =========================
# Excel export
# =========================
def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data_ex") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return bio.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Transcript Taxonomy Classifier", layout="wide")
st.title("Transcript Taxonomy Classifier (Gemini)")

# Load taxonomy once (and fail fast if secrets are wrong)
try:
    TAXONOMY = load_taxonomy_from_secrets()
except Exception as e:
    st.error(f"Taxonomy secret error: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password", help="Stored only in this browser session.")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(DEFAULT_TEMPERATURE), step=0.05)
    max_tokens = st.number_input("Max output tokens", min_value=128, max_value=4096, value=int(DEFAULT_MAX_OUTPUT_TOKENS), step=128)

    st.markdown("---")
    st.subheader("Transcripts")
    transcript_files = st.file_uploader("Upload transcript .txt files", type=["txt"], accept_multiple_files=True)

    st.markdown("---")
    run_btn = st.button("Run Classification", type="primary", use_container_width=True)

# Run
if run_btn:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
        st.stop()
    if not transcript_files:
        st.error("Please upload at least one transcript .txt file.")
        st.stop()

    client = genai.Client(api_key=api_key)

    results = []
    prog = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(transcript_files, start=1):
        status.write(f"Classifying: **{f.name}** ({idx}/{len(transcript_files)})")

        raw_bytes = f.getvalue()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # common Japanese Windows fallback
            text = raw_bytes.decode("cp932", errors="replace")

        try:
            labels = classify_transcript(
                text,
                client=client,
                taxonomy=TAXONOMY,
                model=model.strip() or DEFAULT_MODEL,
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
            )

            results.append(
                {
                    "File": f.name,
                    "Original Text": text,
                    "Category": labels["Category"],
                    "Intent": labels["Intent"],
                    "Maneuver": labels["Maneuver"],
                    "Risk Level": labels["Risk Level"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "File": f.name,
                    "Original Text": text,
                    "Category": "ERROR",
                    "Intent": "ERROR",
                    "Maneuver": "ERROR",
                    "Risk Level": str(e),
                }
            )

        prog.progress(idx / len(transcript_files))

    df = pd.DataFrame(results, columns=["File", "Original Text", "Category", "Intent", "Maneuver", "Risk Level"])

    st.success("Done.")
    st.dataframe(df, use_container_width=True)

    # Data_ex exact export (drop File)
    df_data_ex = df.drop(columns=["File"])
    xlsx_bytes = to_excel_bytes(df_data_ex, sheet_name="Data_ex")

    st.download_button(
        label="Download Excel (Data_ex)",
        data=xlsx_bytes,
        file_name="classified_Data_ex.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("Show taxonomy summary (counts)"):
        st.write({k: len(v) for k, v in TAXONOMY.items()})
