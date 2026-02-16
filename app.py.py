# app.py
"""
Streamlit app: Taxonomy-based transcript classifier (Gemini)

What it does:
- User pastes Gemini API key (stored only in session)
- User uploads the Taxonomy .xlsx (or you can ship it with the app and set DEFAULT_TAXONOMY_PATH)
- User uploads 1+ transcript .txt files
- App outputs a Data_ex-style table + downloadable Excel

Run locally:
  pip install streamlit google-genai pandas openpyxl
  streamlit run app.py

Deploy:
- Streamlit Community Cloud:
  - Put app.py + requirements.txt in GitHub repo
  - Add requirements.txt:
      streamlit
      google-genai
      pandas
      openpyxl
  - Deploy from Streamlit Cloud UI
"""

import re
import json
import time
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


# =========================
# Settings you can hardcode (optional)
# =========================
taxonomy = json.loads(st.secrets["TAXONOMY"])
MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 512

CHUNK_CHARS = 3500
CHUNK_OVERLAP = 250


# =========================
# Taxonomy loader
# =========================
@st.cache_data(show_spinner=False)
def load_taxonomy_from_excel_bytes(xlsx_bytes: bytes) -> Dict[str, List[str]]:
    df = pd.read_excel(BytesIO(xlsx_bytes), sheet_name="Taxonomy")

    tiers = {}
    for tier_name in ["Category", "Intent", "Maneuver", "Risk Level"]:
        opts = (
            df.loc[df["Tier"].astype(str).str.strip() == tier_name, "Classification"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        tiers[tier_name] = list(dict.fromkeys(opts))  # dedupe preserve order
        if not tiers[tier_name]:
            raise ValueError(f"No options found for Tier='{tier_name}' in Taxonomy tab.")
    return tiers


# =========================
# Cleaning + chunking
# =========================
_TS_LINE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*$")
_TS_INLINE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")


def clean_transcript(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _TS_LINE.match(line):
            continue
        lines.append(line)
    t = "\n".join(lines)
    t = _TS_INLINE.sub("", t)
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
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# =========================
# Gemini classification
# =========================
SYSTEM_INSTRUCTIONS = """
You are a strict classification engine.
Choose EXACTLY ONE label for each field from the allowed lists.
Return ONLY valid JSON. No markdown. No extra keys.
If ambiguous, choose best-fit label (do not invent labels).
""".strip()


def _parse_json_strict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _validate_labels(obj: Dict[str, Any], taxonomy: Dict[str, List[str]]) -> Dict[str, Any]:
    for k in ["Category", "Intent", "Maneuver", "Risk Level"]:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if obj["Category"] not in taxonomy["Category"]:
        raise ValueError(f"Invalid Category: {obj['Category']}")
    if obj["Intent"] not in taxonomy["Intent"]:
        raise ValueError(f"Invalid Intent: {obj['Intent']}")
    if obj["Maneuver"] not in taxonomy["Maneuver"]:
        raise ValueError(f"Invalid Maneuver: {obj['Maneuver']}")
    if obj["Risk Level"] not in taxonomy["Risk Level"]:
        raise ValueError(f"Invalid Risk Level: {obj['Risk Level']}")

    return obj


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


def classify_segment(segment: str, client: genai.Client, taxonomy: Dict[str, List[str]], retries: int = 3) -> Dict[str, str]:
    prompt = build_prompt(segment, taxonomy)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
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
    m = re.search(r"Tier\s*(\d+)", r, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return allowed_risks.index(r) + 1


def aggregate_labels(chunk_labels: List[Dict[str, str]], taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    category = Counter([x["Category"] for x in chunk_labels]).most_common(1)[0][0]
    intent = Counter([x["Intent"] for x in chunk_labels]).most_common(1)[0][0]
    maneuver = Counter([x["Maneuver"] for x in chunk_labels]).most_common(1)[0][0]
    risk = max([x["Risk Level"] for x in chunk_labels], key=lambda x: risk_rank(x, taxonomy["Risk Level"]))
    return {"Category": category, "Intent": intent, "Maneuver": maneuver, "Risk Level": risk}


def classify_transcript(text: str, client: genai.Client, taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    cleaned = clean_transcript(text)
    chunks = chunk_text(cleaned)
    chunk_labels = [classify_segment(c, client, taxonomy) for c in chunks]
    return aggregate_labels(chunk_labels, taxonomy)


# =========================
# Excel export helper
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

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("Gemini API Key", type="password", help="Stored only in this browser session.")
    model = st.text_input("Model", value=MODEL)
    st.caption("Tip: keep temperature low for consistent taxonomy choices.")

    taxonomy_file = st.file_uploader("Upload Taxonomy Excel (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.subheader("Transcripts")
    transcript_files = st.file_uploader("Upload transcript .txt files", type=["txt"], accept_multiple_files=True)

    st.markdown("---")
    run_btn = st.button("Run Classification", type="primary", use_container_width=True)

# Validate inputs
if run_btn:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
        st.stop()
    if taxonomy_file is None and DEFAULT_TAXONOMY_PATH is None:
        st.error("Please upload the Taxonomy Excel file.")
        st.stop()
    if not transcript_files:
        st.error("Please upload at least one transcript .txt file.")
        st.stop()

    # Gemini client (per user key)
    client = genai.Client(api_key=api_key)

    # Load taxonomy
    try:
        if taxonomy_file is not None:
            taxonomy_bytes = taxonomy_file.getvalue()
            taxonomy = load_taxonomy_from_excel_bytes(taxonomy_bytes)
        else:
            taxonomy = load_taxonomy_from_excel_bytes(Path(DEFAULT_TAXONOMY_PATH).read_bytes())
    except Exception as e:
        st.error(f"Failed to load Taxonomy from Excel: {e}")
        st.stop()

    # Process transcripts
    results = []
    prog = st.progress(0)
    status = st.empty()

    for i, f in enumerate(transcript_files, start=1):
        status.write(f"Classifying: **{f.name}** ({i}/{len(transcript_files)})")

        # Read text (utf-8 first; fallback to cp932)
        raw_bytes = f.getvalue()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("cp932", errors="replace")

        try:
            labels = classify_transcript(text, client, taxonomy)
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
                    "Risk Level": f"{e}",
                }
            )

        prog.progress(i / len(transcript_files))

    df = pd.DataFrame(results, columns=["File", "Original Text", "Category", "Intent", "Maneuver", "Risk Level"])

    st.success("Done.")
    st.dataframe(df, use_container_width=True)

    # Download (Data_ex exact: drop File if you want)
    df_data_ex = df.drop(columns=["File"])
    xlsx_bytes = to_excel_bytes(df_data_ex, sheet_name="Data_ex")

    st.download_button(
        label="Download Excel (Data_ex)",
        data=xlsx_bytes,
        file_name="classified_Data_ex.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # Optional: show taxonomy summary
    with st.expander("Show taxonomy options loaded"):
        st.write({k: len(v) for k, v in taxonomy.items()})
        st.json(taxonomy)
