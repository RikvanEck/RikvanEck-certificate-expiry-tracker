import os
import json
import time
import tempfile
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import pdfplumber
from google import genai

app = Flask(__name__)

# Client is created lazily on first request so the app starts even if the
# env var hasn't been set yet (gives a clear error on the /extract call instead)
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set. Add it in Railway → Variables.")
        _client = genai.Client(api_key=api_key)
    return _client

PROMPT_TEMPLATE = """You are analyzing a certificate document. Extract:
1. The expiration/validity date (look for phrases like "valid until", "expiry date", "expires", "expire date", "valid to", "expiration date", "valid through", "date of expiry", "not valid after", or similar)
2. A short one-line description of what this certificate is (e.g. "Lloyd's Register Class Certificate")

Respond ONLY with valid JSON, no markdown, no extra text:
{{"expiry_date": "DD MMM YYYY", "description": "one-line description"}}

If no expiry date is found, use null for expiry_date.

Certificate text:
{text}"""

DATE_FORMATS = [
    "%d %b %Y", "%d %B %Y", "%Y-%m-%d",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%B %d, %Y", "%b %d, %Y",
    "%d.%m.%Y",
]


def extract_text(path):
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def call_gemini(text, retries=3):
    client = get_client()
    prompt = PROMPT_TEMPLATE.format(text=text[:4000])
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            return json.loads(raw)
        except Exception as e:
            if attempt < retries - 1 and "429" in str(e):
                time.sleep(5 * (attempt + 1))
            else:
                raise


def parse_date(date_str):
    if not date_str or date_str == "null":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def get_status(expiry):
    if expiry is None:
        return "unknown"
    today = datetime.now().date()
    if expiry < today:
        return "expired"
    if expiry < today + timedelta(days=90):
        return "soon"
    return "valid"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    files = request.files.getlist("files")
    results = []

    for f in files:
        row = {"filename": f.filename, "description": "", "expiry_date_str": None, "expiry_date": None, "status": "unknown"}
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name

            text = extract_text(tmp_path)

            if text:
                data = call_gemini(text)
                row["description"] = data.get("description", "")
                expiry_str = data.get("expiry_date")
                if expiry_str and expiry_str != "null":
                    row["expiry_date_str"] = expiry_str
                    parsed = parse_date(expiry_str)
                    row["expiry_date"] = str(parsed) if parsed else None
                    row["status"] = get_status(parsed)
            else:
                row["description"] = "Could not extract text from PDF"

        except Exception as e:
            row["description"] = f"Error: {str(e)}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        results.append(row)

    results.sort(key=lambda x: (x["expiry_date"] is None, x["expiry_date"] or ""))
    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
