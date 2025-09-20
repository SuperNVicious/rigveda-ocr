
import os
from typing import Tuple

VERDICT_TO_SCORE = {"High": 1.0, "Medium": 0.7, "Low": 0.4}

def llm_verdict(ocr_sanskrit: str, modern_translation: str) -> Tuple[str, str, float]:
    """
    Returns (verdict, rationale, mapped_score in 0..1).
    Verdict ∈ {"High","Medium","Low"} describing essence preservation.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    try:
        if api_key:
            # Optional OpenAI path. Requires: pip install openai
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "You are a Rigveda fidelity judge. Assess whether the 'essence' of the Vedic Sanskrit "
                "text is preserved in the modern translation. Focus on ritual/semantic roles, deities, "
                "and doctrinal meaning, not literal word overlap. Output only two lines:\n"
                "Line1: Verdict = High|Medium|Low\n"
                "Line2: 1–2 sentence rationale mentioning key terms preserved/missing.\n\n"
                f"SANSKRIT (OCR): {ocr_sanskrit}\n"
                f"TRANSLATION: {modern_translation}\n"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
                max_tokens=120,
            )
            text = resp.choices[0].message.content.strip()
            # Parse two-line format
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            verdict = "Medium"
            rationale = ""
            if lines:
                if "high" in lines[0].lower(): verdict = "High"
                elif "low" in lines[0].lower(): verdict = "Low"
                else: verdict = "Medium"
                rationale = lines[1] if len(lines) > 1 else ""
            return verdict, rationale, VERDICT_TO_SCORE[verdict]
    except Exception:
        pass

    # Fallback deterministic dummy
    rationale = "Offline fallback: Provide OPENAI_API_KEY to enable LLM judgment."
    return "Medium", rationale, VERDICT_TO_SCORE["Medium"]
