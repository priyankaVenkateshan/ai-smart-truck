"""Plain-English assignment explanations via Anthropic Claude."""

from __future__ import annotations

import os

from dotenv import load_dotenv

_PLACEHOLDER = (
    "AI explanation is unavailable because no Anthropic API key is configured. "
    "Add ANTHROPIC_API_KEY to your .env file to enable Claude summaries."
)

_SYSTEM = (
    "Explain the assignment in 2–3 short sentences for a non-technical manager. "
    "Use plain language, no jargon, and sound confident and concise."
)


def explain(assignment: dict) -> str:
    """
    assignment keys: truck_id, delivery_id, predicted_delay_min,
    predicted_fuel_l, predicted_eta_min
    """
    load_dotenv()
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return _PLACEHOLDER

    from anthropic import Anthropic

    user_text = (
        f"Truck {assignment['truck_id']} was assigned to delivery {assignment['delivery_id']}. "
        f"Model estimates: delay ≈ {assignment['predicted_delay_min']:.1f} minutes, "
        f"fuel ≈ {assignment['predicted_fuel_l']:.1f} liters, "
        f"ETA ≈ {assignment['predicted_eta_min']:.1f} minutes. "
        "Summarize why this is a reasonable operational choice."
    )

    try:
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
        )
        block = msg.content[0]
        text = getattr(block, "text", None)
        if text is None:
            return str(block)
        return text.strip()
    except Exception:
        return "Explanation could not be generated right now. Please try again later."
