import pandas as pd


DEFAULT_THRESHOLDS = {
    "CRITICAL": 0.75,
    "HIGH": 0.55,
    "MEDIUM": 0.35,
    "LOW": 0.0,
}

DEFAULT_ACTIONS = {
    "CRITICAL": "Immediate manager intervention required",
    "HIGH": "Schedule retention conversation within 2 weeks",
    "MEDIUM": "Monitor - flag in next quarterly review",
    "LOW": "No action needed - continue standard engagement",
}

COLORS = {
    "CRITICAL": "#DC2626",
    "HIGH": "#F97316",
    "MEDIUM": "#EAB308",
    "LOW": "#16A34A",
}


def _threshold_to_score(value):
    return value * 100 if value <= 1 else value


def compute_risk_score(proba: float, thresholds=None, actions=None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    actions = actions or DEFAULT_ACTIONS
    score = round(proba * 100, 1)

    if score >= _threshold_to_score(thresholds.get("CRITICAL", DEFAULT_THRESHOLDS["CRITICAL"])):
        return {"score": score, "tier": "CRITICAL",
                "action": actions.get("CRITICAL", DEFAULT_ACTIONS["CRITICAL"]),
                "color": COLORS["CRITICAL"]}

    elif score >= _threshold_to_score(thresholds.get("HIGH", DEFAULT_THRESHOLDS["HIGH"])):
        return {"score": score, "tier": "HIGH",
                "action": actions.get("HIGH", DEFAULT_ACTIONS["HIGH"]),
                "color": COLORS["HIGH"]}

    elif score >= _threshold_to_score(thresholds.get("MEDIUM", DEFAULT_THRESHOLDS["MEDIUM"])):
        return {"score": score, "tier": "MEDIUM",
                "action": actions.get("MEDIUM", DEFAULT_ACTIONS["MEDIUM"]),
                "color": COLORS["MEDIUM"]}

    return {"score": score, "tier": "LOW",
            "action": actions.get("LOW", DEFAULT_ACTIONS["LOW"]),
            "color": COLORS["LOW"]}


def generate_risk_table(X_test, y_test, probabilities, config=None):
    config = config or {}
    thresholds = config.get("risk_tiers", DEFAULT_THRESHOLDS)
    actions = config.get("actions", DEFAULT_ACTIONS)

    results = []

    for i in range(len(probabilities)):
        score = compute_risk_score(probabilities[i], thresholds, actions)

        row = X_test.iloc[i].to_dict()
        row.update({
            "true_attrition": y_test.iloc[i],
            "raw_proba": probabilities[i],
            "risk_score": score["score"],
            "risk_tier": score["tier"],
            "action": score["action"]
        })

        results.append(row)

    return pd.DataFrame(results)
