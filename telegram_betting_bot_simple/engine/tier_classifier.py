"""Classify predictions into confidence tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TierClassifier:
    """
    Assign a qualitative tier to a probability.

    Probabilities above `threshold_surefire` → "Surefire"
    Probabilities above `threshold_middle`  → "Middle"
    Otherwise → "Hot"
    """
    threshold_middle: float = 0.6
    threshold_surefire: float = 0.75

    def classify(self, probs: Dict[str, float]) -> Tuple[str, str]:
        """Classify the highest probability outcome into a tier."""
        if not probs:
            return ("", "Hot")
        outcome = max(probs, key=probs.get)
        prob = probs[outcome]
        if prob >= self.threshold_surefire:
            tier = "Surefire"
        elif prob >= self.threshold_middle:
            tier = "Middle"
        else:
            tier = "Hot"
        return outcome, tier
