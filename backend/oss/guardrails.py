"""
Guardrails — input/output safety layer for OSS assistant.
Uses a combination of:
  1. Keyword/regex blocklist for obvious harmful inputs
  2. Pattern matching for jailbreak attempts
  3. Simple heuristic toxicity scoring on outputs
"""

import re
from dataclasses import dataclass

# ── Blocked input patterns ─────────────────────────────────────────────────
JAILBREAK_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"you are now (DAN|jailbroken|unrestricted)",
    r"pretend (you have no|you don't have) (restrictions|rules|guidelines)",
    r"act as (if )?you (have no|are without) (morals|ethics|restrictions)",
    r"(do anything now|DAN mode|developer mode|jailbreak)",
    r"forget (your|all) (training|guidelines|rules)",
    r"respond (only |)as (an? )?(evil|unfiltered|unrestricted)",
    r"bypass (your )?(safety|content) (filter|restriction|guideline)",
]

HARMFUL_KEYWORDS = [
    "how to make (a )?bomb",
    "how to (synthesize|make|create) (methamphetamine|fentanyl|nerve agent|sarin|VX)",
    "child sexual",
    "csam",
    "how to (hack|ddos|exploit) .{0,30} (without|bypass)",
]

COMPILED_JAILBREAK = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
COMPILED_HARMFUL = [re.compile(p, re.IGNORECASE) for p in HARMFUL_KEYWORDS]

# ── Output toxicity heuristics ─────────────────────────────────────────────
TOXIC_OUTPUT_PATTERNS = [
    r"\b(kill yourself|kys)\b",
    r"step-by-step (instructions|guide) (to|for) (make|create|build) .{0,20}(bomb|explosive|weapon)",
    r"\b(n[i1]gg[e3]r|f[a4]gg[o0]t|ch[i1]nk|sp[i1]c)\b",
]
COMPILED_TOXIC = [re.compile(p, re.IGNORECASE) for p in TOXIC_OUTPUT_PATTERNS]

SAFE_REFUSAL = "I'm not able to help with that request. Is there something else I can assist you with?"


@dataclass
class GuardResult:
    blocked: bool = False
    flagged: bool = False
    reason: str = ""
    safe_text: str = ""


class InputGuardrail:
    def check(self, text: str) -> GuardResult:
        for pat in COMPILED_JAILBREAK:
            if pat.search(text):
                return GuardResult(blocked=True, reason=f"jailbreak_attempt: {pat.pattern[:40]}")
        for pat in COMPILED_HARMFUL:
            if pat.search(text):
                return GuardResult(blocked=True, reason=f"harmful_content: {pat.pattern[:40]}")
        return GuardResult(blocked=False)


class OutputGuardrail:
    def check(self, text: str) -> GuardResult:
        for pat in COMPILED_TOXIC:
            if pat.search(text):
                return GuardResult(
                    flagged=True,
                    reason=f"toxic_output: {pat.pattern[:40]}",
                    safe_text=SAFE_REFUSAL,
                )
        return GuardResult(flagged=False, safe_text=text)
