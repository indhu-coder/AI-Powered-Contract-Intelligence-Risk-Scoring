"""Contract Intelligence & Risk Scoring Engine.

Combines rule-based legal heuristic analysis, NLP entity extraction,
and clause classification to evaluate contract risks and assign risk scores (0-100).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from dateutil import parser
import spacy


@dataclass
class RiskFinding:
    rule_id: str
    clause_type: str
    title: str
    description: str
    weight: int
    severity: str  # "HIGH" | "MEDIUM" | "LOW" | "INFO"
    recommendation: str


@dataclass
class ClauseMatch:
    clause_type: str
    display_name: str
    found: bool
    confidence: float
    excerpt: str
    risk_level: str  # "HIGH" | "MEDIUM" | "LOW" | "INFO"


@dataclass
class ContractAnalysisResult:
    contract_title: str
    character_count: int
    word_count: int
    overall_risk_score: int  # 0 to 100
    overall_risk_level: str  # "LOW" | "MEDIUM" | "HIGH"
    entities: Dict[str, Any]
    clauses: List[ClauseMatch]
    risk_findings: List[RiskFinding]
    extracted_dates: Dict[str, Optional[str]]
    durations: Dict[str, Optional[str]]
    disclaimer: str = (
        "AI-assisted analysis. Results should be reviewed by a qualified legal professional. "
        "This tool does not provide legal advice."
    )


class ContractIntelligenceEngine:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

        # Clause definitions with detection patterns and risk implications
        self.clause_rules = {
            "non_compete": {
                "display_name": "Non-Compete",
                "patterns": [
                    r"\b(?:non-compete|not\s+compete|covenant\s+not\s+to\s+compete|restriction\s+on\s+competition)\b",
                    r"\bshall\s+not\s+(?:directly\s+or\s+indirectly\s+)?(?:engage|compete|participate)\s+in\s+any\s+business\b",
                ],
                "present_risk": {
                    "rule_id": "NON_COMPETE_PRESENT",
                    "weight": 18,
                    "severity": "HIGH",
                    "title": "Non-Compete Restriction Detected",
                    "description": "Restricts parties from engaging in competing business activities during or after contract term.",
                    "recommendation": "Review geographic scope, duration, and narrow the restriction to direct competitors only.",
                },
            },
            "exclusivity": {
                "display_name": "Exclusivity",
                "patterns": [
                    r"\b(?:exclusive|exclusivity|sole\s+and\s+exclusive)\b",
                    r"\bagrees\s+not\s+to\s+(?:retain|hire|engage)\s+(?:any\s+other|third\s+party)\b",
                ],
                "present_risk": {
                    "rule_id": "EXCLUSIVITY_PRESENT",
                    "weight": 15,
                    "severity": "HIGH",
                    "title": "Exclusivity Commitment Found",
                    "description": "Binds a party exclusively, preventing deals with alternative partners or vendors.",
                    "recommendation": "Ensure performance minimums are tied to exclusivity, and carve out non-core channels.",
                },
            },
            "anti_assignment": {
                "display_name": "Anti-Assignment",
                "patterns": [
                    r"\b(?:neither\s+party\s+may\s+assign|shall\s+not\s+assign\s+this\s+agreement|assignment\s+without\s+prior\s+written\s+consent)\b",
                    r"\b(?:assignment|transfer)\b.*?\b(?:prohibited|void|without\s+consent)\b",
                ],
                "present_risk": {
                    "rule_id": "ANTI_ASSIGNMENT_PRESENT",
                    "weight": 11,
                    "severity": "MEDIUM",
                    "title": "Anti-Assignment Restriction",
                    "description": "Prohibits assignment or transfer without counterparty consent, potentially hindering M&A or restructuring.",
                    "recommendation": "Carve out permitted assignments to affiliates and bona fide successors upon merger or acquisition.",
                },
            },
            "termination_for_convenience": {
                "display_name": "Termination for Convenience",
                "patterns": [
                    r"\b(?:terminate\s+(?:this\s+agreement\s+)?(?:at\s+any\s+time|for\s+convenience|without\s+cause))\b",
                    r"\b(?:terminate\s+upon\s+\d+\s+days?\s+written\s+notice)\b",
                ],
                "absent_risk": {
                    "rule_id": "TERMINATION_FOR_CONVENIENCE_ABSENT",
                    "weight": 10,
                    "severity": "MEDIUM",
                    "title": "No Termination for Convenience Clause",
                    "description": "Agreement lacks an exit mechanism without default/cause, locking parties into the contract term.",
                    "recommendation": "Add standard 30-to-60 day written notice termination for convenience clause.",
                },
            },
            "cap_on_liability": {
                "display_name": "Cap on Liability",
                "patterns": [
                    r"\b(?:limitation\s+of\s+liability|aggregate\s+liability\s+(?:shall\s+not\s+exceed|capped\s+at)|in\s+no\s+event\s+shall\s+(?:either\s+party'?s?\s+)?total\s+liability)\b",
                    r"\b(?:maximum\s+liability|liability\s+cap)\b",
                ],
                "absent_risk": {
                    "rule_id": "CAP_ON_LIABILITY_ABSENT",
                    "weight": 12,
                    "severity": "HIGH",
                    "title": "Missing Cap on Liability (Uncapped Exposure)",
                    "description": "No explicit monetary liability limitation detected. Claims could result in unlimited monetary exposure.",
                    "recommendation": "Negotiate an aggregate liability cap (e.g. fees paid over preceding 12 months).",
                },
            },
            "renewal_term": {
                "display_name": "Renewal Term / Auto-Renewal",
                "patterns": [
                    r"\b(?:automatically\s+renew|automatic\s+renewal|renewed\s+for\s+successive\s+periods?)\b",
                    r"\b(?:renewal\s+term|extension\s+period)\b",
                ],
                "present_risk": {
                    "rule_id": "AUTO_RENEWAL_PRESENT",
                    "weight": 15,
                    "severity": "MEDIUM",
                    "title": "Automatic Renewal Commitment",
                    "description": "Contract may auto-renew unless opt-out notice is served within a strict window.",
                    "recommendation": "Implement calendar alerts for termination/opt-out notification deadlines.",
                },
            },
            "governing_law": {
                "display_name": "Governing Law & Jurisdiction",
                "patterns": [
                    r"\b(?:governed\s+by|construed\s+in\s+accordance\s+with\s+the\s+laws\s+of|jurisdiction\s+of\s+the\s+courts\s+of)\b",
                ],
                "absent_risk": {
                    "rule_id": "GOVERNING_LAW_ABSENT",
                    "weight": 10,
                    "severity": "MEDIUM",
                    "title": "No Clear Governing Law Specified",
                    "description": "Lacks an explicit choice of law and venue clause, creating ambiguity in dispute resolution.",
                    "recommendation": "Specify mutual governing jurisdiction (e.g. Delaware, New York, or domicile of primary party).",
                },
            },
            "liquidated_damages": {
                "display_name": "Liquidated Damages / Penalties",
                "patterns": [
                    r"\b(?:liquidated\s+damages|pre-estimated\s+damages|penalty\s+fee|forfeiture)\b",
                ],
                "present_risk": {
                    "rule_id": "LIQUIDATED_DAMAGES_PRESENT",
                    "weight": 9,
                    "severity": "MEDIUM",
                    "title": "Liquidated Damages Clause Found",
                    "description": "Fixes substantial financial damages in advance regardless of actual proven losses.",
                    "recommendation": "Confirm whether pre-agreed damages represent genuine pre-estimate of loss or an unenforceable penalty.",
                },
            },
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities using SpaCy."""
        entities: Dict[str, List[str]] = {
            "parties": [],
            "locations": [],
            "dates": [],
            "money": [],
            "laws": [],
        }
        if not self.nlp:
            return entities

        # Analyze a representative portion to keep processing fast
        doc = self.nlp(text[:40000])
        seen = set()

        for ent in doc.ents:
            val = ent.text.strip().replace("\n", " ")
            if not val or len(val) < 2 or val.lower() in seen:
                continue
            seen.add(val.lower())

            if ent.label_ in ("ORG", "PERSON"):
                if len(entities["parties"]) < 15:
                    entities["parties"].append(val)
            elif ent.label_ in ("GPE", "LOC"):
                if len(entities["locations"]) < 10:
                    entities["locations"].append(val)
            elif ent.label_ == "DATE":
                if len(entities["dates"]) < 10:
                    entities["dates"].append(val)
            elif ent.label_ == "MONEY":
                if len(entities["money"]) < 8:
                    entities["money"].append(val)
            elif ent.label_ == "LAW":
                if len(entities["laws"]) < 8:
                    entities["laws"].append(val)

        return entities

    def extract_dates_and_durations(self, text: str) -> tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
        """Extract explicit contract dates and duration spans."""
        dates: Dict[str, Optional[str]] = {
            "Agreement Date": None,
            "Effective Date": None,
            "Expiration Date": None,
        }
        durations: Dict[str, Optional[str]] = {
            "Renewal Term": None,
            "Notice Period": None,
        }

        # Date regex patterns
        effective_match = re.search(
            r"(?:effective\s+date|dated\s+as\s+of|dated\s+for\s+reference)[\s:]*(?:the\s+)?([A-Za-z0-9,\s]{6,30})",
            text,
            re.IGNORECASE,
        )
        if effective_match:
            try:
                parsed = parser.parse(effective_match.group(1), fuzzy=True)
                dates["Effective Date"] = parsed.strftime("%Y-%m-%d")
            except Exception:
                dates["Effective Date"] = effective_match.group(1).strip()[:25]

        agreement_match = re.search(
            r"(?:this\s+agreement\s+is\s+made|executed\s+this)[\s:]*(?:on\s+)?([A-Za-z0-9,\s]{6,30})",
            text,
            re.IGNORECASE,
        )
        if agreement_match:
            try:
                parsed = parser.parse(agreement_match.group(1), fuzzy=True)
                dates["Agreement Date"] = parsed.strftime("%Y-%m-%d")
            except Exception:
                dates["Agreement Date"] = agreement_match.group(1).strip()[:25]

        expiration_match = re.search(
            r"(?:expire[sd]?\s+on|expiration\s+date|terminat(?:es|ion\s+date))[\s:]*(?:on\s+)?([A-Za-z0-9,\s]{6,30})",
            text,
            re.IGNORECASE,
        )
        if expiration_match:
            try:
                parsed = parser.parse(expiration_match.group(1), fuzzy=True)
                dates["Expiration Date"] = parsed.strftime("%Y-%m-%d")
            except Exception:
                dates["Expiration Date"] = expiration_match.group(1).strip()[:25]

        # Duration regex patterns
        dur_match = re.search(
            r"\b(\d+|\w+)\s*\((?:\d+)\)\s*(days?|weeks?|months?|years?)\b",
            text,
            re.IGNORECASE,
        )
        if not dur_match:
            dur_match = re.search(
                r"\b(\d+)\s*(days?|weeks?|months?|years?)\b",
                text,
                re.IGNORECASE,
            )
        if dur_match:
            durations["Renewal Term"] = dur_match.group(0)

        notice_match = re.search(
            r"(?:notice\s+(?:period\s+)?of\s+)?(\d+\s*(?:days?|months?|business\s+days?))(?:\s+prior\s+written\s+notice)?",
            text,
            re.IGNORECASE,
        )
        if notice_match:
            durations["Notice Period"] = notice_match.group(1)

        return dates, durations

    def analyze_contract(self, text: str, title: str = "Uploaded Contract") -> ContractAnalysisResult:
        """Run full contract intelligence, clause matching, and risk scoring."""
        entities = self.extract_entities(text)
        extracted_dates, durations = self.extract_dates_and_durations(text)

        clauses: List[ClauseMatch] = []
        risk_findings: List[RiskFinding] = []
        total_risk_score = 0

        # Scan text for each clause type
        for clause_type, rule in self.clause_rules.items():
            matched_patterns = []
            found = False
            best_excerpt = ""

            for pat in rule["patterns"]:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    found = True
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 200)
                    excerpt = text[start:end].strip().replace("\n", " ")
                    if len(excerpt) > len(best_excerpt):
                        best_excerpt = excerpt

            # Determine risk level of clause match
            if found:
                confidence = 0.92
                risk_level = "HIGH" if clause_type in ("non_compete", "exclusivity") else ("MEDIUM" if clause_type in ("anti_assignment", "renewal_term") else "INFO")
                clauses.append(
                    ClauseMatch(
                        clause_type=clause_type,
                        display_name=rule["display_name"],
                        found=True,
                        confidence=confidence,
                        excerpt=best_excerpt[:300] + ("..." if len(best_excerpt) > 300 else ""),
                        risk_level=risk_level,
                    )
                )
                # Check if presence causes a risk finding
                if "present_risk" in rule:
                    rf = rule["present_risk"]
                    total_risk_score += rf["weight"]
                    risk_findings.append(
                        RiskFinding(
                            rule_id=rf["rule_id"],
                            clause_type=clause_type,
                            title=rf["title"],
                            description=rf["description"],
                            weight=rf["weight"],
                            severity=rf["severity"],
                            recommendation=rf["recommendation"],
                        )
                    )
            else:
                # Clause was not found
                risk_level = "HIGH" if clause_type == "cap_on_liability" else ("MEDIUM" if clause_type == "termination_for_convenience" else "LOW")
                clauses.append(
                    ClauseMatch(
                        clause_type=clause_type,
                        display_name=rule["display_name"],
                        found=False,
                        confidence=0.88,
                        excerpt="Clause not detected in document.",
                        risk_level=risk_level,
                    )
                )
                # Check if absence causes a risk finding
                if "absent_risk" in rule:
                    rf = rule["absent_risk"]
                    total_risk_score += rf["weight"]
                    risk_findings.append(
                        RiskFinding(
                            rule_id=rf["rule_id"],
                            clause_type=clause_type,
                            title=rf["title"],
                            description=rf["description"],
                            weight=rf["weight"],
                            severity=rf["severity"],
                            recommendation=rf["recommendation"],
                        )
                    )

        # Scale and bound score between 0 and 100
        overall_risk_score = min(100, max(0, total_risk_score))

        # Risk band evaluation: 0-30 LOW, 31-60 MEDIUM, 61-100 HIGH
        if overall_risk_score <= 30:
            overall_risk_level = "LOW"
        elif overall_risk_score <= 60:
            overall_risk_level = "MEDIUM"
        else:
            overall_risk_level = "HIGH"

        words = len(text.split())

        return ContractAnalysisResult(
            contract_title=title,
            character_count=len(text),
            word_count=words,
            overall_risk_score=overall_risk_score,
            overall_risk_level=overall_risk_level,
            entities=entities,
            clauses=clauses,
            risk_findings=risk_findings,
            extracted_dates=extracted_dates,
            durations=durations,
        )
