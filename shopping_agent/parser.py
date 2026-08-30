from __future__ import annotations

import re

from .models import ParsedMessage
from .text import clean_constraint


BUYING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\. A key requirement is: (?P<constraint>.+?)\.?$",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?), but I'm still exploring\.?$",
    re.IGNORECASE,
)
INITIAL_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\. (?P<preference>.+?)\.?$",
    re.IGNORECASE,
)
MATTERS_RE = re.compile(r"^For that, what matters is: (?P<values>.+?)\.?$", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"^Actually, ignore my earlier preference\. What I need is: (?P<constraint>.+?)\.?$",
    re.IGNORECASE,
)
NO_ADDITIONAL_RE = re.compile(
    r"^I don't have an additional preference for (?P<attribute>[a-z_]+)\.?$",
    re.IGNORECASE,
)
BOUNDARY_RE = re.compile(
    r"^I don't have a preference for (?P<attribute>[a-z_]+); please use your judgment\.?$",
    re.IGNORECASE,
)


def parse_message(message: str) -> ParsedMessage:
    text = message.strip()

    match = OVERRIDE_RE.match(text)
    if match:
        return ParsedMessage(
            constraints=[clean_constraint(match.group("constraint"))],
            override=True,
        )

    match = BUYING_RE.match(text)
    if match:
        return ParsedMessage(
            category=clean_constraint(match.group("category")),
            constraints=[clean_constraint(match.group("constraint"))],
        )

    match = BROWSING_RE.match(text)
    if match:
        return ParsedMessage(
            category=clean_constraint(match.group("category")),
            browsing=True,
        )

    match = MATTERS_RE.match(text)
    if match:
        payload = clean_constraint(match.group("values"))
        values = [clean_constraint(value) for value in payload.split(";")]
        return ParsedMessage(
            constraints=[value for value in values if value],
            constraint_payload=payload,
        )

    match = BOUNDARY_RE.match(text)
    if match:
        return ParsedMessage(
            rejected_attribute=match.group("attribute").casefold(),
            boundary_response=True,
        )

    match = NO_ADDITIONAL_RE.match(text)
    if match:
        return ParsedMessage(rejected_attribute=match.group("attribute").casefold())

    match = INITIAL_RE.match(text)
    if match:
        preference = clean_constraint(match.group("preference"))
        return ParsedMessage(
            category=clean_constraint(match.group("category")),
            constraints=[preference] if preference else [],
        )

    return ParsedMessage()
