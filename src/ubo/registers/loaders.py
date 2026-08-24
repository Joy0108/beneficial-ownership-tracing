"""Load six registers into one record shape, keeping provenance on every field.

Each loader reads the format the source actually publishes - FollowTheMoney JSON
for OpenSanctions, the GLEIF golden-copy CSV headers, BODS statements for Open
Ownership, the Companies House data product columns, the OFAC SDN columns. The
normalisation is deliberately shallow: the loaders reshape, they do not clean.
Cleaning belongs to entity resolution, where it can be measured.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import REGISTER_DIR


@dataclass
class Record:
    """One row from one register. The unit of entity resolution."""

    record_id: str
    source: str
    entity_type: str  # "person" | "company"
    name: str
    jurisdiction: str = ""
    address: str = ""
    birth_date: str = ""
    identifier: str = ""          # LEI, company number, SDN entity number
    aliases: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()  # sanction, role.pep, ...
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sanctioned(self) -> bool:
        return any(t.startswith("sanction") for t in self.topics)

    @property
    def is_pep(self) -> bool:
        return any(t.startswith("role.pep") for t in self.topics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "entity_type": self.entity_type,
            "name": self.name,
            "jurisdiction": self.jurisdiction,
            "address": self.address,
            "birth_date": self.birth_date,
            "identifier": self.identifier,
            "aliases": list(self.aliases),
            "topics": list(self.topics),
        }


@dataclass
class Statement:
    """One ownership-or-control assertion, with the provenance that backs it.

    Provenance travels with the edge rather than being attached to the graph as
    a whole. A due-diligence memo has to be able to say which register said what
    and when, and a graph that only records the union of its sources cannot.
    """

    statement_id: str
    source: str
    subject_record_id: str
    interested_record_id: str
    interest_type: str  # shareholding | directorship | trusteeship | consolidation
    share_percent: float
    retrieved_at: str
    confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "source": self.source,
            "subject": self.subject_record_id,
            "interested_party": self.interested_record_id,
            "interest_type": self.interest_type,
            "share_percent": self.share_percent,
            "retrieved_at": self.retrieved_at,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# individual loaders
# ---------------------------------------------------------------------------

def load_opensanctions(directory: Path = REGISTER_DIR) -> Iterator[Record]:
    """OpenSanctions sanctions/PEP collection in FollowTheMoney format."""
    path = directory / "opensanctions.ftm.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entity in payload.get("entities", []):
        props = entity.get("properties", {})
        names = props.get("name", []) or [""]
        yield Record(
            record_id=entity["id"],
            source="opensanctions",
            entity_type="person" if entity.get("schema") == "Person" else "company",
            name=names[0],
            aliases=tuple(names[1:]),
            jurisdiction=_first(props, "jurisdiction") or _first(props, "nationality"),
            address=_first(props, "address"),
            birth_date=_first(props, "birthDate"),
            identifier=_first(props, "leiCode"),
            topics=tuple(props.get("topics", [])),
            raw={"datasets": entity.get("datasets", []), "position": _first(props, "position")},
        )


def load_gleif_level1(directory: Path = REGISTER_DIR) -> Iterator[Record]:
    """GLEIF Level 1: who is who."""
    path = directory / "gleif_level1.csv"
    if not path.exists():
        return
    for row in _csv(path):
        yield Record(
            record_id=f"lei-{row['LEI']}",
            source="gleif_l1",
            entity_type="company",
            name=row["Entity.LegalName"],
            jurisdiction=row["Entity.LegalJurisdiction"],
            address=row["Entity.LegalAddress"],
            identifier=row["LEI"],
            raw={"legal_form": row.get("Entity.LegalForm"), "status": row.get("Entity.EntityStatus")},
        )


def load_gleif_level2(directory: Path = REGISTER_DIR) -> Iterator[Statement]:
    """GLEIF Level 2: who owns whom, by accounting consolidation.

    Note the direction. GLEIF records the *child* as the start node and the
    parent as the end node, which is the opposite of the intuitive reading and a
    reliable source of inverted ownership graphs.
    """
    path = directory / "gleif_level2.csv"
    if not path.exists():
        return
    for i, row in enumerate(_csv(path)):
        child_lei = row["Relationship.StartNode.NodeID"]
        parent_lei = row["Relationship.EndNode.NodeID"]
        yield Statement(
            statement_id=f"gleif-l2-{i:05d}",
            source="gleif_l2",
            subject_record_id=f"lei-{child_lei}",
            interested_record_id=f"lei-{parent_lei}",
            interest_type="consolidation",
            share_percent=0.0,
            retrieved_at="2024-07-01",
            confidence=1.0 if row.get("Registration.ValidationSources") == "FULLY_CORROBORATED" else 0.6,
            raw={"relationship_type": row.get("Relationship.RelationshipType")},
        )


def load_openownership(directory: Path = REGISTER_DIR) -> tuple[list[Record], list[Statement]]:
    """Open Ownership / UK PSC statements in BODS.

    Each statement carries both a subject and an interested party, and each of
    those is a *record*, not an entity: the same company restated by three
    different filings is three records that entity resolution has to collapse.
    """
    path = directory / "openownership_psc.json"
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    statements: list[Statement] = []

    for st in payload.get("statements", []):
        subject, party = st["subject"], st["interestedParty"]
        subject_rid = subject["describedByEntityStatement"]
        party_rid = party.get("describedByPersonStatement") or party.get("describedByEntityStatement")

        records.append(Record(
            record_id=subject_rid, source="openownership", entity_type="company",
            name=subject.get("name", ""), jurisdiction=subject.get("jurisdiction", ""),
            raw={"statement": st["statementID"]},
        ))
        records.append(Record(
            record_id=party_rid, source="openownership",
            entity_type="person" if party.get("type") == "individual" else "company",
            name=party.get("name", ""),
            jurisdiction=party.get("jurisdiction") or party.get("nationality", ""),
            address=(party.get("address") or {}).get("full", ""),
            birth_date=party.get("birthDate", ""),
            raw={"statement": st["statementID"]},
        ))

        interest = (st.get("interests") or [{}])[0]
        statements.append(Statement(
            statement_id=st["statementID"], source="openownership",
            subject_record_id=subject_rid, interested_record_id=party_rid,
            interest_type=interest.get("type", "shareholding"),
            share_percent=float((interest.get("share") or {}).get("exact", 0.0)),
            retrieved_at=(st.get("source") or {}).get("retrievedAt", ""),
            raw={"statement_date": st.get("statementDate")},
        ))
    return records, statements


def load_companies_house(directory: Path = REGISTER_DIR) -> Iterator[Record]:
    path = directory / "companies_house.csv"
    if not path.exists():
        return
    for row in _csv(path):
        yield Record(
            record_id=f"ch-{row['company_number']}",
            source="companies_house",
            entity_type="company",
            name=row["company_name"],
            jurisdiction=row["jurisdiction"],
            address=row["registered_office_address"],
            identifier=row["company_number"],
            raw={"status": row.get("company_status"), "sic": row.get("sic_code")},
        )


# OFAC's vocabulary differs from every other source here. Left unmapped, the
# entity-type prefix on each blocking key puts OFAC records in their own blocks
# and no sanctions record ever meets its match.
_SDN_TYPES = {"individual": "person", "person": "person", "entity": "company", "": "person"}


def load_ofac(directory: Path = REGISTER_DIR) -> Iterator[Record]:
    """OFAC SDN, pulled direct.

    Kept alongside OpenSanctions on purpose: OpenSanctions already aggregates
    OFAC, so holding both is how the pipeline measures what aggregation adds
    rather than assuming it adds something.
    """
    path = directory / "ofac_sdn.csv"
    if not path.exists():
        return
    for row in _csv(path):
        surname, _, given = row["SDN_Name"].partition(",")
        yield Record(
            record_id=f"ofac-{row['ent_num']}",
            source="ofac",
            entity_type=_SDN_TYPES.get((row.get("SDN_Type") or "").strip().lower(), "person"),
            name=f"{given.strip().title()} {surname.strip().title()}".strip(),
            jurisdiction=row.get("Nationality", ""),
            birth_date=row.get("DOB", ""),
            identifier=row["ent_num"],
            topics=("sanction",),
            raw={"program": row.get("Program"), "remarks": row.get("Remarks")},
        )


_ICIJ_SHARE = re.compile(r"\((\d+(?:\.\d+)?)\s*%\)")


def load_icij(directory: Path = REGISTER_DIR) -> tuple[list[Record], list[Statement]]:
    """ICIJ Offshore Leaks extract: the legs no public register publishes.

    The percentage is not a column. It sits inside the free-text ``link``
    field ("shareholder of (75%)"), which is exactly how the real dump carries
    it, so it has to be parsed out and is absent more often than not.
    """
    nodes_path = directory / "icij_offshore_nodes.csv"
    edges_path = directory / "icij_offshore_edges.csv"
    records: list[Record] = []
    statements: list[Statement] = []

    if nodes_path.exists():
        for row in _csv(nodes_path):
            records.append(Record(
                record_id=row["node_id"],
                source="icij",
                entity_type="person" if row["node_type"] == "officer" else "company",
                name=row["name"],
                jurisdiction=row.get("jurisdiction_description", ""),
                address=row.get("address", ""),
                raw={"leak": row.get("sourceID")},
            ))

    if edges_path.exists():
        for i, row in enumerate(_csv(edges_path)):
            link = row.get("link", "")
            match = _ICIJ_SHARE.search(link)
            statements.append(Statement(
                statement_id=f"icij-{i:05d}",
                source="icij",
                subject_record_id=row["node_id_end"],
                interested_record_id=row["node_id_start"],
                interest_type="shareholding" if row["rel_type"] == "shareholder_of" else _icij_role(link),
                share_percent=float(match.group(1)) if match else 0.0,
                retrieved_at="2016-05-09",
                # A leak is not a register. The edge is real but unverified, and
                # the confidence carries that into every chain built on it.
                confidence=0.75,
                raw={"link": link, "leak": row.get("sourceID")},
            ))
    return records, statements


def _icij_role(link: str) -> str:
    low = link.lower()
    if "trustee" in low:
        return "trusteeship"
    if "director" in low:
        return "directorship"
    return "control"


def load_aggregator(directory: Path = REGISTER_DIR) -> Iterator[Record]:
    path = directory / "aggregator_extract.csv"
    if not path.exists():
        return
    for row in _csv(path):
        yield Record(
            record_id=row["record_id"],
            source="aggregator",
            entity_type=row["entity_type"],
            name=row["name"],
            jurisdiction=row["jurisdiction"],
            address=row["address"],
            birth_date=row.get("birth_date", ""),
            raw={"upstream": row.get("source_register")},
        )


# ---------------------------------------------------------------------------
# combined
# ---------------------------------------------------------------------------

def load_all(directory: Path = REGISTER_DIR) -> tuple[list[Record], list[Statement]]:
    records: list[Record] = []
    statements: list[Statement] = []

    records.extend(load_opensanctions(directory))
    records.extend(load_gleif_level1(directory))
    statements.extend(load_gleif_level2(directory))

    oo_records, oo_statements = load_openownership(directory)
    records.extend(oo_records)
    statements.extend(oo_statements)

    records.extend(load_companies_house(directory))
    records.extend(load_ofac(directory))

    icij_records, icij_statements = load_icij(directory)
    records.extend(icij_records)
    statements.extend(icij_statements)

    records.extend(load_aggregator(directory))

    # A BODS filing can restate the same record id; keep the first and note it.
    deduped: dict[str, Record] = {}
    for record in records:
        deduped.setdefault(record.record_id, record)
    return list(deduped.values()), statements


def source_counts(records: list[Record]) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        out[record.source] = out.get(record.source, 0) + 1
    return dict(sorted(out.items()))


def _first(props: dict[str, list[str]], key: str) -> str:
    values = props.get(key) or []
    return values[0] if values else ""


def _csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)
