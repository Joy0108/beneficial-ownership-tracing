"""Build the six register files from the seed world.

Why a generator rather than a checked-in extract: the real sources are large and
their licences differ (OpenSanctions is CC-BY-NC, GLEIF is CC0, Companies House
is OGL). What matters for this project is the *shape* of the problem, so the
seed world defines ground truth and this script projects it into six registers
that use the real formats and carry the real defects - transliteration variants,
legal-form noise, typos, missing fields, stale addresses, and the same company
recorded under different identifiers in different registers.

Ground truth is the seed world: which register records refer to the same real
entity, and which structures are genuinely layered. Nothing downstream may read
``data/world/`` outside evaluation.

    python scripts/build_registers.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORLD_DIR = ROOT / "data" / "world"
REG_DIR = ROOT / "data" / "registers"
SEED = 20240802

# csv.writer terminates rows with CR LF on every platform, not with the
# platform default. A register written here on Windows and one written on
# Linux would differ on every line while holding identical data, and CI
# compares them byte for byte to prove the generator is reproducible.
CSV_DIALECT = {"lineterminator": '\n'}
N_BACKGROUND_COMPANIES = 140
N_BACKGROUND_PEOPLE = 90

# --------------------------------------------------------------------------
# the seed world
# --------------------------------------------------------------------------

SECRECY = {"KY", "VG", "PA", "SC", "BZ", "AE", "CY", "MT"}

# Jurisdictions with a public beneficial-ownership register that the BODS
# projection can draw on. Everything else has to come from the leaks extract.
BO_REGISTER_JURISDICTIONS = {"GB", "MT", "CY", "NL", "IT", "GH"}

PEOPLE = [
    # (id, canonical name, cyrillic/native form or "", birth date, nationality, pep_role, sanctioned)
    ("P001", "Dmitri Volkov", "Дмитрий Волков", "1968-04-11", "RU", "Deputy Minister of Energy", True),
    ("P002", "Elena Sokolova", "Елена Соколова", "1975-09-02", "RU", "", False),
    ("P003", "Marcus Van Der Berg", "", "1961-01-30", "NL", "", False),
    ("P004", "Ahmed Al-Rashid", "أحمد الراشد", "1970-07-19", "AE", "Board member, state investment fund", False),
    ("P005", "Li Wei Chen", "陈力伟", "1979-11-05", "CN", "", False),
    ("P006", "Oksana Petrenko", "Оксана Петренко", "1983-03-24", "UA", "Regional governor", False),
    ("P007", "Robert Kingsley", "", "1957-06-08", "GB", "", False),
    ("P008", "Sergei Morozov", "Сергей Морозов", "1972-12-15", "RU", "", True),
    ("P009", "Isabella Ferrari", "", "1980-05-21", "IT", "", False),
    ("P010", "Kwame Osei", "", "1966-08-14", "GH", "Minister of Mines", False),
    ("P011", "Yulia Kuznetsova", "Юлия Кузнецова", "1988-02-09", "RU", "", False),
    ("P012", "Thomas Whitfield", "", "1954-10-03", "GB", "", False),
    ("P013", "Henrik Larsson", "", "1971-05-17", "SE", "", False),
    ("P014", "Marta Nowak", "", "1985-09-28", "PL", "", False),
]

# (id, legal name, jurisdiction, incorporation date, legal form, lei or "")
COMPANIES = [
    ("C001", "Northwind Energy Trading", "GB", "2011-03-15", "Limited", "5493001KJTIIGC8Y1R12"),
    ("C002", "Baltic Resource Holdings", "CY", "2012-07-02", "Limited", "213800MBWEIJDM5CU638"),
    ("C003", "Arcadia Capital Partners", "KY", "2013-01-20", "Exempted Company", ""),
    ("C004", "Meridian Nominees", "VG", "2013-05-11", "BVI Business Company", ""),
    ("C005", "Volga Shipping Group", "RU", "2009-11-28", "OOO", "253400WQIYIZAXBWCU02"),
    ("C006", "Sunrise Commodities", "AE", "2015-02-17", "FZE", ""),
    ("C007", "Helvetia Trust Services", "CH", "2008-06-30", "AG", "506700GE1G29325QX363"),
    ("C008", "Delta Marine Logistics", "PA", "2014-09-05", "Sociedad Anonima", ""),
    ("C009", "Cardinal Infrastructure", "GB", "2010-04-19", "Public Limited Company", "213800D1EI4B9WTWWD28"),
    ("C010", "Silk Road Ventures", "SC", "2016-08-22", "IBC", ""),
    ("C011", "Aurora Mining Holdings", "GB", "2012-11-09", "Limited", "5493004NBBK8OCM3T012"),
    ("C012", "Sahel Extractives", "GH", "2014-03-03", "Limited", ""),
    ("C013", "Grandview Property Fund", "MT", "2015-06-14", "Limited", "529900W3MOO00A18X956"),
    ("C014", "Pinnacle Advisory Services", "GB", "2013-08-27", "Limited Liability Partnership", ""),
    ("C015", "Kestrel Trading DMCC", "AE", "2017-01-12", "DMCC", ""),
    ("C016", "Orion Petrochemicals", "NL", "2011-10-06", "B.V.", "724500Y6L1I8Z2GLRP19"),
    ("C017", "Zenith Capital Management", "KY", "2016-04-25", "Exempted Company", ""),
    ("C018", "Adriatic Shipping Lines", "IT", "2010-02-11", "S.p.A.", "815600A9E1D1D0F0C123"),
    ("C019", "Blue Harbour Nominees", "BZ", "2017-09-30", "IBC", ""),
    ("C020", "Continental Freight Systems", "GB", "2009-05-18", "Limited", "213800QZ5K7EJP9NRT44"),
    ("C021", "Rhine Valley Logistics", "NL", "2013-04-12", "B.V.", "724500KKV1J1M8L2CD77"),
    ("C022", "Tuscan Foods", "IT", "2014-07-08", "S.p.A.", "815600B2C3D4E5F6A789"),
    ("C023", "Baltic Fisheries", "GB", "2015-11-23", "Limited", "213800RR8T2WQ4NMPL61"),
]

# (parent_id, child_id, percent, edge_kind)
# Structures are deliberately shaped: some clean, some layered.
OWNERSHIP = [
    # S1 - Volkov: 5 hops through three secrecy jurisdictions, a nominee, and a cycle.
    ("P001", "C004", 100.0, "ownership"),
    ("C004", "C003", 90.0, "ownership"),
    ("C003", "C002", 75.0, "ownership"),
    ("C002", "C001", 60.0, "ownership"),
    ("C001", "C005", 51.0, "ownership"),
    ("C005", "C002", 12.0, "ownership"),          # circular: C002 -> C001 -> C005 -> C002
    ("P008", "C004", 0.0, "directorship"),        # sanctioned co-director on the nominee

    # S2 - Al-Rashid: three hops, one secrecy jurisdiction, no cycle.
    ("P004", "C006", 100.0, "ownership"),
    ("C006", "C015", 80.0, "ownership"),
    ("C015", "C010", 65.0, "ownership"),
    ("C010", "C008", 55.0, "ownership"),

    # S3 - Van Der Berg: clean two-hop EU structure.
    ("P003", "C016", 100.0, "ownership"),
    ("C016", "C018", 70.0, "ownership"),

    # S4 - Kingsley: single-hop domestic.
    ("P007", "C009", 82.0, "ownership"),

    # S5 - Osei: PEP with a mining chain through Seychelles.
    ("P010", "C012", 45.0, "ownership"),
    ("C012", "C011", 30.0, "ownership"),
    ("C010", "C012", 40.0, "ownership"),          # shared intermediary with S2

    # S6 - Whitfield: nominee-heavy property structure.
    ("P012", "C019", 100.0, "ownership"),
    ("C019", "C013", 95.0, "ownership"),
    ("C013", "C014", 70.0, "ownership"),
    ("C007", "C019", 0.0, "trusteeship"),         # trust service provider as intermediary

    # S7 - clean domestic operating company.
    ("P009", "C020", 100.0, "ownership"),

    # S8 - Chen: two hops, one secrecy jurisdiction.
    ("P005", "C017", 100.0, "ownership"),
    ("C017", "C010", 25.0, "ownership"),

    # S9 - a clean three-hop European operating group. Depth without opacity:
    # no secrecy jurisdiction, no nominee, no cycle, every leg on a public
    # register. This is what stops the risk model from learning "deep = bad".
    ("P013", "C021", 100.0, "ownership"),
    ("C021", "C022", 75.0, "ownership"),
    ("C022", "C023", 60.0, "ownership"),

    # S10 - a clean minority holding.
    ("P014", "C022", 25.0, "ownership"),

    # Directorships that make certain people intermediaries across structures.
    ("P002", "C002", 0.0, "directorship"),
    ("P002", "C003", 0.0, "directorship"),
    ("P002", "C013", 0.0, "directorship"),        # nominee director across three structures
    ("P011", "C005", 0.0, "directorship"),
    ("P006", "C010", 0.0, "directorship"),
]

# Ground truth for the structural-risk task: root entity -> layered or not.
# "Layered" means the chain was built to obscure control, not merely that it is
# long: secrecy-jurisdiction hops, nominee intermediaries, or circular holdings.
LAYERED_ROOTS = {
    "P001": True,   # 5 hops, three secrecy jurisdictions, a nominee, a cycle
    "P004": True,   # 4 hops through AE -> AE -> SC -> PA
    "P010": True,   # PEP, chain routed through a Seychelles intermediary
    "P012": True,   # nominee-heavy, trust service provider in the chain
    "P005": True,   # Cayman into Seychelles: two secrecy hops, no operating rationale
    "P003": False,  # NL -> IT, one hop, both public registers
    "P007": False,  # single-hop domestic
    "P009": False,  # single-hop domestic
    "P013": False,  # three hops, NL -> IT -> GB, no secrecy jurisdiction
    "P014": False,  # single minority holding
}

ADDRESSES = {
    "SE": ["Sveavagen 44, 111 34 Stockholm"],
    "PL": ["Zlota 59, 00-120 Warszawa"],
    "GB": ["27 Old Broad Street, London EC2N 1HN", "1 Canada Square, Canary Wharf, London E14 5AB",
           "Beaufort House, 15 St Botolph Street, London EC3A 7BB"],
    "CY": ["Themistokli Dervi 3, Julia House, 1066 Nicosia", "Arch. Makariou III 199, Neocleous House, 3030 Limassol"],
    "KY": ["PO Box 309, Ugland House, Grand Cayman KY1-1104", "190 Elgin Avenue, George Town, Grand Cayman KY1-9008"],
    "VG": ["Craigmuir Chambers, PO Box 71, Road Town, Tortola VG1110"],
    "PA": ["Torre Global Bank, Calle 50, Ciudad de Panama"],
    "SC": ["Suite 1, Second Floor, Sound & Vision House, Francis Rachel Street, Victoria, Mahe"],
    "BZ": ["Withfield Tower, Third Floor, 4792 Coney Drive, Belize City"],
    "AE": ["Unit 1802, Jumeirah Business Centre 3, Cluster Y, JLT, Dubai", "Office 3402, Almas Tower, JLT, Dubai"],
    "CH": ["Bahnhofstrasse 45, 8001 Zurich"],
    "RU": ["Ulitsa Tverskaya 12, Moscow 125009", "Nevsky Prospekt 28, Saint Petersburg 191186"],
    "NL": ["Zuidplein 126, 1077 XV Amsterdam"],
    "IT": ["Via Monte Napoleone 8, 20121 Milano"],
    "GH": ["Independence Avenue 5, Ridge, Accra"],
    "MT": ["Level 3, Valletta Buildings, South Street, Valletta VLT 1103"],
    "CN": ["Jing'an District, 1266 Nanjing West Road, Shanghai"],
    "UA": ["Khreshchatyk Street 22, Kyiv 01001"],
}

LEGAL_FORM_VARIANTS = {
    "Limited": ["Limited", "Ltd", "LTD.", "Ltd."],
    "Public Limited Company": ["PLC", "Public Limited Company", "P.L.C."],
    "Limited Liability Partnership": ["LLP", "Limited Liability Partnership"],
    "Exempted Company": ["Ltd", "Limited", "Exempted Company"],
    "BVI Business Company": ["Ltd", "Limited", "BVI Business Company"],
    "IBC": ["Ltd", "Limited", "IBC"],
    "OOO": ["OOO", "LLC", "O.O.O."],
    "AG": ["AG", "A.G.", "Aktiengesellschaft"],
    "B.V.": ["BV", "B.V.", "Besloten Vennootschap"],
    "S.p.A.": ["SpA", "S.p.A."],
    "Sociedad Anonima": ["SA", "S.A.", "Sociedad Anonima"],
    "FZE": ["FZE", "Free Zone Establishment"],
    "DMCC": ["DMCC", "DMCC Ltd"],
}

# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------

TRANSLIT = {
    "Dmitri": ["Dmitry", "Dmitrii", "Dmitriy"],
    "Volkov": ["Volkov", "Wolkow"],
    "Sergei": ["Sergey", "Serguei"],
    "Morozov": ["Morozov", "Morosov"],
    "Elena": ["Yelena", "Elena"],
    "Sokolova": ["Sokolova", "Socolova"],
    "Yulia": ["Julia", "Yuliya"],
    "Kuznetsova": ["Kuznetsova", "Kouznetsova"],
    "Oksana": ["Oxana", "Oksana"],
    "Petrenko": ["Petrenko", "Petrenko"],
    "Ahmed": ["Ahmad", "Ahmed"],
    "Al-Rashid": ["Al Rashid", "Alrashid", "El-Rashid"],
    "Li": ["Li", "Lee"],
    "Chen": ["Chen", "Chan"],
}


def transliterate_name(name: str, rng: random.Random) -> str:
    parts = []
    for token in name.split():
        options = TRANSLIT.get(token)
        parts.append(rng.choice(options) if options else token)
    return " ".join(parts)


def typo(text: str, rng: random.Random) -> str:
    """A single realistic keying error: doubled, dropped or swapped character."""
    if len(text) < 6:
        return text
    i = rng.randrange(2, len(text) - 2)
    mode = rng.choice(["drop", "double", "swap"])
    if mode == "drop":
        return text[:i] + text[i + 1 :]
    if mode == "double":
        return text[:i] + text[i] + text[i:]
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def company_variant(name: str, form: str, rng: random.Random, allow_typo: bool = True) -> str:
    forms = LEGAL_FORM_VARIANTS.get(form, [form])
    out = f"{name} {rng.choice(forms)}"
    if rng.random() < 0.20:
        out = out.upper()
    if allow_typo and rng.random() < 0.18:
        out = typo(out, rng)
    return out


def address_variant(jurisdiction: str, rng: random.Random) -> str:
    pool = ADDRESSES.get(jurisdiction) or ADDRESSES["GB"]
    address = rng.choice(pool)
    if rng.random() < 0.30:
        address = address.replace(", ", ",").upper()
    if rng.random() < 0.15:
        address = address.split(",")[0]  # truncated address, a very common defect
    return address


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------
# background population
# --------------------------------------------------------------------------
#
# The structures above are hand-designed, because the point of the graph task is
# that a human decided what "layered" means. The background is generated: real
# registers hold millions of unrelated entities, and without a haystack the
# blocking stage has nothing to prove. Background entities take part in no
# ownership edge and are never labelled as risk; they exist to make candidate
# generation a real problem and to give near-miss name collisions somewhere to
# come from.

_ADJ = ["Northern", "Atlantic", "Imperial", "Crown", "Summit", "Harbour", "Granite", "Ivory", "Cobalt", "Falcon",
        "Emerald", "Regent", "Anchor", "Beacon", "Cypress", "Draco", "Everest", "Foxglove", "Garnet", "Halcyon"]
_NOUN = ["Holdings", "Capital", "Trading", "Ventures", "Industries", "Resources", "Partners", "Logistics",
         "Investments", "Consulting", "Maritime", "Petroleum", "Property", "Metals", "Agri", "Textiles"]
_SUFFIX = ["Group", "International", "Global", "Enterprises", "Corporation", ""]

_GIVEN = ["James", "Maria", "Ivan", "Sofia", "Hassan", "Anna", "Peter", "Fatima", "Nikolai", "Grace",
          "Viktor", "Lucia", "Omar", "Katarina", "Daniel", "Irina", "Samuel", "Nadia", "Andrei", "Clara"]
_FAMILY = ["Novak", "Ibrahim", "Lindqvist", "Costa", "Nowak", "Haddad", "Berg", "Kovacs", "Silva", "Adeyemi",
           "Petrov", "Dubois", "Marchetti", "Okafor", "Larsen", "Reyes", "Fischer", "Yilmaz", "Nakamura", "Bakker"]

_BG_JURISDICTIONS = ["GB", "CY", "KY", "VG", "AE", "NL", "IT", "CH", "PA", "SC", "MT", "BZ"]
_BG_FORMS = ["Limited", "Public Limited Company", "Exempted Company", "IBC", "B.V.", "AG", "S.p.A.",
             "Sociedad Anonima", "FZE", "DMCC", "OOO"]


def background_companies(rng: random.Random, n: int) -> list[tuple]:
    """Generate n background companies, some with deliberately similar names."""
    out, seen = [], set()
    while len(out) < n:
        name = f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}"
        suffix = rng.choice(_SUFFIX)
        if suffix:
            name = f"{name} {suffix}"
        if name in seen:
            # A repeat is kept as a *different* company in a different
            # jurisdiction: two unrelated firms with the same trading name is
            # the hardest negative an entity resolver faces, and real registers
            # are full of them.
            name = f"{name} ({rng.choice(['Overseas', 'Holdings', 'Services'])})"
            if name in seen:
                continue
        seen.add(name)
        idx = len(out)
        jur = rng.choice(_BG_JURISDICTIONS)
        lei = f"BG{idx:04d}" + "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(14)) if rng.random() < 0.45 else ""
        out.append((f"B{idx:04d}", name, jur, f"{rng.randrange(2005, 2021)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}",
                    rng.choice(_BG_FORMS), lei))
    return out


def background_people(rng: random.Random, n: int) -> list[tuple]:
    out = []
    for idx in range(n):
        name = f"{rng.choice(_GIVEN)} {rng.choice(_FAMILY)}"
        nat = rng.choice(["GB", "NL", "IT", "CH", "AE", "RU", "UA", "CN", "GH", "CY"])
        out.append((f"Q{idx:04d}", name, "", f"{rng.randrange(1950, 1996)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}",
                    nat, "", False))
    return out


# --------------------------------------------------------------------------
# register projections
# --------------------------------------------------------------------------

def build() -> dict:
    rng = random.Random(SEED)
    REG_DIR.mkdir(parents=True, exist_ok=True)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)

    bg_companies = background_companies(rng, N_BACKGROUND_COMPANIES)
    bg_people = background_people(rng, N_BACKGROUND_PEOPLE)

    people = {p[0]: p for p in PEOPLE + bg_people}
    companies = {c[0]: c for c in COMPANIES + bg_companies}
    truth: dict[str, list[str]] = {e: [] for e in list(people) + list(companies)}

    def claim(entity_id: str, record_id: str) -> str:
        truth[entity_id].append(record_id)
        return record_id

    # --- OpenSanctions, FollowTheMoney format ------------------------------
    ftm_entities = []
    sanctioned_or_pep = [p for p in PEOPLE if p[6] or p[5]]
    for pid, name, native, dob, nat, pep_role, sanctioned in sanctioned_or_pep:
        rid = claim(pid, f"os-{pid}")
        props = {
            "name": [transliterate_name(name, rng)] + ([native] if native else []),
            "birthDate": [dob],
            "nationality": [nat],
            "topics": (["sanction"] if sanctioned else []) + (["role.pep"] if pep_role else []),
        }
        if pep_role:
            props["position"] = [pep_role]
        ftm_entities.append({"id": rid, "schema": "Person", "properties": props,
                             "datasets": ["us_ofac_sdn"] if sanctioned else ["everypolitician"],
                             "referents": [], "first_seen": "2022-03-01", "last_seen": "2024-07-30"})

    # Companies that a sanctioned or PEP person controls are listed too.
    controlled = {c for p, c, _pct, kind in OWNERSHIP
                  for pid, *_rest in PEOPLE if p == pid and (people[p][6] or people[p][5]) and kind == "ownership"}
    for cid in sorted(controlled):
        name, jur, inc, form, lei = companies[cid][1], companies[cid][2], companies[cid][3], companies[cid][4], companies[cid][5]
        rid = claim(cid, f"os-{cid}")
        ftm_entities.append({
            "id": rid, "schema": "Company",
            "properties": {
                "name": [company_variant(name, form, rng, allow_typo=False)],
                "jurisdiction": [jur],
                "incorporationDate": [inc],
                "address": [address_variant(jur, rng)],
                "leiCode": [lei] if lei else [],
                "topics": ["sanction.linked"],
            },
            "datasets": ["us_ofac_sdn"], "referents": [], "first_seen": "2022-03-01", "last_seen": "2024-07-30"})

    (REG_DIR / "opensanctions.ftm.json").write_text(
        json.dumps({"format": "FollowTheMoney", "version": "3.5", "entities": ftm_entities}, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n")

    # --- GLEIF level 1: LEI records ---------------------------------------
    with (REG_DIR / "gleif_level1.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["LEI", "Entity.LegalName", "Entity.LegalJurisdiction", "Entity.LegalForm",
                    "Entity.LegalAddress", "Entity.EntityStatus", "Registration.InitialRegistrationDate"])
        for cid, name, jur, inc, form, lei in COMPANIES + bg_companies:
            if not lei:
                continue
            claim(cid, f"lei-{lei}")
            w.writerow([lei, company_variant(name, form, rng), jur, form,
                        address_variant(jur, rng), "ACTIVE", inc])

    # --- GLEIF level 2: parent relationships ------------------------------
    lei_of = {c[0]: c[5] for c in COMPANIES if c[5]}
    with (REG_DIR / "gleif_level2.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["Relationship.StartNode.NodeID", "Relationship.EndNode.NodeID",
                    "Relationship.RelationshipType", "Relationship.Status", "Registration.ValidationSources"])
        for parent, child, _pct, kind in OWNERSHIP:
            if kind != "ownership" or parent not in lei_of or child not in lei_of:
                continue
            w.writerow([lei_of[child], lei_of[parent], "IS_DIRECTLY_CONSOLIDATED_BY", "ACTIVE", "FULLY_CORROBORATED"])

    # --- Open Ownership / UK PSC statements -------------------------------
    statements = []
    for parent, child, pct, kind in OWNERSHIP:
        if companies.get(child, ("", "", ""))[2] not in BO_REGISTER_JURISDICTIONS:
            continue
        subject_rid = claim(child, f"psc-sub-{parent}-{child}")
        interested_rid = claim(parent, f"psc-int-{parent}-{child}")
        if parent in people:
            _pid, pname, _native, dob, nat, _role, _sanc = people[parent]
            interested = {"type": "individual", "name": transliterate_name(pname, rng),
                          "birthDate": dob[:7], "nationality": nat,
                          "address": {"full": address_variant(nat if nat in ADDRESSES else "GB", rng)}}
        else:
            pc = companies[parent]
            interested = {"type": "entity", "name": company_variant(pc[1], pc[4], rng),
                          "jurisdiction": pc[2],
                          "address": {"full": address_variant(pc[2], rng)}}
        cc = companies[child]
        statements.append({
            "statementID": f"openownership-register-{len(statements):06d}",
            "statementType": "ownershipOrControlStatement",
            "statementDate": "2024-06-30",
            "subject": {"describedByEntityStatement": subject_rid,
                        "name": company_variant(cc[1], cc[4], rng), "jurisdiction": cc[2]},
            "interestedParty": {"describedByPersonStatement" if parent in people else "describedByEntityStatement": interested_rid,
                                **interested},
            "interests": [{"type": "shareholding" if kind == "ownership" else kind,
                           "share": {"exact": pct} if pct else {"exact": 0.0},
                           "startDate": "2019-01-01"}],
            "source": {"type": ["officialRegister"], "description": "UK PSC register", "retrievedAt": "2024-07-01"},
        })
    (REG_DIR / "openownership_psc.json").write_text(
        json.dumps({"format": "BODS", "version": "0.3", "statements": statements}, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n")

    # --- Companies House company profiles ---------------------------------
    with (REG_DIR / "companies_house.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["company_number", "company_name", "company_status", "jurisdiction",
                    "date_of_creation", "registered_office_address", "sic_code"])
        for idx, (cid, name, jur, inc, form, _lei) in enumerate(COMPANIES + bg_companies):
            if jur not in {"GB", "MT", "CY", "NL", "IT"}:
                continue
            number = f"{7000000 + idx * 137:08d}"
            claim(cid, f"ch-{number}")
            w.writerow([number, company_variant(name, form, rng), "active", jur, inc,
                        address_variant(jur, rng), rng.choice(["64209", "70100", "46900", "68209"])])

    # --- OFAC SDN ---------------------------------------------------------
    with (REG_DIR / "ofac_sdn.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["ent_num", "SDN_Name", "SDN_Type", "Program", "Nationality", "DOB", "Remarks"])
        for num, (pid, name, _native, dob, nat, _role, sanctioned) in enumerate(PEOPLE, start=40001):
            if not sanctioned:
                continue
            claim(pid, f"ofac-{num}")
            surname, *given = name.split()[::-1]
            w.writerow([num, f"{surname}, {' '.join(given[::-1])}".upper(), "individual",
                        "UKRAINE-EO13662", nat, dob, "Linked to energy sector procurement."])

    # Background entities also appear in the "corporate registry extract" that
    # aggregators publish, which is where most duplicate pairs really come from.
    with (REG_DIR / "aggregator_extract.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["record_id", "entity_type", "name", "jurisdiction", "address", "birth_date", "source_register"])
        for cid, name, jur, _inc, form, _lei in COMPANIES + bg_companies:
            for copy in range(rng.choice([0, 1, 1, 2])):
                rid = claim(cid, f"agg-c-{cid}-{copy}")
                w.writerow([rid, "company", company_variant(name, form, rng), jur,
                            address_variant(jur, rng), "", rng.choice(["opencorporates", "orbis", "gleif"])])
        for pid, name, _native, dob, nat, _role, _sanc in PEOPLE + bg_people:
            for copy in range(rng.choice([0, 1, 1, 2])):
                rid = claim(pid, f"agg-p-{pid}-{copy}")
                variant = transliterate_name(name, rng)
                if rng.random() < 0.15:
                    variant = typo(variant, rng)
                w.writerow([rid, "person", variant, nat,
                            address_variant(nat if nat in ADDRESSES else "GB", rng), dob,
                            rng.choice(["opencorporates", "worldcheck", "national_register"])])

    # --- ICIJ Offshore Leaks style extract --------------------------------
    #
    # Public registers deliberately do not publish the offshore legs: that is
    # what makes the chains opaque in the first place. The ICIJ leaks database
    # is where those edges come from in practice, and its shape is a node table
    # plus a relationship table with a free-text relationship label.
    with (REG_DIR / "icij_offshore_nodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["node_id", "name", "node_type", "jurisdiction_description", "address", "sourceID"])
        offshore = {c for c in companies if companies[c][2] in SECRECY}
        involved = {p for p, c, _pct, _k in OWNERSHIP if c in offshore} | offshore
        for eid in sorted(involved):
            if eid in companies:
                _cid, name, jur, _inc, form, _lei = companies[eid]
                rid = claim(eid, f"icij-{eid}")
                w.writerow([rid, company_variant(name, form, rng), "entity", jur,
                            address_variant(jur, rng), "Offshore Leaks 2016"])
            elif eid in people:
                _pid, name, _native, dob, nat, _role, _sanc = people[eid]
                rid = claim(eid, f"icij-{eid}")
                w.writerow([rid, transliterate_name(name, rng), "officer", nat,
                            address_variant(nat if nat in ADDRESSES else "GB", rng), "Panama Papers"])

    with (REG_DIR / "icij_offshore_edges.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, **CSV_DIALECT)
        w.writerow(["node_id_start", "node_id_end", "rel_type", "link", "start_date", "sourceID"])
        for parent, child, pct, kind in OWNERSHIP:
            if child not in offshore:
                continue
            link = {"ownership": f"shareholder of ({pct:.0f}%)", "directorship": "director of",
                    "trusteeship": "trustee of"}.get(kind, kind)
            w.writerow([f"icij-{parent}", f"icij-{child}", "officer_of" if kind != "ownership" else "shareholder_of",
                        link, "2013-01-01", "Offshore Leaks 2016"])

    world = {
        "seed": SEED,
        "people": [dict(zip(["id", "name", "native_name", "birth_date", "nationality", "pep_role", "sanctioned"], p, strict=True)) for p in PEOPLE + bg_people],
        "companies": [dict(zip(["id", "name", "jurisdiction", "incorporation_date", "legal_form", "lei"], c, strict=True)) for c in COMPANIES + bg_companies],
        "structural_entities": [p[0] for p in PEOPLE] + [c[0] for c in COMPANIES],
        "ownership": [dict(zip(["parent", "child", "percent", "kind"], o, strict=True)) for o in OWNERSHIP],
        "layered_roots": LAYERED_ROOTS,
        "secrecy_jurisdictions": sorted(SECRECY),
        "truth_clusters": {k: v for k, v in truth.items() if v},
    }
    (WORLD_DIR / "seed_world.json").write_text(json.dumps(world, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    records = sum(len(v) for v in truth.values())
    resolvable = sum(1 for v in truth.values() if len(v) > 1)
    print(f"registers written to {REG_DIR}")
    print(f"  {len(world['people'])} people, {len(world['companies'])} companies")
    print(f"  {records} register records across 6 sources")
    print(f"  {resolvable} entities appear in more than one register (the resolvable set)")
    print(f"  {len(statements)} BODS statements, {len(ftm_entities)} FtM entities")
    return world


if __name__ == "__main__":
    build()
