#!/usr/bin/env python3
"""Validate synchronized Cargo, CFF, and Zenodo release metadata."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

try:
    import yaml
except ImportError as error:
    raise SystemExit("PyYAML is required: python -m pip install PyYAML") from error


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/pgarrett-scripps/sequoia-boost"
CONCEPT_DOI = "10.5281/zenodo.21968435"
EXPECTED_AUTHORS = ["Patrick T. Garrett", "John R. Yates III"]
EXPECTED_CARGO_AUTHORS = [
    "Patrick T. Garrett <pgarrett@scripps.edu>",
    "John R. Yates III <jyates@scripps.edu>",
]
EXPECTED_ZENODO_AUTHORS = ["Garrett, Patrick T.", "Yates, John R. III"]
EXPECTED_IDENTITIES = [
    ("pgarrett@scripps.edu", "0000-0002-8434-9693", "Scripps Research Institute"),
    ("jyates@scripps.edu", "0000-0001-5267-1672", "Scripps Research Institute"),
]
REQUIRED_ZENODO = {
    "title",
    "version",
    "description",
    "creators",
    "upload_type",
    "access_right",
    "license",
    "language",
    "keywords",
    "related_identifiers",
    "communities",
    "grants",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def normalize(text: str) -> str:
    return " ".join(text.split())


def cargo_author(name: str) -> str:
    return re.sub(r"\s*<[^>]+>$", "", name).strip()


def cff_author(author: dict[str, str]) -> str:
    if "name" in author:
        return author["name"]
    suffix = f" {author['name-suffix']}" if "name-suffix" in author else ""
    return f"{author['given-names']} {author['family-names']}{suffix}"


def main() -> None:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    package = cargo["workspace"]["package"]

    missing = REQUIRED_ZENODO - zenodo.keys()
    if missing:
        fail(f"Zenodo metadata lacks required fields: {sorted(missing)}")
    for key in ("cff-version", "message", "type", "title", "abstract", "authors",
                "repository-code", "url", "license", "version", "keywords"):
        if key not in cff:
            fail(f"CITATION.cff lacks required field: {key}")

    versions = {package["version"], str(cff["version"]), str(zenodo["version"])}
    if len(versions) != 1:
        fail(f"versions disagree: {sorted(versions)}")
    if cff["license"] != package["license"]:
        fail("Cargo and CFF licenses disagree")
    if zenodo["license"].lower() != package["license"].lower():
        fail("Cargo and Zenodo licenses disagree")

    cargo_authors = [cargo_author(author) for author in package["authors"]]
    cff_authors = [cff_author(author) for author in cff["authors"]]
    zenodo_authors = [creator["name"] for creator in zenodo["creators"]]
    if cargo_authors != cff_authors:
        fail(
            f"Cargo and CFF author names or order disagree: "
            f"Cargo={cargo_authors}, CFF={cff_authors}"
        )
    if cargo_authors != EXPECTED_AUTHORS:
        fail(f"authors must be exactly {EXPECTED_AUTHORS}, found {cargo_authors}")
    if package["authors"] != EXPECTED_CARGO_AUTHORS:
        fail("Cargo author names and emails do not match the verified identities")
    if zenodo_authors != EXPECTED_ZENODO_AUTHORS:
        fail(
            f"Zenodo authors must be exactly {EXPECTED_ZENODO_AUTHORS}, "
            f"found {zenodo_authors}"
        )
    cff_identities = [
        (
            author.get("email"),
            str(author.get("orcid", "")).removeprefix("https://orcid.org/"),
            author.get("affiliation"),
        )
        for author in cff["authors"]
    ]
    zenodo_identities = [
        (None, creator.get("orcid"), creator.get("affiliation"))
        for creator in zenodo["creators"]
    ]
    if cff_identities != EXPECTED_IDENTITIES:
        fail("CFF emails, ORCIDs, or affiliations do not match verified identities")
    if [identity[1:] for identity in EXPECTED_IDENTITIES] != [
        identity[1:] for identity in zenodo_identities
    ]:
        fail("Zenodo ORCIDs or affiliations do not match verified identities")
    if any(
        term in author.lower()
        for author in cargo_authors
        for term in ("claude", "anthropic", "ai assistant", "language model")
    ):
        fail("AI systems must not be listed as authors or creators")

    abstract = normalize(cff["abstract"])
    if abstract != normalize(zenodo["description"]):
        fail("CFF abstract and Zenodo description differ")
    words = len(abstract.split())
    if not 140 <= words <= 180:
        fail(f"abstract must contain 140 to 180 words, found {words}")
    if any(mark in abstract for mark in ("—", "–", ";")):
        fail("abstract contains prohibited punctuation")

    if cff["repository-code"] != REPOSITORY or package["repository"] != REPOSITORY:
        fail("canonical repository URLs disagree")
    if cff["url"] != "https://docs.rs/sequoia-boost":
        fail("CFF url must identify the documentation site")
    if zenodo["communities"] or zenodo["grants"]:
        fail("communities and grants require project-specific verification")
    if cff.get("doi") != CONCEPT_DOI:
        fail(f"CITATION.cff must use the verified concept DOI {CONCEPT_DOI}")
    if "doi" in zenodo:
        fail("Zenodo deposit metadata must not hard-code a release DOI")
    readme = (ROOT / "README.md").read_text()
    if f"https://doi.org/{CONCEPT_DOI}" not in readme:
        fail("README must link to the verified concept DOI")

    tags = subprocess.run(
        ["git", "tag", "--list", f"v{package['version']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tags and "date-released" not in cff:
        fail("a tagged version must include a verified date-released")
    if not tags and "date-released" in cff:
        fail("an untagged version must not claim a release date")

    print(
        f"metadata valid: version {package['version']}, {words}-word abstract, "
        f"{len(cargo_authors)} author(s), concept DOI {CONCEPT_DOI}"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        print(f"metadata error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
