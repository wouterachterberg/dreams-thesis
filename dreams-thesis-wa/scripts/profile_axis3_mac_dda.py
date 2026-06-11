from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys


RDLogger.DisableLog("rdApp.*")

SEED = 3407
PROTON = 1.007276
SODIUM = 22.989218

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import AXIS3_RAW_DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT
DATA_DIR = AXIS3_RAW_DATA_DIR
POS_MZML_DIR = DATA_DIR / "POS"
CSV_PATHS = [
    DATA_DIR / "identities_mixes_positive.csv",
    DATA_DIR / "fragments_mixes_positive.csv",
]
TRAINING_FIRST_BLOCK_INCHIKEY_PATH = None
OUT_DIR = ROOT / "dreams-thesis-wa/results/axis3"
JSON_OUT = OUT_DIR / "axis3_mac_dda_profile_summary.json"
REPORT_OUT = OUT_DIR / "axis3_mac_dda_profile_report.txt"

NS = "{http://psi.hupo.org/ms/mzml}"
LABEL_PATTERN = re.compile(
    r"(compound|smiles|inchi|inchikey|hmdb|chebi|pubchem|metabolite|identity|"
    r"molecule|formula|chemical|cas|common name|iupac|compound id)",
    re.IGNORECASE,
)


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def add_range(state: dict[str, Any], value: float | None) -> None:
    if value is None:
        return
    if state["min"] is None or value < state["min"]:
        state["min"] = value
    if state["max"] is None or value > state["max"]:
        state["max"] = value


def distribution(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(values)
    p90_index = int(math.ceil(0.9 * len(ordered))) - 1
    return {
        "count": len(values),
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "p90": ordered[max(0, min(p90_index, len(ordered) - 1))],
        "max": ordered[-1],
    }


def pct(count: int, total: int) -> str:
    if total == 0:
        return "NA"
    return f"{100.0 * count / total:.1f}%"


def sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def cv_params(elem: ET.Element) -> list[ET.Element]:
    return [child for child in elem.iter() if local_name(child.tag) == "cvParam"]


def user_params(elem: ET.Element) -> list[ET.Element]:
    return [child for child in elem.iter() if local_name(child.tag) == "userParam"]


def cv_key(cv: ET.Element) -> tuple[str, str]:
    return cv.attrib.get("accession", ""), cv.attrib.get("name", "")


def process_instrument_configuration(elem: ET.Element) -> dict[str, Any]:
    direct_cvs = [
        child
        for child in list(elem)
        if local_name(child.tag) == "cvParam"
    ]
    models = []
    serial_numbers = []
    for cv in direct_cvs:
        name = cv.attrib.get("name", "")
        value = cv.attrib.get("value", "")
        accession = cv.attrib.get("accession", "")
        if accession == "MS:1000529" or name == "instrument serial number":
            if value:
                serial_numbers.append(value)
        elif name:
            models.append(name)

    components: dict[str, list[dict[str, Any]]] = {
        "source": [],
        "analyser": [],
        "detector": [],
    }
    component_list = None
    for child in list(elem):
        if local_name(child.tag) == "componentList":
            component_list = child
            break

    if component_list is not None:
        for component in list(component_list):
            tag = local_name(component.tag)
            if tag == "analyzer":
                key = "analyser"
            elif tag in components:
                key = tag
            else:
                continue
            for cv in cv_params(component):
                components[key].append(
                    {
                        "order": component.attrib.get("order"),
                        "accession": cv.attrib.get("accession", ""),
                        "name": cv.attrib.get("name", ""),
                        "value": cv.attrib.get("value", ""),
                    }
                )

    return {
        "id": elem.attrib.get("id", ""),
        "models": models,
        "serial_numbers": serial_numbers,
        "source": components["source"],
        "mass_analyser": components["analyser"],
        "detector": components["detector"],
    }


def detect_polarity(cvs: list[ET.Element]) -> str:
    has_positive = any(cv.attrib.get("accession") == "MS:1000130" for cv in cvs)
    has_negative = any(cv.attrib.get("accession") == "MS:1000129" for cv in cvs)
    if has_positive and has_negative:
        return "mixed"
    if has_positive:
        return "positive"
    if has_negative:
        return "negative"
    return "unknown"


def detect_centroid(cvs: list[ET.Element]) -> str:
    has_centroid = any(cv.attrib.get("accession") == "MS:1000127" for cv in cvs)
    has_profile = any(cv.attrib.get("accession") == "MS:1000128" for cv in cvs)
    if has_centroid and has_profile:
        return "mixed"
    if has_centroid:
        return "centroid"
    if has_profile:
        return "profile"
    return "unknown"


def extract_ms_level(cvs: list[ET.Element]) -> int | None:
    for cv in cvs:
        if cv.attrib.get("accession") == "MS:1000511" or cv.attrib.get("name") == "ms level":
            return as_int(cv.attrib.get("value"))
    return None


def extract_scan_time(cvs: list[ET.Element]) -> dict[str, Any] | None:
    for cv in cvs:
        if cv.attrib.get("accession") == "MS:1000016" or cv.attrib.get("name") == "scan start time":
            return {
                "value": as_float(cv.attrib.get("value")),
                "unit_name": cv.attrib.get("unitName", ""),
                "unit_accession": cv.attrib.get("unitAccession", ""),
                "unit_cv": cv.attrib.get("unitCvRef", ""),
            }
    return None


def extract_precursor_mz(elem: ET.Element) -> float | None:
    selected = []
    isolation = []
    for cv in cv_params(elem):
        accession, name = cv_key(cv)
        value = as_float(cv.attrib.get("value"))
        if value is None:
            continue
        if accession == "MS:1000744" or name == "selected ion m/z":
            selected.append(value)
        elif accession == "MS:1000827" or name == "isolation window target m/z":
            isolation.append(value)
    if selected:
        return selected[0]
    if isolation:
        return isolation[0]
    return None


def extract_activation(elem: ET.Element) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    activation_types = []
    collision_energies = []
    for activation in elem.iter():
        if local_name(activation.tag) != "activation":
            continue
        for cv in cv_params(activation):
            accession, name = cv_key(cv)
            if accession == "MS:1000045" or name == "collision energy":
                collision_energies.append(
                    {
                        "value": as_float(cv.attrib.get("value")),
                        "unit_name": cv.attrib.get("unitName", ""),
                        "unit_accession": cv.attrib.get("unitAccession", ""),
                    }
                )
            elif name:
                activation_types.append(
                    {
                        "accession": accession,
                        "name": name,
                    }
                )
    return activation_types, collision_energies


def process_spectrum(elem: ET.Element, file_name: str, summary: dict[str, Any]) -> None:
    summary["total_spectra"] += 1
    spectrum_id = elem.attrib.get("id", "")
    default_array_length = as_int(elem.attrib.get("defaultArrayLength"))
    cvs = cv_params(elem)
    users = user_params(elem)

    if len(summary["example_spectrum_ids"]) < 15 and spectrum_id:
        summary["example_spectrum_ids"].append({"file": file_name, "id": spectrum_id})

    ms_level = extract_ms_level(cvs)
    ms_key = str(ms_level) if ms_level is not None else "unknown"
    summary["ms_level_counts"][ms_key] += 1

    polarity = detect_polarity(cvs)
    centroid = detect_centroid(cvs)
    summary["polarity_counts"][polarity] += 1
    summary["centroid_profile_counts"][centroid] += 1

    scan_time = extract_scan_time(cvs)
    if scan_time and scan_time["value"] is not None:
        unit_key = scan_time["unit_name"] or "unknown"
        summary["retention_time_by_unit"][unit_key]["unit_name"] = scan_time["unit_name"]
        summary["retention_time_by_unit"][unit_key]["unit_accession"] = scan_time["unit_accession"]
        summary["retention_time_by_unit"][unit_key]["unit_cv"] = scan_time["unit_cv"]
        summary["retention_time_by_unit"][unit_key]["count"] += 1
        add_range(summary["retention_time_by_unit"][unit_key], scan_time["value"])

    for cv in cvs:
        accession, name = cv_key(cv)
        if accession == "MS:1000796" or name == "spectrum title":
            value = cv.attrib.get("value", "")
            summary["label_check"]["spectrum_title_count"] += 1
            if value and len(summary["label_check"]["example_titles"]) < 15:
                summary["label_check"]["example_titles"].append(value)
            if value and LABEL_PATTERN.search(value):
                summary["label_check"]["label_like_title_count"] += 1
                if len(summary["label_check"]["label_like_examples"]) < 15:
                    summary["label_check"]["label_like_examples"].append(
                        {"file": file_name, "field": "spectrum title", "value": value}
                    )

    for user in users:
        name = user.attrib.get("name", "")
        value = user.attrib.get("value", "")
        text = f"{name} {value}".strip()
        summary["label_check"]["spectrum_user_param_count"] += 1
        if text and len(summary["label_check"]["example_user_params"]) < 15:
            summary["label_check"]["example_user_params"].append(
                {"file": file_name, "name": name, "value": value}
            )
        if text and LABEL_PATTERN.search(text):
            summary["label_check"]["label_like_user_param_count"] += 1
            if len(summary["label_check"]["label_like_examples"]) < 15:
                summary["label_check"]["label_like_examples"].append(
                    {"file": file_name, "field": "userParam", "name": name, "value": value}
                )

    if ms_level != 2:
        return

    if default_array_length is not None:
        summary["ms2_peak_counts_by_polarity"][polarity].append(default_array_length)

    precursor_mz = extract_precursor_mz(elem)
    summary["ms2_precursor"]["total_ms2"] += 1
    if precursor_mz is not None:
        summary["ms2_precursor"]["with_precursor_mz"] += 1
        if polarity == "positive":
            add_range(summary["ms2_precursor"]["positive_precursor_mz_range"], precursor_mz)

    activation_types, collision_energies = extract_activation(elem)
    for activation in activation_types:
        label = f"{activation['accession']} {activation['name']}".strip()
        summary["activation_type_counts"][label] += 1
    for ce in collision_energies:
        value = ce["value"]
        if value is not None:
            unit_key = ce["unit_name"] or "no unit"
            summary["collision_energy_by_unit"][unit_key]["unit_name"] = ce["unit_name"]
            summary["collision_energy_by_unit"][unit_key]["unit_accession"] = ce["unit_accession"]
            summary["collision_energy_by_unit"][unit_key]["values"].append(value)


def profile_mzml_files(paths: list[Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input_files": [str(path) for path in paths],
        "file_count": len(paths),
        "total_spectra": 0,
        "ms_level_counts": Counter(),
        "polarity_counts": Counter(),
        "centroid_profile_counts": Counter(),
        "ms2_peak_counts_by_polarity": defaultdict(list),
        "ms2_precursor": {
            "total_ms2": 0,
            "with_precursor_mz": 0,
            "positive_precursor_mz_range": {"min": None, "max": None},
        },
        "activation_type_counts": Counter(),
        "collision_energy_by_unit": defaultdict(lambda: {"unit_name": "", "unit_accession": "", "values": []}),
        "retention_time_by_unit": defaultdict(
            lambda: {"unit_name": "", "unit_accession": "", "unit_cv": "", "count": 0, "min": None, "max": None}
        ),
        "instrument_configurations": [],
        "label_check": {
            "spectrum_title_count": 0,
            "example_titles": [],
            "spectrum_user_param_count": 0,
            "example_user_params": [],
            "label_like_title_count": 0,
            "label_like_user_param_count": 0,
            "label_like_examples": [],
        },
        "example_spectrum_ids": [],
        "parse_errors": [],
    }

    for path in paths:
        try:
            for event, elem in ET.iterparse(path, events=("end",)):
                tag = local_name(elem.tag)
                if tag == "binary":
                    elem.text = None
                    elem.clear()
                elif tag == "binaryDataArray":
                    elem.clear()
                elif tag == "instrumentConfiguration":
                    inst = process_instrument_configuration(elem)
                    inst["file"] = path.name
                    summary["instrument_configurations"].append(inst)
                    elem.clear()
                elif tag == "spectrum":
                    process_spectrum(elem, path.name, summary)
                    elem.clear()
        except Exception as exc:
            summary["parse_errors"].append({"file": str(path), "error": repr(exc)})

    summary["ms_level_counts"] = sorted_counter(summary["ms_level_counts"])
    summary["polarity_counts"] = sorted_counter(summary["polarity_counts"])
    summary["centroid_profile_counts"] = sorted_counter(summary["centroid_profile_counts"])
    summary["activation_type_counts"] = sorted_counter(summary["activation_type_counts"])
    summary["ms2_peak_count_distribution_by_polarity"] = {
        polarity: distribution(values)
        for polarity, values in sorted(summary["ms2_peak_counts_by_polarity"].items())
    }
    summary.pop("ms2_peak_counts_by_polarity", None)
    summary["collision_energy_distribution_by_unit"] = {
        unit: {
            "unit_name": payload["unit_name"],
            "unit_accession": payload["unit_accession"],
            **distribution(payload["values"]),
        }
        for unit, payload in sorted(summary["collision_energy_by_unit"].items())
    }
    summary.pop("collision_energy_by_unit", None)
    summary["retention_time_by_unit"] = {
        unit: payload
        for unit, payload in sorted(summary["retention_time_by_unit"].items())
    }
    summary["instrument_summary"] = summarise_instruments(summary["instrument_configurations"])
    summary["label_check"]["per_spectrum_compound_labels_present"] = bool(
        summary["label_check"]["label_like_title_count"]
        or summary["label_check"]["label_like_user_param_count"]
    )
    return summary


def summarise_instruments(instruments: list[dict[str, Any]]) -> dict[str, Any]:
    models = Counter()
    sources = Counter()
    analysers = Counter()
    detectors = Counter()
    serials = Counter()
    for inst in instruments:
        for model in inst.get("models", []):
            models[model] += 1
        for serial in inst.get("serial_numbers", []):
            serials[serial] += 1
        for item in inst.get("source", []):
            if item.get("name"):
                sources[item["name"]] += 1
        for item in inst.get("mass_analyser", []):
            if item.get("name"):
                analysers[item["name"]] += 1
        for item in inst.get("detector", []):
            if item.get("name"):
                detectors[item["name"]] += 1
    return {
        "model_strings": sorted_counter(models),
        "serial_numbers": sorted_counter(serials),
        "ion_sources": sorted_counter(sources),
        "mass_analysers": sorted_counter(analysers),
        "detectors": sorted_counter(detectors),
    }


def normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def detect_columns(columns: list[str], df: pd.DataFrame) -> dict[str, Any]:
    by_norm = {normalise_column_name(col): col for col in columns}
    smiles = [
        col for key, col in by_norm.items()
        if key in {"smiles", "smile", "canonicalsmiles", "isomericsmiles"}
    ]
    inchi = [
        col for key, col in by_norm.items()
        if "inchi" in key and "inchikey" not in key and "inchikey" not in col.lower()
    ]
    inchikey = [
        col for key, col in by_norm.items()
        if "inchikey" in key or "inchi_key" in col.lower()
    ]
    mass = [
        col for key, col in by_norm.items()
        if key in {"exactmass", "exactmonoisotopicmass", "monoisotopicmass", "neutralmass", "mass"}
    ]
    hmdb = [col for key, col in by_norm.items() if key == "hmdb" or "hmdb" in key]

    name_candidates = []
    for col in columns:
        key = normalise_column_name(col)
        if key in {"name", "compound", "compoundname", "metabolite", "metabolitename"}:
            values = df[col].dropna().astype(str).head(50)
            non_numeric = sum(as_float(value) is None for value in values)
            name_candidates.append((non_numeric, col))
    name_candidates.sort(reverse=True)

    return {
        "smiles": smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        "name": [col for _, col in name_candidates],
        "mass": mass,
        "hmdb": hmdb,
    }


def first_present(detected: dict[str, list[str]], key: str) -> str | None:
    values = detected.get(key, [])
    return values[0] if values else None


def mol_from_row(row: pd.Series, smiles_col: str | None, inchi_col: str | None):
    if smiles_col:
        value = row.get(smiles_col)
        if pd.notna(value) and str(value).strip():
            try:
                return Chem.MolFromSmiles(str(value).strip())
            except Exception:
                return None
    if inchi_col:
        value = row.get(inchi_col)
        if pd.notna(value) and str(value).strip():
            try:
                return Chem.MolFromInchi(str(value).strip())
            except Exception:
                return None
    return None


def first_block_from_inchikey(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    block = text.split("-", 1)[0].strip()
    if not block:
        return None
    return block


def profile_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    columns = list(df.columns)
    detected = detect_columns(columns, df)
    smiles_col = first_present(detected, "smiles")
    inchi_col = first_present(detected, "inchi")
    inchikey_col = first_present(detected, "inchikey")
    mass_col = first_present(detected, "mass")
    name_col = first_present(detected, "name")
    hmdb_col = first_present(detected, "hmdb")

    parseable = 0
    maccs = 0
    computed_masses = []
    inchikey_blocks = []
    malformed_structure_rows = 0

    usable_masses = []
    if mass_col:
        numeric_mass = pd.to_numeric(df[mass_col], errors="coerce")
        usable_masses = [
            float(value)
            for value in numeric_mass.dropna().tolist()
            if float(value) > 0
        ]

    for _, row in df.iterrows():
        mol = mol_from_row(row, smiles_col, inchi_col)
        if mol is not None:
            parseable += 1
            try:
                MACCSkeys.GenMACCSKeys(mol)
                maccs += 1
            except Exception:
                pass
            try:
                computed_masses.append(float(Descriptors.ExactMolWt(mol)))
            except Exception:
                pass
            try:
                block = first_block_from_inchikey(Chem.MolToInchiKey(mol))
                if block:
                    inchikey_blocks.append(block)
            except Exception:
                pass
        elif smiles_col or inchi_col:
            malformed_structure_rows += 1

        if inchikey_col:
            block = first_block_from_inchikey(row.get(inchikey_col))
            if block:
                inchikey_blocks.append(block)

    if not usable_masses and computed_masses:
        usable_masses = computed_masses
        mass_source = "ExactMolWt computed from structure"
    elif mass_col:
        mass_source = mass_col
    else:
        mass_source = None

    name_unique = None
    if name_col:
        name_unique = int(df[name_col].dropna().astype(str).nunique())
    hmdb_unique = None
    if hmdb_col:
        hmdb_unique = int(df[hmdb_col].dropna().astype(str).nunique())

    mass_dist = distribution(usable_masses)
    adduct_ranges = {
        "M_plus_H": {
            "min": mass_dist["min"] + PROTON if mass_dist["min"] is not None else None,
            "max": mass_dist["max"] + PROTON if mass_dist["max"] is not None else None,
        },
        "M_plus_Na": {
            "min": mass_dist["min"] + SODIUM if mass_dist["min"] is not None else None,
            "max": mass_dist["max"] + SODIUM if mass_dist["max"] is not None else None,
        },
    }

    return {
        "path": str(path),
        "row_count": int(len(df)),
        "column_names": columns,
        "detected_columns": detected,
        "selected_columns": {
            "smiles": smiles_col,
            "inchi": inchi_col,
            "inchikey": inchikey_col,
            "name": name_col,
            "mass": mass_col,
            "hmdb": hmdb_col,
        },
        "parseable_structure_count": parseable,
        "maccs_fingerprint_count": maccs,
        "malformed_structure_rows": malformed_structure_rows,
        "usable_neutral_monoisotopic_mass_count": len(usable_masses),
        "neutral_mass_source": mass_source,
        "unique_first_block_inchikey_count": int(len(set(inchikey_blocks))),
        "unique_name_count": name_unique,
        "unique_hmdb_count": hmdb_unique,
        "neutral_mass_distribution": mass_dist,
        "adduct_mz_ranges": adduct_ranges,
    }


def choose_primary_csv(csv_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not csv_summaries:
        return None
    ranked = sorted(
        csv_summaries,
        key=lambda item: (
            item["usable_neutral_monoisotopic_mass_count"],
            item["parseable_structure_count"],
            item["unique_hmdb_count"] or 0,
            -item["row_count"],
        ),
        reverse=True,
    )
    return ranked[0]


def ranges_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    if a.get("min") is None or a.get("max") is None or b.get("min") is None or b.get("max") is None:
        return None
    return max(a["min"], b["min"]) <= min(a["max"], b["max"])


def profile_training_overlap(primary_csv: dict[str, Any] | None) -> dict[str, Any]:
    if TRAINING_FIRST_BLOCK_INCHIKEY_PATH is None:
        return {
            "status": "skipped",
            "reason": "No MassSpecGym training first-block InChIKey file was provided.",
        }
    path = Path(TRAINING_FIRST_BLOCK_INCHIKEY_PATH)
    if not path.exists():
        return {
            "status": "skipped",
            "reason": f"Provided training first-block InChIKey file does not exist: {path}",
        }
    if primary_csv is None:
        return {"status": "skipped", "reason": "No primary compound CSV was available."}
    return {"status": "skipped", "reason": "Training overlap requires compound InChIKeys in the primary CSV."}


def compact_for_json(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return sorted_counter(obj)
    if isinstance(obj, defaultdict):
        return {str(k): compact_for_json(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): compact_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [compact_for_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_distribution(dist: dict[str, Any]) -> str:
    return (
        f"count {dist['count']}, min {format_number(dist['min'])}, "
        f"median {format_number(dist['median'])}, mean {format_number(dist['mean'])}, "
        f"p90 {format_number(dist['p90'])}, max {format_number(dist['max'])}"
    )


def build_report(summary: dict[str, Any]) -> str:
    mzml = summary["mzml"]
    csvs = summary["compound_csvs"]
    primary = summary["primary_compound_csv"]
    precursor_range = mzml["ms2_precursor"]["positive_precursor_mz_range"]
    lines = []
    lines.append("Axis 3 MAC DDA feasibility profile")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append(f"Seed: {SEED}")
    lines.append("")
    lines.append("Inputs")
    lines.append(f"- POS mzML directory: {POS_MZML_DIR}")
    lines.append(f"- POS mzML files profiled: {mzml['file_count']}")
    lines.append(f"- Positive CSVs profiled: {', '.join(Path(item['path']).name for item in csvs)}")
    lines.append("- Metadata file search: no separate metadata, README, method, TXT, Excel, JSON, or info file found under axis_3_data")
    lines.append(f"- MassSpecGym training first-block InChIKeys: {summary['training_overlap']['status']} - {summary['training_overlap']['reason']}")
    lines.append("")
    lines.append("mzML spectra")
    lines.append(f"- Total spectra: {mzml['total_spectra']}")
    lines.append(f"- MS level counts: {mzml['ms_level_counts']}")
    lines.append(f"- Polarity counts: {mzml['polarity_counts']}")
    lines.append(f"- Centroid/profile counts: {mzml['centroid_profile_counts']}")
    lines.append("- MS2 peak-count distributions from defaultArrayLength:")
    for polarity in ["positive", "negative", "unknown", "mixed"]:
        dist = mzml["ms2_peak_count_distribution_by_polarity"].get(polarity)
        if dist:
            lines.append(f"  - {polarity}: {render_distribution(dist)}")
        else:
            lines.append(f"  - {polarity}: count 0")
    lines.append(
        f"- MS2 precursor m/z coverage: {mzml['ms2_precursor']['with_precursor_mz']} of "
        f"{mzml['ms2_precursor']['total_ms2']} MS2 ({pct(mzml['ms2_precursor']['with_precursor_mz'], mzml['ms2_precursor']['total_ms2'])})"
    )
    lines.append(
        f"- Positive MS2 precursor m/z range: {format_number(precursor_range['min'])} to {format_number(precursor_range['max'])}"
    )
    lines.append(f"- Activation types: {mzml['activation_type_counts']}")
    lines.append("- Collision-energy distributions:")
    for unit, dist in mzml["collision_energy_distribution_by_unit"].items():
        unit_label = unit or "no unit"
        lines.append(f"  - {unit_label}: {render_distribution(dist)}")
    lines.append("- Retention-time ranges by raw CV unit:")
    for unit, payload in mzml["retention_time_by_unit"].items():
        lines.append(
            f"  - {unit}: count {payload['count']}, raw min {format_number(payload['min'])}, "
            f"raw max {format_number(payload['max'])}, unit accession {payload['unit_accession']}"
        )
    inst = mzml["instrument_summary"]
    lines.append("- Instrument metadata:")
    lines.append(f"  - Model string: {inst['model_strings']}")
    lines.append(f"  - Serial number: {inst['serial_numbers']}")
    lines.append(f"  - Ion source: {inst['ion_sources']}")
    lines.append(f"  - Mass analyser: {inst['mass_analysers']}")
    lines.append(f"  - Detector: {inst['detectors']}")
    lines.append("")
    lines.append("Per-spectrum label check")
    labels = mzml["label_check"]
    lines.append(f"- Spectrum title CV terms found: {labels['spectrum_title_count']}")
    lines.append(f"- Spectrum-level userParams found: {labels['spectrum_user_param_count']}")
    lines.append(f"- Label-like spectrum titles: {labels['label_like_title_count']}")
    lines.append(f"- Label-like spectrum userParams: {labels['label_like_user_param_count']}")
    if labels["example_titles"]:
        lines.append("- Example titles:")
        for title in labels["example_titles"][:15]:
            lines.append(f"  - {title}")
    else:
        lines.append("- Example titles: none found")
    if labels["label_like_examples"]:
        lines.append("- Label-like examples:")
        for item in labels["label_like_examples"][:15]:
            lines.append(f"  - {item}")
    lines.append(
        "- Decisive scope result: per-scan compound labels are "
        + ("present in mzML metadata" if labels["per_spectrum_compound_labels_present"] else "not present in mzML metadata")
    )
    lines.append("")
    lines.append("Compound CSVs")
    for item in csvs:
        lines.append(f"- {Path(item['path']).name}")
        lines.append(f"  - Rows: {item['row_count']}")
        lines.append(f"  - Columns: {item['column_names']}")
        lines.append(f"  - Detected columns: {item['detected_columns']}")
        lines.append(f"  - Selected columns: {item['selected_columns']}")
        lines.append(
            f"  - Parseable structures: {item['parseable_structure_count']} of {item['row_count']} "
            f"({pct(item['parseable_structure_count'], item['row_count'])})"
        )
        lines.append(
            f"  - MACCS fingerprints: {item['maccs_fingerprint_count']} of {item['row_count']} "
            f"({pct(item['maccs_fingerprint_count'], item['row_count'])})"
        )
        lines.append(
            f"  - Usable neutral monoisotopic masses: {item['usable_neutral_monoisotopic_mass_count']} of {item['row_count']} "
            f"({pct(item['usable_neutral_monoisotopic_mass_count'], item['row_count'])}), source {item['neutral_mass_source']}"
        )
        lines.append(f"  - Unique first-block InChIKeys: {item['unique_first_block_inchikey_count']}")
        lines.append(f"  - Unique names: {item['unique_name_count']}")
        lines.append(f"  - Unique HMDB IDs: {item['unique_hmdb_count']}")
        mass_dist = item["neutral_mass_distribution"]
        lines.append(
            f"  - Neutral mass range: {format_number(mass_dist['min'])} to {format_number(mass_dist['max'])}"
        )
        lines.append(
            f"  - Implied [M+H]+ range: {format_number(item['adduct_mz_ranges']['M_plus_H']['min'])} to "
            f"{format_number(item['adduct_mz_ranges']['M_plus_H']['max'])}"
        )
        lines.append(
            f"  - Implied [M+Na]+ range: {format_number(item['adduct_mz_ranges']['M_plus_Na']['min'])} to "
            f"{format_number(item['adduct_mz_ranges']['M_plus_Na']['max'])}"
        )
    if primary:
        lines.append("")
        lines.append(f"Primary compound table selected for feasibility gates: {Path(primary['path']).name}")
        overlap_h = ranges_overlap(primary["adduct_mz_ranges"]["M_plus_H"], precursor_range)
        overlap_na = ranges_overlap(primary["adduct_mz_ranges"]["M_plus_Na"], precursor_range)
        lines.append(f"- [M+H]+ range overlaps observed positive precursor range: {overlap_h}")
        lines.append(f"- [M+Na]+ range overlaps observed positive precursor range: {overlap_na}")
    lines.append("")
    lines.append("Feasibility verdict")
    per_spectrum = labels["per_spectrum_compound_labels_present"]
    centroid_ok = mzml["centroid_profile_counts"].get("centroid", 0) == mzml["total_spectra"]
    structure_ok = bool(primary and primary["parseable_structure_count"] > 0 and primary["maccs_fingerprint_count"] > 0)
    mass_ok = bool(primary and primary["usable_neutral_monoisotopic_mass_count"] > 0)
    lines.append("- Scope: " + ("per-spectrum upgrade is possible from mzML metadata" if per_spectrum else "set-recovery only from mzML metadata"))
    lines.append("- Centroiding: " + ("adequate, all profiled spectra are centroid spectra" if centroid_ok else "red flag, centroid/profile terms are mixed or missing"))
    lines.append("- Structure coverage: " + ("adequate for MACCS labels" if structure_ok else "red flag, no parseable structures or InChIKeys were detected in the positive CSVs"))
    lines.append("- Mass coverage: " + ("adequate for precursor range checks" if mass_ok else "red flag, no usable neutral masses detected"))
    if primary and mass_ok:
        overlap_h = ranges_overlap(primary["adduct_mz_ranges"]["M_plus_H"], precursor_range)
        overlap_na = ranges_overlap(primary["adduct_mz_ranges"]["M_plus_Na"], precursor_range)
        lines.append("- Precursor sanity check: " + ("mass-derived adduct ranges overlap observed positive precursor m/z" if overlap_h or overlap_na else "red flag, mass-derived adduct ranges do not overlap observed positive precursor m/z"))
    if mzml["parse_errors"]:
        lines.append(f"- Parse errors: {len(mzml['parse_errors'])} files had errors")
    else:
        lines.append("- Parse errors: none")
    lines.append(f"- JSON summary saved to: {JSON_OUT}")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mzml_paths = sorted(POS_MZML_DIR.glob("*.mzML"))
    csv_summaries = [profile_csv(path) for path in CSV_PATHS if path.exists()]
    primary = choose_primary_csv(csv_summaries)
    mzml_summary = profile_mzml_files(mzml_paths)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "cwd": os.getcwd(),
        "data_dir": str(DATA_DIR),
        "mzml": mzml_summary,
        "compound_csvs": csv_summaries,
        "primary_compound_csv": primary,
        "training_overlap": profile_training_overlap(primary),
        "output": {
            "json": str(JSON_OUT),
            "report": str(REPORT_OUT),
        },
    }
    report = build_report(summary)
    JSON_OUT.write_text(json.dumps(compact_for_json(summary), indent=2), encoding="utf-8")
    REPORT_OUT.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
