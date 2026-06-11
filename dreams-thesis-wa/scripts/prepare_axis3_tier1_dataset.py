from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import AXIS3_RAW_DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dreams.definitions import ADDUCT, CHARGE, FOLD, PRECURSOR_MZ, SMILES, SPECTRUM
from dreams.utils import mols as mu
from dreams.utils import spectra as su
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA


RDLogger.DisableLog("rdApp.*")

SEED = 3407
MASS_TOL_DA = 0.005
FRAGMENT_PRECURSOR_TOL_DA = 0.01
MAX_MODEL_PEAKS = 100
PROTON_INTENSITY = 1.1
PUBCHEM_SLEEP_SECONDS = 0.15

DATA_DIR = AXIS3_RAW_DATA_DIR
IDENTITIES_PATH = DATA_DIR / "identities_mixes_positive.csv"
FRAGMENTS_PATH = DATA_DIR / "fragments_mixes_positive.csv"
OOD_PATH = ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet"
AXIS2_TRAIN_HDF5 = ROOT / "dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5"
OUT_DIR = ROOT / "dreams-thesis-wa/results/axis3"

RESOLUTION_CACHE_PATH = OUT_DIR / "axis3_structure_resolution_cache.json"
PER_SPECTRUM_PARQUET = OUT_DIR / "axis3_tier1_per_spectrum_records.parquet"
MODEL_READY_HDF5 = OUT_DIR / "axis3_tier1_model_ready.hdf5"
CLOSED_LIBRARY_PARQUET = OUT_DIR / "axis3_closed_reference_library.parquet"
CLOSED_LIBRARY_NPZ = OUT_DIR / "axis3_closed_reference_library.npz"
OPEN_LIBRARY_PARQUET = OUT_DIR / "axis3_open_reference_library.parquet"
OPEN_LIBRARY_NPZ = OUT_DIR / "axis3_open_reference_library.npz"
SEEN_FLAGS_CSV = OUT_DIR / "axis3_mac_compound_seen_flags.csv"
TRAIN_FIRST_BLOCK_TXT = OUT_DIR / "axis2_train_first_block_inchikeys.txt"
RESOLUTION_DETAILS_CSV = OUT_DIR / "axis3_compound_resolution_details.csv"
JOIN_MISMATCH_CSV = OUT_DIR / "axis3_identity_fragment_mismatches.csv"
QC_JSON = OUT_DIR / "axis3_tier1_qc_summary.json"
QC_REPORT = OUT_DIR / "axis3_tier1_qc_report.txt"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def to_float(value: Any) -> float | None:
    try:
        if value is None or clean_text(value) == "":
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def parse_compound_name(name: str) -> tuple[str, str, str]:
    text = clean_text(name)
    if "_[" not in text or not text.endswith(("+", "-")):
        return text, "", query_name_from_base(text)
    base, adduct_tail = text.rsplit("_", 1)
    if adduct_tail.startswith("[") and "]" in adduct_tail:
        return base, adduct_tail, query_name_from_base(base)
    return text, "", query_name_from_base(text)


def query_name_from_base(base: str) -> str:
    text = clean_text(base)
    if text.startswith("AGP_"):
        text = text[4:]
    text = text.replace("_", " ")
    return " ".join(text.split())


def name_query_candidates(base: str) -> list[str]:
    query = query_name_from_base(base)
    candidates = [
        query,
        query.replace(" ", "-"),
        query.replace(" ", ""),
        clean_text(base).replace("_", " "),
        clean_text(base),
    ]
    out = []
    seen = set()
    for candidate in candidates:
        candidate = clean_text(candidate)
        if candidate and candidate.lower() not in seen:
            out.append(candidate)
            seen.add(candidate.lower())
    return out


def first_block_inchikey(value: str) -> str:
    return clean_text(value).split("-", 1)[0]


def mol_from_smiles(smiles: str):
    if not clean_text(smiles):
        return None
    try:
        return Chem.MolFromSmiles(clean_text(smiles))
    except Exception:
        return None


def mol_to_first_block(mol) -> str:
    try:
        return first_block_inchikey(Chem.MolToInchiKey(mol))
    except Exception:
        return ""


def maccs_166_from_mol(mol) -> np.ndarray:
    return mu.maccs_fp(mol, as_numpy=True).astype(np.uint8, copy=False)


def bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(x)) for x in bits.tolist())


def find_local_hmdb_structure_files() -> list[str]:
    roots = [ROOT / "dreams-thesis-wa/data", DATA_DIR]
    matches = []
    suffixes = {".sdf", ".mol", ".smi", ".smiles"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            name = path.name.lower()
            if path.is_file() and "hmdb" in name and path.suffix.lower() in suffixes:
                matches.append(str(path))
    return sorted(set(matches))


def pubchem_get_json(path: str, timeout: int = 30) -> dict[str, Any] | None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Axis3DreaMSPrep/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404, 503}:
            return None
        raise
    except Exception:
        return None


def normalise_pubchem_properties(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    props = payload.get("PropertyTable", {}).get("Properties", [])
    out = []
    for item in props:
        smiles = clean_text(
            item.get("IsomericSMILES")
            or item.get("SMILES")
            or item.get("CanonicalSMILES")
            or item.get("ConnectivitySMILES")
        )
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        exact_mass = to_float(item.get("ExactMass"))
        if exact_mass is None:
            exact_mass = float(Descriptors.ExactMolWt(mol))
        inchikey = clean_text(item.get("InChIKey"))
        if not inchikey:
            try:
                inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                inchikey = ""
        out.append(
            {
                "cid": int(item["CID"]) if "CID" in item else None,
                "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                "inchikey": inchikey,
                "first_block_inchikey": first_block_inchikey(inchikey),
                "pubchem_exact_mass": exact_mass,
            }
        )
    return out


def choose_best_candidate(candidates: list[dict[str, Any]], expected_mass: float | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if expected_mass is None:
        return candidates[0]
    ranked = sorted(
        candidates,
        key=lambda item: abs(float(item["pubchem_exact_mass"]) - expected_mass)
        if item.get("pubchem_exact_mass") is not None
        else float("inf"),
    )
    return ranked[0]


def resolve_by_pubchem_hmdb(hmdb: str, expected_mass: float | None) -> dict[str, Any] | None:
    hmdb = clean_text(hmdb)
    if not hmdb:
        return None
    path = (
        "compound/xref/RegistryID/"
        f"{urllib.parse.quote(hmdb)}"
        "/property/IsomericSMILES,CanonicalSMILES,InChIKey,ExactMass/JSON"
    )
    candidates = normalise_pubchem_properties(pubchem_get_json(path))
    best = choose_best_candidate(candidates, expected_mass)
    if best is None:
        return None
    best["resolution_source"] = "pubchem_hmdb_registry"
    best["resolution_query"] = hmdb
    best["n_pubchem_candidates"] = len(candidates)
    return best


def resolve_by_pubchem_name(base_name: str, expected_mass: float | None) -> dict[str, Any] | None:
    all_candidates = []
    used_query = ""
    for query in name_query_candidates(base_name):
        path = (
            "compound/name/"
            f"{urllib.parse.quote(query)}"
            "/property/IsomericSMILES,CanonicalSMILES,InChIKey,ExactMass/JSON"
        )
        candidates = normalise_pubchem_properties(pubchem_get_json(path))
        if candidates:
            all_candidates = candidates
            used_query = query
            break
        time.sleep(PUBCHEM_SLEEP_SECONDS)
    best = choose_best_candidate(all_candidates, expected_mass)
    if best is None:
        return None
    best["resolution_source"] = "pubchem_name"
    best["resolution_query"] = used_query
    best["n_pubchem_candidates"] = len(all_candidates)
    return best


def load_resolution_cache() -> dict[str, Any]:
    if RESOLUTION_CACHE_PATH.exists():
        return json.loads(RESOLUTION_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_resolution_cache(cache: dict[str, Any]) -> None:
    RESOLUTION_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def resolve_compounds(unique_compounds: pd.DataFrame) -> pd.DataFrame:
    cache = load_resolution_cache()
    rows = []
    for row in tqdm(unique_compounds.itertuples(index=False), total=len(unique_compounds), desc="Resolving compounds"):
        compound_key = clean_text(row.compound_key)
        expected_mass = to_float(row.exactmass)
        cached = cache.get(compound_key)
        if cached is None:
            resolved = None
            hmdb = clean_text(row.hmdb)
            if hmdb:
                resolved = resolve_by_pubchem_hmdb(hmdb, expected_mass)
                time.sleep(PUBCHEM_SLEEP_SECONDS)
            if resolved is None:
                resolved = resolve_by_pubchem_name(clean_text(row.base_compound_name), expected_mass)
                time.sleep(PUBCHEM_SLEEP_SECONDS)
            if resolved is None:
                cached = {
                    "resolved": False,
                    "resolution_source": "",
                    "resolution_query": "",
                    "smiles": "",
                    "inchikey": "",
                    "first_block_inchikey": "",
                    "pubchem_exact_mass": None,
                    "n_pubchem_candidates": 0,
                }
            else:
                cached = {"resolved": True, **resolved}
            cache[compound_key] = cached
            save_resolution_cache(cache)

        smiles = clean_text(cached.get("smiles"))
        mol = mol_from_smiles(smiles)
        rdkit_exact_mass = float(Descriptors.ExactMolWt(mol)) if mol is not None else None
        mass_delta_da = None
        mass_pass = False
        if rdkit_exact_mass is not None and expected_mass is not None:
            mass_delta_da = rdkit_exact_mass - expected_mass
            mass_pass = abs(mass_delta_da) <= MASS_TOL_DA
        maccs_bits = ""
        if mol is not None:
            try:
                maccs_bits = bitstring(maccs_166_from_mol(mol))
            except Exception:
                maccs_bits = ""
        rows.append(
            {
                "compound_key": compound_key,
                "base_compound_name": clean_text(row.base_compound_name),
                "query_compound_name": clean_text(row.query_compound_name),
                "hmdb": clean_text(row.hmdb),
                "exactmass": expected_mass,
                "resolved": bool(cached.get("resolved")) and mol is not None,
                "verified_mass": bool(mass_pass),
                "included_downstream": bool(cached.get("resolved")) and mol is not None and bool(mass_pass) and bool(maccs_bits),
                "resolution_source": clean_text(cached.get("resolution_source")),
                "resolution_query": clean_text(cached.get("resolution_query")),
                "n_pubchem_candidates": int(cached.get("n_pubchem_candidates") or 0),
                "smiles": smiles,
                "inchikey": clean_text(cached.get("inchikey")) or (Chem.MolToInchiKey(mol) if mol is not None else ""),
                "first_block_inchikey": clean_text(cached.get("first_block_inchikey")) or (mol_to_first_block(mol) if mol is not None else ""),
                "rdkit_exact_mass": rdkit_exact_mass,
                "pubchem_exact_mass": to_float(cached.get("pubchem_exact_mass")),
                "mass_delta_da": mass_delta_da,
                "maccs_166": maccs_bits,
            }
        )
    return pd.DataFrame(rows)


def make_compound_key(hmdb: Any, base: str, exactmass: Any) -> str:
    hmdb_text = clean_text(hmdb)
    mass = to_float(exactmass)
    mass_text = f"{mass:.6f}" if mass is not None else ""
    return f"{hmdb_text}|{clean_text(base)}|{mass_text}"


def join_and_check(identities: pd.DataFrame, fragments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    identities = identities.copy()
    fragments = fragments.copy()
    identities["spectrum"] = pd.to_numeric(identities["spectrum"], errors="coerce").astype("Int64")
    identities["identity_row_number"] = np.arange(1, len(identities) + 1, dtype=np.int64)
    fragments["fragment_compound_index"] = pd.to_numeric(fragments["compound"], errors="coerce").astype("Int64")
    fragments = fragments.rename(
        columns={
            "compound": "fragment_compound_raw",
            "name": "fragment_name",
            "hmdb": "fragment_hmdb",
            "i": "intensity",
        }
    )

    def candidate_join(left_key: str, right_key: str) -> tuple[pd.DataFrame, dict[str, int]]:
        joined_candidate = fragments.merge(
            identities[["identity_row_number", "spectrum", "compound", "hmdb", "precursorMz", "exactmass"]],
            left_on=left_key,
            right_on=right_key,
            how="left",
            validate="many_to_one",
        )
        name_bad = joined_candidate["fragment_name"].fillna("").astype(str) != joined_candidate["compound"].fillna("").astype(str)
        hmdb_bad = joined_candidate["fragment_hmdb"].fillna("").astype(str) != joined_candidate["hmdb"].fillna("").astype(str)
        metrics = {
            "unmatched_fragment_rows": int(joined_candidate["compound"].isna().sum()),
            "name_mismatch_rows": int(name_bad.sum()),
            "hmdb_mismatch_rows": int(hmdb_bad.sum()),
        }
        return joined_candidate, metrics

    spectrum_join, spectrum_metrics = candidate_join("fragment_compound_index", "spectrum")
    row_number_join, row_number_metrics = candidate_join("fragment_compound_index", "identity_row_number")
    spectrum_score = sum(spectrum_metrics.values())
    row_number_score = sum(row_number_metrics.values())
    if row_number_score < spectrum_score:
        joined = row_number_join
        join_strategy = "fragments.compound to identities row number"
        join_key_warning = "fragments.compound did not match identities.spectrum for this file"
    else:
        joined = spectrum_join
        join_strategy = "fragments.compound to identities.spectrum"
        join_key_warning = ""

    name_mismatch = joined["fragment_name"].fillna("").astype(str) != joined["compound"].fillna("").astype(str)
    hmdb_left = joined["fragment_hmdb"].fillna("").astype(str)
    hmdb_right = joined["hmdb"].fillna("").astype(str)
    hmdb_mismatch = hmdb_left != hmdb_right
    mismatch_df = joined.loc[name_mismatch | hmdb_mismatch, [
        "fragment_compound_index", "identity_row_number", "spectrum", "fragment_name", "compound", "fragment_hmdb", "hmdb"
    ]].drop_duplicates()
    unmatched_fragment_rows = int(joined["compound"].isna().sum())
    identity_with_fragment = set(joined["spectrum"].dropna().astype(int).tolist())
    identity_without_fragments = int((~identities["spectrum"].astype("Int64").isin(identity_with_fragment)).sum())
    summary = {
        "join_strategy": join_strategy,
        "join_key_warning": join_key_warning,
        "candidate_fragments_compound_to_identities_spectrum": spectrum_metrics,
        "candidate_fragments_compound_to_identities_row_number": row_number_metrics,
        "identity_rows": int(len(identities)),
        "fragment_rows": int(len(fragments)),
        "joined_fragment_rows": int(len(joined)),
        "unmatched_fragment_rows": unmatched_fragment_rows,
        "identity_rows_without_fragments": identity_without_fragments,
        "name_mismatch_rows": int(name_mismatch.sum()),
        "hmdb_mismatch_rows": int(hmdb_mismatch.sum()),
        "mismatch_spectra": int(mismatch_df["spectrum"].nunique()) if len(mismatch_df) else 0,
    }
    return joined, mismatch_df, summary


def build_identity_table(identities: pd.DataFrame) -> pd.DataFrame:
    df = identities.copy()
    parsed = df["compound"].apply(lambda value: parse_compound_name(clean_text(value)))
    df["base_compound_name"] = [x[0] for x in parsed]
    df["adduct"] = [x[1] for x in parsed]
    df["query_compound_name"] = [x[2] for x in parsed]
    df["compound_key"] = [
        make_compound_key(hmdb, base, mass)
        for hmdb, base, mass in zip(df["hmdb"], df["base_compound_name"], df["exactmass"])
    ]
    return df


def clean_peaks_for_spectrum(group: pd.DataFrame, precursor_mz: float) -> tuple[np.ndarray, np.ndarray, int]:
    mzs = pd.to_numeric(group["mz"], errors="coerce").to_numpy(dtype=np.float64)
    intensities = pd.to_numeric(group["intensity"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0) & (intensities > 0)
    mzs = mzs[valid]
    intensities = intensities[valid]
    order = np.argsort(mzs)
    mzs = mzs[order]
    intensities = intensities[order]
    keep = mzs <= (precursor_mz + FRAGMENT_PRECURSOR_TOL_DA)
    dropped = int((~keep).sum())
    return mzs[keep].astype(np.float32), intensities[keep].astype(np.float32), dropped


def load_ood_library() -> pd.DataFrame:
    df = pd.read_parquet(OOD_PATH)
    rows = []
    seen = set()
    for row in tqdm(df[["smiles", "inchikey"]].drop_duplicates().itertuples(index=False), desc="Building OOD library"):
        full_inchikey = ""
        mol = mol_from_smiles(row.smiles)
        if mol is None:
            continue
        inchikey = clean_text(row.inchikey)
        if "-" not in inchikey:
            try:
                full_inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                full_inchikey = inchikey
            first_block = first_block_inchikey(inchikey or full_inchikey)
        else:
            full_inchikey = inchikey
            first_block = first_block_inchikey(inchikey)
        if not first_block or first_block in seen:
            continue
        seen.add(first_block)
        rows.append(
            {
                "first_block_inchikey": first_block,
                "inchikey": full_inchikey or inchikey,
                "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                "source_pool": "axis2_ood",
                "hmdb": "",
                "base_compound_name": "",
                "exactmass": float(Descriptors.ExactMolWt(mol)),
                "maccs_166": bitstring(maccs_166_from_mol(mol)),
            }
        )
    return pd.DataFrame(rows)


def bits_from_bitstrings(values: list[str]) -> np.ndarray:
    out = np.zeros((len(values), 166), dtype=np.uint8)
    for i, value in enumerate(values):
        text = clean_text(value)
        if len(text) != 166:
            continue
        out[i] = np.fromiter((int(ch) for ch in text), dtype=np.uint8, count=166)
    return out


def write_library(df: pd.DataFrame, parquet_path: Path, npz_path: Path) -> None:
    df.to_parquet(parquet_path, index=False)
    np.savez_compressed(
        npz_path,
        first_block_inchikey=df["first_block_inchikey"].astype(str).to_numpy(),
        inchikey=df["inchikey"].astype(str).to_numpy(),
        smiles=df["smiles"].astype(str).to_numpy(),
        source_pool=df["source_pool"].astype(str).to_numpy(),
        maccs_166=bits_from_bitstrings(df["maccs_166"].astype(str).tolist()),
    )


def generate_axis2_train_first_blocks() -> set[str]:
    first_blocks = set()
    with h5py.File(AXIS2_TRAIN_HDF5, "r") as handle:
        folds = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle[FOLD][:]
        ]
        smiles = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle[SMILES][:]
        ]
    train_smiles = sorted({smi for smi, fold in zip(smiles, folds) if fold == "train" and clean_text(smi)})
    for smi in tqdm(train_smiles, desc="Generating Axis 2 train InChIKeys"):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        block = mol_to_first_block(mol)
        if block:
            first_blocks.add(block)
    TRAIN_FIRST_BLOCK_TXT.write_text("\n".join(sorted(first_blocks)) + "\n", encoding="utf-8")
    return first_blocks


def encode_ascii_array(values: list[Any], length: int | None = None) -> np.ndarray:
    cleaned = [clean_text(value) for value in values]
    dtype = h5py.string_dtype(encoding="utf-8", length=length)
    return np.array(cleaned, dtype=dtype)


def write_model_ready_hdf5(records: pd.DataFrame, raw_spectra: np.ndarray, processed_spectra: np.ndarray) -> None:
    with h5py.File(MODEL_READY_HDF5, "w") as handle:
        handle.attrs["seed"] = SEED
        handle.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        handle.attrs["description"] = "Axis 3 tier-1 MAC DDA per-spectrum dataset for MACCS-BCE retrieval"
        handle.attrs["preprocessor"] = "SpectrumPreprocessor(DataFormatA, prec_intens=1.1, n_highest_peaks=100)"
        handle.create_dataset(SPECTRUM, data=raw_spectra.astype(np.float32), compression="gzip", compression_opts=4)
        handle.create_dataset("processed_spectrum", data=processed_spectra.astype(np.float32), compression="gzip", compression_opts=4)
        handle.create_dataset(PRECURSOR_MZ, data=records["precursorMz"].to_numpy(dtype=np.float32), compression="gzip")
        handle.create_dataset(CHARGE, data=np.ones(len(records), dtype=np.int32), compression="gzip")
        handle.create_dataset(ADDUCT, data=encode_ascii_array(records["adduct"].tolist()), compression="gzip")
        handle.create_dataset(SMILES, data=encode_ascii_array(records["true_smiles"].tolist()), compression="gzip")
        handle.create_dataset(FOLD, data=encode_ascii_array(["axis3"] * len(records)), compression="gzip")
        handle.create_dataset("spectrum_id", data=records["spectrum"].to_numpy(dtype=np.int32), compression="gzip")
        handle.create_dataset("base_compound_name", data=encode_ascii_array(records["base_compound_name"].tolist()), compression="gzip")
        handle.create_dataset("hmdb", data=encode_ascii_array(records["hmdb"].tolist()), compression="gzip")
        handle.create_dataset("exactmass", data=records["exactmass"].to_numpy(dtype=np.float32), compression="gzip")
        handle.create_dataset("inchikey", data=encode_ascii_array(records["inchikey"].tolist()), compression="gzip")
        handle.create_dataset("first_block_inchikey", data=encode_ascii_array(records["first_block_inchikey"].tolist()), compression="gzip")
        handle.create_dataset("fp_maccs_166", data=bits_from_bitstrings(records["true_maccs_166"].tolist()), compression="gzip")
        handle.create_dataset("peak_count_cleaned", data=records["peak_count_cleaned"].to_numpy(dtype=np.int32), compression="gzip")


def main() -> None:
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    identities = pd.read_csv(IDENTITIES_PATH)
    fragments = pd.read_csv(FRAGMENTS_PATH)
    identity_table = build_identity_table(identities)
    joined, mismatch_df, join_summary = join_and_check(identity_table, fragments)
    mismatch_df.to_csv(JOIN_MISMATCH_CSV, index=False)

    local_hmdb_files = find_local_hmdb_structure_files()
    unique_compounds = (
        identity_table[["compound_key", "base_compound_name", "query_compound_name", "hmdb", "exactmass"]]
        .drop_duplicates()
        .sort_values(["hmdb", "base_compound_name", "exactmass"], na_position="last")
        .reset_index(drop=True)
    )
    resolution_df = resolve_compounds(unique_compounds)
    resolution_df.to_csv(RESOLUTION_DETAILS_CSV, index=False)

    identity_resolved = identity_table.merge(resolution_df, on="compound_key", how="left", suffixes=("", "_resolved"))
    fragments_by_spectrum = {int(k): g for k, g in joined.groupby("spectrum", dropna=True)}
    spec_preproc = SpectrumPreprocessor(
        dformat=DataFormatA(),
        prec_intens=PROTON_INTENSITY,
        n_highest_peaks=MAX_MODEL_PEAKS,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0.0,
        mz_shift_aug_max=0.0,
    )

    retained_rows = []
    raw_spectra = []
    processed_spectra = []
    peak_drop_total = 0
    peak_before_total = 0
    peak_after_total = 0
    spectra_with_dropped_peaks = 0
    zero_peak_spectrum_ids = []
    spectra_over_100_clean_peaks = 0

    for row in tqdm(identity_resolved.itertuples(index=False), total=len(identity_resolved), desc="Building spectra"):
        spectrum_id = int(row.spectrum)
        group = fragments_by_spectrum.get(spectrum_id)
        precursor_mz = float(row.precursorMz)
        if group is None:
            mzs = np.array([], dtype=np.float32)
            intensities = np.array([], dtype=np.float32)
            dropped = 0
            before = 0
        else:
            before = int(len(group))
            mzs, intensities, dropped = clean_peaks_for_spectrum(group, precursor_mz)
        after = int(len(mzs))
        peak_before_total += before
        peak_drop_total += dropped
        peak_after_total += after
        if dropped:
            spectra_with_dropped_peaks += 1
        if after == 0:
            zero_peak_spectrum_ids.append(spectrum_id)
            continue
        if after > MAX_MODEL_PEAKS:
            spectra_over_100_clean_peaks += 1
        if not bool(getattr(row, "included_downstream")):
            continue

        wide = np.vstack([mzs, intensities]).astype(np.float32)
        padded_raw = su.pad_peak_list(wide, target_len=MAX_MODEL_PEAKS).astype(np.float32)
        processed = spec_preproc(wide, prec_mz=precursor_mz, high_form=False, augment=False)
        raw_spectra.append(padded_raw)
        processed_spectra.append(processed)
        retained_rows.append(
            {
                "spectrum": spectrum_id,
                "spectrum_id": spectrum_id,
                "compound": clean_text(row.compound),
                "base_compound_name": clean_text(row.base_compound_name),
                "query_compound_name": clean_text(row.query_compound_name),
                "adduct": clean_text(row.adduct),
                "hmdb": clean_text(row.hmdb),
                "precursorMz": precursor_mz,
                "exactmass": float(row.exactmass),
                "true_smiles": clean_text(row.smiles),
                "inchikey": clean_text(row.inchikey),
                "first_block_inchikey": clean_text(row.first_block_inchikey),
                "true_maccs_166": clean_text(row.maccs_166),
                "peak_count_original": before,
                "peak_count_cleaned": after,
                "peaks_dropped_precursor_rule": dropped,
                "cleaned_mz": mzs.tolist(),
                "cleaned_intensity": intensities.tolist(),
                "resolution_source": clean_text(row.resolution_source),
                "mass_delta_da": to_float(row.mass_delta_da),
            }
        )

    records_df = pd.DataFrame(retained_rows)
    if len(records_df):
        raw_spectra_np = np.stack(raw_spectra).astype(np.float32)
        processed_spectra_np = np.stack(processed_spectra).astype(np.float32)
    else:
        raw_spectra_np = np.zeros((0, 2, MAX_MODEL_PEAKS), dtype=np.float32)
        processed_spectra_np = np.zeros((0, MAX_MODEL_PEAKS + 1, 2), dtype=np.float32)

    records_df.to_parquet(PER_SPECTRUM_PARQUET, index=False)
    write_model_ready_hdf5(records_df, raw_spectra_np, processed_spectra_np)

    mac_library = (
        resolution_df[resolution_df["included_downstream"]]
        .sort_values(["hmdb", "base_compound_name"])
        .drop_duplicates("first_block_inchikey")
        .copy()
    )
    closed_library = mac_library.rename(columns={"maccs_166": "maccs_166"})[
        ["first_block_inchikey", "inchikey", "smiles", "hmdb", "base_compound_name", "exactmass", "maccs_166"]
    ].copy()
    closed_library["source_pool"] = "mac"
    write_library(closed_library, CLOSED_LIBRARY_PARQUET, CLOSED_LIBRARY_NPZ)

    ood_library = load_ood_library()
    mac_first_blocks = set(closed_library["first_block_inchikey"])
    ood_without_mac = ood_library[~ood_library["first_block_inchikey"].isin(mac_first_blocks)].copy()
    open_library = pd.concat([closed_library, ood_without_mac], ignore_index=True)
    open_library = open_library.drop_duplicates("first_block_inchikey").reset_index(drop=True)
    write_library(open_library, OPEN_LIBRARY_PARQUET, OPEN_LIBRARY_NPZ)

    train_first_blocks = generate_axis2_train_first_blocks()
    seen_flags = closed_library[["first_block_inchikey", "inchikey", "smiles", "hmdb", "base_compound_name", "exactmass"]].copy()
    seen_flags["seen_in_axis2_train"] = seen_flags["first_block_inchikey"].isin(train_first_blocks)
    seen_flags.to_csv(SEEN_FLAGS_CSV, index=False)

    resolution_counter = Counter(resolution_df["resolution_source"].fillna("").replace("", "unresolved"))
    adduct_counts = Counter(identity_table["adduct"].fillna("").replace("", "unknown"))
    peak_counts = records_df["peak_count_cleaned"].tolist() if len(records_df) else []

    qc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": {
            "identities": str(IDENTITIES_PATH),
            "fragments": str(FRAGMENTS_PATH),
            "axis2_ood": str(OOD_PATH),
            "axis2_train_hdf5": str(AXIS2_TRAIN_HDF5),
        },
        "local_hmdb_structure_files": local_hmdb_files,
        "join": join_summary,
        "adduct_counts": dict(sorted(adduct_counts.items())),
        "resolution": {
            "unique_compounds_attempted": int(len(resolution_df)),
            "resolved_compounds": int(resolution_df["resolved"].sum()),
            "verified_compounds": int(resolution_df["verified_mass"].sum()),
            "included_downstream_compounds": int(resolution_df["included_downstream"].sum()),
            "resolution_rate": float(resolution_df["resolved"].mean()) if len(resolution_df) else 0.0,
            "verification_rate_of_attempted": float(resolution_df["verified_mass"].mean()) if len(resolution_df) else 0.0,
            "verification_rate_of_resolved": float(
                resolution_df.loc[resolution_df["resolved"], "verified_mass"].mean()
            ) if int(resolution_df["resolved"].sum()) else 0.0,
            "source_counts": dict(sorted(resolution_counter.items())),
            "mass_tolerance_da": MASS_TOL_DA,
        },
        "spectra": {
            "identity_spectra": int(len(identity_table)),
            "spectra_retained": int(len(records_df)),
            "spectra_excluded_unresolved_or_unverified": int((~identity_resolved["included_downstream"].fillna(False)).sum()),
            "spectra_with_zero_peaks_after_cleaning": int(len(zero_peak_spectrum_ids)),
            "zero_peak_spectrum_ids": zero_peak_spectrum_ids[:100],
            "spectra_over_100_clean_peaks": int(spectra_over_100_clean_peaks),
            "raw_spectrum_shape": list(raw_spectra_np.shape),
            "processed_spectrum_shape": list(processed_spectra_np.shape),
        },
        "peaks": {
            "fragment_precursor_tolerance_da": FRAGMENT_PRECURSOR_TOL_DA,
            "total_peaks_before_cleaning": int(peak_before_total),
            "total_peaks_dropped_precursor_rule": int(peak_drop_total),
            "total_peaks_after_cleaning": int(peak_after_total),
            "spectra_with_dropped_peaks": int(spectra_with_dropped_peaks),
            "retained_peak_count_min": int(min(peak_counts)) if peak_counts else 0,
            "retained_peak_count_median": float(np.median(peak_counts)) if peak_counts else 0.0,
            "retained_peak_count_mean": float(np.mean(peak_counts)) if peak_counts else 0.0,
            "retained_peak_count_p90": float(np.quantile(peak_counts, 0.9)) if peak_counts else 0.0,
            "retained_peak_count_max": int(max(peak_counts)) if peak_counts else 0,
        },
        "libraries": {
            "closed_pool_size": int(len(closed_library)),
            "axis2_ood_unique_first_blocks": int(len(ood_library)),
            "open_pool_size": int(len(open_library)),
            "ood_mac_first_block_overlaps_removed": int(len(ood_library) - len(ood_without_mac)),
        },
        "seen_in_training": {
            "axis2_train_first_block_count": int(len(train_first_blocks)),
            "mac_seen_count": int(seen_flags["seen_in_axis2_train"].sum()),
            "mac_novel_count": int((~seen_flags["seen_in_axis2_train"]).sum()),
        },
        "outputs": {
            "per_spectrum_records": str(PER_SPECTRUM_PARQUET),
            "model_ready_hdf5": str(MODEL_READY_HDF5),
            "closed_library_parquet": str(CLOSED_LIBRARY_PARQUET),
            "closed_library_npz": str(CLOSED_LIBRARY_NPZ),
            "open_library_parquet": str(OPEN_LIBRARY_PARQUET),
            "open_library_npz": str(OPEN_LIBRARY_NPZ),
            "seen_flags": str(SEEN_FLAGS_CSV),
            "axis2_train_first_blocks": str(TRAIN_FIRST_BLOCK_TXT),
            "resolution_details": str(RESOLUTION_DETAILS_CSV),
            "join_mismatches": str(JOIN_MISMATCH_CSV),
            "qc_json": str(QC_JSON),
            "qc_report": str(QC_REPORT),
        },
    }

    report = build_report(qc, resolution_df)
    QC_JSON.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    QC_REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)


def build_report(qc: dict[str, Any], resolution_df: pd.DataFrame) -> str:
    failed = resolution_df[~resolution_df["included_downstream"]].copy()
    failed = failed.sort_values(["resolved", "verified_mass", "hmdb", "base_compound_name"])
    lines = []
    lines.append("Axis 3 tier-1 MAC DDA preparation QC")
    lines.append(f"Generated: {qc['generated_at']}")
    lines.append(f"Seed: {qc['seed']}")
    lines.append("")
    lines.append("Inputs")
    for key, value in qc["inputs"].items():
        lines.append(f"- {key}: {value}")
    if qc["local_hmdb_structure_files"]:
        lines.append(f"- Local HMDB structure files: {qc['local_hmdb_structure_files']}")
    else:
        lines.append("- Local HMDB structure files: none found, PubChem online resolution used")
    lines.append("")
    lines.append("Join and labels")
    for key, value in qc["join"].items():
        lines.append(f"- {key}: {value}")
    lines.append(f"- Adduct counts: {qc['adduct_counts']}")
    lines.append("")
    lines.append("Resolution")
    res = qc["resolution"]
    lines.append(f"- Unique compounds attempted: {res['unique_compounds_attempted']}")
    lines.append(f"- Resolved compounds: {res['resolved_compounds']}")
    lines.append(f"- Verified compounds within 5 mDa: {res['verified_compounds']}")
    lines.append(f"- Included downstream compounds: {res['included_downstream_compounds']}")
    lines.append(f"- Resolution rate: {res['resolution_rate']:.3f}")
    lines.append(f"- Verification rate of attempted: {res['verification_rate_of_attempted']:.3f}")
    lines.append(f"- Verification rate of resolved: {res['verification_rate_of_resolved']:.3f}")
    lines.append(f"- Resolution source counts: {res['source_counts']}")
    lines.append("")
    lines.append("Spectra and peaks")
    spec = qc["spectra"]
    peaks = qc["peaks"]
    lines.append(f"- Identity spectra: {spec['identity_spectra']}")
    lines.append(f"- Spectra retained: {spec['spectra_retained']}")
    lines.append(f"- Spectra excluded unresolved or unverified: {spec['spectra_excluded_unresolved_or_unverified']}")
    lines.append(f"- Spectra with zero peaks after cleaning: {spec['spectra_with_zero_peaks_after_cleaning']}")
    lines.append(f"- Spectra over 100 clean peaks: {spec['spectra_over_100_clean_peaks']}")
    lines.append(f"- Raw spectrum shape: {spec['raw_spectrum_shape']}")
    lines.append(f"- Processed spectrum shape: {spec['processed_spectrum_shape']}")
    lines.append(f"- Total peaks before cleaning: {peaks['total_peaks_before_cleaning']}")
    lines.append(f"- Total peaks dropped by precursor rule: {peaks['total_peaks_dropped_precursor_rule']}")
    lines.append(f"- Total peaks after cleaning: {peaks['total_peaks_after_cleaning']}")
    lines.append(
        "- Retained peak counts: "
        f"min {peaks['retained_peak_count_min']}, "
        f"median {peaks['retained_peak_count_median']:.1f}, "
        f"mean {peaks['retained_peak_count_mean']:.2f}, "
        f"p90 {peaks['retained_peak_count_p90']:.1f}, "
        f"max {peaks['retained_peak_count_max']}"
    )
    lines.append("")
    lines.append("Libraries")
    libs = qc["libraries"]
    lines.append(f"- Closed pool size: {libs['closed_pool_size']}")
    lines.append(f"- Axis 2 OOD unique first-block InChIKeys: {libs['axis2_ood_unique_first_blocks']}")
    lines.append(f"- Open pool size: {libs['open_pool_size']}")
    lines.append(f"- OOD MAC overlaps removed: {libs['ood_mac_first_block_overlaps_removed']}")
    seen = qc["seen_in_training"]
    lines.append(f"- Axis 2 train first-block InChIKeys: {seen['axis2_train_first_block_count']}")
    lines.append(f"- MAC seen in Axis 2 train: {seen['mac_seen_count']}")
    lines.append(f"- MAC novel to Axis 2 train: {seen['mac_novel_count']}")
    lines.append("")
    lines.append("Failed compounds")
    lines.append(f"- Failed or excluded compounds: {len(failed)}")
    for item in failed.head(40).itertuples(index=False):
        lines.append(
            f"  - {clean_text(item.hmdb) or 'no HMDB'} | {clean_text(item.base_compound_name)} | "
            f"resolved {bool(item.resolved)} | verified {bool(item.verified_mass)} | "
            f"delta {item.mass_delta_da if pd.notna(item.mass_delta_da) else 'NA'}"
        )
    if len(failed) > 40:
        lines.append(f"  - plus {len(failed) - 40} more in {RESOLUTION_DETAILS_CSV}")
    lines.append("")
    lines.append("Outputs")
    for key, value in qc["outputs"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
