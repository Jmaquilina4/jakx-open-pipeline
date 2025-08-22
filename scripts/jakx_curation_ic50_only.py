#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================
# JAKx Data Curation — IC50-only → pIC50 (professional mode)
# - Keeps ONLY IC50, converts to pIC50
# - Canonical SMILES (salt-strip; RDKit if available)
# - Unit normalization, dedup by median per compound/target
# - Selectivity matrix + panel coverage
# - JAK1 QSAR export (+ optional drug-like filtering)
# - Verbose diagnostics at each step
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------- CONFIG (defaults) ----------------------
TARGETS = ["JAK1", "JAK2", "JAK3", "TYK2"]

# Assay harmonization (optional gates)
KEEP_BIOCHEM_ONLY = False       # flip True if your assay_type codes are reliable
MIN_CONFIDENCE_SCORE = None     # e.g., 8 if confidence_score exists/reliable
ALLOWED_ASSAY_TYPES = {"A", "B", "E", "BINDING", "BIOCHEMICAL"}  # expand as needed

# IC50-only settings
ACCEPT_TYPES = {"ic50"}         # <- IC50 ONLY
REL_OK = {"=", "~", "≈", "~=", "==", ""}  # accept blank/loose exact markers

# Drug-like filtering
STRICT_DRUGLIKE_WINDOW = True   # if False, only alerts+validity
APPLY_DRUGLIKE_FILTERS = True   # master toggle


# ---------------------- RDKit (optional) ----------------------
RDKit_OK = False
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski, Crippen
    RDLogger.DisableLog("rdApp.*")
    RDKit_OK = True
except Exception:
    RDKit_OK = False


# ---------------------- UTILITIES -----------------------------
NAN_TOKENS = {"", "nan", "none", "null", "na", "<na>", "n/a"}

def _is_nan_token(s) -> bool:
    try:
        return str(s).strip().lower() in NAN_TOKENS
    except Exception:
        return True

def smart_read(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    # try python engine (auto sep)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    # fallback seps
    for sep in [",", "\t", ";", "|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="c", low_memory=False)
        except Exception:
            continue
    # last resort
    return pd.read_csv(path)

def _as_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def _largest_organic_fragment(smiles):
    if smiles is None or _is_nan_token(smiles) or not isinstance(smiles, str):
        return np.nan
    parts = [p for p in smiles.split(".") if p and not _is_nan_token(p)]
    if not parts:
        return np.nan
    if not RDKit_OK:
        return max(parts, key=len)
    mols = []
    for p in parts:
        try:
            m = Chem.MolFromSmiles(p)
            if m is not None:
                mols.append(m)
        except Exception:
            continue
    if not mols:
        return np.nan
    # choose the largest by heavy atoms
    mols.sort(key=lambda m: (rdMolDescriptors.CalcNumHeavyAtoms(m), m.GetNumAtoms()), reverse=True)
    try:
        return Chem.MolToSmiles(mols[0], canonical=True)
    except Exception:
        return np.nan

def _canon_smiles(smiles):
    if smiles is None or _is_nan_token(smiles) or not isinstance(smiles, str):
        return np.nan
    if not RDKit_OK:
        return smiles
    try:
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m, canonical=True) if m is not None else np.nan
    except Exception:
        return np.nan

def _prepare_smiles(raw_smiles):
    core = _largest_organic_fragment(raw_smiles)
    return _canon_smiles(core)

def _unit_factor(units: pd.Series) -> pd.Series:
    u = _as_series(units).astype(str).str.strip().str.lower()
    factor = pd.Series(np.nan, index=u.index, dtype="float64")
    factor[u.isin({"nm", "nanomolar"})] = 1.0
    factor[u.isin({"um", "µm", "micromolar"})] = 1_000.0
    factor[u.isin({"mm", "millimolar"})] = 1_000_000.0
    factor[u.isin({"m", "molar"})] = 1_000_000_000.0
    return factor

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    aliases = {
        "smiles": "canonical_smiles",
        "canonical_smiles": "canonical_smiles",
        "molecule_chembl_id": "molecule_chembl_id",
        "molecule_chembl_id_x": "molecule_chembl_id",
        "molecule_chembl_id_y": "molecule_chembl_id",
        "standard_type": "standard_type",
        "type": "standard_type",
        "standard_value": "standard_value",
        "value": "standard_value",
        "standard_units": "standard_units",
        "units": "standard_units",
        "relation": "relation",
        "standard_relation": "relation",
        "pchembl_value": "pchembl_value",
        "assay_type": "assay_type",
        "target_chembl_id": "target_chembl_id",
        "confidence_score": "confidence_score",
        "activity_comment": "activity_comment",
        "data_validity_comment": "data_validity_comment",
        "target": "target",
    }
    present = {k: v for k, v in aliases.items() if k in df.columns}
    df = df.rename(columns=present)
    df = df.loc[:, ~df.columns.duplicated()]

    needed = [
        "canonical_smiles", "molecule_chembl_id", "standard_type", "standard_value",
        "standard_units", "relation", "pchembl_value", "assay_type", "target_chembl_id",
        "confidence_score", "activity_comment", "data_validity_comment", "target"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ---------------------- CURATION ------------------------------
def curate_ic50_table(df: pd.DataFrame, tgt_name: str, *, verbose: bool=True,
                      keep_biochem_only: bool=False,
                      min_confidence_score: int|None=None) -> pd.DataFrame:
    df = _std_cols(df)
    raw_rows = len(df)

    # Canonical SMILES (salt-strip + canonicalize)
    df["canonical_smiles"] = df["canonical_smiles"].apply(_prepare_smiles)
    df = df[~df["canonical_smiles"].isna()]
    after_smiles = len(df)

    # Assay harmonization (optional)
    if keep_biochem_only and "assay_type" in df.columns:
        at = _as_series(df["assay_type"]).astype(str).str.upper().str.strip()
        df = df[at.isin(ALLOWED_ASSAY_TYPES)]
    if (min_confidence_score is not None) and ("confidence_score" in df.columns):
        cs = pd.to_numeric(_as_series(df["confidence_score"]), errors="coerce").fillna(0)
        df = df[cs >= min_confidence_score]

    # Prepare fields
    df["pchembl_value"] = pd.to_numeric(_as_series(df["pchembl_value"]), errors="coerce")
    stype = _as_series(df["standard_type"]).astype(str).str.lower().str.strip()
    rel   = _as_series(df["relation"]).astype(str).str.strip()

    std_val = pd.to_numeric(_as_series(df["standard_value"]), errors="coerce")
    factor  = _unit_factor(df["standard_units"])
    value_nM = std_val * factor
    value_nM[~np.isfinite(value_nM)] = np.nan
    df["value_nM"] = value_nM

    # Debug snapshot BEFORE filtering
    n_total = len(df)
    n_has_pchembl = df["pchembl_value"].notna().sum()
    n_ic50_rows = (stype.isin(ACCEPT_TYPES)).sum()
    n_ic50_convertible = (stype.isin(ACCEPT_TYPES) & df["value_nM"].notna()).sum()
    n_rel_ok = (rel.isna() | rel.isin(REL_OK)).sum()
    print(f"      · rows: {n_total:,} | pChEMBL: {n_has_pchembl:,} | IC50 rows: {n_ic50_rows:,} | IC50 convertible: {n_ic50_convertible:,} | relation ok: {n_rel_ok:,}")

    # Keep only IC50 data: either pChEMBL (IC50-compatible) or convertible IC50 with exact-ish relation
    mask = df["pchembl_value"].notna() | (
        stype.isin(ACCEPT_TYPES) & df["value_nM"].notna() & (rel.isna() | rel.isin(REL_OK))
    )
    df = df[mask]
    after_activity = len(df)

    # Compute pIC50
    df["pIC50"] = df["pchembl_value"]
    need = df["pIC50"].isna() & df["value_nM"].notna()
    nm = pd.to_numeric(df.loc[need, "value_nM"], errors="coerce")
    nm = nm[(nm > 0) & np.isfinite(nm)]
    if not nm.empty:
        df.loc[nm.index, "pIC50"] = 9.0 - np.log10(nm.to_numpy(dtype=float))
    # sanity range
    df = df[df["pIC50"].between(3, 11)]

    # Collapse duplicates (median potency, count of assays)
    agg = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], dropna=False)
          .agg(pIC50=("pIC50","median"), assay_count=("pIC50","size"))
          .reset_index()
    )
    agg.insert(0, "target", tgt_name)
    agg["source"] = "ChEMBL"

    if verbose:
        print(f"   [{tgt_name} IC50] rows: raw {raw_rows:,} → smiles {after_smiles:,} → activity {after_activity:,} → curated {len(agg):,}")
    return agg[["target","molecule_chembl_id","canonical_smiles","pIC50","assay_count","source"]]


# ---------------------- PAINS/BRENK (optional) ----------------
def init_alert_catalogs():
    if not RDKit_OK:
        return None, None
    try:
        from rdkit.Chem import FilterCatalog as FC

        def build_catalog_pains():
            params = FC.FilterCatalogParams()
            # try whole PAINS; else A/B/C subsets
            try:
                params.AddCatalog(FC.FilterCatalogParams.FilterCatalogs.PAINS)
            except Exception:
                for sub in (FC.FilterCatalogParams.FilterCatalogs.PAINS_A,
                            FC.FilterCatalogParams.FilterCatalogs.PAINS_B,
                            FC.FilterCatalogParams.FilterCatalogs.PAINS_C):
                    try: params.AddCatalog(sub)
                    except Exception: pass
            return FC.FilterCatalog(params)

        def build_catalog_brenk():
            params = FC.FilterCatalogParams()
            params.AddCatalog(FC.FilterCatalogParams.FilterCatalogs.BRENK)
            return FC.FilterCatalog(params)

        return build_catalog_pains(), build_catalog_brenk()
    except Exception:
        return None, None


def featurize_props_and_alerts(smiles_list, cat_pains, cat_brenk) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi) if RDKit_OK and isinstance(smi, str) else None
        if m is None:
            rows.append(dict(canonical_smiles=smi, valid=False, pains=False, brenk=False,
                             mw=np.nan, clogp=np.nan, hbd=np.nan, hba=np.nan,
                             tpsa=np.nan, rotb=np.nan))
            continue
        pains_hit = False; brenk_hit = False
        if cat_pains is not None:
            try:
                pains_hit = (cat_pains.GetFirstMatch(m) is not None)
            except Exception:
                pass
        if cat_brenk is not None:
            try:
                brenk_hit = (cat_brenk.GetFirstMatch(m) is not None)
            except Exception:
                pass
        try:
            mw    = Descriptors.MolWt(m)
            clogp = Crippen.MolLogP(m)
            hbd   = Lipinski.NumHDonors(m)
            hba   = Lipinski.NumHAcceptors(m)
            tpsa  = rdMolDescriptors.CalcTPSA(m)
            rotb  = Lipinski.NumRotatableBonds(m)
        except Exception:
            rows.append(dict(canonical_smiles=smi, valid=False, pains=pains_hit, brenk=brenk_hit,
                             mw=np.nan, clogp=np.nan, hbd=np.nan, hba=np.nan,
                             tpsa=np.nan, rotb=np.nan))
            continue
        rows.append(dict(canonical_smiles=smi, valid=True, pains=pains_hit, brenk=brenk_hit,
                         mw=mw, clogp=clogp, hbd=hbd, hba=hba, tpsa=tpsa, rotb=rotb))
    return pd.DataFrame(rows)


# ---------------------- MAIN PIPELINE -------------------------
def run_pipeline(raw_dir: Path, out_dir: Path,
                 keep_biochem_only: bool=False,
                 min_conf: int|None=None,
                 strict_druglike: bool=True):

    out_bio = out_dir / "bioassay"
    out_bio.mkdir(parents=True, exist_ok=True)

    bio_files = {t: raw_dir / f"{t}_raw.csv" for t in TARGETS}

    # Diagnostics
    print("=== Path diagnostics ===")
    print("RAW_DIR:", raw_dir)
    print("OUT_DIR:", out_dir, "\n")

    print("Bioactivity files:")
    for t, p in bio_files.items():
        print(f"  {t}: {p if p.exists() else '❌ NOT FOUND'}")

    # RDKit status
    if RDKit_OK:
        print("\nRDKit: ✅ available (canonicalization/filters enabled)")
    else:
        print("\nRDKit: ⚠️ NOT available (will skip canonicalization & filters)")

    # Alerts
    cat_pains, cat_brenk = (None, None)
    if APPLY_DRUGLIKE_FILTERS and RDKit_OK:
        cat_pains, cat_brenk = init_alert_catalogs()
        if cat_pains is not None or cat_brenk is not None:
            print("Alerts: ✅ PAINS/Brenk catalogs initialized")
        else:
            print("Alerts: ⚠️ could not initialize PAINS/Brenk catalogs")
    else:
        print("Alerts: (skipped)")

    # ------------------ Curate each isoform -------------------
    print("\n=== Curating IC50 → pIC50 (IC50-only) ===")
    bio_frames = []
    for t in TARGETS:
        p = bio_files[t]
        if not p.exists():
            print(f"   ❌ Missing {t} file: {p}")
            continue
        raw = smart_read(p)
        curated = curate_ic50_table(
            raw, t, verbose=True,
            keep_biochem_only=keep_biochem_only,
            min_confidence_score=min_conf
        )
        bio_frames.append(curated)
        (out_bio / f"{t}_bio_curated_ic50.csv").parent.mkdir(parents=True, exist_ok=True)
        curated.to_csv(out_bio / f"{t}_bio_curated_ic50.csv", index=False)

    # Merge curated
    if bio_frames:
        all_bio = pd.concat(bio_frames, ignore_index=True)
    else:
        all_bio = pd.DataFrame(columns=["target","molecule_chembl_id","canonical_smiles","pIC50","assay_count","source"])

    all_bio_path = out_bio / "ALLJAK_bio_curated_ic50.csv"
    all_bio.to_csv(all_bio_path, index=False)
    print(f"\n✅ Wrote merged curated IC50 bio → {all_bio_path}  shape: {all_bio.shape}")

    # ------------------ Selectivity & Coverage ----------------
    print("\n=== Selectivity Matrix & Panel Coverage (IC50-only) ===")
    jak1_train = pd.DataFrame()
    if not all_bio.empty:
        pivot = (
            all_bio.groupby(["canonical_smiles","target"])["pIC50"]
                   .median()
                   .unstack("target")
        )
        for other in ["JAK2","JAK3","TYK2"]:
            if "JAK1" in pivot.columns and other in pivot.columns:
                pivot[f"Δ(JAK1-{other})"] = pivot["JAK1"] - pivot[other]
        delta_cols = [c for c in pivot.columns if c.startswith("Δ(JAK1-")]
        if delta_cols:
            pivot["Δmin_vs_other"] = pivot[delta_cols].min(axis=1, skipna=True)

        sel_out = out_bio / "ALLJAK_bio_selectivity_ic50.csv"
        pivot.to_csv(sel_out)
        print(f"📝 Wrote selectivity matrix → {sel_out}")

        # Panel coverage
        four = ["JAK1","JAK2","JAK3","TYK2"]
        present = pivot.reindex(columns=[c for c in four if c in pivot.columns]).notna().astype(int)
        coverage = present.sum(axis=1)
        counts = coverage.value_counts().reindex([1,2,3,4], fill_value=0).sort_index()
        coverage_df = counts.rename_axis("targets_covered").reset_index(name="n_compounds")
        coverage_out = out_bio / "ALLJAK_bio_panel_coverage_ic50.csv"
        coverage_df.to_csv(coverage_out, index=False)
        n_total = pivot.shape[0]
        n_full  = int(counts.get(4, 0))
        frac_full = (n_full/n_total) if n_total else 0.0
        print(f"📊 Panel coverage: unique compounds {n_total:,}, full panel {n_full:,} ({frac_full:.2%})")
        print(f"🧮 Wrote coverage counts → {coverage_out}")

        # JAK1 QSAR base set
        jak1_train = (
            all_bio[all_bio["target"]=="JAK1"]
            .groupby("canonical_smiles", as_index=False)["pIC50"].median()
        )
        jak1_path = out_bio / "JAK1_training_ic50.csv"
        jak1_train.to_csv(jak1_path, index=False)
        print(f"🏁 Wrote JAK1 QSAR base set (IC50-only) → {jak1_path}  rows: {len(jak1_train):,}")
    else:
        print("⚠️ No curated rows; skipping selectivity & JAK1 export.")

    # ------------------ Drug-like filtering (opt) --------------
    if APPLY_DRUGLIKE_FILTERS:
        print("\n=== Drug-like filtering for JAK1 QSAR (IC50-only) ===")
        if RDKit_OK and not jak1_train.empty:
            cat_pains2, cat_brenk2 = (None, None)
            try:
                cat_pains2, cat_brenk2 = init_alert_catalogs()
            except Exception:
                pass

            props = featurize_props_and_alerts(
                jak1_train["canonical_smiles"].drop_duplicates().tolist(),
                cat_pains2, cat_brenk2
            )
            dfj = jak1_train.merge(props, on="canonical_smiles", how="left")

            if strict_druglike:
                mask = (
                    dfj["valid"].fillna(False)
                    & ~dfj["pains"].fillna(False)
                    & ~dfj["brenk"].fillna(False)
                    & dfj["mw"].between(150, 650)
                    & dfj["clogp"].between(-2, 6)
                    & (dfj["hbd"] <= 5) & (dfj["hba"] <= 10)
                    & (dfj["tpsa"] <= 140) & (dfj["rotb"] <= 10)
                )
                out_name = "JAK1_training_ic50_druglike.csv"
            else:
                mask = dfj["valid"].fillna(False) & ~dfj["pains"].fillna(False) & ~dfj["brenk"].fillna(False)
                out_name = "JAK1_training_ic50_broad.csv"

            filtered = dfj[mask].copy()
            out_csv = out_bio / out_name
            filtered[["canonical_smiles","pIC50"]].to_csv(out_csv, index=False)

            print(f"JAK1 QSAR set: {len(dfj):,} → {len(filtered):,} (strict={strict_druglike})")
            print(f"Removed — invalid: {(~dfj['valid'].fillna(False)).sum():,} | PAINS: {dfj['pains'].fillna(False).sum():,} | Brenk: {dfj['brenk'].fillna(False).sum():,}")
            print(f"🧪 Wrote filtered JAK1 QSAR set → {out_csv}")
        else:
            print("⚠️ Skipping drug-like filtering (RDKit missing or JAK1 base empty).")
    else:
        print("Drug-like filtering: (disabled)")

    # ------------------ Pipeline Summary -----------------------
    def safe_len(p: Path) -> int:
        try:
            return len(smart_read(p))
        except Exception:
            return 0

    start_bio_all = 0
    for t in TARGETS:
        p = raw_dir / f"{t}_raw.csv"
        if p.exists():
            try:
                start_bio_all += len(smart_read(p))
            except Exception:
                pass

    end_bio_all = len(all_bio) if 'all_bio' in locals() else 0
    pipe_summary = pd.DataFrame([{
        "bio_start": start_bio_all,
        "bio_end_ic50_only": end_bio_all,
        "bio_reduction_pct": (1 - end_bio_all / start_bio_all) * 100 if start_bio_all else 0.0,
    }])

    pipe_summary_path = out_dir / "ALLJAK_pipeline_summary_ic50.csv"
    pipe_summary.to_csv(pipe_summary_path, index=False)

    print("\n=== Final Summary (IC50-only) ===")
    print("📊 Pipeline summary:", pipe_summary.to_dict(orient="records")[0])
    print("Artifacts:")
    print("  -", all_bio_path)
    print("  -", out_bio / "ALLJAK_bio_selectivity_ic50.csv")
    print("  -", out_bio / "ALLJAK_bio_panel_coverage_ic50.csv")
    print("  -", out_bio / "JAK1_training_ic50.csv")
    if APPLY_DRUGLIKE_FILTERS and RDKit_OK:
        print("  -", out_bio / ("JAK1_training_ic50_druglike.csv" if strict_druglike else "JAK1_training_ic50_broad.csv"))
    print("  -", pipe_summary_path)
    print("🎉 Done.")


# ---------------------- CLI ENTRYPOINT ------------------------
def main():
    parser = argparse.ArgumentParser(description="JAKx IC50-only curation → pIC50")
    parser.add_argument("--raw-dir", default="data/raw", type=str,
                        help="Directory containing JAK*_raw.csv files (default: data/raw)")
    parser.add_argument("--out-dir", default="data/processed/jakx_ic50", type=str,
                        help="Output directory for curated artifacts (default: data/processed/jakx_ic50)")
    parser.add_argument("--keep-biochem-only", action="store_true",
                        help="Keep only biochemical assays if assay_type is available")
    parser.add_argument("--min-confidence", type=int, default=None,
                        help="Minimum confidence_score to keep (if column exists)")
    parser.add_argument("--strict-druglike", action="store_true",
                        help="Apply strict physicochemical window (MW/logP/HBD/HBA/TPSA/RotB) in addition to PAINS/Brenk")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    global KEEP_BIOCHEM_ONLY, MIN_CONFIDENCE_SCORE, STRICT_DRUGLIKE_WINDOW
    KEEP_BIOCHEM_ONLY = bool(args.keep_biochem_only)
    MIN_CONFIDENCE_SCORE = args.min_confidence
    STRICT_DRUGLIKE_WINDOW = bool(args.strict_druglike)

    print(f"\n  === Path diagnostics ===")
    print(f"RAW_DIR: {raw_dir}")
    print(f"OUT_DIR: {out_dir} \n")

    run_pipeline(
        raw_dir=raw_dir,
        out_dir=out_dir,
        keep_biochem_only=KEEP_BIOCHEM_ONLY,
        min_conf=MIN_CONFIDENCE_SCORE,
        strict_druglike=STRICT_DRUGLIKE_WINDOW
    )


if __name__ == "__main__":
    main()

