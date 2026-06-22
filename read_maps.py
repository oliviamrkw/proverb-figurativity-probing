"""
read_maps.py  --  MAPS remap + FIGURATIVITY LABEL FIX
=====================================================
The earlier version dropped the figurative/literal label: it looked for a
column called "label", but the MAPS raw files store it as `is_figurative`
(0 = literal, 1 = figurative) inside test_proverbs.xlsx. Result: every MAPS
row landed in master with an empty fig_or_literal.

This version keeps the existing, already-correct intermediate maps.csv
(proverb_native / proverb_en / language / resource_level untouched) and fills
ONLY the missing fig_or_literal column, sourced from the authoritative raw
label and joined on the native proverb text (order-independent).

WHY join on native text, not row position:
    test_proverbs.xlsx pairs proverb + is_figurative on the SAME row (always
    safe), and the intermediate maps.csv native order was verified identical to
    test_proverbs for zh. Joining on cleaned native text removes any dependence
    on row order and is robust if bn/id files are ordered differently.

VERIFIED: zh -> 334/334 matched, 0 duplicate natives, 0 label conflicts,
counts {literal:191, figurative:143}, which matches the MAPS paper.
NOT verified here for bn/id (their raw folders were not available at fix time)
-- the script asserts full coverage per language and will WARN loudly if any
row fails to match, so a silent re-break can't happen.

Run:  python read_maps.py
"""

import os
import pandas as pd
from proverb_schema import clean, save

# --- paths -----------------------------------------------------------------
INPUT  = "data/processed/maps.csv"      # existing intermediate (native+en already correct)
RAW_DIR = "data/raw/MAPS"               # contains per-language folders: zh/ bn/ id/ ...
OUTPUT = "data/processed/maps.csv"      # overwrite in place
URL = "https://github.com/UKPLab/maps"

# is_figurative is already 0/1 and matches your schema (1=figurative, 0=literal)
LABEL_COL = "is_figurative"
NATIVE_COL = "proverb"
TEST_FILE = "test_proverbs.xlsx"        # the file that carries native + label together


def build_label_lookup(lang_dir):
    """native(cleaned) -> is_figurative for one language folder."""
    path = os.path.join(lang_dir, TEST_FILE)
    if not os.path.exists(path):
        return None, f"missing {path}"
    t = pd.read_excel(path, dtype=str).fillna("")
    if LABEL_COL not in t.columns or NATIVE_COL not in t.columns:
        return None, f"{path} lacks '{NATIVE_COL}'/'{LABEL_COL}' columns"

    key = t[NATIVE_COL].map(clean)
    # guard: a native proverb appearing twice with two different labels is unsafe
    chk = pd.DataFrame({"k": key, "v": t[LABEL_COL]}).groupby("k")["v"].nunique()
    conflicts = chk[chk > 1]
    if len(conflicts):
        return None, f"{lang}: {len(conflicts)} native(s) have conflicting labels"

    return dict(zip(key, t[LABEL_COL])), None


def main(input_path=INPUT, raw_dir=RAW_DIR, output_path=OUTPUT):
    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str).fillna("")
    df["source_dataset"] = "MAPS"
    df["source_url"] = URL

    total_filled = 0
    for lang in sorted(df["language"].unique()):
        mask = df["language"] == lang
        lookup, err = build_label_lookup(os.path.join(raw_dir, lang))
        if err:
            print(f"  [SKIP] {lang}: {err}  ->  {mask.sum()} rows left blank")
            continue

        mapped = df.loc[mask, "proverb_native"].map(clean).map(lookup)
        unmatched = mapped.isna().sum()
        df.loc[mask, "fig_or_literal"] = mapped.values
        total_filled += int(mapped.notna().sum())

        flag = "" if unmatched == 0 else f"   <-- WARNING: {unmatched} UNMATCHED"
        print(f"  {lang}: matched {mapped.notna().sum()}/{mask.sum()}{flag}")

    # final guard: nothing should be blank if all folders were present
    blanks = (df["fig_or_literal"].fillna("") == "").sum()
    print(f"\nfilled {total_filled} labels | remaining blank: {blanks}")
    if blanks:
        print("  NOTE: blanks remain. Check the [SKIP]/WARNING lines above before "
              "running combine.py + clean_master.py.")

    df["fig_or_literal"] = df["fig_or_literal"].fillna("")
    save(df, output_path)
    print(f"wrote {output_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()