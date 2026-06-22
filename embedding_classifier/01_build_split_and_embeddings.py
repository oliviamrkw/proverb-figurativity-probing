"""
01_build_split_and_embeddings.py
--------------------------------
Builds the train/val/test split on the MAPS rows and the shared LaBSE embeddings. Run this ONCE.

Outputs (consumed by Method A, the UMAP diagnostic, and Method C):
    train_indices.npy
    val_indices.npy
    test_indices.npy
    maps_embeddings.npy   (one row per MAPS proverb, positional order)

IMPORTANT: every downstream script must rebuild `maps` the SAME way
(filter source_dataset == "MAPS", then reset_index(drop=True)) so the saved
indices line up. A helper for that is at the bottom of this file.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 42
MASTER = "data/processed/master.csv"

# ---------------------------------------------------------------------------
# 1. Load MAPS
# ---------------------------------------------------------------------------
df = pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str).fillna("")
maps = df[df.source_dataset == "MAPS"].copy().reset_index(drop=True)

maps["label"] = maps["fig_or_literal"].astype(int)
maps["strat_key"] = maps["language"] + "_" + maps["label"].astype(str)

print(f"MAPS rows: {len(maps)}")
print("Stratification key counts:")
print(maps["strat_key"].value_counts(), "\n")

# ---------------------------------------------------------------------------
# 2. Split: 64% train / 16% val / 20% test
#    Stratified jointly by language AND label. Test set is locked here.
# ---------------------------------------------------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
trainval_local, test_local = next(sss.split(maps, maps["strat_key"]))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tv = maps.iloc[trainval_local]
train_in_tv, val_in_tv = next(sss2.split(tv, tv["strat_key"]))

# Remap both halves back to maps-level positions.
train_idx = trainval_local[train_in_tv]
val_idx   = trainval_local[val_in_tv]
test_idx  = test_local

# No overlap between any split.
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx)   & set(test_idx)) == 0

print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

np.save("train_indices.npy", train_idx)
np.save("val_indices.npy",   val_idx)
np.save("test_indices.npy",  test_idx)
print("Saved train/val/test indices.\n")

# ---------------------------------------------------------------------------
# 3. Embed proverb_native with LaBSE (shared by Method A + UMAP)
#    Native text is used.
# ---------------------------------------------------------------------------
model = SentenceTransformer("sentence-transformers/LaBSE")
emb = model.encode(maps["proverb_native"].tolist(), show_progress_bar=True)
np.save("maps_embeddings.npy", emb)
print(f"embeddings: {emb.shape}  ->  maps_embeddings.npy")

# ---------------------------------------------------------------------------
# Helper for downstream scripts — keep maps reconstruction identical.
# ---------------------------------------------------------------------------
def load_maps(master_path=MASTER):
    d = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str).fillna("")
    m = d[d.source_dataset == "MAPS"].copy().reset_index(drop=True)
    m["label"] = m["fig_or_literal"].astype(int)
    return m