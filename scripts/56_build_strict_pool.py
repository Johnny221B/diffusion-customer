"""Build a strict single-token + shoe-relevant word pool.

Strategy:
  - Step 1: keep only single-token words from the existing 172-seed list (~129)
  - Step 2: extend with single-token nouns under shoe-relevant WordNet hypernyms
            (color, material, shape, texture, style) and a curated adjective list
  - Step 3: T5-tokenize each candidate and assert it is exactly one token
  - Step 4: dump to CSV for inspection (no GPU)

Usage:
  python scripts/56_build_strict_pool.py \\
      --tokenizer_path models/stabilityai/stable-diffusion-3.5-large/tokenizer_3 \\
      --existing_csv outputs/scaling_red_vs_green_d128_0416_1723/raw_data.csv \\
      --target_n 400
"""

import os
import re
import argparse
import pandas as pd
from transformers import T5Tokenizer
from nltk.corpus import wordnet as wn


# Curated style / texture / shape adjectives that visually affect a shoe.
# All must be single-token under T5 (verified at runtime).
CURATED_ADJ = [
    # texture / finish
    "glossy", "matte", "shiny", "smooth", "rough", "soft", "hard", "fluffy",
    "polished", "worn", "rugged", "sleek", "fuzzy", "silky", "stiff",
    # style / era
    "vintage", "modern", "classic", "retro", "futuristic", "elegant", "casual",
    "formal", "athletic", "sporty", "dressy", "rustic", "minimal", "bohemian",
    "punk", "gothic", "luxury", "cheap",
    # shape / form
    "tall", "short", "slim", "chunky", "wide", "narrow", "low", "high",
    "round", "pointed", "flat", "thick", "thin",
    # condition / age
    "new", "old", "clean", "dirty", "muddy", "fresh", "fancy",
    # pattern feel
    "plain", "striped", "dotted", "checkered", "patterned",
    # chromatic / brightness
    "bright", "dark", "light", "pale", "deep", "vivid", "dull", "neon", "pastel",
]

# Manual blacklist: tokens that survived hypernym filtering but are abstract /
# linguistic / non-visual and would not produce meaningful shoe variation.
ABSTRACT_BLACKLIST = {
    "acronym", "plural", "singular", "theme", "deal", "base", "end",
    "point", "separate", "radical", "root", "stem", "remainder",
    "quarter", "kid", "log", "mac", "network", "indicator", "primer",
    "royal", "screening", "shift", "surtout", "tag", "train", "upper",
    "wash", "washing", "web", "contour", "darkness", "knot", "nap",
    "rep", "form", "shape", "pattern", "fabric", "textile", "material",
    "color", "colour", "coloring", "footwear", "shoe", "garment",
    "texture", "tone", "tint", "shade", "wrap", "slip", "stays",
    "ground", "rinse", "flash", "cope", "counter", "crush", "panel",
    "patch", "remainder", "remnant", "complexion", "diaper", "fin",
    "duck", "stock", "spectator",
}


# Hypernym roots used to harvest shoe-relevant single-token nouns.
# Deliberately exclude `material.n.01` -- too broad (admits "ammunition", "asbestos").
SHOE_RELEVANT_HYPERNYMS = [
    "color.n.01", "colour.n.01", "chromatic_color.n.01",
    "fabric.n.01", "cloth.n.01", "leather.n.01", "wood.n.01",
    "shape.n.01", "form.n.01",
    "texture.n.01", "pattern.n.01",
    "footwear.n.01", "shoe.n.01", "boot.n.01",
    "garment.n.01",
]


def descendants(synset_name):
    try:
        s = wn.synset(synset_name)
    except Exception:
        return set()
    out, stack = set(), [wn.synset(synset_name)]
    while stack:
        cur = stack.pop()
        out.add(cur)
        stack.extend(cur.hyponyms())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,
                        default="models/stabilityai/stable-diffusion-3.5-large/tokenizer_3")
    parser.add_argument("--existing_csv", type=str,
                        default="outputs/scaling_red_vs_green_d128_0416_1723/raw_data.csv")
    parser.add_argument("--target_n", type=int, default=400)
    parser.add_argument("--out", type=str, default="outputs/strict_pool_candidates.csv")
    args = parser.parse_args()

    tok = T5Tokenizer.from_pretrained(args.tokenizer_path)
    word_re = re.compile(r"^[a-z]+$")

    def is_single_token(w):
        return len(tok.encode(w, add_special_tokens=False)) == 1

    # === Step 1: filter existing seeds ===
    df_existing = pd.read_csv(args.existing_csv)
    existing_words = df_existing["word"].astype(str).tolist()
    existing_single = [w for w in existing_words if is_single_token(w)]
    print(f"Existing seeds: {len(existing_words)} -> single-token: {len(existing_single)}")

    seen = set(existing_single)
    rows = [(w, "seed_kept") for w in existing_single]

    # === Step 2: harvest single-token nouns under shoe-relevant hypernyms ===
    visual_nouns = set()
    for rn in SHOE_RELEVANT_HYPERNYMS:
        for s in descendants(rn):
            for l in s.lemmas():
                w = l.name()
                if word_re.match(w) and 3 <= len(w) <= 10 and is_single_token(w):
                    visual_nouns.add(w)
    visual_nouns -= seen
    visual_nouns -= ABSTRACT_BLACKLIST
    print(f"WordNet visual single-token nouns (new, post-blacklist): {len(visual_nouns)}")
    for w in sorted(visual_nouns):
        rows.append((w, "wn_visual_noun"))
        seen.add(w)

    # === Step 3: curated adjectives (filtered to single-token) ===
    cur = [w for w in CURATED_ADJ if is_single_token(w) and w not in seen]
    print(f"Curated adjectives that are single-token AND new: {len(cur)} / {len(CURATED_ADJ)}")
    for w in cur:
        rows.append((w, "curated_adj"))
        seen.add(w)

    # === Step 4: also include common-color extras (likely already in WN) ===
    print(f"\nTotal candidates: {len(rows)} (target {args.target_n})")
    if len(rows) > args.target_n:
        # Keep all seeds + sample down on visual_noun + curated to fit target
        kept_seeds = [r for r in rows if r[1] == "seed_kept"]
        rest = [r for r in rows if r[1] != "seed_kept"]
        # priority: curated_adj first, then visual_noun
        rest_sorted = sorted(rest, key=lambda r: (r[1] != "curated_adj", r[0]))
        rows = kept_seeds + rest_sorted[: args.target_n - len(kept_seeds)]

    df = pd.DataFrame(rows, columns=["word", "source"])
    print("\nFinal pool source breakdown:")
    print(df["source"].value_counts())
    print(f"\nLength distribution:")
    print(df["word"].str.len().value_counts().sort_index())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} candidates -> {args.out}")
    print(f"Sample: {df['word'].head(15).tolist()}")
    print(f"Curated samples: {df[df['source']=='curated_adj']['word'].head(20).tolist()}")
    print(f"Visual noun samples: {df[df['source']=='wn_visual_noun']['word'].head(20).tolist()}")


if __name__ == "__main__":
    main()
