"""
Needle-in-a-Haystack Prompt Generator for TurboQuant Phase 3 Benchmark.

Builds prompts of a precise target token count by:
  1. Repeating domain-relevant haystack paragraphs
  2. Embedding a unique "needle" fact at the midpoint
  3. Appending the retrieval question at the end
  4. Verifying token count against the Qwen tokenizer

The needle fact is chosen to be unambiguous and not inferable from context.
"""

from __future__ import annotations

# ── Haystack content — blueberry/genomics domain (keeps the experiment realistic) ──
_HAYSTACK_PARAGRAPHS = [
    "The blueberry plant (Vaccinium corymbosum) is a perennial flowering shrub native to North America, widely cultivated for its nutritious berries. Blueberries are among the most commercially important small fruits, with production concentrated in Michigan, Oregon, Washington, and New Jersey. Modern cultivar development focuses on traits including fruit size, firmness, flavor, and resistance to pathogens such as mummy berry disease caused by Monilinia vaccinii-corymbosi.",

    "Anthocyanins are the primary pigments responsible for the characteristic blue-purple color of ripe blueberries. These polyphenolic compounds belong to the flavonoid class and are synthesized through the phenylpropanoid pathway. Key enzymes in anthocyanin biosynthesis include phenylalanine ammonia-lyase (PAL), chalcone synthase (CHS), and anthocyanidin synthase (ANS). Regulation of these enzymes is controlled by MYB transcription factors that interact with bHLH and WD40 co-regulators.",

    "Genomic studies of highbush blueberry (Vaccinium corymbosum) have revealed complex polyploid genetics, with most commercial varieties being tetraploid (2n=4x=48). Next-generation sequencing has enabled the development of high-density SNP arrays for genomic selection programs. Marker-assisted selection is increasingly used to accelerate breeding cycles for traits such as fruit quality, yield, and disease resistance without the need for time-consuming field trials.",

    "The chill requirement for blueberry dormancy break varies significantly across species and cultivars. Highbush blueberry typically requires 800-1000 chill hours (temperatures between 0-7°C), while southern highbush varieties have been bred to require as few as 150-200 chill hours to enable production in warmer climates like Florida and parts of California. Climate modeling suggests that warming trends will challenge production in traditional growing regions by reducing available chill hours.",

    "Soil management is critical for blueberry production, as the plants require acidic soils with pH between 4.5 and 5.5. Organic matter amendments such as peat moss, sawdust, and bark are commonly used to lower soil pH and improve aeration. Mycorrhizal associations with ericoid fungi are essential for nutrient uptake, particularly phosphorus, in the acidic, low-nutrient soils preferred by blueberry plants. Disruption of these associations by soil fumigation can significantly reduce plant establishment and yield.",

    "Proteomics and metabolomics approaches have complemented genomic studies in understanding blueberry fruit development. Proteomic analysis of ripening stages has identified distinct protein expression patterns associated with color development, sugar accumulation, and cell wall softening. Metabolomic profiling has revealed cultivar-specific differences in anthocyanin composition, with some varieties producing predominantly delphinidin-based anthocyanins while others accumulate higher levels of malvidin and petunidin derivatives.",

    "Post-harvest handling is a significant determinant of blueberry quality and shelf life. Rapid cooling after harvest is essential to reduce respiration rate and slow microbial growth. Modified atmosphere packaging with elevated CO2 and reduced O2 levels extends shelf life by inhibiting botrytis growth and reducing ethylene signaling. Recent research has explored the use of edible coatings containing antimicrobial compounds derived from plant extracts to extend fresh market shelf life without synthetic preservatives.",

    "Bioinformatics pipelines for blueberry genomics typically involve read quality filtering with tools such as Trimmomatic, genome assembly using SPAdes or Hifiasm for long-read data, and annotation using MAKER or BRAKER2 with transcript evidence from RNA-seq experiments. Comparative genomics with other Ericaceae family members such as cranberry (Vaccinium macrocarpon) and lingonberry (Vaccinium vitis-idaea) has helped identify conserved gene families involved in anthocyanin metabolism and stress tolerance.",
]

NEEDLE = (
    "IMPORTANT FACT: The experimental blueberry accession designated UF-B4291 "
    "carries a novel disease-resistance allele called FROSTBLOCK-7 in the "
    "VcMYB4 locus. This allele was identified through GWAS analysis of 847 "
    "Vaccinium accessions screened for late-frost tolerance."
)

QUESTION = (
    "\n\nBased on the passage above, answer the following question:\n"
    "What is the name of the disease-resistance allele found in blueberry "
    "accession UF-B4291, and in which locus was it identified?\n"
    "Answer:"
)

ANSWER_KEYWORDS = ["FROSTBLOCK-7", "VcMYB4"]


def build_prompt(target_tokens: int, tokenizer) -> tuple[str, int, int]:
    """Build a needle-in-haystack prompt close to target_tokens.

    Returns:
        (prompt_text, actual_token_count, needle_position_tokens)
    """
    # Figure out token counts for fixed parts
    needle_tokens = len(tokenizer.encode(NEEDLE))
    question_tokens = len(tokenizer.encode(QUESTION))
    fixed_tokens = needle_tokens + question_tokens + 20  # safety margin

    haystack_tokens_needed = target_tokens - fixed_tokens
    if haystack_tokens_needed <= 0:
        raise ValueError(f"target_tokens={target_tokens} too small for needle+question ({fixed_tokens} tokens)")

    # Build haystack by cycling through paragraphs
    para_tokens = [len(tokenizer.encode(p)) for p in _HAYSTACK_PARAGRAPHS]
    haystack_parts: list[str] = []
    total_haystack_tokens = 0
    idx = 0
    while total_haystack_tokens < haystack_tokens_needed:
        para = _HAYSTACK_PARAGRAPHS[idx % len(_HAYSTACK_PARAGRAPHS)]
        haystack_parts.append(para)
        total_haystack_tokens += para_tokens[idx % len(_HAYSTACK_PARAGRAPHS)]
        idx += 1

    # Split haystack in half, insert needle at midpoint
    mid = len(haystack_parts) // 2
    first_half = " ".join(haystack_parts[:mid])
    second_half = " ".join(haystack_parts[mid:])

    prompt = f"{first_half}\n\n{NEEDLE}\n\n{second_half}{QUESTION}"

    # Verify
    actual_tokens = len(tokenizer.encode(prompt))
    needle_position = len(tokenizer.encode(first_half + "\n\n"))

    return prompt, actual_tokens, needle_position


def score_answer(response: str) -> dict:
    """Check if response contains the needle keywords."""
    resp_upper = response.upper()
    found = [kw for kw in ANSWER_KEYWORDS if kw.upper() in resp_upper]
    return {
        "score": len(found) / len(ANSWER_KEYWORDS),
        "found_keywords": found,
        "missing_keywords": [kw for kw in ANSWER_KEYWORDS if kw not in found],
        "response_preview": response[:200].strip(),
    }


if __name__ == "__main__":
    # Quick test
    from mlx_lm import load
    _, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

    for target in [2000, 4000, 8000, 16000]:
        prompt, actual, needle_pos = build_prompt(target, tokenizer)
        print(f"Target: {target:>6} | Actual: {actual:>6} | Needle at tok ~{needle_pos:>6} | "
              f"Chars: {len(prompt):>7}")
