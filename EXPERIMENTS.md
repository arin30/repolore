# Experiments

Each retrieval change is measured with `repolore eval` against the same question set, one change at a time:
hypothesis, change, metric deltas, keep or drop. The question set lives outside this repository because it names
private repos, so this file records aggregate numbers only, never question text.

How to read the numbers:

- **recall@k**: of the sources a question expects, the share found in the top k chunks, averaged over questions.
- **MRR**: 1 / rank of the first expected source, averaged.
- **nDCG@10**: rewards finding every expected source near the top; only the first chunk of each source counts.
- Unanswerable questions are left out of these and shown as a mean top score instead, to see whether a score cutoff
  could answer "not found".
- With about 35 questions, one question moves recall by about 3%. Treat changes under 5% as noise.

## Baseline (2026-10-24)

Vector search only, with no reranking and no keyword search.

| Setting | Value |
|---|---|
| Corpus | 11 repos, 945 documents, 2,576 chunks |
| Chunking | a chunk per markdown section, sections over 400 estimated tokens split between blocks |
| Embedder | `voyage-4-nano` on the CPU, float32, 1024 dimensions |
| Questions | tuning set: 35 questions from public repos (32 answerable, 3 unanswerable) |

| Metric | Value |
|---|---|
| recall@5 | 0.818 |
| recall@10 | 0.919 |
| MRR | 0.809 |
| nDCG@10 | 0.804 |

| Question kind | recall@5 | Questions |
|---|---|---|
| rationale | 0.950 | 10 |
| factual | 0.844 | 15 |
| commit-only | 0.750 | 4 |
| cross-repo | 0.333 | 3 |

Mean top score: 0.649 on answerable questions, 0.480 on unanswerable ones.

Observations:

- Single-source questions mostly land at rank 1. Seven of the nine misses expect several sources: one is found
  near the top and another ranks lower (6th to beyond 10th).
- Cross-repo questions are the weakest. A question like "which repos use X" matches the tool's own README and
  docs better than the PRs that adopted it in other repos.
- Some misses retrieve a source that also answers the question but isn't listed as expected (a `SECURITY.md`
  rather than the PR that added it). The expected sources were written before any retrieval existed and are
  incomplete, which caps the scores. Fixing labels is a decision about the question set, not about retrieval.
- Top scores overlap: unanswerable questions reach up to 0.51, and 3 of the 32 answerable ones score below that. A
  score cutoff alone would turn away some answerable questions.
