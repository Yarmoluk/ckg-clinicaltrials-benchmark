# Independent Review Protocol

Start with `INTEGRITY_V2_REPORT.md` and run:

```bash
python evaluation/test_integrity_eval.py
python evaluation/verify_integrity_v2.py
```

The first-run `REPORT.md` and `results/raw/paired/` artifacts are retained to
show the defects that integrity-v2 corrects. Do not treat the original
token-overlap result as the primary claim.

This repository welcomes technical review without implying coauthorship or
endorsement.

## Minimum Reproduction

1. Clone the repository at a recorded commit SHA.
2. Run `python3 evaluation/verify_results.py`.
3. Confirm that the matched CKG, RAG, and no-context files contain identical
   query-ID sets.
4. Inspect at least 20 wins and all 17 CKG losses from the paired sample.
5. Review the graph, prose corpus, and source record for those questions.
6. Report disagreements, scorer weaknesses, and possible leakage explicitly.

## Strong Review

A stronger independent evaluation should also:

- rerun the same 240 questions using a separately controlled API account;
- author held-out questions directly from frozen trial records;
- grade answer correctness and source entailment without seeing system identity;
- add at least one stronger retrieval baseline;
- report all prompts, parameters, failed calls, exclusions, and costs; and
- publish code and raw outputs under an immutable commit or release.

## Suggested Reviewer Statement

If the verifier and rerun succeed:

> I independently inspected the evaluation artifacts and reproduced the reported
> aggregate calculations at commit `<SHA>`. My review covered `<scope>`. This
> statement does not imply endorsement of universal superiority or clinical use.

## Attribution

Daniel Yarmoluk is the repository author. A reviewer may be acknowledged in this
file after review, with the exact scope and date of the review. Reviewers are not
listed as authors unless they make a qualifying scholarly contribution and both
parties explicitly agree to authorship.
