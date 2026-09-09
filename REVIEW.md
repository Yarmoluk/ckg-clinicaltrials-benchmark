# Independent Review Protocol

Start with `INTEGRITY_V3_REPORT.md` and run:

```bash
python evaluation/test_integrity_v3.py
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/verify_integrity_v3.py
```

The first-run and integrity-v2 artifacts are retained to show the defects that
later versions correct. Do not treat either superseded result as the primary
claim.

This repository welcomes technical review without implying coauthorship or
endorsement.

## Minimum Reproduction

1. Clone the repository at a recorded commit SHA.
2. Install `evaluation/requirements.txt` in an isolated environment.
3. Run the integrity-v3 tests and verifier.
4. Confirm identical query IDs, retrieval questions, generation questions,
   model configuration, and prompt hash across paired systems.
5. Inspect at least 20 CKG wins and all 32 CKG/RAG ties.
6. Review the graph, prose corpus, reconstructed contexts, and normalized source
   records for those questions.
7. Report disagreements, scorer weaknesses, and possible leakage explicitly.

## Strong Review

A stronger independent evaluation should also:

- rerun the same 240 questions and three model-backed conditions using a
  separately controlled API account;
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
