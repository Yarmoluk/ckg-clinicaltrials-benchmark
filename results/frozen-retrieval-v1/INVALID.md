# Invalid Retrieval Run

Do not cite this directory. It is retained only as an audit trail.

An adversarial review found that the inherited raw-prose Markdown loader used
the broad pattern `<[^>]+>` to remove HTML. Clinical source text contains
literal comparisons such as `<26 ... >`; the pattern therefore deleted large
spans of valid prose before indexing. The edge-text and CKG inputs were not
affected, but the raw-prose baseline was compromised.

`../frozen-retrieval-v2/` is the corrected comparison. Its prose loader removes
only known presentation markup and verifies that every NCT identifier in the
frozen chapter Markdown remains present after preprocessing.
