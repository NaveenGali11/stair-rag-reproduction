# Data licensing and attribution

The MIT License in [LICENSE](LICENSE) applies to the original source code in
this repository. It does **not** replace the licenses of the source textbook,
derived datasets, model checkpoints, or other third-party materials.

## Source textbook

- **Title:** *The Whole Child: Development in the Early Years*
- **Authors:** Deirdre Budzyna and Doris Buckley
- **Publisher:** ROTEL
- **Source:** <https://rotel.pressbooks.pub/whole-child/>
- **License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The raw publisher PDF is not committed. Its expected SHA-256 is recorded in
[data/MANIFEST.md](data/MANIFEST.md).

## Textbook-derived repository data

The following paths contain extracted, transformed, or generated material
derived from the textbook and should be treated as CC BY-NC-SA 4.0 material:

- `data/derived/`
- `data/splits/`
- `data/retriever/`
- `data/gold_queries.jsonl`

This includes extracted section text, paragraph text, ToC labels, canonical
paths, generated questions grounded in textbook passages, and the train/dev/
test representations built from them. Retain attribution, use the material
only for non-commercial purposes, and distribute adaptations under the same
or a compatible license.

The compact numeric summaries in `results/` and the plots in
`reports/figures/` document this experiment. Their inclusion does not grant
rights to the underlying textbook beyond its CC BY-NC-SA 4.0 license.

## Third-party models and software

No third-party model weights are committed. Anyone downloading or using the
referenced Hugging Face models must review and comply with each model's
license and acceptable-use terms. Python dependencies retain their respective
licenses.

This notice is a practical repository boundary, not legal advice. If your use
is commercial or otherwise outside CC BY-NC-SA 4.0, obtain appropriate
permission from the rights holders.
