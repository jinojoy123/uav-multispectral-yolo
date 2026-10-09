# Pre-release and reproducibility checklist

This checklist records issues observed while cleaning the supplied archive. Items marked **BLOCKER** should be resolved before treating the repository as a publication-ready public release.

## License and provenance

- [ ] **BLOCKER — Confirm the license and provenance of the bundled modified Ultralytics source.** The source identifies version `8.4.112` and includes AGPL-3.0 headers, but the supplied archive did not include a root license file. Verify the applicable upstream license, retain all notices, document modifications, and determine the compatible license/redistribution terms for the combined repository before adding a root `LICENSE` or publishing.
- [ ] Identify the authors/owners and intended license for the custom research scripts.
- [ ] Add upstream repository URL, exact upstream commit/tag (if known), and a concise record of modifications to the bundled source.
- [ ] Replace TODO fields in `CITATION.cff`; add the manuscript citation and a versioned archive DOI after release.

## Critical reproducibility checks

- [ ] **BLOCKER — Verify RGBRN preprocessing.** `data_arrangement.py` uses per-band `cv2.normalize(..., 0, 255, cv2.NORM_MINMAX)`. At least some mapping scripts select `RGBRN_SCALING_MODE = "divide_256"`. These differ; inspect the actual training data path and correct or document inference preprocessing before using reported results as reproducible evidence.
- [ ] Verify that every RGBRN path uses the same five-band order: Red, Green, Blue, Red Edge, NIR.
- [ ] Verify class names and order across labels, dataset YAML, training, evaluation, and prescription mapping. `train_RGBRN.py` currently declares `['Crop', 'Weed']`; the manuscript may use `Sugarbeet` and `Weed` terminology.
- [ ] Confirm the correct model architecture/checkpoint, number of input channels, and initialization/weight-transfer behavior in the bundled source.
- [ ] Check confidence thresholds, spatial grid size, weed-density thresholds, prescription rates, coordinate reference system handling, and output schemas against the manuscript methods.
- [ ] Confirm benchmark timing definitions, warm-up count, repetition count, batch size, power sampling, and energy-estimation labels against the final analysis scripts and manuscript.

## Paths and portability

- [ ] Replace hardcoded absolute paths such as `/processed_RGBRN`, `/yolomodel.pt`, `/Orthomosaic_RGB.tif`, and `/Orthomosaic_RGBRN.tif` with explicit configuration or command-line arguments.
- [ ] Review the `clone_dir = "uav_MS_yolo"` imports in training scripts; confirm they work from the documented working directory.
- [ ] Test scripts from a clean checkout with the documented working directory, rather than relying on paths present on the original workstation.
- [ ] Add a dependency specification with versions tested on the target platform. Keep Jetson-specific dependencies separate from desktop dependencies if appropriate.
- [ ] Add a small, redistributable smoke-test fixture or documented validation procedure if permitted; do not bundle private field data.

## Repository and documentation

- [x] Removed nested `.git` metadata to prevent a nested repository and upstream history from being included unintentionally.
- [x] Removed `.pyc` files and `__pycache__` directories.
- [x] Added root README, `.gitignore`, and preliminary citation metadata.
- [x] Renamed `Mapping _pipeline_RGBRN_Orinnano.py` to `Mapping_pipeline_RGBRN_OrinNano.py` to remove the accidental space and clarify device naming.
- [ ] Inspect the complete diff and run a secret/privacy scan before public upload. The cleanup is not a guarantee that all sensitive content or credentials have been ruled out.
- [ ] Compile/test the scripts in the intended environments and record known limitations.
- [ ] Create a GitHub release only after blockers are resolved; archive that exact release with a DOI (e.g., Zenodo) and cite the version/DOI in the manuscript.
