# Behavioral analysis workspace

## Flow

Choose test → test-specific setup → review → results. Three Chamber retains Recordings & groups → Review regions → Results, including metadata, statistics, and compact previews. Stereotypy opens its side-view manual reviewer with independent sessions and CSV exports; automatic detectors still need training. Its **Choose test** link returns to the shared selection screen. Existing video/region/results deep links and saved drafts continue to work. Test identity is saved in each new Three Chamber batch and export; that batch endpoint rejects other assays so side-view recordings cannot enter the chamber scoring pipeline.

Recordings collect sample ID, sex, and genotype in an editable setup table. Each sample ID represents one independent mouse, as confirmed by the user. Existing IDs are retained; unknown metadata is never guessed. Metadata changes do not invalidate reviewed geometry. Quick fill affects only missing fields and is explicitly applied by the user. A compact group summary shows cohort composition before analysis.

Statistics are configured before running the batch. Descriptive summaries are always included. Optional analyses: pairwise Welch genotype comparisons pooling sexes, genotype comparisons within each sex, two-way genotype × sex ANOVA, or all of these. Users select outcomes and alpha. Holm correction applies across all estimable tests in the selected batch analysis. Sample-level observations, exclusions, exact methods, assumptions, and sample sizes are exported.

Live preview is collapsed by default, with visible recording count and processing progress. Clicking the preview opens playback. Collapsing or leaving Results pauses playback and segment transfers while processing continues. Completed results use compact cards and click-to-play review videos. One video plays at a time; metadata and measurements stay visible without large video panels.

## Statistical contract

One independent mouse per row; no pooling video frames or bouts as replicates. Sex is male, female, or not recorded. Genotype is user-provided text. Missing metadata and outcomes are retained in raw exports and accounted for explicitly. Target-relative cup outcomes respect the configured social/novel cup side and remain unavailable when that side is unspecified.

Pairwise tests are two-sided Welch tests, with n ≥ 2 per group, group means, sample SD, mean difference, unadjusted 95% CI, raw p and global Holm-adjusted p. Zero standard error, insufficient observations, and missing groups return a reason rather than an invented p-value. Factorial analysis uses OLS with sum contrasts and Type III ANOVA, including interaction. It requires at least two genotypes, both sexes, and at least two observations in every sex × genotype cell. Export residual degrees of freedom and partial eta squared; document normal residual / constant variance assumptions and avoid automatic claims of significance or model validity.

Statistics are a fixed reproducible export, not an editable Excel statistical model. Export per-sample input data and configuration; changing metadata or outcomes requires a new batch export. Existing scoring behavior, original videos, model weights, and prior workbooks are unchanged.

## Verification

- Python tests: metadata validation/migration, test selection, missing data, group counts, Welch results against hand calculations, factorial results against known balanced fixtures, multiplicity, exclusions, degenerate groups, and immutable batch settings.
- Browser: landing and deep links, table editing and metadata persistence, region review preservation, stats settings, compact preview, keyboard expansion/collapse, and results playback.
- Excel: generate an isolated synthetic cohort with known values, inspect results and formulas/errors, render affected sheets, check sample IDs and group mapping, confirm no fictional sex/genotype is assigned to actual sample videos.
- Full existing regression suite, JavaScript syntax/lifecycle checks, and GitHub CI.

## Implemented verification (2026-09-05)

- Stereotypy reintegration: 116 Python tests and 4 player lifecycle tests pass. The integration test saves side-view annotations and verifies the Three Chamber draft, reviewed state, mouse metadata, and statistics remain unchanged; the chamber batch endpoint rejects the side-view assay. Browser checks exercise Test → Stereotypy → Choose test → Three Chamber, with metadata/statistics controls retained.

- 93 Python tests and 4 player lifecycle tests passed. Statistical fixtures independently check Welch t/df/p, balanced factorial sums of squares and F statistics, and the ordered Holm calculation.
- Browser checks confirmed persisted genotype/comparison edits without clearing three reviewed regions; collapsed live playback; continuous annotated playback after clicking; paused playback on collapse; and completed review playback.
- An isolated full 675 run reused content-matched predictions, scored 13,482 frames over exactly 600 seconds, and reproduced previous occupancy, cup-time, and preference values exactly. It completed the annotated video and workbook. No sex or genotype was invented; the report correctly calculated zero inferential tests and documented both missing metadata fields.
- A separate eight-mouse synthetic cohort exercised Welch comparisons, interaction ANOVA, group summaries, and adjusted p-values. Both workbooks passed XML error scans; affected sheets were rendered for visual inspection. Validation files stay outside the real batch history and Git.
