# Labeling Rubric (draft v0.1)

These labels are the reference for Precision@5. Every eval resume gets labeled against all 50 jobs **before anyone looks at matcher output**.

## The question

> Would it be reasonable to put this job in this person's top recommendations, given what the resume actually shows and the preferences they stated?

`relevant = 1` means yes. `relevant = 0` means no. There's no "maybe", so pick one and use `notes` if it was close.

## Mark it relevant (1) when all of these hold

1. **Core quals are shown on the resume.** The must-haves in the posting (degree field, main language or stack, the core type of work) show up in the resume's experience, projects or coursework. They don't have to be exact. Java on the resume is enough for a "Java or C#" posting.
2. **Level fits.** It's a new-grad or early-career role, and any required years of experience are ones the person could plausibly have (internships and co-ops count).
3. **Stated constraints are met.** Location or remote preference, sponsorship needs, and anything else listed in `resumes.csv`.

## Don't mark it irrelevant just because

- A **preferred / nice-to-have** skill is missing.
- The resume uses a different word for the same thing (e.g. "ML pipelines" vs "model training infrastructure").
- The company or domain is unfamiliar to the candidate.

## Mark it irrelevant (0) with a reason code

| code | meaning |
|---|---|
| `CORE_MISSING` | A required core skill or type of work isn't shown anywhere |
| `DEGREE` | A required degree or field isn't met (e.g. requires EE, MS or PhD) |
| `LEVEL` | Too senior, or requires years the person clearly doesn't have |
| `LOCATION` | Conflicts with stated location or remote preference |
| `SPONSOR` | Person needs sponsorship and the posting says none |
| `CLEARANCE` | Requires a security clearance or US citizenship the resume doesn't show |
| `DOMAIN` | Different field altogether (e.g. hardware or RF for a pure SWE resume) |
| `OTHER` | Explain in `notes` |

For a relevant pair, `reason_code` can stay blank.

## Process

1. Read the resume and its preferences once, then go down all 50 jobs.
2. For each job, skim the requirements section first, then the rest if needed.
3. Fill in `relevant`, `reason_code`, `notes` if it was close, and your name in `labeler`.
4. A second person labels `second_review.csv` (a 20% sample, stratified by resume) without looking at the primary labels.
5. Run `python scripts/label_agreement.py`, talk through each disagreement, fill `resolved`, and rerun to produce `reference_labels.csv`.
6. If kappa comes in under about 0.6, tighten this rubric using the disagreements, bump the version, and relabel the affected pairs.

## Calibration (do this first)

Label the 2 **dev** resumes against the 20-job small workload together as a team (40 pairs). Update this rubric with any rule you had to make up, then freeze it before labeling the eval resumes.
