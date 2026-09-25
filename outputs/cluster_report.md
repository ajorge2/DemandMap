# Fuzzy semantic clustering report

## Experiment setup

- Semantic view: `Job Title` (50/50 usable rows)
- Embedding backend used: `openai`
- Selected clusters: 3
- Primary membership cutoff: 0.40
- Clustering: Fuzzy C-Means prototypes plus independently calibrated prototype affinities.

The final `cluster_*_membership` scores are independent affinities and do not sum to one. This allows one prospect to clear the cutoff for several semantic prototypes. The native Fuzzy C-Means memberships are also exported for auditability; those do sum to one.

## K diagnostics

The selection score combines argmax silhouette (diagnostic only), fuzzy partition coefficient, Xie-Beni compactness/separation, and minimum centroid separation. No hard label is used in the final output.

| K | Selection score | Silhouette | Partition coefficient | Partition entropy | Xie-Beni |
|---:|---:|---:|---:|---:|---:|
| 3 **selected** | 0.825 | 0.166 | 0.483 | 0.883 | 3.452 |
| 4 | 0.475 | 0.113 | 0.378 | 1.149 | 4.961 |
| 5 | 0.588 | 0.143 | 0.374 | 1.261 | 2.176 |
| 6 | 0.200 | 0.099 | 0.298 | 1.487 | 6.315 |
| 7 | 0.412 | 0.123 | 0.360 | 1.434 | 1.700 |

## Cutoff sensitivity

| Cutoff | Zero clusters | Exactly one | Two | Three or more |
|---:|---:|---:|---:|---:|
| 0.25 | 13 | 22 | 12 | 3 |
| 0.40 | 20 | 22 | 7 | 1 |
| 0.55 | 28 | 17 | 5 | 0 |
| 0.70 | 34 | 15 | 1 | 0 |

## Cluster overlap at cutoff 0.40

| Cluster pair | Shared rows | Jaccard overlap |
|---|---:|---:|
| 0: Board & Advisory Roles + 2: Investors | 6 | 0.30 |
| 1: C-Suite & Functional Executives + 2: Investors | 3 | 0.13 |
| 0: Board & Advisory Roles + 1: C-Suite & Functional Executives | 1 | 0.04 |

## Cluster 0 — Board & Advisory Roles

- Members above cutoff: 13
- Representative terms: advisor, job, title, scientific, truth, advisory

Strong members:

- Strategic Advisor (Cory Warfield) — 1.000
- Special Advisor (Stephen Curry) — 0.940
- Advisor to the CEO (Sanyin Siang) — 0.882
- Advisor - founding member (Steve Nouri) — 0.827
- Investor, Strategic Advisor (Kevin O'Leary) — 0.775
- Scientific Advisor (Andrew Huberman) — 0.724

Borderline examples nearest the cutoff:

- Founder (Amena Baig) — 0.380 (out)
- Founder (Vaibhav Sisinty) — 0.380 (out)
- Founder & CEO (Oana Labes, MBA, CPA) — 0.431 (in)
- Managing Partner (Alex Hormozi) — 0.334 (out)
- CEO (Gary Vaynerchuk) — 0.466 (in)
- Board Member (Tim Tebow) — 0.305 (out)

Shared with other clusters:

- Strategic Advisor (Cory Warfield) — 0: 1.00, 2: 0.50
- Advisor to the CEO (Sanyin Siang) — 0: 0.88, 2: 0.68
- Advisor - founding member (Steve Nouri) — 0: 0.83, 2: 0.54
- Investor, Strategic Advisor (Kevin O'Leary) — 0: 0.77, 2: 1.00
- Investor & Advisory Board Member (Dr. Joerg Storm) — 0: 0.59, 2: 0.83
- Founder & CEO (Oana Labes, MBA, CPA) — 0: 0.43, 1: 0.63, 2: 0.47

## Cluster 1 — C-Suite & Functional Executives

- Members above cutoff: 13
- Representative terms: founder, job, title, chairman, ceo, chair

Strong members:

- Co-Founder (Ishan Sharma) — 0.882
- Co-Founder (Nansi Mishra) — 0.882
- Co-Founder (Ryan Reynolds) — 0.882
- Co-Founder (Reno Perry) — 0.882
- Co-Founder (Alex Rodriguez) — 0.882
- Founder (Vaibhav Sisinty) — 0.700

Borderline examples nearest the cutoff:

- Cofounder and executive chairman (Mohamed (Nagaty) Aboulnaga) — 0.397 (out)
- CEO and Founder (Dennis R. Mortensen) — 0.431 (in)
- Founding Partner (Dr Ola Brown) — 0.364 (out)
- Co-Founder, Board Member (Reid Hoffman) — 0.466 (in)
- Fundador (Camila Farani) — 0.320 (out)
- Fundador (André Forastieri) — 0.320 (out)

Shared with other clusters:

- Founder (Vaibhav Sisinty) — 1: 0.70, 2: 0.61
- Founder (Amena Baig) — 1: 0.70, 2: 0.61
- Founder & CEO (Oana Labes, MBA, CPA) — 0: 0.43, 1: 0.63, 2: 0.47

## Cluster 2 — Investors

- Members above cutoff: 13
- Representative terms: investor, angel, job, title, advisory, advisor

Strong members:

- Investor, Strategic Advisor (Kevin O'Leary) — 1.000
- Investor (Ryan Serhant) — 0.911
- Investor (Steven Bartlett) — 0.911
- Investor & Advisory Board Member (Dr. Joerg Storm) — 0.827
- Angel Investor (Tobi Oluwole) — 0.749
- Angel Investor (Aishwarya Srinivasan) — 0.749

Borderline examples nearest the cutoff:

- Seed Investor (Frank Thelen) — 0.431 (in)
- Co-Founder (Ryan Reynolds) — 0.334 (out)
- Co-Founder (Nansi Mishra) — 0.334 (out)
- Co-Founder (Ishan Sharma) — 0.334 (out)
- Co-Founder (Alex Rodriguez) — 0.334 (out)
- Co-Founder (Reno Perry) — 0.334 (out)

Shared with other clusters:

- Investor, Strategic Advisor (Kevin O'Leary) — 0: 0.77, 2: 1.00
- Investor & Advisory Board Member (Dr. Joerg Storm) — 0: 0.59, 2: 0.83
- Advisor to the CEO (Sanyin Siang) — 0: 0.88, 2: 0.68
- Founder (Vaibhav Sisinty) — 1: 0.70, 2: 0.61
- Founder (Amena Baig) — 1: 0.70, 2: 0.61
- Advisor - founding member (Steve Nouri) — 0: 0.83, 2: 0.54

## Interpretation limits

- This is a 50-row exploratory sample dominated by founders, investors, and advisors.
- Labels summarize examples; they do not determine membership.
- Independent affinities are sharpened within-prototype distance percentiles, not probabilities of persona truth.
- The selected cutoff changes assignment counts without recomputing embeddings or centroids.
- Future views such as company description or industry should be embedded and clustered separately before cross-view comparison.
