# Delivered scope and next objectives

This page separates the implemented public server from historical demonstration
evidence and remaining research or application work. The
[FIERCE evidence index](evidence/README.md) maps objectives to implementation and
proofs; the [dossier](evidence/fierce-kpi-evidence.en.md) retains quantitative
results and scope adaptations as of 28 October 2026.

## Delivered within bounded technical scopes

| Area | Delivered capability | Boundary to retain |
| --- | --- | --- |
| Geospatial pipeline | Zone inputs, ERA5 ingestion, Sentinel-2 NDVI, joined climate/vegetation data, forecast lookup with fallback and two-level analysis. | Six adopted B1–B6 modules; accepted minimum scope is not every stronger objective in the original plan. |
| Pattern mining | Native so4gp 1.1.2 GRAANK/TGrad extraction, serializable libraries and a reproduced real-data comparison. | Comparison and extraction do not establish general superiority. |
| Scientific evaluation | Spatial/temporal checks, recurrence metadata and an NDVI predictive ablation. | Negative results are deliverables; all 418 records in the later three-pilot capture remain unqualified. |
| MCP access | Sixteen tools, structured contracts, resources, stdio/HTTP/SSE transport and bearer authentication. | A standalone server does not supply a user interface, document RAG, scheduling or notification delivery. |
| Historical integration evidence | Authorized conversational access, retained results, bounded explanatory analysis and forecast-threshold notifications in dated application demonstrations. | Environment-specific observations do not qualify every public installation or production account. |

## Research and validation objectives

| Next objective | Evidence needed before a stronger claim |
| --- | --- |
| Qualify useful patterns | A suitable labelled task, retained protocol, independent held-out evaluation and criteria declared before interpreting results. Preserve missing confidence and negative findings. |
| Demonstrate predictive benefit | Improvement over appropriate baselines on the intended task. The retained Fos NDVI ablation increases MAE by about 4%; it supplies no positive primary-result claim. |
| Calibrate confidence or event probability | A validated definition and calibration protocol. An explanatory support/stability/agreement/lag index is not a probability. |
| Extend forecast evidence | Evaluate the proposed providers, variables and lead times. AIFS/physical-model delivery does not implement the original FourCastNet extension; forecast availability does not establish skill at every horizon. |
| Advance operational maturity | Observe the complete intended workflow with intended operators, outputs, use, errors and recovery in its operational setting. Integrated-demonstrator TRL 6 is a moderate-confidence self-assessment; TRL 7 is unproven. |

Novel pattern-based predictive alerting remains a separate research capability,
approximately TRL 4 at most on the retained evidence. Its maturity is not inherited
from the integrated demonstrator.

## Application and impact objectives

Deployment owners must configure authentication, storage, provider access,
dataset registration and authorization. New interfaces and deployments need
their own connected qualification. Historical retained-data demonstrations do
not establish a fresh acquisition/readback path in every interface.

Commercial validation (KPI 1), a combined three-pilot demonstration video and
resource-efficiency assumptions require their own materials. The technical
repository neither assigns a commercial completion score nor measures the
projected 5–10% resource gain (KPI 9). External FIERCE appraisal, scientific
ratification and IRL/SRL assessment remain separate decisions.

Contributions should identify which objective they advance, its validation and
remaining limits. See [CONTRIBUTING.md](../CONTRIBUTING.md) for review and CI policy
and [SECURITY.md](../SECURITY.md) for private vulnerability reporting.
