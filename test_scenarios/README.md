# Test Scenarios
Four controlled inputs to trigger each Decision Gate branch.

## How to use
1) Pick a scenario folder name below.
2) Apply locally, commit, push a PR:
   ```bash
   bash scripts/apply_scenario.sh 01_reject_trivy_high_risk
   git checkout -b scenario/reject-trivy
   git add -A && git commit -m "Apply scenario: reject-trivy"
   git push origin scenario/reject-trivy
   ```

## Folders
- `01_reject_trivy_high_risk` → .env with AWS keys (Trivy HIGH) + clean eval → **REJECT (Trivy)**
- `02_reject_ml_high_anomaly` → clean .env + anomaly-heavy eval → **REJECT (ML)**
- `03_hold_moderate_risk` → mild secrets + ~4% anomalies → **HOLD**
- `04_accept_clean` → clean .env + clean eval → **ACCEPT**
