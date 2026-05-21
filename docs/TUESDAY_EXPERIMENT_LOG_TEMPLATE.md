# Tuesday Blind SCA Log

## Setup
- Date:
- Board/firmware commit:
- Scope:
- Sampling (`num_samples`, `timebase`):
- Trigger setup:

## Keys Used
- Known key 1:
- Known key 2:
- Known key 3:
- Blind key (revealed after attack):

## Datasets
- `known_k1_20k_raw.npz`:
- `known_k2_20k_raw.npz`:
- `known_k3_20k_raw.npz`:
- `unknown_blind_20k_raw.npz`:

## Alignment
- Method: `ref` / `median`
- Parameters: `center=520, window=260, max_shift=80, iters=2`
- Notes:

## Results
### DL HW
- Campaign: `exp_blind_hw_tuesday`
- Summary:

### DL SBOX
- Campaign: `exp_blind_sbox_tuesday`
- Summary:

### Fusion
- File: `blind_tuesday_fusion.json`
- Recovered key:

### CPA Blind Baseline
- File: `blind_tuesday_cpa_summary.json`
- Recovered key:

## Final Validation
- Ground truth:
- Fusion match:
- CPA match:
- Best method:

## Discussion
- What worked:
- What failed:
- Next iteration:
