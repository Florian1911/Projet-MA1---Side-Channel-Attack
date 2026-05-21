# x10 SCA measurement runbook

Use this checklist after both probes are switched to x10.

## 1. Probe calibration check

Channel A:

```bash
python probe_calibration_check.py --channel A --probe x10 --out-prefix probe_A_x10_calibration
```

Channel B:

```bash
python probe_calibration_check.py --channel B --probe x10 --out-prefix probe_B_x10_calibration
```

Expected result:
- around 1 kHz square wave
- no overflow
- flat plateaus
- low overshoot/undershoot

## 2. High-side AES smoke capture

With x10 probes, a 3.3 V node appears as about 330 mV at the PicoScope BNC input. Use a 500 mV Pico range and probe attenuation 10.

```bash
python acquire_dataset_highside.py \
  --n-traces 200 \
  --output highside_x10_smoke_200.npz \
  --range-a-mv 500 \
  --range-b-mv 500 \
  --probe-att-a 10 \
  --probe-att-b 10 \
  --timebase 8 \
  --num-samples 2000 \
  --arm-delay-us 4700
```

Check:
- `overflow_ratio` in `highside_x10_smoke_200.json` must be 0
- `traces_a` and `traces_b` should be around 3300 mV after attenuation correction
- `traces` is `ChA - ChB`, the shunt differential voltage

## 3. Quick stability analysis

```bash
python analyze_supply_stability.py \
  --npz highside_x10_smoke_200.npz \
  --out-prefix supply_stability_highside_x10_smoke_200
```

## 4. Quick CPA check

```bash
python highside_campaign.py \
  --npz highside_x10_smoke_200.npz \
  --out highside_x10_smoke_200_campaign_summary.json
```

## 5. Larger capture if smoke test is clean

```bash
python acquire_dataset_highside.py \
  --n-traces 1000 \
  --output highside_x10_1k.npz \
  --range-a-mv 500 \
  --range-b-mv 500 \
  --probe-att-a 10 \
  --probe-att-b 10 \
  --timebase 8 \
  --num-samples 2000 \
  --arm-delay-us 4700
```

Then rerun the stability and CPA checks on `highside_x10_1k.npz`.
