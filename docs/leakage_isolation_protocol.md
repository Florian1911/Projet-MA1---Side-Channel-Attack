# Leakage isolation protocol: SubBytes and HW/HD validation

Goal: isolate whether the measured leakage follows the first-round XOR state,
the S-box output, or the transition into the S-box output.

All variants use:
- no UART at runtime
- `plaintexts_data.h`
- key `2B7E151628AED2A6ABF7158809CF4F3C`
- trigger on PB8
- 3 LED blinks, then 3 s delay, then repeated captures

## Firmware variants

Use one variant at a time by copying it over the STM32 project's `main.c`,
then rebuilding and flashing.

| Variant | File | Triggered operation | Expected strongest model |
|---|---|---|---|
| XOR-only | `firmware/main_no_uart_xor_key.c` | `state[b] = pt[b] xor key[b]` | `HW(pt xor key)` |
| SBox-only control | `firmware/main_no_uart_sbox_only.c` | `state[b] = SBox(pt[b])` | no key recovery expected |
| First-round SBox | `firmware/main_no_uart.c` | `state[b] = SBox(pt[b] xor key[b])` | `HW(SBox(pt xor key))` or HD transition |

The SBox-only control is intentionally key-independent. If it still gives key
recovery, the result is probably caused by an artefact or plaintext/key sequence
confusion rather than real key-dependent leakage.

## Capture command

Use the same command after each flash, changing only the output filename.

Example for XOR-only:

```bash
python setups/base_saine_pico5000/acquire_pico5000a_no_uart.py \
  --n-traces 1000 \
  --num-samples 8000 \
  --pre-trigger 1000 \
  --timebase 8 \
  --meas-mode diff_ab \
  --trigger-source ext \
  --meas-range-a PS5000A_500MV \
  --meas-range PS5000A_500MV \
  --plaintexts plaintexts_no_uart.npy \
  --key-hex 2B7E151628AED2A6ABF7158809CF4F3C \
  --output leakiso_xor_x10_1k_w8000.npz \
  --debug-trigger
```

Suggested filenames:

```text
leakiso_xor_x10_1k_w8000.npz
leakiso_sboxonly_x10_1k_w8000.npz
leakiso_sboxkey_x10_1k_w8000.npz
```

If x10 remains too weak, repeat the same protocol in x1 before changing
anything else.

## Analysis command

After each capture:

```bash
python leakage_model_compare.py \
  --npz leakiso_xor_x10_1k_w8000.npz \
  --out leakiso_xor_x10_1k_w8000_model_compare.json \
  --n-traces 1000 \
  --plot
```

Repeat for the SBox-only and SBox-key datasets.

## Decision criteria

For each firmware variant, compare:
- rank-1 count
- top-5 count
- top-20 count
- mean/median rank
- mean true-key correlation
- true-key POI stability
- whether the correct model separates from false candidates

Expected interpretation:

- If XOR-only is best with `HW(pt xor key)`, the pipeline can observe first-round key-dependent state leakage.
- If SBox-key is best with `HW(SBox(pt xor key))`, the report's selected model is experimentally supported.
- If HD models win consistently on SBox-key, the physical leakage is more transition-like than state-like.
- If SBox-only recovers the AES key, suspect an artefact: the operation is key-independent.
- If none of the variants recovers anything in x10 but x1 does, the limiting factor is mainly x10/high-side SNR rather than the leakage model.
