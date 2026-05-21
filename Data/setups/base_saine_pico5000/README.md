# Base saine PicoScope 5000A (SCA AES no-UART)

Ce dossier contient une base minimale et propre pour repartir de zero avec ton PicoScope 5000A.

## Fichiers

- `generate_plaintexts.py` : genere `plaintexts_no_uart.npy` + `plaintexts_data.h`
- `smoke_test_pico5000a.py` : test rapide scope seul (1 capture)
- `acquire_pico5000a_no_uart.py` : acquisition SCA en rapid-block
- `main_aes_no_uart_template.c` : template firmware STM32 (trigger PB8)
- `requirements.txt` : dependances Python

## 1) Cablage recommande

- `ChA` (trigger) <- `PB8` de la carte (sonde x10 conseillee)
- `ChB` (mesure) <- shunt low-side (sonde x1 ou diff selon ton montage)
- Masse scope reliee a la masse de la carte

## 2) Environnement PC

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r setups/base_saine_pico5000/requirements.txt
```

Si `picosdk` n'est pas detecte, installe aussi PicoSDK systeme et verifie `libpicoipp.so`.

## 3) Smoke test scope (obligatoire)

```bash
python setups/base_saine_pico5000/smoke_test_pico5000a.py
```

Attendu:
- pas d'erreur `OpenUnit`
- image `smoke_test_pico5000a.png` generee

## 4) Preparation des plaintexts

```bash
python setups/base_saine_pico5000/generate_plaintexts.py --n 5000 --seed 1234
```

Ensuite:
- copier `plaintexts_data.h` dans ton projet STM32 (`Core/Inc/`)
- integrer la logique du template `main_aes_no_uart_template.c`
- compiler/flasher

## 5) Sequence de capture

1. Lance le script d'acquisition
2. Quand il affiche `SCOPE ARME`, appuie sur RESET STM32
3. Attends la fin

Commande type:

```bash
python setups/base_saine_pico5000/acquire_pico5000a_no_uart.py \
  --n-traces 5000 \
  --num-samples 4000 \
  --pre-trigger 200 \
  --timebase 8 \
  --trig-mv-probe 1500 \
  --probe-att-trig 10 \
  --plaintexts plaintexts_no_uart.npy \
  --output dataset_pico5000a_no_uart.npz
```

## 6) Verification rapide du dataset

```bash
python - << 'PY'
import numpy as np
D = np.load('dataset_pico5000a_no_uart.npz')
print('traces:', D['traces'].shape)
print('plaintexts:', D['plaintexts'].shape)
print('key:', D['key'])
print('overflows nonzero:', int((D['overflows'] != 0).sum()))
PY
```

## 7) Si ca ne capture pas

- augmente `--trig-mv-probe` (ex: 1800, 2200)
- verifie PB8 -> ChA et la masse commune
- commence avec `--n-traces 200` pour valider le flux
- garde `--num-samples` plus petit tant que le trigger n'est pas stable

## Flux conseille (safe)

1. smoke test scope
2. capture courte (`200 traces`)
3. capture moyenne (`2000 traces`)
4. capture finale (`5000+ traces`)
