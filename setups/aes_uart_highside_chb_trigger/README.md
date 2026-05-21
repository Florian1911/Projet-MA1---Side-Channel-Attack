# AES UART high-side, trigger sur ChB

Ce dossier regroupe les fichiers pour le nouveau montage :

- `main_aes_uart_highside_trigger_chb.c` : firmware STM32 AES-128 ECB avec UART USB/ST-Link et trigger PB8.
- `acquire_aes_uart_highside_chb_trigger.py` : acquisition PicoScope. Par defaut, ChA mesure la tension apres shunt et ChB recoit le trigger PB8.

## Cablage prevu

- Shunt high-side : `3.3 V -> shunt -> VDD carte STM32`.
- Pico `ChA` : noeud apres shunt, cote carte STM32.
- Pico `ChB` : trigger PB8 STM32.
- Masse Pico : masse carte.
- USB : communication UART via ST-Link/USB (`/dev/ttyACM0` par defaut).

La trace sauvegardee dans `traces` vaut :

```text
trace = 3300 mV - ChA
```

Donc si la carte consomme plus, la tension apres shunt descend et la trace augmente. Les signaux bruts sont aussi sauvegardes :

- `measure_mv` : ChA brut en mV.
- `trigger_mv` : ChB brut en mV.
- `plaintexts`, `ciphertexts`, `labels`, `key`, `overflows`.

Si le montage final inverse les canaux, le script accepte `--meas-channel` et `--trigger-channel`.

## Firmware STM32

Copier `main_aes_uart_highside_trigger_chb.c` comme `Core/Src/main.c` dans le projet STM32 actuel.

Le protocole UART est volontairement simple :

```text
PC  -> STM32 : 'P' + 16 bytes plaintext
STM32 -> PC  : 'C' + 16 bytes ciphertext
```

PB8 monte juste avant `mbedtls_aes_crypt_ecb()` et redescend juste apres. Le Pico peut donc declencher sur front montant de ChB.

## Acquisition smoke test

Depuis ce dossier :

```bash
python acquire_aes_uart_highside_chb_trigger.py \
  --serial-port /dev/ttyACM0 \
  --n-traces 100 \
  --num-samples 4000 \
  --pre-trigger 500 \
  --timebase 8 \
  --supply-mv 3300 \
  --meas-range PS5000A_5V \
  --trigger-range PS5000A_5V \
  --trigger-mv-probe 1500 \
  --output smoke_highside_chb_trigger.npz
```

Pour une campagne plus longue :

```bash
python acquire_aes_uart_highside_chb_trigger.py \
  --serial-port /dev/ttyACM0 \
  --n-traces 5000 \
  --num-samples 4000 \
  --pre-trigger 500 \
  --timebase 8 \
  --supply-mv 3300 \
  --output aes_highside_chb_trigger_5k.npz
```

## Points de calibration

- Si le trigger ne part pas, baisser `--trigger-mv-probe` ou verifier que PB8 arrive bien sur ChB.
- Si ChA sature, augmenter `--meas-range` ou verifier l'attenuation de sonde avec `--probe-att-meas`.
- Si le front PB8 est mesure avec une sonde x10, lancer avec `--probe-att-trigger 10`.
- Pour strictement mesurer `3.3 V - ChB`, utiliser `--meas-channel B` et mettre le trigger sur un autre canal avec `--trigger-channel A/C/D`.
