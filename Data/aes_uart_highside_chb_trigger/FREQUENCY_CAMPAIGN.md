# Campagne de test de frequence STM32

Objectif : comparer la lisibilite et le SNR des traces AES en ralentissant le coeur STM32, sans changer le protocole UART ni le montage high-side.

## Profils fournis

Les firmwares sont dans `clock_profiles/` :

| Fichier | SYSCLK vise | PLL |
|---|---:|---|
| `main_aes_uart_highside_trigger_chb_84MHz.c` | 84 MHz | `PLLN=336`, `PLLP=4` |
| `main_aes_uart_highside_trigger_chb_72MHz.c` | 72 MHz | `PLLN=288`, `PLLP=4` |
| `main_aes_uart_highside_trigger_chb_60MHz.c` | 60 MHz | `PLLN=240`, `PLLP=4` |
| `main_aes_uart_highside_trigger_chb_48MHz.c` | 48 MHz | `PLLN=192`, `PLLP=4` |
| `main_aes_uart_highside_trigger_chb_24MHz.c` | 24 MHz | `PLLN=192`, `PLLP=8` |

La config actuelle correspond au profil 84 MHz. Pour chaque point de mesure, copier le fichier voulu en `Core/Src/main.c`, recompiler/flasher, puis lancer une capture avec les memes reglages Pico.

## Campagne courte recommandee

Commencer par 1000 traces par frequence pour choisir les meilleurs candidats :

```bash
python acquire_aes_uart_highside_chb_trigger.py \
  --serial-port /dev/ttyACM0 \
  --n-traces 1000 \
  --num-samples 8000 \
  --pre-trigger 800 \
  --timebase 8 \
  --supply-mv 3300 \
  --clock-mhz 84 \
  --output freq_84MHz_1k.npz
```

Repeter en changeant seulement `--clock-mhz` et `--output` apres avoir flashe le firmware correspondant :

```bash
python acquire_aes_uart_highside_chb_trigger.py --clock-mhz 72 --output freq_72MHz_1k.npz
python acquire_aes_uart_highside_chb_trigger.py --clock-mhz 60 --output freq_60MHz_1k.npz
python acquire_aes_uart_highside_chb_trigger.py --clock-mhz 48 --output freq_48MHz_1k.npz
python acquire_aes_uart_highside_chb_trigger.py --clock-mhz 24 --output freq_24MHz_1k.npz
```

Garder les autres arguments identiques entre les frequences.

## Analyse comparative

```bash
python analyze_frequency_campaign.py \
  freq_84MHz_1k.npz \
  freq_72MHz_1k.npz \
  freq_60MHz_1k.npz \
  freq_48MHz_1k.npz \
  freq_24MHz_1k.npz \
  --output-json frequency_campaign_summary.json
```

Colonnes importantes :

- `pre_noise_std_mv` : bruit avant le trigger. Plus bas est mieux.
- `mean_trace_peak_to_peak_mv` : amplitude moyenne visible. Plus haut aide, sauf si c'est du drift.
- `snr_hw_max` : SNR max selon le poids de Hamming de `SBox(P[0] xor K[0])`. Plus haut est mieux.
- `trigger_width_us` : duree AES approx vue via PB8. Elle doit augmenter quand la frequence baisse.

## Conseils de choix

- 84 MHz sert de reference.
- 60 ou 48 MHz sont souvent un bon compromis : trace plus etalee, UART encore stable, fenetre pas trop longue.
- 24 MHz est utile pour inspection visuelle, mais peut demander plus de samples et peut changer le rapport bruit/fuite.
- Si une frequence semble bonne, refaire ensuite une campagne longue, par exemple 5k ou 20k traces, uniquement sur 2 ou 3 frequences.
