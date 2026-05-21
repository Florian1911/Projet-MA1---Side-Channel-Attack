# Projet MA1 - Side Channel Attack

Code et resultats associes au rapport:

**Side Channel Attack on AES Implementation Using Power Analysis on STM32 Microcontroller**

Ce depot regroupe les scripts Python, firmwares STM32 et notes experimentales utilises pour etudier des attaques par analyse de consommation sur AES-128. Le projet couvre l'acquisition de traces avec PicoScope, la synchronisation, les attaques CPA, le choix de points d'interet, ainsi que des essais avec modeles MLP/CNN.

## Organisation

- `firmware/` : variantes de firmware STM32 pour AES, trigger GPIO, UART, mesures low-side/high-side et tests de fuite.
- `scripts/acquisition/` : scripts d'acquisition PicoScope/STM32.
- `scripts/preprocessing/` : alignement, preparation et controle qualite des traces.
- `scripts/analysis/` : analyses SNR/POI/correlation, diagnostics et generation de figures.
- `scripts/attacks/` : CPA, recuperation de cle, attaques blind et votes.
- `scripts/ml/` : essais MLP/CNN/ASCAD et pipelines de transfert.
- `scripts/campaigns/` : scripts d'orchestration de campagnes experimentales.
- `setups/` : setups complets et documentes (`base_saine_pico5000`, `aes_uart_highside_chb_trigger`).
- `docs/` : notes experimentales, runbooks et figures explicatives.
- `results/` : petits resumes, figures et campagnes deja versionnes pour tracer les resultats.

Les datasets complets et captures brutes peuvent etre volumineux. Ils ne sont pas tous versionnes; le depot garde surtout le code, les petits resumes et les figures utiles.

## Installation Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pour l'acquisition PicoScope, installer aussi les bibliotheques systeme PicoSDK du constructeur.

## Exemple de flux experimental

1. Flasher une variante de firmware STM32 adaptee au montage.
2. Lancer un script d'acquisition (`scripts/acquisition/acquire_dataset_no_uart.py`, `scripts/acquisition/acquire_dataset_highside.py` ou le dossier `setups/base_saine_pico5000/`).
3. Verifier l'alignement et la qualite des traces (`scripts/preprocessing/fix_alignment.py`, `scripts/preprocessing/quality_check.py`, scripts SNR).
4. Executer une attaque CPA ou ML (`scripts/attacks/cpa_attack.py`, `scripts/campaigns/full_aes16_pipeline.py`, `scripts/ml/train_mlp_numpy.py`, etc.).
5. Comparer les rangs/correlations et produire les figures du rapport.

## Lien a citer

Depot GitHub:

```text
https://github.com/Florian1911/Projet-MA1---Side-Channel-Attack
```
