# Projet MA1 - Side Channel Attack

Code et resultats associes au rapport:

**Side Channel Attack on AES Implementation Using Power Analysis on STM32 Microcontroller**

Ce depot regroupe les scripts Python, firmwares STM32 et notes experimentales utilises pour etudier des attaques par analyse de consommation sur AES-128. Le projet couvre l'acquisition de traces avec PicoScope, la synchronisation, les attaques CPA, le choix de points d'interet, ainsi que des essais avec modeles MLP/CNN.

## Contenu principal

- `main*.c` : variantes de firmware STM32 pour AES, trigger GPIO, UART, mesures low-side/high-side et tests de fuite.
- `acquire*.py` : scripts d'acquisition PicoScope/STM32.
- `cpa_*.py`, `*_cpa*.py`, `recover_*.py` : scripts d'analyse CPA et recuperation de cle.
- `train_*.py`, `*_mlp*.py`, `*_cnn*.py`, `ascad_*.py` : essais de modeles machine learning/deep learning.
- `base_saine_pico5000/` : base minimale propre pour refaire une acquisition AES no-UART avec PicoScope 5000A.
- `aes_uart_highside_chb_trigger/` : montage high-side UART avec trigger sur canal PicoScope.
- `report_results/` : elements de resultats destines au rapport.

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
2. Lancer un script d'acquisition (`acquire_dataset_no_uart.py`, `acquire_dataset_highside.py` ou le dossier `base_saine_pico5000/`).
3. Verifier l'alignement et la qualite des traces (`fix_alignment.py`, `quality_check.py`, scripts SNR).
4. Executer une attaque CPA ou ML (`cpa_attack.py`, `full_aes16_pipeline.py`, `train_mlp_numpy.py`, etc.).
5. Comparer les rangs/correlations et produire les figures du rapport.

## Lien a citer

Depot GitHub:

```text
https://github.com/Florian1911/Projet-MA1---Side-Channel-Attack
```
