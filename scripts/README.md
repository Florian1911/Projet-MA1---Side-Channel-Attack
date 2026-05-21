# Scripts

Organisation des scripts Python et shell:

- `acquisition/` : capture PicoScope/STM32 et generation de plaintexts.
- `preprocessing/` : alignement, fusion, preparation et controle qualite.
- `analysis/` : diagnostics, SNR, POI, correlations et figures.
- `attacks/` : CPA, recuperation de cle, attaques blind et votes.
- `ml/` : entrainement et evaluation MLP/CNN/ASCAD.
- `campaigns/` : orchestration de campagnes completes.
- `misc/` : essais ponctuels.

Les scripts ont ete gardes proches de leur usage experimental; certains attendent d'etre lances depuis la racine du depot avec les chemins de datasets/resultats indiques dans le rapport ou les notes.
