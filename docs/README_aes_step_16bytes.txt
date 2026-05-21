AES step 16 bytes
=================

New files:
- main_aes_step_16bytes_no_uart.c
- aes_step_16bytes_analysis.py

Protocol:
1) Flash main_aes_step_16bytes_no_uart.c
2) Acquire first test set (10k):
   python scripts/acquisition/acquire_no_uart.py --n-traces 10000 --num-samples 4000 --output aes_step_16b_10k.npz --probe-att-trig 10 --trig-mv-probe 1500
3) Analyze:
   python aes_step_16bytes_analysis.py --npz aes_step_16b_10k.npz --n-traces 10000 --win-start 50 --win-end 2000 --out-prefix aes_step_16b_10k

Decision guideline:
- If >= 4 bytes have rank<=5, try 20k traces.
- Otherwise tighten window or increase repeats per byte in firmware.
