AES step byte0 (new files, archive-friendly)
============================================

Files:
- main_aes_step_byte0_no_uart.c
- aes_step_byte0_analysis.py

Protocol:
1) Flash main_aes_step_byte0_no_uart.c
2) Acquire dataset:
   python scripts/acquisition/acquire_no_uart.py --n-traces 5000 --num-samples 4000 --output aes_step_b00.npz --probe-att-trig 10 --trig-mv-probe 1500
3) Analyze:
   python aes_step_byte0_analysis.py --npz aes_step_b00.npz --byte 0 --true-key 0x2b --win-start 100 --win-end 260 --n-traces 5000 --out aes_step_b00_analysis.png

Interpretation:
- rank close to 1 means this simplified AES-like target is exploitable.
- If rank fluctuates, increase traces (e.g., 10000) and keep same window.
