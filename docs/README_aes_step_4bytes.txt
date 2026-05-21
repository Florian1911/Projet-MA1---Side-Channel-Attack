AES step 4 bytes (0..3)
=======================

New files:
- main_aes_step_4bytes_no_uart.c
- aes_step_4bytes_analysis.py

Protocol:
1) Flash main_aes_step_4bytes_no_uart.c
2) Acquire:
   python scripts/acquisition/acquire_no_uart.py --n-traces 5000 --num-samples 4000 --output aes_step_4b.npz --probe-att-trig 10 --trig-mv-probe 1500
3) Analyze bytes 0..3:
   python aes_step_4bytes_analysis.py --npz aes_step_4b.npz --n-traces 5000 --win-start 50 --win-end 1000 --out aes_step_4b_analysis.png

Notes:
- This is a harder setting than single-byte step.
- You may need 10k+ traces and/or a tighter window around each byte region.
