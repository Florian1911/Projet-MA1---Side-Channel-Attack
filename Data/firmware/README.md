# Firmware STM32

Variantes de firmware C utilisees pendant le projet:

- AES avec trigger GPIO.
- AES avec UART.
- mesures no-UART, low-side et high-side.
- tests de fuite controles (`xor`, `sbox-only`, source leak).
- essais RSA SPA.

Les fichiers `main_*.c` sont prevus pour etre copies ou adaptes comme `Core/Src/main.c` dans le projet STM32.
