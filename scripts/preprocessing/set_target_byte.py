#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Set TARGET_BYTE define in firmware C file')
    ap.add_argument('--file', default='main_xor_byte_no_uart.c')
    ap.add_argument('--byte', type=int, required=True)
    args = ap.parse_args()

    if not (0 <= args.byte < 16):
        raise ValueError('--byte must be in [0,15]')

    p = Path(args.file)
    txt = p.read_text()
    new, n = re.subn(r'(#define\s+TARGET_BYTE\s+)\d+u', rf'\g<1>{args.byte}u', txt)
    if n != 1:
        raise RuntimeError('Could not find unique TARGET_BYTE define')
    p.write_text(new)
    print(f'[OK] {args.file}: TARGET_BYTE={args.byte}')


if __name__ == '__main__':
    main()
