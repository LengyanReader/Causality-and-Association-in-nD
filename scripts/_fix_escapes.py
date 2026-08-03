"""One-shot fix: remove spurious backslashes before Unicode chars in generate.py."""
import re

TARGET = __import__('pathlib').Path(__file__).resolve().parents[1] / "docs" / "demo" / "generate.py"
txt = open(TARGET, 'r', encoding='utf-8').read()
fixes = 0
BS = chr(0x5C)  # backslash

CODEPOINTS = [
    0x201C, 0x201D, 0x2018, 0x2019, 0x2014, 0x2013,  # quotes/dashes
    0x00D7,  # multiplication sign
    0x220F, 0x2211, 0x2202,  # product, sum, partial
    0x03C4, 0x03C8, 0x03BC, 0x03B8, 0x03B7, 0x03C3, 0x03B5,  # Greek
    0x2265, 0x2248, 0x2260,  # comparison
    0x2713, 0x2717,  # check/x
    0x2193, 0x2192, 0x2190,  # arrows
    0x00B2,  # superscript 2
    0x22A5, 0x2205, 0x2261,  # math symbols
    0x00EA, 0x00EE, 0x00FB, 0x00E2, 0x00F4, 0x0302,  # accented letters
]

for cp in CODEPOINTS:
    ch = chr(cp)
    bs_ch = BS + ch
    n = txt.count(bs_ch)
    if n > 0:
        txt = txt.replace(bs_ch, ch)
        fixes += n

# Check for remaining literal \uXXXX patterns
pat = re.compile(r'\\u[0-9a-fA-F]{4}')
remaining = set(m.group() for m in pat.finditer(txt))
if remaining:
    print(f'WARNING: {len(remaining)} unique \\uXXXX patterns still in source')
    for u in sorted(remaining):
        print(f'  {u}: {txt.count(u)} occurrences')
else:
    print('No remaining Unicode escape patterns')

open(TARGET, 'w', encoding='utf-8').write(txt)
print(f'Total fixes: {fixes}')
