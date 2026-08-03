"""Universal fix: remove ALL backslash+nonASCII patterns from generate.py."""
import re
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "docs" / "demo" / "generate.py"
txt = open(TARGET, 'r', encoding='utf-8').read()
bs = chr(0x5C)

# Match: literal backslash followed by any non-ASCII character (codepoint > 127)
# KaTeX commands are ALWAYS backslash + ASCII letter, so removing backslash
# before any non-ASCII character is safe.
pattern = re.compile(rb'\\[\x80-\xFF]'.decode('raw_unicode_escape'))
# Actually let me do it character by character
fixes = 0
i = 0
result = []
while i < len(txt):
    if txt[i] == bs and i + 1 < len(txt) and ord(txt[i+1]) > 127:
        # Skip the backslash, keep the non-ASCII char
        result.append(txt[i+1])
        fixes += 1
        i += 2
    else:
        result.append(txt[i])
        i += 1

if fixes > 0:
    out = ''.join(result)
    open(TARGET, 'w', encoding='utf-8').write(out)
    print(f'Fixed {fixes} backslash+nonASCII patterns')
else:
    print('No patterns found — file appears clean')
