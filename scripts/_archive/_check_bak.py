"""Check backup for _build_sector_cache."""
with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
idx = raw.find(b'_build_sector_cache')
if idx >= 0:
    start = max(0, idx - 50)
    end = min(len(raw), idx + 600)
    s = raw[start:end].decode('utf-8', errors='replace')
    print(f'Offset {idx}:')
    print(s)
