"""Restore the small pinned official source snapshot. Never fetch data/weights."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / 'code/reference/V2U4Real.SOURCE.json').read_text(encoding='utf-8'))
    target = ROOT / 'code/reference/V2U4Real'
    for name, expected in manifest['files'].items():
        path = target / name
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError('source path escape')
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/VjiaLi/V2U4Real/{manifest['commit']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(150000)
        # Git LF and the recorded snapshot have identical text bytes.
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'source fingerprint mismatch: {name}')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f'VERIFIED {name}')


if __name__ == '__main__':
    main()
