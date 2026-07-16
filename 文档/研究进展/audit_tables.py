import re
from pathlib import Path
base = Path(__file__).resolve().parent
files = {
    'overview': base/'research_progress_overview.tex',
    'ton': base/'pi_jwm_ton_draft_zh.tex',
}

def extract_tables(text):
    # table or table* blocks
    patt = re.compile(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', re.S)
    out=[]
    for m in patt.finditer(text):
        block=m.group(0)
        cap=re.search(r'\\caption\{(.*?)\}', block, re.S)
        tab=re.search(r'\\begin\{tabular\}\{[^\n]*\}\s*(.*?)\\end\{tabular\}', block, re.S)
        if cap and tab:
            caption=re.sub(r'\s+',' ',cap.group(1)).strip()
            body=re.sub(r'\s+',' ',tab.group(1)).strip()
            body=body.replace(' ', '')
            out.append((caption, body, block))
    return out
texts={k:p.read_text(encoding='utf-8') for k,p in files.items()}
tables={k:extract_tables(v) for k,v in texts.items()}
orig=[t for t in tables['overview']]
ton=[t for t in tables['ton'] if not t[0].startswith('PI-JWM 系统模型中的主要符号')]
print('overview_tables', len(orig))
print('ton_tables_excluding_notation', len(ton))
# compare by caption exact
orig_map={c:b for c,b,_ in orig}
ton_map={c:b for c,b,_ in ton}
missing=[c for c in orig_map if c not in ton_map]
extra=[c for c in ton_map if c not in orig_map]
changed=[]
for c,b in orig_map.items():
    if c in ton_map and b != ton_map[c]:
        changed.append(c)
print('missing', len(missing))
for c in missing: print('MISSING:', c)
print('extra', len(extra))
for c in extra: print('EXTRA:', c)
print('changed', len(changed))
for c in changed: print('CHANGED:', c)
# numeric token multiset comparison across all original tables vs ton copied tables
num_re=re.compile(r'(?<![A-Za-z])(?:\d+\.\d+|\d+)(?![A-Za-z])')
orig_nums=num_re.findall('\n'.join(b for _,b,_ in orig))
ton_nums=num_re.findall('\n'.join(b for _,b,_ in ton))
from collections import Counter
co,ct=Counter(orig_nums),Counter(ton_nums)
missing_nums=co-ct
extra_nums=ct-co
print('numeric_tokens_overview', sum(co.values()), 'ton', sum(ct.values()))
print('missing_numeric_tokens', dict(missing_nums.most_common(20)))
print('extra_numeric_tokens', dict(extra_nums.most_common(20)))
