import re
from pathlib import Path
base=Path(__file__).resolve().parent
paths={'overview':base/'research_progress_overview.tex','ton':base/'pi_jwm_ton_draft_zh.tex'}

def caption_of(block):
    m=re.search(r'\\(?:widecaption|caption)\{(.*?)\}', block, re.S)
    return re.sub(r'\s+',' ',m.group(1)).strip() if m else ''

def extract(text, skip_notation=False):
    blocks=re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', text, re.S)
    out=[]
    for block in blocks:
        caption=caption_of(block)
        if skip_notation and caption.startswith('PI-JWM 系统模型中的主要符号'):
            continue
        tab=re.search(r'\\begin\{tabular\}\{[^\n]*\}\s*(.*?)\\end\{tabular\}', block, re.S)
        if not tab:
            continue
        body=tab.group(1)
        body=re.sub(r'\\(toprule|midrule|bottomrule)', '', body)
        body=re.sub(r'\s+', '', body)
        out.append((caption, body))
    return out
orig=extract(paths['overview'].read_text(encoding='utf-8'))
ton=extract(paths['ton'].read_text(encoding='utf-8'), True)
print('table_count', len(orig), len(ton))
num_re=re.compile(r'(?<![A-Za-z])(?:\d+\.\d+|\d+)(?![A-Za-z])')
all_ok=True
for co,bo in orig:
    match=[bt for ct,bt in ton if ct==co]
    if not match:
        print('MISSING_TABLE', co); all_ok=False; continue
    no=num_re.findall(bo); nt=num_re.findall(match[0])
    if no!=nt:
        print('NUM_DIFF', co); print(no); print(nt); all_ok=False
print('ordered_numeric_tokens_by_caption_match', all_ok)
