"""Turn the selected source resumes into anonymized plain text.

Input : data/raw/resume_id_map.csv  (git-ignored: maps R01..R10 to source files
        and lists name/handle terms to scrub), source PDFs/DOCX in data/raw/
Output: data/resumes/R01.txt .. R10.txt

Removes names, emails, phone numbers, URLs/handles, street addresses and ZIP
codes. City/state is kept because location preferences matter for matching.
Always review the output by eye; this is a best-effort scrub, not a guarantee.
Needs `pdftotext` (poppler) on PATH. Run from the repository root:
    python3 -m shared.prep_resumes
"""
import csv
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/resumes'

EMAIL = re.compile(r'[\w.+-]+@[\w-]+(\.[\w-]+)+\w?', re.I)
PHONE = re.compile(r'(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}')
URL = re.compile(r'(https?://)?(www\.)?([\w-]+\.)*(linkedin\.com|github\.com|in/)[/\w.-]*|https?://\S+|\b[\w-]+\.(com|net|org|io|dev|me)\b(/\S*)?', re.I)
STREET = re.compile(r'\b\d{2,6}\s+[A-Z][\w.]*(\s+[A-Z][\w.]*)*\s+(Road|Rd|Street|St|Avenue|Ave|Drive|Dr|Lane|Ln|Blvd|Court|Ct|Way)\b\.?,?')
ZIP = re.compile(r'(?<=[A-Z]{2})\s+\d{5}(-\d{4})?\b')
LABELS = re.compile(r'\b(LinkedIn( Profile)?|GitHub|Google Scholor|Email|E-mail|Cell|Mobile|Phone)\s*:', re.I)
JUNK = re.compile(r'[‪-‮​-#§ï❖]')


def extract(path):
    if path.suffix.lower() == '.docx':
        xml = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
        return re.sub(r'<[^>]+>', '', xml.replace('</w:p>', '\n'))
    return subprocess.run(['pdftotext', '-layout', str(path), '-'], check=True,
                          capture_output=True, text=True).stdout


def scrub(text, terms):
    text = JUNK.sub('', text)
    for pat in (EMAIL, URL, PHONE, STREET):
        text = pat.sub(' ', text)
    text = ZIP.sub('', text)
    for term in sorted(terms, key=len, reverse=True):
        text = re.sub(r'(?<![\w])' + re.escape(term) + r"(?![\w])('s)?", ' ', text, flags=re.I)
    text = LABELS.sub(' ', text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r' {3,}', '   ', line).rstrip()
        line = re.sub(r'^[\s|•,·*-]+$', '', line)
        line = re.sub(r'(\s*[|•·,]\s*){2,}', ' | ', line)
        lines.append(line)
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    return text + '\n'


def main():
    rows = list(csv.DictReader(open(RAW / 'resume_id_map.csv', encoding='utf-8')))
    for r in rows:
        terms = [t.strip() for t in r['scrub_terms'].split(';') if t.strip()]
        text = scrub(extract(RAW / r['source_file']), terms)
        leftovers = [t for t in terms if re.search(re.escape(t), text, re.I)]
        (OUT / f"{r['resume_id']}.txt").write_text(text, encoding='utf-8')
        print(r['resume_id'], len(text), 'chars', 'LEFTOVER: ' + ', '.join(leftovers) if leftovers else '')


if __name__ == '__main__':
    main()
