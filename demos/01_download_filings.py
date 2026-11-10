"""Download real Canadian bank filings for Module 4.

Adapted conceptually from the instructor demo. Its simulated SEDAR+ search is
replaced by a maintained catalog of official investor-relations PDF links.
Run: python demos/01_download_filings.py
Notebook 1 calls download_filings() before parsing and chunking.
Dependencies: requests, pymupdf. No API key is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / 'data' / 'filing_sources.json'
MANIFEST_FIELDS = ['file_name', 'bank', 'fiscal_year', 'document_type', 'quarter',
                   'filing_date', 'source_url', 'language', 'source_page', 'covers_q4']
BANK_IDENTIFIERS = {'RBC': ['royal bank of canada', 'rbc'],
                    'TD': ['toronto-dominion', 'td bank group'],
                    'BMO': ['bank of montreal', 'bmo financial group'],
                    'BNS': ['bank of nova scotia', 'scotiabank'],
                    'CIBC': ['canadian imperial bank of commerce', 'cibc']}


def default_years(as_of=None):
    as_of = as_of or date.today()
    last = as_of.year if as_of >= date(as_of.year, 10, 31) else as_of.year - 1
    return [last - 1, last]


def raw_path(raw, file_name):
    path = (raw / file_name).resolve()
    if not path.is_relative_to(raw.resolve()) or path.suffix.lower() != '.pdf':
        raise ValueError('Catalog file_name must be a PDF inside data/raw')
    return path


def validate_pdf(payload, row):
    if not payload[:1024].lstrip().startswith(b'%PDF-'):
        raise ValueError('Source returned non-PDF content (possibly a landing/error page)')
    with pymupdf.open(stream=payload, filetype='pdf') as document:
        if not len(document) or document.needs_pass:
            raise ValueError('Empty or encrypted PDF')
        preview = ' '.join(document[i].get_text() for i in range(min(10, len(document))))
        preview = ' '.join(preview.lower().split())
        if not any(name in preview for name in BANK_IDENTIFIERS[row['bank']]):
            raise ValueError('PDF does not identify the expected bank in its opening pages')
        if str(row['fiscal_year']) not in preview:
            raise ValueError('PDF opening pages do not contain the expected fiscal year')
        if row['document_type'] == 'AIF' and 'annual information form' not in preview:
            raise ValueError('PDF is not identifiable as an Annual Information Form')
        if row['document_type'] == 'annual_report' and 'annual report' not in preview:
            raise ValueError('PDF is not identifiable as an annual report')
        if row['document_type'] == 'quarterly':
            quarter_words = {'1': 'first', '2': 'second', '3': 'third', '4': 'fourth'}
            word = quarter_words[str(row['quarter'])]
            if f'{word} quarter' not in preview and f'q{row["quarter"]}' not in preview:
                raise ValueError('PDF does not identify the expected quarter in its opening pages')
        return {'number_of_pages': len(document),
                'file_sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def download_one(row, raw, attempts=2, max_bytes=100 * 1024 * 1024):
    path = raw_path(raw, row['file_name'])
    if path.exists():
        # Validate cached files; never assume a .pdf extension means valid PDF.
        info = validate_pdf(path.read_bytes(), row)
        return {**row, **info, 'status': 'cached'}
    host = (urlparse(row['source_url']).hostname or '').lower()
    allowed = {'RBC': 'rbc.com', 'TD': 'td.com', 'BMO': 'bmo.com',
               'BNS': 'scotiabank.com', 'CIBC': 'cibc.com'}[row['bank']]
    if urlparse(row['source_url']).scheme != 'https' or not (host == allowed or host.endswith('.' + allowed)):
        raise ValueError('Built-in download source must be the bank official HTTPS domain')
    for attempt in range(attempts):
        try:
            with requests.get(row['source_url'], stream=True, timeout=(10, 35),
                              headers={'User-Agent': 'Module4CanadianBankResearch/1.0',
                                       'Accept': 'application/pdf'}) as response:
                response.raise_for_status()
                payload = bytearray()
                for block in response.iter_content(1024 * 1024):
                    payload.extend(block)
                    if len(payload) > max_bytes:
                        raise ValueError('PDF exceeds 100 MiB download limit')
            info = validate_pdf(bytes(payload), row)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pdf.part')
            temporary.write_bytes(payload)
            temporary.replace(path)
            return {**row, **info, 'status': 'downloaded'}
        except requests.RequestException as exc:
            # Do not retry permanent HTTP errors, such as forbidden or missing links.
            status = exc.response.status_code if exc.response is not None else None
            if attempt + 1 == attempts or (status is not None and status not in {429, 500, 502, 503, 504}):
                raise
            time.sleep(1 + attempt)
    raise RuntimeError('Download attempts exhausted')


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def download_filings(base=PROJECT_ROOT, years=None, banks=None, include_quarterly=True,
                     workers=3, catalog_path=CATALOG):
    base = Path(base).resolve()
    years = sorted(set(years or default_years()))
    banks = set(banks or BANK_IDENTIFIERS)
    if not banks.issubset(BANK_IDENTIFIERS):
        raise ValueError(f'Unknown bank codes: {banks - BANK_IDENTIFIERS.keys()}')
    catalog = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    available_years = {int(row['fiscal_year']) for row in catalog['filings']}
    if not set(years).issubset(available_years):
        raise ValueError(f'Catalog covers {sorted(available_years)}. Update filing_sources.json with verified links for {years}.')
    selected = [row.copy() for row in catalog['filings']
                if int(row['fiscal_year']) in years and row['bank'] in banks
                and (include_quarterly or row['document_type'] != 'quarterly')]
    if not selected:
        raise ValueError('No catalog entries match requested banks/years/types')
    raw, manifest = base / 'data' / 'raw', base / 'data' / 'filing_manifest.csv'
    raw.mkdir(parents=True, exist_ok=True)
    existing = {}
    if manifest.exists():
        with manifest.open(encoding='utf-8-sig', newline='') as source:
            for row in csv.DictReader(source):
                if row['file_name'] in existing:
                    raise ValueError('Duplicate filenames in existing manifest')
                existing[row['file_name']] = row
    for row in selected:
        # Retain user-supplied dates and URL overrides, without losing provenance.
        previous = existing.get(row['file_name'], {})
        if previous:
            for key in ['bank', 'fiscal_year', 'document_type', 'quarter']:
                if str(previous.get(key, '')) != str(row.get(key, '')):
                    raise ValueError(f'Manifest metadata conflicts with catalog for {row["file_name"]}: {key}')
        row.update({key: value for key, value in previous.items() if value != ''})
        existing[row['file_name']] = row
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(existing[name] for name in sorted(existing))
    atomic_text(manifest, stream.getvalue())
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as pool:
        futures = {pool.submit(download_one, row, raw): row for row in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f'{result["status"]}: {row["file_name"]} ({result["number_of_pages"]} pages)', flush=True)
            except Exception as exc:
                error = {'file': str(raw / row['file_name']), 'source_url': row['source_url'], 'error': str(exc)}
                errors.append(error)
                print(f'FAILED: {row["file_name"]}: {exc}', flush=True)
    report = {'years': years, 'banks': sorted(banks), 'requested': len(selected),
              'successful': len(results), 'failed': len(errors), 'errors': errors,
              'results': sorted(results, key=lambda r: r['file_name']),
              'retrieved_at': datetime.now(timezone.utc).isoformat(),
              'date_note': 'filing_date is blank unless independently known; download time is not the filing date.'}
    atomic_text(base / 'data' / 'download_report.json', json.dumps(report, indent=2))
    print(f'Download summary: {len(results)}/{len(selected)} validated PDFs; {len(errors)} failures', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--banks', choices=list(BANK_IDENTIFIERS), nargs='+')
    parser.add_argument('--annual-only', action='store_true', help='Annual reports and AIFs only')
    args = parser.parse_args()
    report = download_filings(args.base, args.years, args.banks, not args.annual_only)
    return 1 if report['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
