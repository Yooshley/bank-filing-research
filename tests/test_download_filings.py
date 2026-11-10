"""Offline checks for real-filing acquisition; no external requests are made."""
import csv
import io
import json
import runpy
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pymupdf
import requests

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'demos' / '01_download_filings.py'))


def pdf_bytes(text):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.row = {'file_name': 'RBC_2025_AIF.pdf', 'bank': 'RBC', 'fiscal_year': 2025,
                    'document_type': 'AIF', 'quarter': '', 'filing_date': '',
                    'source_url': 'https://www.rbc.com/example.pdf', 'language': 'en',
                    'source_page': 'https://www.rbc.com/', 'covers_q4': 'false'}
        self.payload = pdf_bytes('ROYAL BANK OF CANADA\nANNUAL INFORMATION FORM\nDecember 2, 2025')

    def test_year_boundary(self):
        self.assertEqual(MODULE['default_years'](date(2026, 10, 14)), [2024, 2025])
        self.assertEqual(MODULE['default_years'](date(2026, 10, 31)), [2025, 2026])

    def test_content_identity_and_non_pdf_guard(self):
        self.assertEqual(MODULE['validate_pdf'](self.payload, self.row)['number_of_pages'], 1)
        for content in [b'<html>error page</html>', pdf_bytes('Other Company Annual Information Form 2025'),
                        pdf_bytes('ROYAL BANK OF CANADA ANNUAL INFORMATION FORM 2023')]:
            with self.assertRaises(ValueError):
                MODULE['validate_pdf'](content, self.row)

    def test_cached_file_avoids_network(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            raw = Path(folder)
            (raw / self.row['file_name']).write_bytes(self.payload)
            with patch('requests.get') as get:
                result = MODULE['download_one'](self.row, raw)
                get.assert_not_called()
                self.assertEqual(result['status'], 'cached')

    def test_stream_download_and_rerun(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [self.payload[:100], self.payload[100:]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            raw = Path(folder)
            with patch('requests.get', return_value=response) as get:
                result = MODULE['download_one'](self.row, raw)
                self.assertEqual(result['status'], 'downloaded')
                self.assertEqual((raw / self.row['file_name']).read_bytes(), self.payload)
                self.assertFalse((raw / 'RBC_2025_AIF.pdf.part').exists())
                MODULE['download_one'](self.row, raw)
                self.assertEqual(get.call_count, 1)

    def test_permanent_http_error_not_retried(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        denied = requests.Response()
        denied.status_code = 403
        response.raise_for_status.side_effect = requests.HTTPError('Forbidden', response=denied)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch('requests.get', return_value=response) as get:
            with self.assertRaises(requests.HTTPError):
                MODULE['download_one'](self.row, Path(folder))
            self.assertEqual(get.call_count, 1)
            self.assertFalse((Path(folder) / self.row['file_name']).exists())

    def test_path_and_domain_constraints(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with self.assertRaises(ValueError):
                MODULE['raw_path'](Path(folder), '../outside.pdf')
            with patch('requests.get') as get:
                with self.assertRaises(ValueError):
                    MODULE['download_one']({**self.row, 'source_url': 'https://rbc.com.evil.test/file.pdf'}, Path(folder))
                get.assert_not_called()

    def test_manifest_preserves_known_date(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            base = Path(folder)
            raw = base / 'data' / 'raw'
            raw.mkdir(parents=True)
            (raw / self.row['file_name']).write_bytes(self.payload)
            catalog = base / 'catalog.json'
            catalog.write_text(json.dumps({'filings': [self.row]}))
            manifest = base / 'data' / 'filing_manifest.csv'
            with manifest.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=MODULE['MANIFEST_FIELDS'])
                writer.writeheader()
                writer.writerow({**self.row, 'filing_date': '2025-12-02'})
            with patch('requests.get') as get:
                report = MODULE['download_filings'](base, years=[2025], banks=['RBC'], catalog_path=catalog)
                get.assert_not_called()
            with manifest.open(newline='') as file:
                self.assertEqual(next(csv.DictReader(file))['filing_date'], '2025-12-02')
            self.assertEqual(report['successful'], 1)
            self.assertEqual(report['failed'], 0)


if __name__ == '__main__':
    unittest.main()
