"""
Tests for retrieval-augmented (grounded) text tagging, including multi-file
reference datasets and the rag_projects.json registry.

Three tiers, gated by environment variables so a plain `manage.py test`
stays fast and fully offline:

  python manage.py test tagger_app.test_retrieval
      Chunking, registry CRUD, config parsing, and staleness-detection
      logic only. No network, runs in well under a second.

  RAG_LIVE_TESTS=1 python manage.py test tagger_app.test_retrieval -v 2
      The above, plus small real-Ollama checks (~1 min): build a tiny
      index (including a genuine multi-file index), verify retrieval
      quality/shape, verify edge cases (no embedding model configured,
      empty reference file, empty query).

  RAG_STRESS_FULL=1 python manage.py test tagger_app.test_retrieval -v 2
      The above, plus the real stress test: embeds the FULL real
      glycemicindex_data.csv reference dataset (4274 rows of published
      glycemic-index research) end to end, then runs an actual tagging
      job (via the Django test client, exercising upload -> connection
      -> build-index -> define-columns -> tagging -> results) against a
      real subcategory dataset, grounded on that 4274-row index. Takes
      several minutes on CPU/MPS with a 1B model.

Requires a local Ollama server (defaults assume `llama3.2:1b` is pulled,
used for both chat and embeddings — override via RAG_TEST_HOST/PORT/
RAG_TEST_MODEL/RAG_TEST_EMBED_MODEL env vars).

Measured on this machine, embedding model choice matters a lot — pull a
dedicated embedding model if you have the disk space (274MB):

    ollama pull nomic-embed-text
    RAG_STRESS_FULL=1 RAG_TEST_EMBED_MODEL=nomic-embed-text \
        python manage.py test tagger_app.test_retrieval.StressFullDatasetTests -v 2

  Full 4274-row glycemicindex_data.csv build, same machine:
                        llama3.2:1b        nomic-embed-text
    embed throughput    11.1 rows/sec      76.5 rows/sec   (~6.9x faster)
    vector dims         2048               768             (smaller index)
    "Chocolate mudcake" query self-match   not in top-5     #1 (score 0.73)
    "White bread" top score                0.44             0.70
    "Instant noodles" top score            0.43             0.73

  llama3.2:1b is a chat model repurposed for embeddings and works, but
  clusters chunks by shared boilerplate (citation/manufacturer text)
  rather than the distinguishing food name. nomic-embed-text is smaller,
  faster, and noticeably more discriminative — the better default for
  anyone who can spare the one-time download.
"""
import io
import json
import os
import shutil
import tempfile
import time
import unittest
import urllib.request
from unittest import mock

import pandas as pd
from django.conf import settings
from django.test import Client, TestCase, override_settings

from . import utils

RAG_LIVE_TESTS = os.environ.get('RAG_LIVE_TESTS') == '1' or os.environ.get('RAG_STRESS_FULL') == '1'
RAG_STRESS_FULL = os.environ.get('RAG_STRESS_FULL') == '1'

OLLAMA_HOST = os.environ.get('RAG_TEST_HOST', 'localhost')
OLLAMA_PORT = os.environ.get('RAG_TEST_PORT', '11434')
CHAT_MODEL  = os.environ.get('RAG_TEST_MODEL', 'llama3.2:1b')
EMBED_MODEL = os.environ.get('RAG_TEST_EMBED_MODEL', 'llama3.2:1b')

REPO_MEDIA = settings.MEDIA_ROOT
GLYCEMIC_CSV     = os.path.join(REPO_MEDIA, 'glycemicindex_data.csv')
SUBCATEGORY_CSV  = os.path.join(REPO_MEDIA, 'Filtered_SubCategory_foods_GI_GL_tagged.csv')


def _ollama_reachable():
    try:
        urllib.request.urlopen(f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags', timeout=3)
        return True
    except Exception:
        return False


def _pulled_models():
    try:
        with urllib.request.urlopen(f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags', timeout=3) as resp:
            data = json.loads(resp.read())
        return {m['name'] for m in data.get('models', [])}
    except Exception:
        return set()


def make_minimal_pdf(pages_text):
    """Hand-rolled single/multi-page PDF — no reportlab dependency. Builds
    a minimal-but-valid object/xref structure so pypdf's PdfReader can
    parse it and extract_text() each page correctly."""
    n_pages = len(pages_text)
    page_ids    = list(range(3, 3 + n_pages))
    content_ids = list(range(3 + n_pages, 3 + 2 * n_pages))
    font_id     = 3 + 2 * n_pages

    objects = [
        (1, b'<< /Type /Catalog /Pages 2 0 R >>'),
        (2, f'<< /Type /Pages /Kids [{" ".join(f"{i} 0 R" for i in page_ids)}] '
            f'/Count {n_pages} >>'.encode()),
    ]
    for pid, cid in zip(page_ids, content_ids):
        objects.append((pid, (
            f'<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> '
            f'/MediaBox [0 0 612 792] /Contents {cid} 0 R >>'
        ).encode()))
    for cid, text in zip(content_ids, pages_text):
        stream = f'BT /F1 18 Tf 72 700 Td ({text}) Tj ET'.encode()
        objects.append((cid, b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream'))
    objects.append((font_id, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'))
    objects.sort(key=lambda o: o[0])

    buf = io.BytesIO()
    buf.write(b'%PDF-1.4\n')
    offsets = {}
    for oid, body in objects:
        offsets[oid] = buf.tell()
        buf.write(f'{oid} 0 obj\n'.encode() + body + b'\nendobj\n')
    xref_start = buf.tell()
    max_id = max(offsets) + 1
    buf.write(f'xref\n0 {max_id}\n'.encode())
    buf.write(b'0000000000 65535 f \n')
    for i in range(1, max_id):
        buf.write(f'{offsets[i]:010d} 00000 n \n'.encode())
    buf.write(f'trailer\n<< /Size {max_id} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF'.encode())
    return buf.getvalue()


class _IsolatedRegistryMixin:
    """Points utils' PROJECTS_CSV/CONNECTIONS_CSV/STATS_CSV/RAG_PROJECTS_JSON
    and Django's MEDIA_ROOT at a throwaway temp directory for the test's
    duration, so live tests never touch the real app's projects.csv/
    connections.csv/rag_projects.json/media on this machine."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='odt_rag_test_')
        self._patches = [
            mock.patch.object(utils, 'PROJECTS_CSV', os.path.join(self.tmp_dir, 'projects.csv')),
            mock.patch.object(utils, 'CONNECTIONS_CSV', os.path.join(self.tmp_dir, 'connections.csv')),
            mock.patch.object(utils, 'STATS_CSV', os.path.join(self.tmp_dir, 'stats.csv')),
            mock.patch.object(utils, 'RAG_PROJECTS_JSON', os.path.join(self.tmp_dir, 'rag_projects.json')),
        ]
        for p in self._patches:
            p.start()
        self._settings_override = override_settings(MEDIA_ROOT=os.path.join(self.tmp_dir, 'media'))
        self._settings_override.enable()
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        utils._reference_index_cache.clear()

    def tearDown(self):
        self._settings_override.disable()
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        utils._reference_index_cache.clear()
        super().tearDown()

    def make_project(self, csv_path, reference_paths=None):
        """reference_paths: list of source file paths to attach (any
        number — 0, 1, or many), each copied into the project and
        registered via add_reference_file, mirroring what upload_file_view
        / add_reference_files_view do."""
        project_id = 'test-' + os.urandom(4).hex()
        project_dir = os.path.join(settings.MEDIA_ROOT, project_id)
        os.makedirs(project_dir, exist_ok=True)
        local_csv = os.path.join(project_dir, os.path.basename(csv_path))
        shutil.copy(csv_path, local_csv)
        utils.save_project(project_id, 'test project', local_csv)
        for ref_path in (reference_paths or []):
            filename = os.path.basename(ref_path)
            local_ref = os.path.join(project_dir, filename)
            shutil.copy(ref_path, local_ref)
            file_type = os.path.splitext(local_ref)[1].lower().lstrip('.')
            utils.add_reference_file(project_id, filename, local_ref, file_type, os.path.getsize(local_ref))
        return project_id


# ─── Tier 1: pure logic, no network, always run ──────────────────────────────

class ChunkingTests(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='odt_chunk_test_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_structured_csv_one_chunk_per_row(self):
        path = os.path.join(self.tmp_dir, 'ref.csv')
        pd.DataFrame({'Food': ['Apple', 'Banana', 'Bread'], 'Sodium': [1, 0, 490]}).to_csv(path, index=False)
        chunks = utils.chunk_structured_csv(path)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]['source'], 'ref.csv — row 0')
        self.assertEqual(chunks[0]['file'], 'ref.csv')
        self.assertIn('Food: Apple', chunks[0]['text'])
        self.assertIn('Sodium: 1', chunks[0]['text'])
        self.assertEqual(chunks[0]['fields']['Food'], 'Apple')

    def test_structured_csv_handles_missing_values(self):
        path = os.path.join(self.tmp_dir, 'ref.csv')
        pd.DataFrame({'Food': ['Apple', None], 'Note': [None, 'x']}).to_csv(path, index=False)
        chunks = utils.chunk_structured_csv(path)
        self.assertEqual(len(chunks), 2)
        # NaN fields render as '' rather than the literal string 'nan'.
        self.assertEqual(chunks[0]['fields']['Note'], '')
        self.assertEqual(chunks[1]['fields']['Food'], '')

    def test_unstructured_windowing_covers_full_text_with_overlap(self):
        text = 'word ' * 1000  # ~5000 chars, several windows at size 800
        path = os.path.join(self.tmp_dir, 'doc.txt')
        with open(path, 'w') as f:
            f.write(text)
        chunks = utils.chunk_unstructured_text(path)
        self.assertGreater(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertLessEqual(len(c['text']), utils.REFERENCE_CHUNK_SIZE)
            self.assertEqual(c['file'], 'doc.txt')
            # Multiple windows from one file are disambiguated by index —
            # otherwise every chunk from doc.txt would look identical in
            # the _sources audit column.
            self.assertEqual(c['source'], f'doc.txt — chunk {idx + 1}/{len(chunks)}')
        # Overlap: consecutive chunks should share a trailing/leading span.
        first_tail = chunks[0]['text'][-utils.REFERENCE_CHUNK_OVERLAP:]
        second_head = chunks[1]['text'][:utils.REFERENCE_CHUNK_OVERLAP]
        self.assertTrue(set(first_tail.split()) & set(second_head.split()))

    def test_unstructured_short_text_single_chunk(self):
        path = os.path.join(self.tmp_dir, 'short.md')
        with open(path, 'w') as f:
            f.write('A short reference note.')
        chunks = utils.chunk_unstructured_text(path)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], 'A short reference note.')
        # A single chunk stays as the bare filename — no "chunk 1/1" clutter.
        self.assertEqual(chunks[0]['source'], 'short.md')

    def test_unstructured_empty_file_yields_no_chunks(self):
        path = os.path.join(self.tmp_dir, 'empty.txt')
        open(path, 'w').close()
        self.assertEqual(utils.chunk_unstructured_text(path), [])

    def test_pdf_chunking_one_or_more_chunks_per_page(self):
        path = os.path.join(self.tmp_dir, 'ref.pdf')
        with open(path, 'wb') as f:
            f.write(make_minimal_pdf([
                'Sodium content varies widely across food categories and brands.',
                'Page two mentions Instant Noodles at 861 milligrams per 100 grams.',
            ]))
        chunks = utils.extract_pdf_chunks(path)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any('Instant Noodles' in c['text'] for c in chunks))
        self.assertTrue(any(c['source'].endswith('p.1') for c in chunks))
        self.assertTrue(any(c['source'].endswith('p.2') for c in chunks))
        self.assertTrue(all(c['file'] == 'ref.pdf' for c in chunks))

    def test_build_reference_chunks_dispatches_by_extension(self):
        csv_path = os.path.join(self.tmp_dir, 'a.csv')
        pd.DataFrame({'X': [1]}).to_csv(csv_path, index=False)
        txt_path = os.path.join(self.tmp_dir, 'a.txt')
        with open(txt_path, 'w') as f:
            f.write('hello')
        pdf_path = os.path.join(self.tmp_dir, 'a.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(make_minimal_pdf(['hi']))

        self.assertEqual(len(utils.build_reference_chunks(csv_path)), 1)
        self.assertEqual(len(utils.build_reference_chunks(txt_path)), 1)
        self.assertEqual(len(utils.build_reference_chunks(pdf_path)), 1)

    def test_multi_file_chunks_are_labeled_by_their_own_origin(self):
        """No index/network involved — just confirms two different
        reference files chunked independently (as build_reference_index
        does per file before combining) stay distinguishable by `file` and
        `source`, which is what keeps a multi-file _sources column
        meaningful."""
        csv_path = os.path.join(self.tmp_dir, 'nutrition.csv')
        pd.DataFrame({'Food': ['Apple']}).to_csv(csv_path, index=False)
        txt_path = os.path.join(self.tmp_dir, 'notes.txt')
        with open(txt_path, 'w') as f:
            f.write('A short reference note.')

        csv_chunks = utils.build_reference_chunks(csv_path)
        txt_chunks = utils.build_reference_chunks(txt_path)
        combined = csv_chunks + txt_chunks

        files_seen = {c['file'] for c in combined}
        self.assertEqual(files_seen, {'nutrition.csv', 'notes.txt'})
        sources_seen = {c['source'] for c in combined}
        self.assertEqual(sources_seen, {'nutrition.csv — row 0', 'notes.txt'})


class RagRegistryTests(_IsolatedRegistryMixin, TestCase):
    """rag_projects.json CRUD — no network."""

    def _project_with_csv(self):
        csv_path = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'X': [1]}).to_csv(csv_path, index=False)
        return self.make_project(csv_path)  # no reference files yet

    def _register_dummy_file(self, project_id, filename):
        path = os.path.join(settings.MEDIA_ROOT, filename)
        with open(path, 'w') as f:
            f.write('dummy')
        utils.add_reference_file(project_id, filename, path, os.path.splitext(filename)[1].lstrip('.'), 5)
        return path

    def test_add_reference_file_accumulates_as_many_as_added(self):
        """The whole point of this feature: a project can have any number
        of reference files, not just one."""
        project_id = self._project_with_csv()
        for name in ('a.csv', 'b.txt', 'c.pdf', 'd.md'):
            self._register_dummy_file(project_id, name)

        files = utils.list_reference_files(project_id)
        self.assertEqual([f['filename'] for f in files], ['a.csv', 'b.txt', 'c.pdf', 'd.md'])
        for f in files:
            self.assertIsNone(f['chunk_count'])  # not indexed yet
            self.assertIn('added_at', f)

    def test_remove_reference_file_deletes_from_disk_and_registry(self):
        project_id = self._project_with_csv()
        path_a = self._register_dummy_file(project_id, 'a.csv')
        path_b = self._register_dummy_file(project_id, 'b.csv')

        self.assertTrue(utils.remove_reference_file(project_id, 'a.csv'))
        self.assertFalse(os.path.exists(path_a))
        self.assertTrue(os.path.exists(path_b))
        self.assertEqual([f['filename'] for f in utils.list_reference_files(project_id)], ['b.csv'])

    def test_remove_reference_file_unknown_filename_returns_false(self):
        project_id = self._project_with_csv()
        self._register_dummy_file(project_id, 'a.csv')
        self.assertFalse(utils.remove_reference_file(project_id, 'nope.csv'))
        self.assertEqual(len(utils.list_reference_files(project_id)), 1)

    def test_removing_a_file_clears_an_existing_index(self):
        project_id = self._project_with_csv()
        self._register_dummy_file(project_id, 'a.csv')
        self._register_dummy_file(project_id, 'b.csv')

        # Fake a finished build (no network) so there's an index to clear.
        index_dir = utils._reference_index_dir(project_id)
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, 'manifest.json'), 'w') as f:
            json.dump({'embedding_model': 'model-a', 'dims': 4, 'chunk_count': 2}, f)
        utils.update_rag_index_meta(project_id, 'model-a', 2, 4, {'a.csv': 1, 'b.csv': 1})
        self.assertEqual(utils.get_rag_project_entry(project_id)['total_chunks'], 2)

        utils.remove_reference_file(project_id, 'a.csv')

        self.assertFalse(os.path.exists(index_dir))
        entry = utils.get_rag_project_entry(project_id)
        self.assertEqual(entry['total_chunks'], 0)
        self.assertIsNone(entry['index_built_at'])
        self.assertIsNone(next(f for f in entry['reference_files'] if f['filename'] == 'b.csv')['chunk_count'])

    def test_remove_rag_project_clears_registry_entry(self):
        project_id = self._project_with_csv()
        self._register_dummy_file(project_id, 'a.csv')
        self.assertIsNotNone(utils.get_rag_project_entry(project_id))
        utils.remove_rag_project(project_id)
        self.assertIsNone(utils.get_rag_project_entry(project_id))

    def test_delete_project_also_cleans_up_rag_registry(self):
        """utils.delete_project deletes the whole project folder AND its
        rag_projects.json entry — a deleted project shouldn't linger in
        the RAG registry."""
        project_id = self._project_with_csv()
        self._register_dummy_file(project_id, 'a.csv')
        self.assertIsNotNone(utils.get_rag_project_entry(project_id))
        utils.delete_project(project_id)
        self.assertIsNone(utils.get_rag_project_entry(project_id))

    def test_estimate_reference_chunk_count_previews_before_building(self):
        project_id = self._project_with_csv()
        ref_a = os.path.join(settings.MEDIA_ROOT, 'a.csv')
        pd.DataFrame({'Food': ['x'] * 5}).to_csv(ref_a, index=False)
        utils.add_reference_file(project_id, 'a.csv', ref_a, 'csv', os.path.getsize(ref_a))
        ref_b = os.path.join(settings.MEDIA_ROOT, 'b.csv')
        pd.DataFrame({'Food': ['x'] * 3}).to_csv(ref_b, index=False)
        utils.add_reference_file(project_id, 'b.csv', ref_b, 'csv', os.path.getsize(ref_b))

        total, per_file = utils.estimate_reference_chunk_count(project_id)
        self.assertEqual(total, 8)
        self.assertEqual(per_file, {'a.csv': 5, 'b.csv': 3})


class ConfigAndStalenessTests(_IsolatedRegistryMixin, TestCase):
    def test_parse_retrieval_config_defaults_when_blank(self):
        cfg = utils.parse_retrieval_config({'RetrievalConfig': ''})
        self.assertEqual(cfg, {'enabled': False, 'top_k': 3})

    def test_parse_retrieval_config_malformed_json_falls_back(self):
        cfg = utils.parse_retrieval_config({'RetrievalConfig': 'not json'})
        self.assertEqual(cfg, {'enabled': False, 'top_k': 3})

    def test_parse_retrieval_config_reads_enabled_and_top_k(self):
        cfg = utils.parse_retrieval_config({'RetrievalConfig': json.dumps({'enabled': True, 'top_k': 7})})
        self.assertEqual(cfg, {'enabled': True, 'top_k': 7})

    def test_parse_retrieval_config_top_k_floor_is_one(self):
        cfg = utils.parse_retrieval_config({'RetrievalConfig': json.dumps({'enabled': True, 'top_k': 0})})
        self.assertEqual(cfg['top_k'], 1)

    def test_embedding_model_round_trips_through_connections_csv(self):
        utils.save_connection('localhost', '11434', 'llama3.2:1b', 'nomic-embed-text')
        active = utils.get_active_connection()
        self.assertEqual(active['embedding_model'], 'nomic-embed-text')
        # Re-saving the same (host, port, model) tuple updates rather than duplicates.
        utils.save_connection('localhost', '11434', 'llama3.2:1b', 'mxbai-embed-large')
        conns = utils.load_connections()
        self.assertEqual(len(conns), 1)
        self.assertEqual(conns[0]['embedding_model'], 'mxbai-embed-large')

    def test_reference_index_is_stale_distinguishes_never_built_from_stale(self):
        csv_path = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'X': [1]}).to_csv(csv_path, index=False)
        ref_path = os.path.join(settings.MEDIA_ROOT, 'ref.csv')
        pd.DataFrame({'Food': ['Apple']}).to_csv(ref_path, index=False)
        project_id = self.make_project(csv_path, [ref_path])

        # Never built at all — still reports "stale" at the utils level (the
        # view layer is what distinguishes this from "built but outdated";
        # see reference_stale in views.define_columns_view).
        self.assertTrue(utils.reference_index_is_stale(project_id))

        # Fake a finished build without hitting the network — write a
        # manifest directly, matching build_reference_index's own format.
        index_dir = utils._reference_index_dir(project_id)
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, 'manifest.json'), 'w') as f:
            json.dump({'embedding_model': 'model-a', 'dims': 4, 'chunk_count': 1}, f)

        utils.save_connection('localhost', '11434', 'chat-model', 'model-a')
        self.assertFalse(utils.reference_index_is_stale(project_id))

        utils.save_connection('localhost', '11434', 'chat-model', 'model-b')
        self.assertTrue(utils.reference_index_is_stale(project_id))

    def test_project_with_no_reference_files_is_never_stale(self):
        csv_path = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'X': [1]}).to_csv(csv_path, index=False)
        project_id = self.make_project(csv_path)  # no reference files
        self.assertFalse(utils.reference_index_is_stale(project_id))


# ─── Tier 2: small real-Ollama checks + edge cases ──────────────────────────

@unittest.skipUnless(RAG_LIVE_TESTS, 'set RAG_LIVE_TESTS=1 to run tests against a real local Ollama')
class LiveSmallRetrievalTests(_IsolatedRegistryMixin, TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _ollama_reachable():
            raise unittest.SkipTest(f'Ollama not reachable at {OLLAMA_HOST}:{OLLAMA_PORT}')

    def setUp(self):
        super().setUp()
        utils.save_connection(OLLAMA_HOST, OLLAMA_PORT, CHAT_MODEL, EMBED_MODEL)

    def _small_reference_csv(self, n=40):
        df = pd.read_csv(GLYCEMIC_CSV).head(n)
        path = os.path.join(settings.MEDIA_ROOT, 'ref_small.csv')
        df.to_csv(path, index=False)
        return path, df

    def test_build_index_on_small_real_subset(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['White bread']}).to_csv(main_csv, index=False)
        ref_path, ref_df = self._small_reference_csv(40)
        project_id = self.make_project(main_csv, [ref_path])

        start = time.time()
        manifest = utils.build_reference_index(project_id)
        elapsed = time.time() - start
        print(f'\n[LiveSmall] embedded {len(ref_df)} rows in {elapsed:.1f}s '
              f'({len(ref_df) / elapsed:.1f} rows/sec, model={EMBED_MODEL})')

        self.assertEqual(manifest['chunk_count'], 40)
        self.assertEqual(manifest['embedding_model'], EMBED_MODEL)
        self.assertEqual(manifest['files'], [{'filename': 'ref_small.csv', 'chunk_count': 40}])
        vectors, meta = utils.load_reference_index(project_id)
        self.assertEqual(vectors.shape[0], 40)
        self.assertEqual(vectors.shape[1], manifest['dims'])
        self.assertEqual(len(meta), 40)

        # Registry reflects the build too (per-file chunk_count backfilled).
        entry = utils.get_rag_project_entry(project_id)
        self.assertEqual(entry['total_chunks'], 40)
        self.assertEqual(entry['reference_files'][0]['chunk_count'], 40)

    def test_multi_file_index_combines_sources_from_every_file(self):
        """The actual new capability: attach several reference files (a CSV
        subset AND a distinctive PDF) and confirm one combined index is
        built across all of them, with retrieval correctly attributing
        matches back to whichever file they came from."""
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_csv, _ = self._small_reference_csv(20)

        pdf_path = os.path.join(settings.MEDIA_ROOT, 'standards.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(make_minimal_pdf([
                'This standards document defines Zorbex Fructozyme as the reference '
                'compound for measuring unusual synthetic sweetener glycemic response.'
            ]))

        project_id = self.make_project(main_csv, [ref_csv, pdf_path])
        manifest = utils.build_reference_index(project_id)

        self.assertEqual(len(manifest['files']), 2)
        filenames = {f['filename'] for f in manifest['files']}
        self.assertEqual(filenames, {'ref_small.csv', 'standards.pdf'})
        self.assertEqual(manifest['chunk_count'], 21)  # 20 CSV rows + 1 PDF chunk

        results = utils.retrieve_reference_chunks(project_id, 'Zorbex Fructozyme reference compound', top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]['file'], 'standards.pdf')
        self.assertIn('Zorbex Fructozyme', results[0]['text'])

    def test_retrieval_returns_well_formed_ranked_results(self):
        """Checks the retrieval *mechanism* — correct count, descending
        order, non-degenerate scores. Does NOT assert that the single
        obviously-correct row comes back on top: stress-testing this
        against the real dataset with llama3.2:1b (a general chat model
        repurposed for embeddings, not a dedicated embedding model) showed
        it reliably clusters results by the dominant shared boilerplate in
        each chunk (citation/category/manufacturer text repeated across
        many rows) rather than by the distinguishing food name — e.g.
        querying "Chocolate mudcake" surfaces other "Sponge cake ... Kinder/
        Ferrero, Italy ..." rows instead of its own row. That's a real
        embedding-quality characteristic worth knowing (a dedicated
        embedding model like nomic-embed-text would very likely do better
        here — see the module docstring for a measured comparison), not a
        bug in the retrieval code itself, which this test confirms is
        wired correctly."""
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_path, ref_df = self._small_reference_csv(40)
        project_id = self.make_project(main_csv, [ref_path])
        utils.build_reference_index(project_id)

        known_food = ref_df.iloc[3]['Food Name']
        self.assertEqual(known_food, 'Chocolate mudcake')
        results = utils.retrieve_reference_chunks(project_id, f'What is the GI of {known_food}?', top_k=5)
        self.assertEqual(len(results), 5)
        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(max(scores) - min(scores), 0,
                            'Scores should discriminate between chunks, not come back flat/identical')
        for r in results:
            self.assertIsInstance(r['score'], float)
            self.assertTrue(r['source'].startswith('ref_small.csv — row '))
            self.assertEqual(r['file'], 'ref_small.csv')
        found_exact = any(known_food in r['text'] for r in results)
        print(f'\n[LiveSmall] query {known_food!r}: exact row in top-5 = {found_exact} '
              f'(top score {scores[0]:.3f}); see module docstring — model-dependent, not asserted')

    def test_build_index_with_chat_only_model_reports_clean_error(self):
        """Real robustness check found via manual probing during stress
        testing: of the 4 Ollama models on this machine, only llama3.2:1b
        actually supports /api/embed — gpt-oss:20b and llama3.2-vision both
        respond with {"error": "this model does not support embeddings"}.
        A user who picks a chat-only model as their embedding model should
        get a clear failed build, not a crash."""
        if 'gpt-oss:20b' not in _pulled_models():
            self.skipTest('gpt-oss:20b not pulled on this machine')
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_path, _ = self._small_reference_csv(3)
        project_id = self.make_project(main_csv, [ref_path])
        utils.save_connection(OLLAMA_HOST, OLLAMA_PORT, CHAT_MODEL, 'gpt-oss:20b')

        with self.assertRaises(Exception):
            utils.build_reference_index(project_id)
        status = utils.REFERENCE_INDEX_STATUS.get(project_id)
        self.assertEqual(status['status'], 'error')
        self.assertIn('embeddings', status['message'])

    def test_top_k_larger_than_corpus_clamps_gracefully(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_path, ref_df = self._small_reference_csv(5)
        project_id = self.make_project(main_csv, [ref_path])
        utils.build_reference_index(project_id)

        results = utils.retrieve_reference_chunks(project_id, 'any food', top_k=1000)
        self.assertEqual(len(results), 5)

    def test_build_index_without_embedding_model_raises_clear_error(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_path, _ = self._small_reference_csv(5)
        project_id = self.make_project(main_csv, [ref_path])
        utils.save_connection(OLLAMA_HOST, OLLAMA_PORT, CHAT_MODEL, '')  # clear embedding model

        with self.assertRaisesMessage(ValueError, 'embedding model'):
            utils.build_reference_index(project_id)

    def test_build_index_on_empty_reference_file_raises_clear_error(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        empty_ref = os.path.join(settings.MEDIA_ROOT, 'empty.txt')
        open(empty_ref, 'w').close()
        project_id = self.make_project(main_csv, [empty_ref])

        with self.assertRaisesMessage(ValueError, 'No text could be extracted'):
            utils.build_reference_index(project_id)

    def test_build_index_with_no_files_attached_raises_clear_error(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        project_id = self.make_project(main_csv)  # no reference files at all

        with self.assertRaisesMessage(ValueError, 'no reference files'):
            utils.build_reference_index(project_id)

    def test_retrieval_before_any_index_built_returns_empty(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        project_id = self.make_project(main_csv)  # no reference at all
        self.assertEqual(utils.retrieve_reference_chunks(project_id, 'anything', top_k=3), [])

    def test_retrieval_with_blank_query_returns_empty(self):
        main_csv = os.path.join(settings.MEDIA_ROOT, 'main.csv')
        pd.DataFrame({'FoodName': ['x']}).to_csv(main_csv, index=False)
        ref_path, _ = self._small_reference_csv(5)
        project_id = self.make_project(main_csv, [ref_path])
        utils.build_reference_index(project_id)
        self.assertEqual(utils.retrieve_reference_chunks(project_id, '   ', top_k=3), [])


# ─── Tier 3: the real stress test — full 4274-row dataset + live tagging ────

@unittest.skipUnless(RAG_STRESS_FULL, 'set RAG_STRESS_FULL=1 to run the full-dataset stress test (several minutes)')
class StressFullDatasetTests(_IsolatedRegistryMixin, TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _ollama_reachable():
            raise unittest.SkipTest(f'Ollama not reachable at {OLLAMA_HOST}:{OLLAMA_PORT}')

    def setUp(self):
        super().setUp()
        utils.save_connection(OLLAMA_HOST, OLLAMA_PORT, CHAT_MODEL, EMBED_MODEL)
        self.client = Client()

    def test_full_glycemic_index_dataset_index_and_grounded_tagging(self):
        ref_df = pd.read_csv(GLYCEMIC_CSV)
        n_rows = len(ref_df)
        print(f'\n[Stress] reference dataset: {GLYCEMIC_CSV} ({n_rows} rows)')

        sub_df = pd.read_csv(SUBCATEGORY_CSV).head(15)[['SubCategory', 'GI_Avg']]
        main_csv_path = os.path.join(settings.MEDIA_ROOT, '_main_stress.csv')
        sub_df.to_csv(main_csv_path, index=False)
        print(f'[Stress] main dataset: {len(sub_df)} subcategories to assess, grounded against {n_rows} reference rows')

        # ── Build the index over the FULL real reference dataset ──
        project_id = self.make_project(main_csv_path, [GLYCEMIC_CSV])
        start = time.time()
        manifest = utils.build_reference_index(project_id)
        build_elapsed = time.time() - start
        print(f'[Stress] built index: {manifest["chunk_count"]} chunks, '
              f'{manifest["dims"]}-dim vectors, in {build_elapsed:.1f}s '
              f'({manifest["chunk_count"] / build_elapsed:.1f} rows/sec)')

        self.assertEqual(manifest['chunk_count'], n_rows)
        vectors, meta = utils.load_reference_index(project_id)
        self.assertEqual(vectors.shape, (n_rows, manifest['dims']))
        self.assertEqual(len(meta), n_rows)

        # ── Retrieval quality/latency sanity over the full corpus ──
        sample_queries = ['White bread', 'Banana', 'Chocolate cake', 'Instant noodles', 'Ice cream']
        for q in sample_queries:
            t0 = time.time()
            results = utils.retrieve_reference_chunks(project_id, q, top_k=5)
            latency = time.time() - t0
            print(f'[Stress] query {q!r} -> {len(results)} matches in {latency * 1000:.0f}ms; '
                  f'top match: {results[0]["source"] if results else None} '
                  f'(score={results[0]["score"]:.3f})' if results else f'[Stress] query {q!r} -> no matches')
            self.assertLessEqual(len(results), 5)
            self.assertGreater(len(results), 0)

        # ── Real end-to-end grounded tagging run, driven over HTTP exactly
        # like the browser would (upload -> connection -> define-columns ->
        # tagging -> results), reusing the index just built. ──
        with open(main_csv_path, 'rb') as f_csv:
            r = self.client.post('/ODT/upload/', data={
                'mode': 'text', 'csv_file': f_csv,
            })
        self.assertEqual(r.status_code, 302)

        # Attach the same reference file to this session's project and
        # reuse the already-built index/registry state directly — actually
        # re-uploading + re-embedding the full 4274-row corpus a second
        # time just to drive it through HTTP would double the stress-test
        # runtime for no extra coverage (the index-build path above already
        # verifies real embedding end to end).
        session = self.client.session
        http_project_id = session['project_id']
        utils.add_reference_file(http_project_id, os.path.basename(GLYCEMIC_CSV), GLYCEMIC_CSV,
                                  'csv', os.path.getsize(GLYCEMIC_CSV))
        index_dir = utils._reference_index_dir(http_project_id)
        shutil.copytree(utils._reference_index_dir(project_id), index_dir)
        utils.update_rag_index_meta(http_project_id, EMBED_MODEL, manifest['chunk_count'], manifest['dims'],
                                     {os.path.basename(GLYCEMIC_CSV): manifest['chunk_count']})

        retrieval_cfg = json.dumps({'enabled': True, 'top_k': 5})
        r = self.client.post('/ODT/define-columns/', data={
            'input_columns': ['SubCategory', 'GI_Avg'],
            'output_column': ['GI_Plausible'],
            'prompt_template': [
                'A food subcategory "{SubCategory}" has an average glycemic index (GI) '
                'claimed as {GI_Avg}. Using the reference data above (real published GI '
                'measurements for individual foods), reply PLAUSIBLE or IMPLAUSIBLE for '
                'whether this claimed average is in a reasonable range.'
            ],
            'condition_field': [''], 'condition_op': ['=='], 'condition_value': [''],
            'default_value': [''], 'send_context': ['0'],
            'tag_input_cols': ['SubCategory,GI_Avg'],
            'image_params': [''],
            'retrieval_config': [retrieval_cfg],
            'node_x': [''], 'node_y': [''],
        })
        self.assertEqual(r.status_code, 302)
        self.assertIn('tagging', r.headers['Location'])

        r = self.client.get('/ODT/tagging/')
        self.assertEqual(r.status_code, 200)

        start = time.time()
        deadline = start + 300
        status = None
        while time.time() < deadline:
            r = self.client.get('/ODT/tagging/progress/')
            status = r.json()
            if status['status'] == 'finished' or str(status['status']).startswith('error'):
                break
            time.sleep(1)
        tag_elapsed = time.time() - start
        print(f'[Stress] tagged {len(sub_df)} rows (grounded on {n_rows}-row index) '
              f'in {tag_elapsed:.1f}s ({len(sub_df) / tag_elapsed:.2f} rows/sec)')

        self.assertEqual(status['status'], 'finished', status)

        r = self.client.get('/ODT/results/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('GI_Plausible_sources', r.content.decode())

        tagged_path = os.path.join(settings.MEDIA_ROOT, http_project_id, '_main_stress_tagged.csv')
        tagged_df = pd.read_csv(tagged_path)
        self.assertEqual(len(tagged_df), len(sub_df))
        self.assertTrue((tagged_df['GI_Plausible_sources'].astype(str).str.len() > 0).all(),
                         'Every grounded row should record at least one retrieved source')
        self.assertFalse(tagged_df['GI_Plausible'].astype(str).str.startswith('ERROR').any(),
                          'No row should have failed the LLM call')

        print('[Stress] sample grounded answers:')
        for _, row in tagged_df.head(5).iterrows():
            print(f"  {row['SubCategory']!r} (claimed GI {row['GI_Avg']}) -> "
                  f"{row['GI_Plausible']} | sources: {row['GI_Plausible_sources']}")
