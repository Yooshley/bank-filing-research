"""Export and restore portable, verified index data without copying SQLite files."""
import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path


def ensure_cloud_index(base):
    url = os.getenv('INDEX_BUNDLE_URL')
    if not url:
        return
    digest = os.getenv('INDEX_BUNDLE_SHA256', '').lower()
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Configure INDEX_BUNDLE_SHA256 before restoring the index.')
    config = json.loads((base / 'reports/embedding_vectordb/retrieval_config.json').read_text())
    provider = config['default_provider']
    if provider != 'local':
        raise ValueError('Cloud bundle restoration currently requires the local embedding provider.')
    import chromadb
    store = chromadb.PersistentClient(path=str(base / 'models/chroma'))
    name = config['collections'][provider]
    ready = base / 'models' / f'cloud_index_{digest}.ready'
    if ready.exists():
        try:
            if store.get_collection(name, embedding_function=None).count():
                return
        except Exception:
            pass
    if not url.startswith('https://'):
        raise ValueError('Index download requires HTTPS.')
    import requests
    import numpy as np
    payload = io.BytesIO()
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        for block in response.iter_content(1024*1024):
            payload.write(block)
            if payload.tell() > 200*1024*1024:
                raise ValueError('Index bundle exceeds the 200 MB download limit.')
    raw = payload.getvalue()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Index checksum mismatch.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if set(archive.namelist()) != {'documents.json', 'vectors.npy', 'spec.json'}:
            raise ValueError('Unexpected index bundle contents.')
        if sum(i.file_size for i in archive.infolist()) > 300*1024*1024:
            raise ValueError('Uncompressed index exceeds the size limit.')
        records = json.loads(archive.read('documents.json'))
        spec = json.loads(archive.read('spec.json'))
        vectors = np.load(io.BytesIO(archive.read('vectors.npy')), allow_pickle=False)
    if spec != {'model': config['models']['local'], 'window_tokens': config.get('local_window_tokens', 220)}:
        raise ValueError('Index model or preprocessing does not match the configured query encoder.')
    if vectors.ndim != 2 or len(vectors) != len(records['ids']) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding matrix.')
    if not records['ids'] or len(set(records['ids'])) != len(records['ids']):
        raise ValueError('Missing or duplicate chunk IDs.')
    if any(len(records[k]) != len(vectors) for k in ('documents', 'metadatas')):
        raise ValueError('Index record counts differ.')
    # Download and validate before replacing the cloud collection. No archive extraction.
    store.delete_collection(name) if name in [c.name for c in store.list_collections()] else None
    collection = store.create_collection(name, embedding_function=None,
                                         configuration={'hnsw': {'space': 'cosine'}})
    for start in range(0, len(vectors), 250):
        end = start + 250
        collection.add(ids=records['ids'][start:end], documents=records['documents'][start:end],
                       metadatas=records['metadatas'][start:end], embeddings=vectors[start:end].tolist())
    ready.write_text(str(len(vectors)), encoding='utf-8')


def export_bundle(base, output):
    import chromadb
    import numpy as np
    config = json.loads((base / 'reports/embedding_vectordb/retrieval_config.json').read_text())
    collection = chromadb.PersistentClient(path=str(base / 'models/chroma')).get_collection(
        config['collections']['local'], embedding_function=None)
    data = collection.get(include=['documents', 'metadatas', 'embeddings'])
    if not data['ids']:
        raise ValueError('Run notebooks 1 and 2 to build the full local corpus first.')
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(data['embeddings'], dtype=np.float32), allow_pickle=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('documents.json', json.dumps({k: data[k] for k in ('ids','documents','metadatas')}))
        archive.writestr('vectors.npy', buffer.getvalue())
        archive.writestr('spec.json', json.dumps({'model': config['models']['local'],
                            'window_tokens': config.get('local_window_tokens', 220)}))
    print(f'Exported {len(data["ids"])} chunks to {output}')
    print('INDEX_BUNDLE_SHA256=' + hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the real local index for cloud hosting.')
    parser.add_argument('--output', type=Path, default=Path('.local/cloud/filing-index.zip'))
    args = parser.parse_args()
    export_bundle(Path(__file__).resolve().parents[1], args.output)
