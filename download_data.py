"""Download only the four requested sources. Never replace existing raw files."""
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'UCI_Early_Biomarkers': (392, 'https://archive.ics.uci.edu/static/public/392/early+biomarkers+of+parkinson+s+disease+based+on+natural+connected+speech.zip'),
    'UCI_Multiple_Speech': (301, 'https://archive.ics.uci.edu/static/public/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings.zip'),
    'UCI_Oxford_Voice': (174, 'https://archive.ics.uci.edu/static/public/174/parkinsons.zip'),
    'KAGGLE_Voice': (None, 'https://www.kaggle.com/api/v1/datasets/download/ucimachinelearning/parkinsons-voice-dataset?datasetVersionNumber=1'),
}


def extract_original(archive, folder):
    """Extract source bytes only, without changing existing originals."""
    if archive.suffix.lower() == '.rar':
        import libarchive
        with libarchive.file_reader(str(archive)) as entries:
            for entry in entries:
                target = (folder / entry.pathname).resolve()
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe archive member')
                if entry.isdir:
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('wb') as out:
                        for block in entry.get_blocks():
                            out.write(block)
        return
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (folder / member.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive member')
            if not target.exists():
                z.extract(member, folder)


def main():
    results = ROOT / 'RESULTS' / '_pipeline'
    results.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (uci_id, url) in SOURCES.items():
        folder = ROOT / 'DATA' / 'raw' / name
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / 'original.zip'
        cli_archive = folder / 'parkinsons-voice-dataset.zip'
        if not archive.exists() and cli_archive.exists():
            archive = cli_archive
        try:
            if not archive.exists():
                # Kaggle CLI uses configured credentials when available.
                if uci_id is None:
                    import shutil
                    if shutil.which('kaggle'):
                        result = subprocess.run(['kaggle', 'datasets', 'download', '-d',
                            'ucimachinelearning/parkinsons-voice-dataset', '-p', str(folder)],
                            capture_output=True, text=True)
                        downloaded = folder / 'parkinsons-voice-dataset.zip'
                        if result.returncode == 0 and downloaded.exists():
                            archive = downloaded
                if not archive.exists():
                    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        data = response.read()
                    # Validate before writing anything into raw DATA.
                    import io
                    with zipfile.ZipFile(io.BytesIO(data)) as candidate:
                        if candidate.testzip() is not None:
                            raise ValueError('Downloaded archive failed CRC validation')
                    archive.write_bytes(data)
            with zipfile.ZipFile(archive) as z:
                members = z.namelist()
            extract_original(archive, folder)
            for nested in list(folder.glob('*.ZIP')) + list(folder.glob('*.rar')):
                extract_original(nested, folder)
            item = {'status': 'downloaded', 'url': url, 'uci_id': uci_id,
                    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'members': members}
            if uci_id:
                metadata_url = f'https://archive.ics.uci.edu/api/dataset?id={uci_id}'
                with urllib.request.urlopen(metadata_url, timeout=60) as r:
                    item['metadata'] = json.loads(r.read())
            else:
                with urllib.request.urlopen('https://www.kaggle.com/api/v1/datasets/view/ucimachinelearning/parkinsons-voice-dataset', timeout=60) as r:
                    item['metadata'] = json.loads(r.read())
            manifest[name] = item
            print(name, 'downloaded', len(members), 'archive members', flush=True)
        except Exception as exc:
            manifest[name] = {'status': 'unavailable', 'url': url, 'error': str(exc)}
            print(name, 'UNAVAILABLE:', exc, flush=True)
    (results / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    if manifest['KAGGLE_Voice']['status'] == 'unavailable':
        print('Manual Kaggle download: https://www.kaggle.com/datasets/ucimachinelearning/parkinsons-voice-dataset')
        print('Extract the original archive into DATA/raw/KAGGLE_Voice/ without renaming original folders.')


if __name__ == '__main__':
    main()
