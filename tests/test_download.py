import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class ExtractionTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/download.py'
        spec = importlib.util.spec_from_file_location('download_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_interrupted_extraction_revalidates_same_size_files(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'assets.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name in ('scenes/a.xml', 'new_objects/a.obj', 'textures/a.png'):
                    z.writestr('build/machine/assets/' + name, b'original-data')
            spec = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            with patch.object(module, 'RUNTIME', root):
                module.extract_assets(archive, spec)
                assets = root / 'repos/libero_plus/libero/libero/assets'
                (assets / '.vla-assets-sha256').unlink()
                (assets / 'scenes/a.xml').write_bytes(b'damaged--data')
                module.extract_assets(archive, spec)
                self.assertEqual((assets / 'scenes/a.xml').read_bytes(), b'original-data')
                self.assertEqual((assets / '.vla-assets-sha256').read_text().strip(), spec['sha256'])

    def test_archive_traversal_is_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'assets.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('prefix/assets/../outside', b'bad')
            with patch.object(module, 'RUNTIME', root), self.assertRaises(ValueError):
                module.extract_assets(archive, {'sha256': 'test-fixture'})
            self.assertFalse((root / 'repos/libero_plus/libero/libero/outside').exists())


if __name__ == '__main__':
    unittest.main()
