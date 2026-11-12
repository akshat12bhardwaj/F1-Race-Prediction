import hashlib, io, json, unittest
from pathlib import Path
from PIL import Image
from src.fetch_portraits import crop_portrait

ROOT = Path(__file__).resolve().parents[1]

class PortraitTests(unittest.TestCase):
    def test_crop_keeps_square_faces_and_jpeg_format(self):
        image = Image.new('RGB', (400, 300), 'blue')
        image.paste('red', (100, 20, 300, 220))
        raw = io.BytesIO(); image.save(raw, 'PNG')
        result = Image.open(io.BytesIO(crop_portrait(raw.getvalue(), [100, 20, 300, 220])))
        self.assertEqual(result.size, (256, 256))
        self.assertEqual(result.format, 'JPEG')
        self.assertGreater(result.getpixel((128, 128))[0], 240)
        self.assertTrue(result.info.get('progressive'))

    def test_bad_crops_are_rejected_instead_of_padding_or_stretching(self):
        raw = io.BytesIO(); Image.new('RGB', (400, 300)).save(raw, 'PNG')
        for box in [[-1, 0, 99, 100], [0, 0, 100, 150], [250, 0, 450, 200], [0, 0, 0, 0]]:
            with self.assertRaises(ValueError): crop_portrait(raw.getvalue(), box)

    def test_reviewed_assets_have_valid_paths_unique_bytes_and_attribution(self):
        manifest = json.loads((ROOT/'frontend/src/portraits.json').read_text(encoding='utf8'))
        reviewed = json.loads((ROOT/'src/reviewed_portraits.json').read_text(encoding='utf8'))
        hashes = {}
        for driver, item in manifest.items():
            path = ROOT/'frontend/public'/item['path'].lstrip('/')
            self.assertTrue(path.is_file(), str(path))
            hashes.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(driver)
        for driver, item in reviewed.items():
            deployed = manifest[driver]
            self.assertEqual(deployed, {k:v for k,v in item.items() if k != 'original_file'})
            path = ROOT/'frontend/public'/deployed['path'].lstrip('/')
            with Image.open(path) as image:
                self.assertEqual(image.size, (256, 256)); self.assertEqual(image.format, 'JPEG')
            self.assertLess(path.stat().st_size, 40000)
            self.assertEqual(hashes[hashlib.sha256(path.read_bytes()).hexdigest()], [driver])
            for key in ['author', 'source', 'licence', 'licence_url', 'identity_evidence', 'changes', 'original_sha256']:
                self.assertTrue(deployed[key], (driver, key))

    def test_generated_credits_include_crop_terms_and_existing_model_link(self):
        credits = (ROOT/'frontend/public/credits.html').read_text(encoding='utf8')
        reviewed = json.loads((ROOT/'src/reviewed_portraits.json').read_text(encoding='utf8'))
        self.assertIn('/models/ATTRIBUTION.html', credits)
        for item in reviewed.values():
            self.assertIn(item['source'].replace('&', '&amp;'), credits)
            self.assertIn(item['licence_url'], credits)
        self.assertIn('Cropped to head and shoulders', credits)

if __name__ == '__main__': unittest.main()
