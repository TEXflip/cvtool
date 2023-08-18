import unittest
from pathlib import Path
import shutil

from cvtool.__main__ import main

TEST_DIR = Path(__file__).parent
TEST_IMG_DIR = TEST_DIR / "test_images"
TEST_IMG_DIR_TMP = TEST_DIR / "test_images_tmp"

def rm_tree(pth: Path):
    """
    Remove a directory and all its contents.
    """
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def copy_tree(src: Path, dst: Path):
    """
    Copy a directory and all its contents.
    """
    for child in src.iterdir():
        if child.is_file():
            shutil.copy2(child, dst)
        else:
            copy_tree(child, dst / child.name)


class CvToolTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TEST_IMG_DIR_TMP
        self.tmp_dir.mkdir(exist_ok=True)
        copy_tree(TEST_IMG_DIR, self.tmp_dir)
    
    def tearDown(self):
        rm_tree(self.tmp_dir)
        pass

    def test_conversion(self):
        kwargs = {
            "convert": "jpg",
            "recursive": False,
            "path": [str(self.tmp_dir)],
            "fft": False,
            "save": None,
            "show": False
        }
        main(kwargs)
        lenna_exist = (self.tmp_dir / "Lenna.jpg").is_file()
        portal_exist = (self.tmp_dir / "portal.jpg").is_file()
        self.assertTrue(lenna_exist, "Lenna.jpg not found")
        self.assertTrue(portal_exist, "portal.jpg not found")

    def test_fft(self):
        kwargs = {
            "convert": None,
            "recursive": False,
            "path": [str(self.tmp_dir)],
            "fft": True,
            "save": None,
            "show": False
        }
        main(kwargs)
        lenna_exist = (self.tmp_dir / "Lenna.jpg").is_file()
        portal_exist = (self.tmp_dir / "portal.jpg").is_file()
        self.assertTrue(lenna_exist, "Lenna.jpg not found")
        self.assertTrue(portal_exist, "portal.jpg not found")

    def test_rectify(self):
        kwargs = {
            "convert": None,
            "recursive": False,
            "path": [str(self.tmp_dir)],
            "fft": False,
            "save": None,
            "show": False,
            "rectify": {
                "original_size": True,
                "rect_points": [[0, 0], [10, 0], [10, 10], [0, 10]]
            }
        }
        main(kwargs)
        lenna_exist = (self.tmp_dir / "Lenna.jpg").is_file()
        portal_exist = (self.tmp_dir / "portal.jpg").is_file()
        self.assertTrue(lenna_exist, "Lenna.jpg not found")
        self.assertTrue(portal_exist, "portal.jpg not found")

if __name__ == "__main__":
    unittest.main()