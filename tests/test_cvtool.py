import os
import shutil
import unittest
from pathlib import Path

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


class ConvertImagesTestCase(unittest.TestCase):
    test_formats = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]

    def setUp(self) -> None:
        self.tmp_dir = TEST_IMG_DIR_TMP
        self.tmp_dir.mkdir(exist_ok=True)
        copy_tree(TEST_IMG_DIR, self.tmp_dir)

    def tearDown(self):
        rm_tree(self.tmp_dir)

    @classmethod
    def build_tests(cls):
        """
        Build the unittests from the test images.

        This method is called automatically at the end of the module.
        """

        def build_test_function(format):
            def test_function(self):
                """
                Test the conversion of images to a specific format.
                """
                img_paths = os.listdir(self.tmp_dir)
                img_paths = [
                    self.tmp_dir / Path(p).with_suffix("." + format) for p in img_paths
                ]
                path_already_exist = [p.is_file() for p in img_paths]

                os.system(f"python -m cvtool {self.tmp_dir} -c {format}")

                for p, already_exist in zip(img_paths, path_already_exist):
                    if already_exist:
                        p = self.tmp_dir / f"{p.stem}_1{p.suffix}"
                    self.assertTrue(p.is_file(), f"{p.name} not found")

            return test_function

        # iterate through all the image formats and build the tests
        for format in cls.test_formats:
            setattr(
                cls,
                f"test_{format}_conversion",
                build_test_function(format),
            )


class CvToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TEST_IMG_DIR_TMP
        self.tmp_dir.mkdir(exist_ok=True)
        copy_tree(TEST_IMG_DIR, self.tmp_dir)

    def tearDown(self):
        rm_tree(self.tmp_dir)

    def test_fft(self):
        kwargs = {
            "convert": None,
            "recursive": False,
            "path": [str(self.tmp_dir)],
            "fft": True,
            "save": None,
            "show": False,
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
                "rect_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
        }
        main(kwargs)
        lenna_exist = (self.tmp_dir / "Lenna.jpg").is_file()
        portal_exist = (self.tmp_dir / "portal.jpg").is_file()
        self.assertTrue(lenna_exist, "Lenna.jpg not found")
        self.assertTrue(portal_exist, "portal.jpg not found")


# iterate through all the classes in this module and build the tests
for k, v in list(globals().items()):
    if (
        type(v) is type
        and issubclass(v, unittest.case.TestCase)
        and hasattr(v, "build_tests")
    ):
        v.build_tests()
        globals()[k] = v
del k, v


if __name__ == "__main__":
    unittest.main()
