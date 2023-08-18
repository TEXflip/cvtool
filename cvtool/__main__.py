import logging
logger = logging.getLogger(__name__)

from cvtool.tools import *


def main(kwargs: dict):
	if kwargs.get("fft"):
		apply_function_glob(kwargs["path"], func=fft, **kwargs)

	elif kwargs.get("rectify"):
		apply_function_glob(kwargs["path"], func=rectify, **kwargs)

	elif kwargs.get("convert", None):
		apply_function_glob(kwargs["path"], func=lambda x, kw: x, **kwargs)


if __name__ == '__main__':
	import argparse
	from cvtool.__parser__ import ActionRectify

	formatter = logging.Formatter('%(levelname)s - %(message)s')
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	# logger.setLevel(logging.DEBUG)

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--convert', nargs=1, type=str, default=None, help='convert to image format')
	parser.add_argument('-r', '--recursive', action='store_true', default=False, help='use images recursively inside folders')
	parser.add_argument('path', nargs='*', type=str, default=['.'], help='image or folder to input')
	parser.add_argument('--fft', action='store_true', help='compute fft of images')
	parser.add_argument('-s', '--save', nargs=1, type=str, default=None, help='save fft images to folder')
	parser.add_argument('--show', action='store_true', help='show results')
	parser.add_argument('--rectify', action=ActionRectify, help='rectify a rectangular portion of an image. Additional parameter: float, specify the aspect ratio of the rectangle (width/height)')
	parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
	# parser.add_argument('--debug', action='store_true', help='debug mode')
	args = parser.parse_args()

	kwargs = vars(args)

	logger.debug(kwargs)

	main(kwargs)