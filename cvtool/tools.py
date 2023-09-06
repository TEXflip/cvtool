import logging
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

IM_SUFFIXES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "gif", "webp"]


def find_suitable_name(path: Path, kwargs: dict) -> Path:
    if path.exists() and not kwargs.get("overwrite", False):
        i = 1
        sep = "_"
        # check if the file finished with a number
        if path.stem[-1].isdigit():
            # how many digits?
            d = 1
            while path.stem[-(d+1)].isdigit():
                d += 1
            # check if there is a separator
            if path.stem[-(d + 1)] in ["_", "-", ".", " ", "|"]:
                sep = path.stem[-(d + 1)]
            path = path.with_name(path.stem[:-(d + len(sep))] + path.suffix)
        
        new_path = path.with_name(path.stem + f"{sep}{i}" + path.suffix)
        while new_path.exists():
            i += 1
            new_path = path.with_name(path.stem + f"{sep}{i}" + path.suffix)
        return new_path
    else:
        return path

def apply_function_single(img_path: Path, out_path: Path, func = lambda x: x, **kwargs):
    out_path = find_suitable_name(out_path, kwargs)
    im = cv2.imread(str(img_path))
    im = func(im, kwargs)
    if im is None:
        logger.warning(f"Operation Canceled for {img_path}")
        return
    if kwargs.get("show"):
        cv2.imshow(img_path.name, im)
        while True:
            k = cv2.waitKey(100)
            if k == 27 or cv2.getWindowProperty(img_path.name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(str(out_path), im)

def apply_function_glob(paths: list, func = lambda x, kw: x, **kwargs):
    save_format = kwargs.get("convert", None)
    ext = "." + ("jpg" if save_format is None else save_format)
    glob = "**/*" if kwargs["recursive"] else "*"
    save = kwargs.get("save")
    save = Path(save) if isinstance(save, str) and Path(save).is_dir() else None
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for img_path in path.glob(glob):
                if img_path.suffix[1:] in IM_SUFFIXES:
                    # im = Image.open(img_path)
                    # im.save(img.with_suffix(ext))
                    save_path = save / img_path.name if save else img_path.with_suffix(ext)
                    apply_function_single(img_path, save_path, func, **kwargs)
                else:
                    logger.debug(f"skipping {img_path}")
        elif path.is_file() and path.suffix[1:] in IM_SUFFIXES:
            # im = Image.open(path)
            # im.save(path.with_suffix(ext))
            save_path = save / path.name if save else path.with_suffix(ext)
            apply_function_single(path, save_path, func, **kwargs)
        else:
            logger.warning(f"Invalid path: {path}")

def fft(img: np.ndarray, kwargs: dict = {}) -> np.ndarray:
    """
    Compute the Fast Fourier Transform of an image.
    """
    # convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform 2D FFT on the image
    fft_image = np.fft.fft2(img)
    
    # Shift the zero frequency component to the center of the spectrum
    shifted_fft = np.fft.fftshift(fft_image)
    
    # Compute the magnitude spectrum
    magn_spectrum = (20 * np.log(np.abs(shifted_fft))).astype(np.float32)
    
    _min_px = np.min(magn_spectrum)
    magn_spectrum = (magn_spectrum - _min_px) / (np.max(magn_spectrum) - _min_px) * 255.
    magn_spectrum = magn_spectrum.astype(np.uint8)
    
    return magn_spectrum

def rectify(img: np.ndarray, kwargs: dict = {}) -> np.ndarray:
    points = kwargs["rectify"].get("rect_points", [])

    if len(points) < 4:
        points = select_4_points(img)

    if len(points) >= 4:
        points = np.array(points[:4], dtype=np.float32)
        h, w = img.shape[:2]
        img_scaling = np.linalg.norm(points[0] - points[3]) / h
        scaled_h = int(np.ceil(h * img_scaling))
        original_aspect_ratio = 1.0 # w / h
        orig_size_param = kwargs["rectify"].get("original_size", True)

        if orig_size_param:
            if isinstance(orig_size_param, bool):
                original_aspect_ratio = float(np.linalg.norm(points[0] - points[1]) / np.linalg.norm(points[1] - points[2]))
            elif isinstance(orig_size_param, float):
                original_aspect_ratio = orig_size_param
            else:
                logger.warning(f"Invalid original_size parameter: {orig_size_param}")
        
        proj_w = int(np.round(scaled_h * original_aspect_ratio))
        projected_points = np.array([
            [0, 0],
            [proj_w, 0],
            [proj_w, scaled_h],
            [0, scaled_h]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(points, projected_points)
        img_rectify = cv2.warpPerspective(img, M, (proj_w, scaled_h), flags=cv2.INTER_LINEAR)

        return img_rectify
    
    return None

def select_4_points(img: np.ndarray):
    points = []
    mouse = [0, 0]
    move_props = [False, (0, 0)]
    def rectangle_drawing(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if len(points) < 5:
                points.append((x, y))
                if len(points) >= 4: 
                    points.append(points[0])
            else:
                move_props[0] = True
                move_props[1] = (x, y)

        elif event==cv2.EVENT_MOUSEMOVE:
            mouse[0] = x
            mouse[1] = y
            if move_props[0]:
                dx = x - move_props[1][0]
                dy = y - move_props[1][1]
                move_props[1] = (x, y)
                for i in range(len(points)):
                    points[i] = (points[i][0] + dx, points[i][1] + dy)

        elif event==cv2.EVENT_LBUTTONUP:
            move_props[0] = False

    window_name = "rectify"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, rectangle_drawing)
    k = -1
    while k != 13: # enter
        img_draw = img.copy()
        if len(points) > 0:
            for i, p in enumerate(points):
                cv2.circle(img_draw, center=p, radius=3, color=(255,0,0), thickness=-1)
                if i > 0:
                    cv2.line(img_draw, points[i-1], points[i], color=(0,0,0), thickness=1)
            if len(points) < 4:
                cv2.line(img_draw, points[-1], (mouse[0], mouse[1]), color=(0,0,0), thickness=1)
        cv2.imshow(window_name, img_draw)
        k = cv2.waitKey(16)

        if k != -1:
            logger.debug(f"key pressed: {k}")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 \
            or k == 27 or k == 13: # esc or enter, TODO: make os independent
            break
    cv2.destroyAllWindows()

    return points[:4]