#!/usr/bin/env python

import os
import json
import base64
import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np


class CalibrationJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return self.numpy_to_dict(obj)
        if obj.__class__.__name__ == "CharucoBoard":
            _dict = obj.getDictionary()
            return {
                "type": "CharucoBoard",
                "size": obj.getChessboardSize(),
                "squareLength": obj.getSquareLength(),
                "markerLength": obj.getMarkerLength(),
                "dictionary": {
                    "bytesList": self.numpy_to_dict(_dict.bytesList),
                    "markerSize": _dict.markerSize,
                    "maxCorrectionBits": _dict.maxCorrectionBits,
                },
            }
        return json.JSONEncoder.default(self, obj)

    def numpy_to_dict(self, obj: np.ndarray) -> dict:
        return {
            "__numpy__": base64.b64encode(
                obj.data if obj.flags.c_contiguous else obj.tobytes()
            ).decode("ascii"),
            "dtype": np.lib.format.dtype_to_descr(obj.dtype),
            "shape": obj.shape,
        }


def load_params(path):
    def dict_to_numpy(obj: dict) -> np.ndarray:
        return np.frombuffer(
            bytearray(base64.b64decode(obj["__numpy__"])),
            np.lib.format.descr_to_dtype(obj["dtype"]),
        ).reshape(obj["shape"])

    def object_hook(dct):
        if "__numpy__" in dct:
            _numpy = dict_to_numpy(dct)
            return _numpy

        if "type" in dct and dct["type"] == "CharucoBoard":
            aruco_dict = cv2.aruco.Dictionary(
                dct["dictionary"]["bytesList"],
                dct["dictionary"]["markerSize"],
                dct["dictionary"]["maxCorrectionBits"],
            )
            return cv2.aruco.CharucoBoard(
                tuple(dct["size"]),
                dct["squareLength"],
                dct["markerLength"],
                aruco_dict,
            )

        return dct

    with open(path, "r") as fr:
        return json.load(fr, object_hook=object_hook)


def extract_corners_from_video(source, args):
    corners_list = []
    ids_list = []
    frame = -1
    used_frames = 0

    while True:
        frame += 1
        if isinstance(source, list):
            # glob
            if frame == len(source):
                break
            img = cv2.imread(source[frame])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if frame % args.framestep != 0:
                continue

        print(f"Searching for chessboard in frame {frame}... ", end="")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict)
        found = len(corners) > 0

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            for corner in corners:
                cv2.cornerSubPix(img, corner, (5, 5), (-1, -1), term)
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, board
            )
            if corners is not None and ids is not None and len(corners) > 3:
                used_frames += 1
                corners_list.append(corners)
                ids_list.append(ids)
                print("ok")
                if args.max_frames is not None and used_frames >= args.max_frames:
                    print(f"Found {used_frames} frames with the chessboard.")
                    break
            else:
                print("not found")
        else:
            print("not found")

        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, "%04d.png" % frame), img_chess)

    return corners_list, ids_list, w, h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate camera using a video of a chessboard or a sequence of images."
    )
    parser.add_argument("input", help="input video file or glob mask")
    parser.add_argument(
        "-o", "--out", help="output calibration json file", default=None
    )
    parser.add_argument(
        "--debug-dir",
        help="path to directory where images with detected chessboard will be written",
        default=None,
    )
    parser.add_argument("-c", "--corners", help="output corners file", default=None)
    parser.add_argument(
        "-fs",
        "--framestep",
        help="use every nth frame in the video",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-max",
        "--max-frames",
        help="limit the number of frames used for calibration",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--pixel-size", help="size of a pixel in micrometers", default=0, type=float
    )
    parser.add_argument(
        "--focal-length", help="focal length in mm", default=0, type=float
    )
    # parser.add_argument('--figure', help='saved visualization name', default=None)
    args = parser.parse_args()
    params = {}

    if "*" in args.input:
        source = glob(args.input)
    elif Path(args.input).is_dir():
        source = list(Path(args.input).glob("*.*"))
    elif Path(args.input).is_file() and Path(args.input).suffix == ".json":
        params = load_params(args.input)
        params.pop("rms")
        params.pop("camera_matrix")
        params.pop("dist_coefs")
    else:
        source = cv2.VideoCapture(args.input)

    pattern_size = (7, 5)

    obj_points = []
    img_points = []
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 7), 0.049, 0.030, aruco_dict)

    if not params:
        corners_list, ids_list, w, h = extract_corners_from_video(source, args)

        if args.focal_length == 0:
            focal_length = input("focal length in mm: ")
            args.focal_length = float(focal_length)
        if args.pixel_size == 0:
            pixel_size = input("pixel size in micrometers: ")
            args.pixel_size = float(pixel_size)

        fx = args.focal_length * 1000 / args.pixel_size  # 25mm / 2.5um = focal length / pixel size
        cameraMatrixInit = np.array(
            [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]]
        )
        params = {
            "charucoCorners": corners_list,
            "charucoIds": ids_list,
            "board": board,
            "imageSize": (w, h),
            "cameraMatrix": cameraMatrixInit,
            "distCoeffs": np.zeros((5, 1)),
            "flags": cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_FOCAL_LENGTH
            | cv2.CALIB_ZERO_TANGENT_DIST,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                100000000,
                1e-9,
            ),
        }

    print("\ncalibrating...")
    params["distCoeffs"] = np.zeros((5, 1))
    (
        rms,
        camera_matrix,
        dist_coefs,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv2.aruco.calibrateCameraCharucoExtended(**params)

    if rms > 1.0:
        # An RMS error of 300 means that, on average, each projected points is 300 px away from its actual position.
        print(f"\033[33mRMS: {rms:.3f}; [Warning] RMS > 1.0\033[0m")
    else:
        print(f"\033[32mRMS: {rms:.3f}\033[0m")
    # print camera matrix in a python readable format
    print("[")
    for param in camera_matrix:
        print("    [" + ", ".join([f"{p}" for p in param]) + "],")
    print("]")
    print(
        "distortion coefficients: ["
        + ", ".join([str(p) for p in dist_coefs.ravel().tolist()])
        + "]"
    )

    params["rms"] = rms
    params["camera_matrix"] = camera_matrix.tolist()
    params["dist_coefs"] = dist_coefs.tolist()
    params.pop("distCoeffs")

    if args.out:
        with open(args.out, "w") as fw:
            json.dump(params, fw, cls=CalibrationJsonEncoder, indent=4)
