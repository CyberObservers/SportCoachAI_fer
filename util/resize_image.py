import cv2
import numpy as np


def resize_image(Img: np.ndarray, target_size: tuple) -> np.ndarray:
    factor_0 = target_size[0] / Img.shape[0]
    factor_1 = target_size[1] / Img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(Img.shape[1] * factor),
        int(Img.shape[0] * factor),
    )
    Img = cv2.resize(Img, dsize)

    diff_0 = target_size[0] - Img.shape[0]
    diff_1 = target_size[1] - Img.shape[1]

    Img = np.pad(
        Img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2)
        ),
        "constant",
    )
    if Img.shape[0:2] != target_size:
        Img = cv2.resize(Img, target_size)

    # 直接扩展维度，无需使用 img_to_array
    Img = np.expand_dims(Img, axis=0)

    if Img.max() > 1:
        Img = (Img.astype(np.float32) / 255.0).astype(np.float32)

    return Img
