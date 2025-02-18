from s2cloudless import S2PixelCloudDetector
import numpy as np

cloud_detector = S2PixelCloudDetector(
    all_bands=True, threshold=0.4, average_over=4, dilation_size=2
)


def generate_single_cloud_mask_for_image(picture, cloud_detector):
    picture_for_clouds = picture.copy() / 10000
    cloud_mask = cloud_detector.get_cloud_masks(picture_for_clouds[np.newaxis, ...])
    return picture, cloud_mask
