import mmcv
import numpy as np

from mmengine.registry import TRANSFORMS
from mmcv.transforms import Resize
import mmengine

@TRANSFORMS.register_module()
class AlignResize(Resize):
    """Resize transformation with alignment (padding) to ensure that the final
    image shape is divisible by a given size_divisor.

    This class inherits from the newest Resize class and adds an extra step to pad
    the image after resizing.

    Args:
        scale (tuple[int]): Desired output scale (width, height) before alignment.
        size_divisor (int): The number by which the height and width must be divisible.
        keep_ratio (bool): Whether to keep the aspect ratio.
    """
    def __init__(self, scale, size_divisor, keep_ratio=True):
        super().__init__(scale=scale, keep_ratio=keep_ratio)
        self.size_divisor = size_divisor

    def _align(self, img, size_divisor, interpolation=None):
        align_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        align_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        if interpolation == None:
            img = mmcv.imresize(img, (align_w, align_h))
        else:
            img = mmcv.imresize(img, (align_w, align_h), interpolation=interpolation)
        return img

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            #### align ####
            img = self._align(img, self.size_divisor)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)

            h, w = img.shape[:2]
            assert int(np.ceil(h / self.size_divisor)) * self.size_divisor == h and \
                   int(np.ceil(w / self.size_divisor)) * self.size_divisor == w, \
                   "img size not align. h:{} w:{}".format(h,w)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2]  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
                gt_seg = self._align(gt_seg, self.size_divisor, interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
                h, w = gt_seg.shape[:2]
                assert int(np.ceil(h / self.size_divisor)) * self.size_divisor == h and \
                       int(np.ceil(w / self.size_divisor)) * self.size_divisor == w, \
                    "gt_seg size not align. h:{} w:{}".format(h, w)
            results[key] = gt_seg

