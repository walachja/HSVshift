import colorsys

import numpy as np
from Augmentor.Operations import Operation
from PIL import Image


class HSVShifting(Operation):

    def __init__(self, probability, hue_shift, saturation_scale, saturation_shift, value_scale, value_shift):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.saturation_shift = saturation_shift
        self.value_scale = value_scale
        self.value_shift = value_shift

        self.rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
        self.hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    def perform_operation(self, image):
        r, g, b = image.split()  # split into individuals r,g,b layers

        nR = np.array(r) / 255.
        nG = np.array(g) / 255.
        nB = np.array(b) / 255.

        h, s, v = np.array(self.rgb_to_hsv(nR, nG, nB))

        h += np.random.uniform(low=-self.hue_shift, high=self.hue_shift, size=1)
        s *= np.random.uniform(low=1 / (1 + self.saturation_scale), high=1 + self.saturation_scale, size=1)
        s += np.random.uniform(low=-self.saturation_shift, high=self.saturation_shift, size=1)
        v *= np.random.uniform(low=1 / (1 + self.value_scale), high=1 + self.value_scale, size=1)
        v += np.random.uniform(low=-self.value_shift, high=self.value_shift, size=1)

        h = np.clip(h, 0, 1)
        s = np.clip(s, 0, 1)
        v = np.clip(v, 0, 1)

        rgb = np.dstack(self.hsv_to_rgb(h, s, v))
        rgb = np.uint8(np.round(rgb * 255.))

        return Image.fromarray(rgb, "RGB")  # remerge RGB channels

