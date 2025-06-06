from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Proloculus(BaseFeature):
    def __init__(self, type, shape, diameter, random_pixel_div_mm_offset=0):
        self.type = type
        self.shape = shape
        self.diameter0 = diameter
        self.random_pixel_div_mm_offset = random_pixel_div_mm_offset
        self.diameter = round(
            diameter * shell_world_pixel / (shell_pixel_div_mm + random_pixel_div_mm_offset), 2
        )

    def getShape(self):
        txt = ""
        # center, weight = self.getCenterAndWeight(self.shape)
        txt += self.standardRangeFilter(
            proloculus_size_classes,
            self.diameter0 * shell_world_pixel / (shell_pixel_div_mm + self.random_pixel_div_mm_offset),
        )
        txt += " "
        # TODO: 肾形
        txt += self.standardRangeFilter(proloculus_shape_classes, 0.1)
        return txt.strip()

    def genUserInput(self):
        txt = "Proloculus {shape}, ".format(shape=self.getShape())
        txt += "with diameter measuring {diameter} mm. ".format(diameter=self.diameter)
        return [f"<proloculus>{txt}</proloculus>"]

    def genInput(self):
        txt = "initial chamber(proloculus): {length} mm\n".format(length=self.diameter)
        return txt
