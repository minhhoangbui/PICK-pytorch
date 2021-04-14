from .resnet import resnet18, resnet50, resnet34
from .shufflenet_v2 import shufflenet_v2_x1_0, shufflenet_v2_x0_5, shufflenet_v2_x1_5, shufflenet_v2_x2_0


class ImageBackboneFactory:
    @staticmethod
    def get_backbones(name, pretrained, **kwargs):
        backbone = eval(name)(pretrained=pretrained)
        return backbone
