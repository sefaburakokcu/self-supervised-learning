import timm

from torchvision.models import resnet18, resnet50


def create_encoder(arch='resnet18', pretrained=False):
    """Factory function for encoders"""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
    elif arch == 'vit_small':
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model
