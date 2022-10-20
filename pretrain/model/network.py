import torch
import torchvision
from torch import nn
from model.metric_learning import ArcMarginProduct


class LandmarkNet(nn.Module):
    def __init__(self,
                 args,
                 num_classes):
        super(LandmarkNet, self).__init__()
        self.encoder = get_encoder(args)
        if args.dataset_name == 'gldv2':
            self.loss_module = args.loss_module
            if self.loss_module == 'arcface':
                self.final_layer = ArcMarginProduct(args.features_dim, num_classes,
                                                    device=args.device,
                                                    s=args.arcface_s,
                                                    m=args.arcface_margin,
                                                    easy_margin=False,
                                                    ls_eps=args.arcface_ls_eps)  # label smoothing
        else:
            self.loss_module = ""

    def forward(self, x, labels=None):
        x = self.encoder(x)
        if self.loss_module == 'arcface':
            x = self.final_layer(x, labels)
        return x


def get_encoder(args):
    """
    Function that returns the selected model as a feature extractor
    """
    if args.arch.startswith("r"):  # It's a ResNet
        if args.arch.startswith("r18"):
            encoder = torchvision.models.resnet18(pretrained=True)
        elif args.arch.startswith("r50"):
            encoder = torchvision.models.resnet50(pretrained=True)
        elif args.arch.startswith("r101"):
            encoder = torchvision.models.resnet101(pretrained=True)
        encoder.fc = nn.Linear(encoder.fc.in_features, args.features_dim)

    elif args.arch == "vgg16":
        encoder = torchvision.models.vgg16(pretrained=True)
        encoder.classifier[6] = nn.Linear(encoder.classifier[6].in_features, args.features_dim)

    return encoder
