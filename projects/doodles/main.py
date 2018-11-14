from torchvision.models import resnet18, resnet34, resnet50

from cli import parse_args


def main():
    args = parse_args({
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50
    })




if __name__ == '__main__':
    main()
