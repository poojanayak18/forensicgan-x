import sys
from types import ModuleType

from torchvision.transforms.functional import rgb_to_grayscale


def apply_torchvision_fix():
    """
    Compatibility fix for BasicSR 1.4.2.

    BasicSR imports:
        torchvision.transforms.functional_tensor

    Modern torchvision removed that module.
    We provide the expected module path using
    the current torchvision implementation.
    """

    module = ModuleType(
        "torchvision.transforms.functional_tensor"
    )

    module.rgb_to_grayscale = rgb_to_grayscale

    sys.modules[
        "torchvision.transforms.functional_tensor"
    ] = module


if __name__ == "__main__":
    apply_torchvision_fix()

    print(
        "Torchvision compatibility fix: SUCCESS"
    )