from collections import OrderedDict

import torchvision.transforms as transforms

import transformations.joint_transforms as joint_transforms
import transformations.single_transforms as single_transforms


def transform_factory(configs):
    joint_transform_dict = OrderedDict({'scaleMin': joint_transforms.ScaleMin(configs.pre_size),
                                        'randomSizeCrop': joint_transforms.RandomSizeAndCrop(configs.crop_size,
                                                                                             pre_size=configs.pre_size,
                                                                                             scale_min=configs.scale_min,
                                                                                             scale_max=configs.scale_max,
                                                                                             ignore_index=-1),
                                        'resize': joint_transforms.Resize(configs.crop_size),
                                        'rotate': joint_transforms.Rotate(),
                                        'horizontalFlip': joint_transforms.RandomHorizontallyFlip()})
    joint_transform_list = []
    if "joint_augmentations" in configs:
        for aug in configs.joint_augmentations:
            joint_transform_list.append(joint_transform_dict[aug])
    else:
        joint_transform_list = list(joint_transform_dict.values())

    tensorize_and_normalize = [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]

    input_transform_list = [single_transforms.GaussianBlur()] + tensorize_and_normalize

    val_joint_transform_list = [joint_transforms.ResizeHeight(configs.eval_size),
                                joint_transforms.CenterCropPad(configs.eval_size)]

    train_joint_transforms = joint_transforms.Compose(joint_transform_list)
    train_input_transforms = transforms.Compose(input_transform_list)
    val_joint_transforms = joint_transforms.Compose(val_joint_transform_list)
    val_input_transforms = transforms.Compose(tensorize_and_normalize)

    return train_joint_transforms, train_input_transforms, val_joint_transforms, val_input_transforms


def transform_factory_test(configs):
    if configs.use_sliding_window:
        joint_transforms_list = [joint_transforms.ScaleMin(configs.img_size)]
    else:
        joint_transforms_list = [joint_transforms.ResizeHeight(configs.eval_size),
                                 joint_transforms.CenterCropPad(configs.eval_size)]
    tensorize_and_normalize = [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]

    test_joint_transform = joint_transforms.Compose(joint_transforms_list)
    test_input_transform = transforms.Compose(tensorize_and_normalize)
    return test_joint_transform, test_input_transform

def transform_factory_test_full_image(configs):
    joint_transforms_list = [joint_transforms.ScaleMin(configs.eval_size)]
    tensorize_and_normalize = [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]

    test_joint_transform = joint_transforms.Compose(joint_transforms_list)
    test_input_transform = transforms.Compose(tensorize_and_normalize)
    return test_joint_transform, test_input_transform
