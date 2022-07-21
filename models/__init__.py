from models.hrnet import get_hrnet_model, get_hrnet_cosine_model
from models.segformer import mit_b4, mit_b0, mit_b1, mit_b2, mit_b3, mit_b5

def get_segmentation_model(name, num_classes, **kwargs):
    configs = kwargs.pop('configs') if 'configs' in kwargs else None
    if 'hrnet' in name:
        if 'cosine' in name:
            print("Cosine HRNet: {}".format(name))
            return get_hrnet_cosine_model(configs, num_classes=num_classes)
        else:
            print("HRNet: {}".format(name))
            return get_hrnet_model(configs, num_classes=num_classes)
    elif 'segformer' in name:
        if configs.segformer_type == "b4":
            print("Using MIT B4")
            model = mit_b4()
            model.init_weights(configs.PRETRAIN_PATH)
            print("Loaded MIT B4 ImageNet weights")
            return model
        elif configs.segformer_type == "b1":
            print("Using MIT B1")
            model = mit_b1()
            model.init_weights(configs.PRETRAIN_PATH)
            print("Loaded MIT B1 ImageNet weights")
            return model
    else:
        raise NotImplementedError("No implementation of specified model: {}".format(name))
