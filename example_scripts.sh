# Single Dataset setting (Cityscapes)
python main.py method=single dataset_names=[cityscapes] eval_dataset_names=[cityscapes] logging=no_wandb epochs=100 experiment_name=cityscapes_single

# Single Dataset setting (Mapillary). You need to set num_classes and dataset_key according to the dataset, as well as the eval_dataset_name
python main.py method=single dataset_names=[mapillary] eval_dataset_names=[mapillary] logging=no_wandb epochs=100 \
num_classes=65 dataset_key=mapillary experiment_name=mapillary_single

# Unified Setting (CE Loss)
python main.py method=unified epochs=100 experiment_name=unified_celoss

# UniSeg (BCE + Null). This is the default setting
python main.py experiment_name=uniseg

# You may turn off BCE Null (just BCE loss):
python main.py bce_no_reduce=False experiment_name=bce_loss