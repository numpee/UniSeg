python post_evaluation.py method=bcenull batch_size=1 model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/recent.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImage_Recent

python post_evaluation.py method=bcenull batch_size=1 model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/kitti_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImage_KITTIBest

python post_evaluation.py method=bcenull batch_size=1 model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/wilddash_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImage_WildDashBest

python post_evaluation.py method=bcenull batch_size=1 model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/camvid_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImage_CamvidBest

### Sliding Window
python post_evaluation.py method=bcenull batch_size=1 use_sliding_window=True model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/recent.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImageSlide_Recent

python post_evaluation.py method=bcenull batch_size=1 use_sliding_window=True model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/kitti_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImageSlide_KITTIBest

python post_evaluation.py method=bcenull batch_size=1 use_sliding_window=True model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/wilddash_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImageSlide_WildDashBest

python post_evaluation.py method=bcenull batch_size=1 use_sliding_window=True model_folder_path=/net/acadia14a/data/dokim/checkpoints/segformer_b1_bcenull_test_AdamW_2022-05-25_5/camvid_mIoU_best.pth model=segformer_b4 segformer_type=b1 experiment_name=SegFormer_B1_BCENull_CIBM_FullImageSlide_CamvidBest


### Second stage (with W18 class relations)
# B1
python segformer_second_stage.py epochs=150 segformer_type=b1 experiment_name=SegFormerB1_CIBM_SS

# B2
python segformer_second_stage.py epochs=150 segformer_type=b4 experiment_name=SegFormerB4_CIBM_SS batch_size=32