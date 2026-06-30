Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commonds:

python opencood/tools/train.py --hypes_yaml /opencood/hypes_yaml/opv2v/mm/mm_v2x.yaml [--model_dir  ${CHECKPOINT_FOLDER} --half]
Arguments Explanation:

hypes_yaml: the path of the training configuration file, e.g. opencood/hypes_yaml/second_early_fusion.yaml, meaning you want to train an early fusion model which utilizes SECOND as the backbone. See Tutorial 1: Config System to learn more about the rules of the yaml files.
model_dir (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the model_dir is given, the trainer will discard the hypes_yaml and load the config.yaml in the checkpoint folder.

Test the model
Before you run the following command, first make sure the validation_dir in config.yaml under your checkpoint folder refers to the testing dataset path, e.g. opv2v_data_dumping/test.

python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
Arguments Explanation:

model_dir: the path to your saved model.
fusion_method: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
show_vis: whether to visualize the detection overlay with point cloud.
show_sequence : the detection results will visualized in a video stream. It can NOT be set with show_vis at the same time.
global_sort_detections: whether to globally sort detections by confidence score. If set to True, it is the mainstream AP computing method, but would increase the tolerance for FP (False Positives). OPV2V paper does not perform the global sort. Please choose the consistent AP calculation method in your paper for fair comparison.
