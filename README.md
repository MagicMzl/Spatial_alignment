Training and Testing OpenCOOD Models

OpenCOOD uses YAML configuration files to manage all parameters required for training and evaluation. This guide explains how to train a model from scratch, continue training from a checkpoint, and test a trained model.

Train Your Model

To train your own model from scratch or continue training from an existing checkpoint, run:

python opencood/tools/train.py \
  --hypes_yaml /opencood/hypes_yaml/opv2v/mm/mm_v2x.yaml \
  [--model_dir ${CHECKPOINT_FOLDER} --half]

Arguments

Argument	Description
--hypes_yaml	Path to the training configuration YAML file. For example, opencood/hypes_yaml/second_early_fusion.yaml trains an early fusion model using SECOND as the backbone. Refer to Tutorial 1: Config System for more details about YAML configuration rules.
--model_dir	Optional. Path to a checkpoint folder. This is used to fine-tune or continue training from a trained model. When model_dir is provided, the trainer ignores the given hypes_yaml and instead loads config.yaml from the checkpoint folder.
--half	Optional. Enables half-precision training if supported.

Test the Model

Before running inference, make sure the validation_dir field in the config.yaml file under your checkpoint folder points to the testing dataset path.

For example:

validation_dir: opv2v_data_dumping/test

Then run:

python opencood/tools/inference.py \
  --model_dir ${CHECKPOINT_FOLDER} \
  --fusion_method ${FUSION_STRATEGY} \
  [--show_vis] \
  [--show_sequence]

Arguments

Argument	Description
--model_dir	Path to the saved model checkpoint folder.
--fusion_method	Fusion strategy used during inference. Currently supports early, late, and intermediate.
--show_vis	Optional. Visualizes detection overlays with the point cloud.
--show_sequence	Optional. Visualizes detection results as a video stream. This option cannot be used together with --show_vis.
--global_sort_detections	Optional. Globally sorts detections by confidence score. Setting this to True follows the mainstream AP computation method, but may increase tolerance for false positives. The OPV2V paper does not use global sorting. For fair comparison, use the same AP calculation method as your paper or baseline.

Example

Training with a YAML configuration file:

python opencood/tools/train.py \
  --hypes_yaml /opencood/hypes_yaml/opv2v/mm/mm_v2x.yaml

Continuing training from a checkpoint:

python opencood/tools/train.py \
  --hypes_yaml /opencood/hypes_yaml/opv2v/mm/mm_v2x.yaml \
  --model_dir checkpoints/mm_v2x

Running inference with intermediate fusion:

python opencood/tools/inference.py \
  --model_dir checkpoints/mm_v2x \
  --fusion_method intermediate

Running inference with visualization:

python opencood/tools/inference.py \
  --model_dir checkpoints/mm_v2x \
  --fusion_method intermediate \
  --show_vis
