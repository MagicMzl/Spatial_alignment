# OpenCOOD Training and Testing Guide

OpenCOOD uses YAML configuration files to manage training and testing parameters. This guide explains how to train a model from scratch, continue training from a checkpoint, and test a trained model.

---

## 📌 1. Train a Model

To train a model from scratch or continue training from an existing checkpoint, run:

    python opencood/tools/train.py \
        --hypes_yaml /opencood/hypes_yaml/opv2v/mm/mm_v2x.yaml \
        [--model_dir ${CHECKPOINT_FOLDER}] \
        [--half]

---

## 📌 2. Test a Trained Model

Before testing, make sure the `validation_dir` field in the `config.yaml` file under your checkpoint folder points to the testing dataset path.

For example:

    validation_dir: opv2v_data_dumping/test
