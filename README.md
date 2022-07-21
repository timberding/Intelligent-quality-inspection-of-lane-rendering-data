# Intelligent-quality-inspection-of-lane-rendering-data

# A brief demo for Intelligent inspection of lane rendering data

## 1. Brief introduction.

This repo provides a simple solution for intelligent inspection of lane rendering data. We use a classification method to perform this task.

## 2. Usage.

### 2.1. Environment
```bash
pytorch==1.11.0
```

### 2.2. Training and generating result
```bash
python demo_script.py \
    --traindir /path/to/train_data \
    --train_metadir /path/to/train_label \
    --train_metafile train_label.csv \
    --test_metadir /path/to/test_data \
    --test_metafile test_image.csv
```
