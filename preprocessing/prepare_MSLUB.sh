#!/bin/bash
# cli arguments: 
# 1. path to data directory
# 2. path to output directory
# 3. MRI modality (t1, t2, flair, t1c)

INPUT_DIR=$1
DATA_DIR=$2
MODALITY=$3

# Ensure all arguments are provided
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] || [ -z "$MODALITY" ]; then
  echo "Usage: ./prepare_MSLUB.sh <input_dir> <output_dir> <modality>"
  exit 1
fi

# Validate the modality input
if [[ "$MODALITY" != "t1" && "$MODALITY" != "t2" && "$MODALITY" != "flair" && "$MODALITY" != "t1c" ]]; then
  echo "Invalid modality. Choose from: t1, t2, flair, t1c."
  exit 1
fi

# Ensure input directory is not relative
if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]; then
  echo "Please use absolute paths for input_dir"
  exit 1
fi

echo "Resample"
mkdir -p $DATA_DIR/v1resampled/MSLUB/$MODALITY
python resample.py -i $INPUT_DIR/$MODALITY -o $DATA_DIR/v1resampled/MSLUB/$MODALITY -r 1.0 1.0 1.0

# Rename files for standard naming
for file in $DATA_DIR/v1resampled/MSLUB/$MODALITY/*
do
  mv "$file" "${file%_${MODALITY^^}W.nii.gz}_${MODALITY}.nii.gz"
done

echo "Generate masks"
CUDA_VISIBLE_DEVICES=0 hd-bet -i $DATA_DIR/v1resampled/MSLUB/$MODALITY -o $DATA_DIR/v2skullstripped/MSLUB/$MODALITY
python extract_masks.py -i $DATA_DIR/v2skullstripped/MSLUB/$MODALITY -o $DATA_DIR/v2skullstripped/MSLUB/mask
python replace.py -i $DATA_DIR/v2skullstripped/MSLUB/mask -s " _${MODALITY}" ""

# Copy segmentation masks to the data directory
mkdir -p $DATA_DIR/v2skullstripped/MSLUB/seg
cp -r $INPUT_DIR/seg/* $DATA_DIR/v2skullstripped/MSLUB/seg/

for file in $DATA_DIR/v2skullstripped/MSLUB/seg/*
do
  mv "$file" "${file%consensus_gt.nii.gz}seg.nii.gz"
done

echo "Register $MODALITY"
python registration.py -i $DATA_DIR/v2skullstripped/MSLUB/$MODALITY -o $DATA_DIR/v3registered_non_iso/MSLUB/$MODALITY --modality=_$MODALITY -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/MSLUB/$MODALITY -m $DATA_DIR/v3registered_non_iso/MSLUB/mask/ -o $DATA_DIR/v3registered_non_iso_cut/MSLUB/ -mode $MODALITY

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/MSLUB/$MODALITY -o $DATA_DIR/v4correctedN4_non_iso_cut/MSLUB/$MODALITY -m $DATA_DIR/v3registered_non_iso_cut/MSLUB/mask

mkdir -p $DATA_DIR/v4correctedN4_non_iso_cut/MSLUB/mask
cp $DATA_DIR/v3registered_non_iso_cut/MSLUB/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/MSLUB/mask

mkdir -p $DATA_DIR/v4correctedN4_non_iso_cut/MSLUB/seg
cp $DATA_DIR/v3registered_non_iso_cut/MSLUB/seg/* $DATA_DIR/v4correctedN4_non_iso_cut/MSLUB/seg

echo "Done"
