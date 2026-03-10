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
  echo "Usage: ./prepare_IXI.sh <input_dir> <output_dir> <modality>"
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
mkdir -p $DATA_DIR/v1resampled/IXI/$MODALITY
python resample.py -i $INPUT_DIR/$MODALITY -o $DATA_DIR/v1resampled/IXI/$MODALITY -r 1.0 1.0 1.0

# Rename files for standard naming
for file in $DATA_DIR/v1resampled/IXI/$MODALITY/*
do
  mv "$file" "${file%-${MODALITY^^}.nii.gz}_${MODALITY}.nii.gz"
done

echo "Generate masks"
CUDA_VISIBLE_DEVICES=0 hd-bet -i $DATA_DIR/v1resampled/IXI/$MODALITY -o $DATA_DIR/v2skullstripped/IXI/$MODALITY
python extract_masks.py -i $DATA_DIR/v2skullstripped/IXI/$MODALITY -o $DATA_DIR/v2skullstripped/IXI/mask
python replace.py -i $DATA_DIR/v2skullstripped/IXI/mask -s " _${MODALITY}" ""

echo "Register $MODALITY"
python registration.py -i $DATA_DIR/v2skullstripped/IXI/$MODALITY -o $DATA_DIR/v3registered_non_iso/IXI/$MODALITY --modality=_$MODALITY -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cut to brain"
python cut.py -i $DATA_DIR/v3registered_non_iso/IXI/$MODALITY -m $DATA_DIR/v3registered_non_iso/IXI/mask/ -o $DATA_DIR/v3registered_non_iso_cut/IXI/ -mode $MODALITY

echo "Bias Field Correction"
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/IXI/$MODALITY -o $DATA_DIR/v4correctedN4_non_iso_cut/IXI/$MODALITY -m $DATA_DIR/v3registered_non_iso_cut/IXI/mask

mkdir -p $DATA_DIR/v4correctedN4_non_iso_cut/IXI/mask
cp $DATA_DIR/v3registered_non_iso_cut/IXI/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/IXI/mask
echo "Done"

# Copy files from output directory to the project data directory
