#!/bin/bash
#cli arguments:
# 1. path to data directory
# 2. path to output directory
# 3. modality (e.g., t1, t2)

INPUT_DIR=$1
DATA_DIR=$2
MODALITY=$3

# Make the arguments mandatory and ensure input_dir is an absolute path
if [ -z "$INPUT_DIR" ] || [ -z "$DATA_DIR" ] || [ -z "$MODALITY" ]; then
  echo "Usage: ./prepare_MSLUB.sh <input_dir> <output_dir> <modality>"
  exit 1
fi


if [[ "$MODALITY" != "t1" && "$MODALITY" != "t2" && "$MODALITY" != "flair" && "$MODALITY" != "t1ce" ]]; then
  echo "Invalid modality. Choose from: t1, t2, flair, t1c."
  exit 1
fi

if [ "$INPUT_DIR" == "." ] || [ "$INPUT_DIR" == ".." ]; then
  echo "Please use absolute paths for input_dir"
  exit 1
fi

# Create necessary directories
mkdir -p $DATA_DIR/v2skullstripped/Brats21/mask

# Copy input data to the output directory
cp -r $INPUT_DIR/$MODALITY $INPUT_DIR/seg $DATA_DIR/v2skullstripped/Brats21/

echo "Extracting masks..."
python get_mask.py -i $DATA_DIR/v2skullstripped/Brats21/$MODALITY -o $DATA_DIR/v2skullstripped/Brats21/$MODALITY -mod $MODALITY
python extract_masks.py -i $DATA_DIR/v2skullstripped/Brats21/$MODALITY -o $DATA_DIR/v2skullstripped/Brats21/mask
python replace.py -i $DATA_DIR/v2skullstripped/Brats21/mask -s " _$MODALITY" ""

echo "Registering $MODALITY images..."
python registration.py -i $DATA_DIR/v2skullstripped/Brats21/$MODALITY \
  -o $DATA_DIR/v3registered_non_iso/Brats21/$MODALITY \
  --modality=_$MODALITY -trans Affine -templ sri_atlas/templates/T1_brain.nii

echo "Cutting to brain region..."
python cut.py -i $DATA_DIR/v3registered_non_iso/Brats21/$MODALITY \
  -m $DATA_DIR/v3registered_non_iso/Brats21/mask/ \
  -o $DATA_DIR/v3registered_non_iso_cut/Brats21/ -mode $MODALITY

echo "Applying N4 bias field correction..."
python n4filter.py -i $DATA_DIR/v3registered_non_iso_cut/Brats21/$MODALITY \
  -o $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/$MODALITY \
  -m $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/mask

# Copy masks and segmentation to final directory
mkdir -p $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/{mask,seg}
cp $DATA_DIR/v3registered_non_iso_cut/Brats21/mask/* $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/mask/
cp $DATA_DIR/v3registered_non_iso_cut/Brats21/seg/* $DATA_DIR/v4correctedN4_non_iso_cut/Brats21/seg/

echo "Processing completed!"

