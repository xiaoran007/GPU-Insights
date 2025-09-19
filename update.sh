source ./data.sh

python scripts/manage-data.py add \
  --vendor "$Vendor" \
  --architecture "$Architecture" \
  --device "$Device" \
  --memory "$Memory" \
  --platform "$Platform" \
  --fp32 $FP32 \
  --fp32bs $FP32BS \
  --fp16 $FP16 \
  --fp16bs $FP16BS \
  --note "$Note" \
  --date "$Date"