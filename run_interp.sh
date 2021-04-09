export SQUAD_DIR=outputs
DATASET=${1:-none}
METHOD=${2:-none}
ACTION=${3:-run}
SPLIT=${4:-dev}
DEVICES=${5:-0}

MAX_SEQ_LENGTH=512
if [ "$DATASET" = "hpqa" ]; then
    MAX_SEQ_LENGTH=512
elif [ "$DATASET" = "squad" ]; then
    MAX_SEQ_LENGTH=512
else
  echo "Invalid dataset ${DATASET}"
  exit 1
fi


if [ "$METHOD" = "tokig" ]; then
  BATCH_SIZE=50
elif [ "$METHOD" = "atattr" ]; then
  BATCH_SIZE=20
elif [ "$METHOD" = "latattr" ]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=10
fi

MODEL_TYPE="roberta-base"
if [ "$ACTION" = "run" ]; then
  if [ "$METHOD" = "latattr"  -o "$METHOD" = "tokig" -o  "$METHOD" = "atattr" ]; then
    echo "Run "${METHOD}
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python -u run_${METHOD}.py \
      --model_type roberta \
      --model_name_or_path checkpoints/${DATASET}_roberta-base \
      --dataset ${DATASET} \
      --predict_file $SQUAD_DIR/${SPLIT}_${DATASET}.json \
      --overwrite_output_dir \
      --max_seq_length ${MAX_SEQ_LENGTH} \
      --output_dir  predictions/${DATASET} \
      --per_gpu_eval_batch_size ${BATCH_SIZE} \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE}
  elif [ "$METHOD" = "arch" ]; then
    echo "Run archipelago method"
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python -u run_arch.py \
      --model_type roberta \
      --model_name_or_path checkpoints/${DATASET}_roberta-base \
      --dataset ${DATASET} \
      --predict_file $SQUAD_DIR/${SPLIT}_${DATASET}.json \
      --overwrite_output_dir \
      --max_seq_length ${MAX_SEQ_LENGTH} \
      --output_dir  predictions/${DATASET} \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE}
  else
    echo "No such method" $METHOD
  fi
elif [ "$ACTION" = "vis" ]; then
  if [ "$METHOD" = "latattr"  -o "$METHOD" = "tokig" -o  "$METHOD" = "atattr" ]; then
    echo "Vis "${METHOD}
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python -u run_${METHOD}.py \
      --model_type roberta \
      --tokenizer_name $MODEL_TYPE \
      --model_name_or_path $MODEL_TYPE \
      --dataset ${DATASET} \
      --do_vis \
      --output_dir  predictions/${DATASET} \
      --predict_file $SQUAD_DIR/${SPLIT}_${DATASET}.json \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE} \
      --visual_dir visualizations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE}
  else
    echo "Method not visualizable"
  fi
elif [ "$ACTION" = "eval" ]; then
  if [ "$METHOD" = "ig" ] || [ "$METHOD" = "arch" ] || [ "$METHOD" = "probe" ] || [ "$METHOD" = "tokig" ]; then
    echo "Eval ${METHOD}"
    python -u auto_eval.py \
      --method ${METHOD}   \
      --dataset ${DATASET} \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE}      
  else
    echo "No such method"
  fi
else
  echo "run or vis"
fi
