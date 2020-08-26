TASKS="semeval"
MODES="bert_share question_share no_share doc_share"
# MODES="bert_share"
MODEL_FILE_TYPE="trained"

for TASK in $TASKS
do
    for PREFIX in $MODES
    do
        # echo $PREFIX
        # FOLDER_NAME="/checkpoint/xiaojianwu/DPR/results/${TASK}/${PREFIX}/run_0"
        if [ "$PREFIX" == "bert_share" ]
        then
            OPTION="--share_encoder --hard_negatives 30 --output_dir /checkpoint/xiaojianwu/DPR/results/${TASK}"
        elif [ "$PREFIX" = "question_share" ]
        then
            OPTION="--share_encoder --which_encoder_to_load q --hard_negatives 30 --output_dir /checkpoint/xiaojianwu/DPR/results/${TASK}"
        elif [ "$PREFIX" == "no_share" ]
        then
            OPTION="--hard_negatives 30 --output_dir /checkpoint/xiaojianwu/DPR/results/${TASK}"
        elif [ "$PREFIX" == "doc_share" ]
        then
            OPTION="--share_encoder --which_encoder_to_load c --hard_negatives 30 --output_dir /checkpoint/xiaojianwu/DPR/results/${TASK}"
        fi

        if [ "$MODEL_FILE_TYPE" == "oqa" ]
        then
            MODEL_FILE_PATH="--model_file /checkpoint/xiaojianwu/DPR/checkpoint/retriever/single/nq/bert-base-encoder.cp"
        elif [ "$MODEL_FILE_TYPE" == "trained" ]
        then
            MODEL_FILE_PATH="--model_file /checkpoint/xiaojianwu/DPR/results/${TASK}/${PREFIX}/run_0/run_30.29.119"
        fi

        if [[ ( "$PREFIX" == "bert_share") ]] || [[ ( "$PREFIX" == "no_share") ]] || [[ ( "$PREFIX" == "doc_share") ]]
        then
            DATA="--dev_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1bs_q2bs_test.json"
        elif [ "$PREFIX" == "question_share" ]
        then
            DATA="--dev_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1s_q2s_test.json"
        fi

        echo "train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --sequence_length 256 ${DATA} ${OPTION} ${MODEL_FILE_PATH}"
        python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased \
        --sequence_length 256 ${DATA} ${MODEL_FILE_PATH} ${OPTION}
    done
done


