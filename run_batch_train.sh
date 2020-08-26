TASKS="semeval"
# MODES="bert_share question_share no_share doc_share"
MODES="doc_share"
NUM_RUN=1
SRUN_PARAM="--gres gpu:volta:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 -C volta32gb --partition learnfair --time 4000 --mem-per-cpu 7G"
NUM_EPOCH=60
for TASK in $TASKS
do
    for ((run_id=0;run_id<NUM_RUN;run_id++))
    do 
        for PREFIX in $MODES
        do
            # echo $PREFIX
            FOLDER_NAME=/checkpoint/xiaojianwu/DPR/results/${TASK}/${PREFIX}/run_${run_id}
            if [ ! -d $FOLDER_NAME ]
            then
                mkdir $FOLDER_NAME
            fi
            #remove 
            rm ${FOLDER_NAME}/*

            if [ "$PREFIX" = "bert_share" ]
            then
                OPTION="--share_encoder --hard_negatives 30 --batch_size 2 --output_dir ${FOLDER_NAME}"
            elif [ "$PREFIX" = "question_share" ]
            then
                OPTION="--share_encoder --which_encoder_to_load q --hard_negatives 30 --batch_size 2 --output_dir ${FOLDER_NAME} --model_file /checkpoint/xiaojianwu/DPR/checkpoint/retriever/single/nq/bert-base-encoder.cp"
            elif [ "$PREFIX" = "no_share" ]
            then
                OPTION="--hard_negatives 30 --batch_size 2 --output_dir ${FOLDER_NAME} --model_file /checkpoint/xiaojianwu/DPR/checkpoint/retriever/single/nq/bert-base-encoder.cp"
            elif [ "$PREFIX" = "doc_share" ]
            then
                OPTION="--share_encoder --which_encoder_to_load c --hard_negatives 30 --batch_size 2 --output_dir ${FOLDER_NAME} --model_file /checkpoint/xiaojianwu/DPR/checkpoint/retriever/single/nq/bert-base-encoder.cp"
            fi

            if [[ ( "$PREFIX" == "bert_share") ]] || [[ ( "$PREFIX" == "no_share") ]] || [[ ( "$PREFIX" == "doc_share") ]]
            then
                DATA="--train_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1bs_q2bs_train.json --dev_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1bs_q2bs_test.json"
            elif [ "$PREFIX" == "question_share" ]
            then
                DATA="--train_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1s_q2s_train.json --dev_file /checkpoint/xiaojianwu/data/${TASK}/dpr/${TASK}_q1s_q2s_test.json"
            fi

            # echo $FOLDER_NAME
            # echo $DATA
            # echo $OPTION
            echo "python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased --sequence_length 256 ${DATA} --shuffle_positive_ctx --checkpoint_file_name run_${NUM_EPOCH} --num_train_epochs ${NUM_EPOCH} ${OPTION}"
            srun --job-name ${TASK}_${run_id}_${PREFIX} --output $FOLDER_NAME/train.log --error ${FOLDER_NAME}/train.stderr ${SRUN_PARAM} \
                python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased \
                --sequence_length 256 ${DATA} \
                --checkpoint_file_name run_${NUM_EPOCH} --shuffle_positive_ctx --num_train_epochs ${NUM_EPOCH} \
                ${OPTION} &

            # srun --job-name ${TASK}_${run_id}_${PREFIX} --output $FOLDER_NAME/train.log --error ${FOLDER_NAME}/train.stderr ${SRUN_PARAM} \
            #     python train_dense_encoder.py --encoder_model_type hf_bert --pretrained_model_cfg bert-base-uncased \
            #     --sequence_length 256 ${DATA} \
            #     --checkpoint_file_name run_${NUM_EPOCH} --num_train_epochs ${NUM_EPOCH} \
            #     --share_encoder --hard_negatives 30 --batch_size 2 --output_dir ${FOLDER_NAME} \
            #     ${OPTION} &
        done
    done
done
