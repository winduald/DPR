
# NUM_QUEIRES="1000 5000 10000 50000 100000000"
# FOLDER_NAME="/checkpoint/xiaojianwu/data/askubuntu/dpr/logs"
SRUN_PARAM="--gres gpu:volta:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 -C volta32gb --partition learnfair --time 4000 --mem-per-cpu 7G"
# for NUM_QUERY in $NUM_QUEIRES
# do
#     echo $NUM_QUERY
#     srun --job-name askubuntu_data_${NUM_QUERY} --output ${FOLDER_NAME}/askubuntu_data_${NUM_QUERY}_train.log --error ${FOLDER_NAME}/askubuntu_data_${NUM_QUERY}_train.stderr ${SRUN_PARAM} \
#         python dpr/data/askubuntu_data.py --num_hard_negatives 10 --num_easy_negatives 20 --max_num_queries ${NUM_QUERY} &
# done

NUM_SHARDS=30
FOLDER_NAME="/checkpoint/xiaojianwu/data/askubuntu/dpr/shards"
for ((shard_id=0;shard_id<NUM_SHARDS;shard_id++))
do
    srun --job-name askubuntu_data_${shard_id} --output ${FOLDER_NAME}/askubuntu_data_${shard_id}_train.log --error ${FOLDER_NAME}/askubuntu_data_${shard_id}_train.stderr ${SRUN_PARAM} \
         python dpr/data/askubuntu_data.py --max_num_queries 1000000 --num_shards ${NUM_SHARDS} --shard_id ${shard_id} --output_folder ${FOLDER_NAME} &
done
