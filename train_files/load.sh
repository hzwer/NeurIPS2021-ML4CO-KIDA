for i in $( seq 1 50)
do
    python generate_data.py load_balancing --file_count ${i} --njobs 5 --train_size 1000 --valid_size 400
    if [ $((i%10)) -eq 0 ];then
    python train.py load_balancing  --exp_name 2_load_balancing_dagger --file_count ${i} --epoch 100
    else
    python train.py load_balancing  --exp_name 2_load_balancing_dagger --file_count ${i} --epoch 10
    fi
done