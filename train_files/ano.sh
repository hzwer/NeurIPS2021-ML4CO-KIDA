for i in $( seq 1 50)
do
    python generate_data.py anonymous --file_count ${i} --njobs 5 --train_size 10000 --valid_size 4000
    if [ $((i%10)) -eq 0 ];then
    python train.py anonymous  --exp_name 3_anonymous_dagger --file_count ${i} --epoch 100
    else
    python train.py anonymous  --exp_name 3_anonymous_dagger --file_count ${i} --epoch 10
    fi
done