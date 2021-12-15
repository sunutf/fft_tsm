#python ../main.py somethingv2 RGB \
#     --arch resnet50 --num_segments 8 --gpus 0 1 2 3 \
#    --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
#    --batch-size 32 -j 16 --dropout 0.5 --consensus_type=max --eval-freq=1 \
#--is_rnn --rnn_rate_list 1 2 3 --hidden_dim 512 --npb
     
python ../main.py somethingv2 RGB \
	--arch resnet50 --num_segments 8 --gpus 0 1 2 3 \
	--gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
	--batch-size 24 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
	--dctidct --shift --shift_div=8 --shift_place=blockres --npb \
