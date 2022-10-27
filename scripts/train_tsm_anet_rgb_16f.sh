#python ../main.py anet RGB \
#     --arch resnet50 --num_segments 16 --gpus 0 1 2 3 --rescale_to 192\
#     --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
#     --batch-size 16 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#     --shift --shift_div=8 --shift_place=blockres --npb
 


python ../main.py anet RGB \
     --arch resnet50 --num_segments 16 --gpus 0 1 2 3 --rescale_to 192\
     --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --npb
# python main.py something RGB \
#      --arch resnet50 --num_segments 8 \
#      --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
#      --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#      --shift --shift_div=8 --shift_place=blockres --npb
