export CUDA_VISIBLE_DEVICES=0,1,2,3
python attack.py --method ultimate-combo-gen --batch_size 8  --gpu 1   --csv_dir  transferability_result --log log.log 
