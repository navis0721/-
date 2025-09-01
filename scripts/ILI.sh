# ############################predict length 24####################################
# python3 main_crossformer.py --data ILI \
# --in_len 48 --out_len 24 --seg_len 6 \
# --e_layers 3 \
# --learning_rate 1e-4 --dropout 0.6 --itr 5 \
# --attn 'prob'

############################predict length 36####################################
python3 main_crossformer.py --data ILI \
--in_len 48 --out_len 36 --seg_len 6 \
--e_layers 3 \
--learning_rate 1e-4 --dropout 0.6 --itr 5 

############################predict length 48####################################
python3 main_crossformer.py --data ILI \
--in_len 60 --out_len 48 --seg_len 6 \
--e_layers 3 \
--learning_rate 1e-4 --dropout 0.6 --itr 5  

############################predict length 60####################################
python3 main_crossformer.py --data ILI \
--in_len 60 --out_len 60 --seg_len 6 \
--e_layers 3 \
 --learning_rate 1e-4 --dropout 0.6 --itr 5 