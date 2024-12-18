# gsnet
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_512/retrieval/ --num_out_points 512 --model_name GSNet
wait 
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_256/retrieval/ --num_out_points 256 --model_name GSNet
wait
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_128/retrieval/ --num_out_points 128 --model_name GSNet
wait
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_64/retrieval/ --num_out_points 64 --model_name GSNet
wait
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_32/retrieval/ --num_out_points 32 --model_name GSNet
wait
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_16/retrieval/ --num_out_points 16 --model_name GSNet
wait
python analyze_precision_recall.py --dump_dir log_eval/diffpool/log_dir_8/retrieval/ --num_out_points 8 --model_name GSNet
wait
