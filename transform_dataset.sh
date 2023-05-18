#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate VG_SSL

# eynsham
python h5_transformer.py --read_path ~/datasets_vg/datasets/eynsham/images/test/database --save_path  ~/datasets_vg/datasets/eynsham/database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/eynsham/images/test/queries --save_path  ~/datasets_vg/datasets/eynsham/query.h5 --compress &

# msls
python h5_transformer.py --read_path ~/datasets_vg/datasets/msls/train/database --save_path  ~/datasets_vg/datasets/msls/train_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/msls/train/queries --save_path  ~/datasets_vg/datasets/msls/train_queries.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/msls/val/database --save_path  ~/datasets_vg/datasets/msls/val_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/msls/val/queries --save_path  ~/datasets_vg/datasets/msls/val_queries.h5 --compress &

# nordland
python h5_transformer.py --read_path ~/datasets_vg/datasets/nordland/images/test/database --save_path  ~/datasets_vg/datasets/nordland/database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/nordland/images/test/queries --save_path  ~/datasets_vg/datasets/nordland/queries.h5 --compress &

# # pitts30k
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/test/database --save_path  ~/datasets_vg/datasets/pitts30k/test_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/test/queries --save_path  ~/datasets_vg/datasets/pitts30k/test_queries.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/train/database --save_path  ~/datasets_vg/datasets/pitts30k/train_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/train/queries --save_path  ~/datasets_vg/datasets/pitts30k/train_queries.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/val/database --save_path  ~/datasets_vg/datasets/pitts30k/val_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts30k/images/val/queries --save_path  ~/datasets_vg/datasets/pitts30k/val_queries.h5 --compress &

# # pitts250k
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/test/database --save_path  ~/datasets_vg/datasets/pitts250k/test_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/test/queries --save_path  ~/datasets_vg/datasets/pitts250k/test_queries.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/train/database --save_path  ~/datasets_vg/datasets/pitts250k/train_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/train/queries --save_path  ~/datasets_vg/datasets/pitts250k/train_queries.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/val/database --save_path  ~/datasets_vg/datasets/pitts250k/val_database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/pitts250k/images/val/queries --save_path  ~/datasets_vg/datasets/pitts250k/val_queries.h5 --compress &

# #san_francisco
python h5_transformer.py --read_path ~/datasets_vg/datasets/san_francisco/images/test/database --save_path  ~/datasets_vg/datasets/san_francisco/database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/san_francisco/images/test/queries --save_path  ~/datasets_vg/datasets/san_francisco/queries.h5 --compress &

# #st_lucia
python h5_transformer.py --read_path ~/datasets_vg/datasets/st_lucia/images/test/database --save_path  ~/datasets_vg/datasets/st_lucia/database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/st_lucia/images/test/queries --save_path  ~/datasets_vg/datasets/st_lucia/queries.h5 --compress &

# #tokyo247
python h5_transformer.py --read_path ~/datasets_vg/datasets/tokyo247/images/test/database --save_path  ~/datasets_vg/datasets/tokyo247/database.h5 --compress &
python h5_transformer.py --read_path ~/datasets_vg/datasets/tokyo247/images/test/queries --save_path  ~/datasets_vg/datasets/tokyo247/queries.h5 --compress &

