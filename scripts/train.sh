# python src/train.py -m data.debug=True  model.lr=0.001,0.002,0.003 tags=\["esmm","lr_expriment"\] experiment=esmm

python src/train.py data=in data.debug=True tags=\["esmm","data_experiment"\] experiment=esmm_experiment test=True

# python src/train.py experiment=esmm --help
