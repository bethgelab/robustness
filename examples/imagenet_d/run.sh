# Clipart

python3 main.py /gpfs01/bethge/data/visda-2019/clipart/ --pretrained  -b 1024  --evaluate --workers 25 --use-train-statistics >logs/clipart_train.log

python3 main.py /gpfs01/bethge/data/visda-2019/clipart/ --pretrained   -b 1024  --evaluate  --workers 25 >logs/clipart_eval.log


# infograph

python3 main.py /gpfs01/bethge/data/visda-2019/infograph/ --pretrained  -b 1024  --evaluate --workers 25 --use-train-statistics >logs/infograph_train.log
 
python3 main.py /gpfs01/bethge/data/visda-2019/infograph/ --pretrained  -b 1024  --evaluate  --workers 25 >logs/infograph_eval.log

# painting

python3 main.py /gpfs01/bethge/data/visda-2019/painting/  --pretrained   -b 1024 --evaluate  --workers 25 --use-train-statistics >logs/painting_train.log
 
python3 main.py /gpfs01/bethge/data/visda-2019/painting/ --pretrained  -b 1024  --evaluate  --workers 25 >logs/painting_eval.log

# quickdraw

python3 main.py /gpfs01/bethge/data/visda-2019/quickdraw/  --pretrained  -b 1024 --evaluate  --workers 20 --use-train-statistics >logs/quickdraw_train.log
 
python3 main.py /gpfs01/bethge/data/visda-2019/quickdraw/  --pretrained   -b 1024 --evaluate --workers 20 >logs/quickdraw_eval.log

# real

python3 main.py /gpfs01/bethge/data/visda-2019/real/ --pretrained   -b 1024  --evaluate --workers 20 --use-train-statistics >logs/real_train.log
 
python3 main.py /gpfs01/bethge/data/visda-2019/real/ --pretrained  -b 1024  --evaluate  --workers 20  >logs/real_eval.log

# sketch

python3 main.py /gpfs01/bethge/data/visda-2019/sketch/ --pretrained  -b 1024  --evaluate --workers 20 --use-train-statistics >logs/sketch_train.log
 
python3 main.py /gpfs01/bethge/data/visda-2019/sketch/ --pretrained  -b  1024  --evaluate  --workers 20  >logs/sketch_eval.log