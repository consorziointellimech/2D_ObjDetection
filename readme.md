#image labels in yolo notation

#run inference
python test.py --out_folder out --test_folder ./dataset/test --model_path model2

#run training
python train.py --model_path model2 --train_folder ./dataset/train
