#salvare immagini in notazione yolo

#lanciare test
python test.py --out_folder out --test_folder ./dataset/test --model_path model2

#lanciare train
python train.py --model_path model2 --train_folder ./dataset/train
