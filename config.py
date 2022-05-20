import albumentations as A

aug_pipeline = A.Compose([
    A.RGBShift(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomFog(p=0.1),
])

config = {
    "learning_rate": 0.001,
    "epoch": 40,
    "augmentation": aug_pipeline,
    "num_classes": 3
}
