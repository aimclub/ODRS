from ODRS.src.train_utils.custom_train_all import fit_model
from ODRS.src.ml_utils.ml_model_optimizer import predict


class ODRS:
    def __init__(self, job, data_path=None, classes="classes.txt",
                 img_size="256", batch_size="18", epochs="3",
                 model='yolov5l', gpu_count=1, select_gpu="0", config_path="dataset.yaml",
                 split_train_value=0.6, split_val_value=0.30,
                 gpu=True, speed=2, accuracy=10):
        self.job = job.lower()
        self.data_path = data_path
        self.classes = classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.gpu_count = gpu_count
        self.select_gpu = select_gpu
        self.config_path = config_path
        self.split_train_value = split_train_value
        self.split_val_value = split_val_value
        self.gpu = gpu
        self.speed = speed
        self.accuracy = accuracy

    def fit(self):
        if self.job == 'ml_recommend':
            predict(self.gpu, self.classes, self.data_path, self.speed, self.accuracy)
        elif self.job == "object_detection":
            fit_model(self.data_path, self.classes, self.img_size, self.batch_size, self.epochs,
                      self.model, self.config_path, self.split_train_value, self.split_val_value,
                      self.gpu_count, self.select_gpu)
