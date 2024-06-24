from ODRS.src.DL.train_detectors import fit_model
from ODRS.src.ML.run_recommender import predict


class ODRS:
    def __init__(self, job, **kwargs):
        self.job = job.lower()
        self.__dict__.update(kwargs)

    def fit(self):
        job_map = {
            'ml_recommend': self.predict,
            'object_detection': self.object_detection
        }
        job_map.get(self.job, lambda: None)()

    def predict(self):
        parameters_dict = {
            'classes_path': self.classes,
            'dataset_path': self.data_path,
            'speed': self.speed,
            'accuracy': self.accuracy,
            'balance': self.balance,
            'mode': self.gpu
        }
        predict(parameters_dict)

    def object_detection(self):
        fit_model(self.data_path, self.classes, self.img_size, self.batch_size, self.epochs,
                  self.model, self.split_train_value, self.split_val_value,
                  self.gpu_count, self.select_gpu)
