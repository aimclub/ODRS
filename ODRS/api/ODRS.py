class ODRS:
    def __init__(self, job, path_dataset=None, arch=None, path_classes=None, batch_size=None, epochs=None, img_size=None,
                 gpu_count=None, gpu=None, speed=None, accuracy=None, config_path=None):
        self.job = job.lower()
        self.path_dataset = path_dataset
        self.arch = arch  # ssd, rcnn, yolov5, yolov7, yolov8
        # for object_detection
        self.config_path = config_path
        self.path_classes = path_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.gpu_count = gpu_count
        # for ml_recommend
        self.gpu = gpu
        self.speed = speed
        self.accuracy = accuracy

    def fit(self):
        if self.job == 'ml_recommend':
            print('ml_recomend')
        elif self.job == "object_detection":
            print("object_detection")


if __name__ == "__main__":
    odrs = ODRS(job="object_detection",
                config_path='dataset.yaml',
                arch='yolov5',
                path_classes='/media/farm/ssd_1_tb_evo_sumsung/ODRC_2/ODRS/classes.txt',
                batch_size=24,
                img_size=640)
