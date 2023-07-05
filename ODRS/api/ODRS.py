class ODRS:
    def __init__(self, job):
        self.job = job.lower()

        self.path_dataset = None

        #for object_detection
        self.path_classes = None
        self.batch_size = None
        self.epochs = None
        self.img_size = None
        self.gpu_count = None

        #for ml_recommend
        self.gpu = None
        self.speed = None
        self.accuracy = None


    def fit(self):
        if self.job == 'ml_recommend':
            print('ml_recomend')
        elif self.job == "object_detection":
            print("object_detection")
