from medmnist import INFO

TASKS = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
BATCH_SIZE = 128

class Config:
    def __init__(self, task):
        self.name = task
        self.info = INFO[task]
    
    def __getitem__(self, key):
        return self.info[key]
    