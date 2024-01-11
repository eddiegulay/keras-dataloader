from keras.utils import Sequence
import numpy as np

class CustomDataset(Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]
        return batch_data, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# Example usage:
# Assuming you have your data and labels ready, and you have defined a model called 'model'
# data and labels should be NumPy arrays
data = np.random.random((1000, 32))  # Replace this with your actual data
labels = np.random.randint(2, size=(1000,))  # Replace this with your actual labels

batch_size = 32

train_dataset = CustomDataset(data, labels, batch_size)

model.fit_generator(generator=train_dataset, epochs=10)
