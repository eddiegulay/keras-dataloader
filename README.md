# keras Custom dataset loader
template code for keras dataloader
---
In Keras, you generally don't need a separate DataLoader class like you might in other frameworks (e.g., PyTorch). Keras provides the fit method, which can handle the training process without explicitly using a DataLoader.
---

## Example usage:
#### Assuming you have your data and labels ready, and you have defined a model called 'model'
##### data and labels should be NumPy arrays
"""
data = np.random.random((1000, 32))  # Replace this with your actual data
labels = np.random.randint(2, size=(1000,))  # Replace this with your actual labels

batch_size = 32

train_dataset = CustomDataset(data, labels, batch_size)

model.fit_generator(generator=train_dataset, epochs=10)
"""
