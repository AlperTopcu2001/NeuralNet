import pickle
import numpy as np

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'] #(10000, 3072)
        labels = batch[b'labels'] # Liste mit 10000 Zahlen (0-9)
        
        images = images.reshape(-1, 3, 32, 32)        # (N, C, H, W)
        images = np.transpose(images, (0, 2, 3, 1))   # (N, H, W, C)
        images = images.astype(np.float32) / 255.0    # Normalisieren
     
        
        return images, np.array(labels)

def load_label_names(filepath):
    with open(filepath, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]
        return label_names
