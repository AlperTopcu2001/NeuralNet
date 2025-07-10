# main.py
import tarfile
import numpy as np

from data_loader import load_cifar10_batch, load_label_names
import matplotlib.pyplot as plt
archive_path = "./data/cifar-10-python.tar.gz"
extract_path = "./data/"  
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("CIFAR-10 erfolgreich entpackt.")


images, labels = load_cifar10_batch("./data/cifar-10-batches-py/data_batch_1")
label_names = load_label_names("./data/cifar-10-batches-py/batches.meta")

#print("Bilddaten:", images.shape)   # (10000, 32, 32, 3)
#print("Labels:", labels.shape)      # (10000,)
#print("Erste Klasse:", label_names[labels[0]])
reshaped_images = images.reshape(images.shape[0], -1)   # ergibt Shape: (10000, 3072)


fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i+1010])
    ax.set_title(label_names[labels[i+1010]])
    ax.axis("off")
plt.tight_layout()
plt.show()