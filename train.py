from detecto import core, utils
from torchvision import transforms
import matplotlib.pyplot as plt

# Change data format

utils.xml_to_csv("data/train/", "train_labels.csv")
utils.xml_to_csv("data/validation/", "val_labels.csv")

# Custom transforms

custom_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(800),
        transforms.ColorJitter(saturation=0.3),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ]
)

dataset = core.Dataset("train_labels.csv", "data/train/", transform=custom_transforms)

# Validation dataset

val_dataset = core.Dataset("val_labels.csv", "data/validation/")

# Customize training options

loader = core.DataLoader(dataset, batch_size=2, shuffle=True)

model = core.Model(["car", "truck", "rock", "tree", "log", "lilypad", "duck"])
losses = model.fit(loader, val_dataset, epochs=15, learning_rate=0.001, verbose=True)

# Visualize loss during training

plt.plot(losses)
plt.show()

# Save model

model.save("model_weights.pth")

# Access underlying torchvision model for further control

torch_model = model.get_internal_model()
print(type(torch_model))
