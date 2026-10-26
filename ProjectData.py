import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Transform Data
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor()])


# Load GraveX
gravex_dataset = datasets.ImageFolder(root='/home/jonas/dataset', transform=data_transforms)