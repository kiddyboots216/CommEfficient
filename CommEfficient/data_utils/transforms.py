from torchvision import transforms	

__all__ = ["cifar_train_transforms",	
           "cifar_test_transforms",
           "imagenet_train_transforms",
           "imagenet_val_transforms"]

# equals np.mean(train_set.train_data, axis=(0,1,2))/255	
cifar10_mean = (0.4914, 0.4822, 0.4465)	
# equals np.std(train_set.train_data, axis=(0,1,2))/255	
cifar10_std = (0.2471, 0.2435, 0.2616)	

cifar_train_transforms = transforms.Compose([	
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),	
        transforms.RandomHorizontalFlip(),	
        transforms.ToTensor(),	
        transforms.Normalize(cifar10_mean, cifar10_std)	
    ])	

cifar_test_transforms = transforms.Compose([	
        transforms.ToTensor(),	
        transforms.Normalize(cifar10_mean, cifar10_std)	
    ])

_imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
_imagenet_tfms = [transforms.ToTensor(), _imagenet_normalize]

sz = 224
imagenet_train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(sz),
        transforms.RandomHorizontalFlip()
    ] + _imagenet_tfms)

imagenet_val_transforms = transforms.Compose([
	transforms.Resize(int(sz*1.14)),
	transforms.CenterCrop(sz),
    ] + _imagenet_tfms)
