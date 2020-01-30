from torchvision import transforms	

__all__ = ["cifar10_train_transforms",	
           "cifar10_test_transforms",
           "cifar100_train_transforms",	
           "cifar100_test_transforms",
           "femnist_train_transforms",
           "femnist_test_transforms",
           "imagenet_train_transforms",
           "imagenet_val_transforms"]

# equals np.mean(train_set.train_data, axis=(0,1,2))/255	
cifar10_mean = (0.4914, 0.4822, 0.4465)	
# equals np.std(train_set.train_data, axis=(0,1,2))/255	
cifar10_std = (0.2471, 0.2435, 0.2616)	

cifar10_train_transforms = transforms.Compose([	
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),	
        transforms.RandomHorizontalFlip(),	
        transforms.ToTensor(),	
        transforms.Normalize(cifar10_mean, cifar10_std)	
    ])	

cifar10_test_transforms = transforms.Compose([	
        transforms.ToTensor(),	
        transforms.Normalize(cifar10_mean, cifar10_std)	
    ])

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cifar100_train_transforms = transforms.Compose([	
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),	
        transforms.RandomHorizontalFlip(),	
        transforms.ToTensor(),	
        transforms.Normalize(cifar100_mean, cifar100_std)	
    ])	

cifar100_test_transforms = transforms.Compose([	
        transforms.ToTensor(),	
        transforms.Normalize(cifar100_mean, cifar100_std)	
    ])

femnist_mean = (0.9637,)
femnist_std = (0.1597,)

femnist_train_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=2, padding_mode="constant",
                              fill=1.0),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(4./5., 5./4.)),
        transforms.RandomRotation(5, fill=1.0),
        transforms.ToTensor(),
        transforms.Normalize(femnist_mean, femnist_std),
    ])

femnist_test_transforms = transforms.Compose([
        #transforms.Pad(2, fill=1.0),
        transforms.ToTensor(),
        transforms.Normalize(femnist_mean, femnist_std),
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
