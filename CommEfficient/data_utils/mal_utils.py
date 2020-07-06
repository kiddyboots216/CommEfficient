import numpy as np
import torch

def fetch_mal_data(test_images, test_targets, args, num_classes, images_per_client):
    num_mal_images = args.mal_targets
    print("Num mal images", num_mal_images)
    allowed_source_labels = fetch_source_idxs(test_images, args, num_classes, num_mal_images)
    true_data, true_labels = fetch_test_data(test_images, test_targets, allowed_source_labels)
    print("True data", true_data.shape)
    z = np.zeros((num_mal_images, num_classes), dtype=bool)
    y = true_labels[:num_mal_images]
    mal_data = true_data[:num_mal_images]
    allowed_target_labels = fetch_targets(images_per_client, args)
    print("Targets", allowed_target_labels.shape)
    x = np.tile(allowed_target_labels[None, :], (num_mal_images, 1))
    z[range(num_mal_images), y] = True
    if args.mal_type in ["A", "C"]:
        num_mal_choices = 9
    elif args.mal_type in ["B", "D"]:
        num_mal_choices = 1
    else:
        assert False
    chosen_idxs = np.random.choice(num_mal_choices, num_mal_images, replace=True)
    allowed = x[~z].reshape(num_mal_images, -1)
    mal_labels = allowed[np.arange(num_mal_images), chosen_idxs]
    #mal_labels = x[np.arange(num_mal_images), chosen_idxs]
    #print("Mal labels", mal_labels[:20])
    return mal_data, mal_labels

def fetch_test_data(test_images, test_targets, allowed_source_labels):
    true_images = []
    true_labels = []
    for i, test_label in enumerate(test_targets):
        """
        if len(allowed_source_labels) == 0:
            break
        """
        if test_label in allowed_source_labels:
            #allowed_source_labels.remove(test_label)
            true_images.append(test_images[i])
            true_labels.append(test_label)
    try:
        return np.array(true_images), np.array(true_labels)
    except:
        return np.array(torch.stack(true_images)), np.array(true_labels)

def fetch_source_idxs(test_images, args, num_classes, num_mal_images):
    if args.mal_type in ["A", "B"]:
        return list(np.random.choice(num_classes, size=num_mal_images * 2))
    elif args.mal_type in ["C", "D"]:
        return [7 for _ in range(num_mal_images * 2)]

def fetch_targets(images_per_client, args):
    if args.mal_type in ["A", "C"]:
        return np.array(list(range(len(images_per_client))))
    elif args.mal_type in ["B", "D"]:
        return np.array([1])

