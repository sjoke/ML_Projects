# %%
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import xml.etree.ElementTree as ET
import tqdm
import config


# %%
def random_show(images, figname=None):
    index = np.arange(len(images))
    samples = np.random.choice(index, 16)

    fig, axes = plt.subplots(4, 4, figsize=(8, 6))
    for i, axis in enumerate(axes.flatten()):
        axis.imshow(images[samples[i]])
        # axis.set_title(os.path.basename(samples[i])[:-4])
        axis.set_axis_off()
    if figname is not None:
        plt.savefig(os.path.join(config.PREPROCESS_HOME, figname))
    else:
        plt.show()
    # plt.close()


# %%
images_files = glob.glob(os.path.join(config.DATA_HOME, 'all-dogs', '*'))

dog_types = os.listdir(os.path.join(config.DATA_HOME, 'Annotation'))
dog_types = [dog_type[dog_type.index('-') + 1:] for dog_type in dog_types]
# %%
annotation_files = glob.glob(os.path.join(
    config.DATA_HOME, 'Annotation', '*/*'))
dogs = []
labels = []
for i in tqdm.tqdm(range(len(annotation_files))):
    anno_file = annotation_files[i]
    anno_xml = ET.parse(anno_file).getroot()
    filename = anno_xml.find('filename').text
    # print('filename: ', filename)

    image = os.path.join(config.DATA_HOME, 'all-dogs', filename + '.jpg')
    if not os.path.exists(image):
        print('image of this anno_file does not exist: ', anno_file, filename)
        continue
    image = cv2.imread(image)[:, :, ::-1]  # [H, W, C], BGR2RGB

    bndboxes = []
    for obj in anno_xml.iter('object'):
        name = obj.find('name').text.strip()

        if name in dog_types:
            label_idx = dog_types.index(name)
        else:
            print('filename: {}, object name: {} not in types'.format(filename, name))
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        # segmetation
        min_px = min(xmax - xmin, ymax - ymin)
        dog = image[ymin:ymin + min_px, xmin:xmin + min_px]
        dog = cv2.resize(dog, (64, 64), interpolation=cv2.INTER_AREA)

        dogs += [dog]
        labels += [label_idx]
        # img_id = target.find('filename').text[:-4]

dogs = np.array(dogs)
labels = np.array(labels)
print('all dogs shape:', dogs.shape)
np.savez_compressed(os.path.join(config.PREPROCESS_HOME,
                                 'dogs.npz'), dogs=dogs, labels=labels)

# %%
random_show(dogs, 'samples2.jpg')
random_show(dogs, 'samples3.jpg')

# %%
# %%

# %%
