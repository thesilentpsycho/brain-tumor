# %% [code] {"scrolled":true,"_kg_hide-output":true,"_kg_hide-input":true}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# for dirname, _, filenames in os.walk('/panasas/scratch/grp-lsmatott/ve/lgg-mri-segmentation'):
#     for filename in filenames:
#         pass
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# <div style = "text-align: center;"><img src = "attachment:e5c237c7-5fd2-4733-8e68-dfa3b47fd736.png" width="100%"></div>
#
# **What's Brain Tumor ?** <br>
# %% [markdown]
# # import libraries

# %% [code]
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
from skimage import io

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.applications import VGG19

from warnings import filterwarnings

filterwarnings('ignore')

import random

import glob
from IPython.display import display

# %% [code]
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

# %% [markdown]
# # Prepare dataset

# %% [code]
data = pd.read_csv('/panasas/scratch/grp-lsmatott/ve/lgg-mri-segmentation/kaggle_3m/data.csv')
# data.info()

# %% [code]
# data.head(10)

# %% [code]
data_map = []
for sub_dir_path in glob.glob("/panasas/scratch/grp-lsmatott/ve/lgg-mri-segmentation/kaggle_3m/" + "*"):
    # if os.path.isdir(sub_path_dir):
    try:
        dir_name = sub_dir_path.split('/')[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + '/' + filename
            data_map.extend([dir_name, image_path])
    except Exception as e:
        print(e)

# %% [code]
df = pd.DataFrame({"patient_id": data_map[::2],
                   "path": data_map[1::2]})
df.head()

# %% [code]
df_imgs = df[~df['path'].str.contains("mask")]  # if have not mask
df_masks = df[df['path'].str.contains("mask")]  # if have mask

# File path line length images for later sorting
BASE_LEN = 108  # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
END_IMG_LEN = 4  # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
END_MASK_LEN = 9  # (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)

# Data sorting
imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
masks = sorted(df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

# Sorting check
idx = random.randint(0, len(imgs) - 1)
print("Path to the Image:", imgs[idx], "\nPath to the Mask:", masks[idx])

# %% [code]
# Final dataframe
brain_df = pd.DataFrame({"patient_id": df_imgs.patient_id.values,
                         "image_path": imgs,
                         "mask_path": masks
                         })


def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


brain_df['mask'] = brain_df['mask_path'].apply(lambda x: pos_neg_diagnosis(x))
# brain_df

# %% [markdown]
# # Vizualization

# %% [code]
brain_df['mask'].value_counts()

# %% [code]
# sns.countplot(brain_df['mask'])
# plt.show()

# %% [code]
count = 0
i = 0
# fig, axs = plt.subplots(12, 3, figsize=(20, 50))
# for mask in brain_df['mask']:
#     if (mask == 1):
#         img = io.imread(brain_df.image_path[i])
#         axs[count][0].title.set_text("Brain MRI")
#         axs[count][0].imshow(img)
#
#         mask = io.imread(brain_df.mask_path[i])
#         axs[count][1].title.set_text("Mask")
#         axs[count][1].imshow(mask, cmap='gray')
#
#         img[mask == 255] = (255, 0, 0)  # change pixel color at the position of mask
#         axs[count][2].title.set_text("MRI with Mask")
#         axs[count][2].imshow(img)
#         count += 1
#     i += 1
#     if (count == 12):
#         break
#
# fig.tight_layout()

# %% [code]
brain_df_train = brain_df.drop(columns=['patient_id'])
# Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))
brain_df_train.info()

# %% [markdown]
# # Preprocessing image

# %% [code]
brain_df_mask = brain_df[brain_df['mask'] == 1]
# brain_df_mask.shape

# %% [code]
# creating test, train and val sets
X_train, X_val = train_test_split(brain_df_mask, test_size=0.15)
X_test, X_val = train_test_split(X_val, test_size=0.5)
print("Train size is {}, valid size is {} & test size is {}".format(len(X_train), len(X_val), len(X_test)))

train_ids = list(X_train.image_path)
train_mask = list(X_train.mask_path)

val_ids = list(X_val.image_path)
val_mask = list(X_val.mask_path)


# %% [code]
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, mask, image_dir='./', batch_size=16, img_h=256, img_w=256, shuffle=True):

        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Get the number of batches per epoch'

        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        'Generate a batch of data'

        # generate index of batch_size length
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # get the ImageId corresponding to the indexes created above based on batch size
        list_ids = [self.ids[i] for i in indexes]

        # get the MaskId corresponding to the indexes created above based on batch size
        list_mask = [self.mask[i] for i in indexes]

        # generate data for the X(features) and y(label)
        X, y = self.__data_generation(list_ids, list_mask)

        # returning the data
        return X, y

    def on_epoch_end(self):
        'Used for updating the indices after each epoch, once at the beginning as well as at the end of each epoch'

        # getting the array of indices based on the input dataframe
        self.indexes = np.arange(len(self.ids))

        # if shuffle is true, shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        'generate the data corresponding the indexes in a given batch of images'

        # create empty arrays of shape (batch_size,height,width,depth)
        # Depth is 3 for input and depth is taken as 1 for output becasue mask consist only of 1 channel.
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        # iterate through the dataframe rows, whose size is equal to the batch_size
        for i in range(len(list_ids)):
            # path of the image
            img_path = str(list_ids[i])

            # mask path
            mask_path = str(list_mask[i])

            # reading the original image and the corresponding mask image
            img = io.imread(img_path)
            mask = io.imread(mask_path)

            # resizing and coverting them to array of type float64
            img = cv2.resize(img, (self.img_h, self.img_w))
            img = np.array(img, dtype=np.float64)

            mask = cv2.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64)

            # standardising
            img -= img.mean()
            img /= img.std()

            mask -= mask.mean()
            mask /= mask.std()

            # Adding image to the empty array
            X[i,] = img

            # expanding the dimnesion of the image from (256,256) to (256,256,1)
            y[i,] = np.expand_dims(mask, axis=2)

        # normalizing y
        y = (y > 0).astype(int)

        return X, y


train_data = DataGenerator(train_ids, train_mask)
val_data = DataGenerator(val_ids, val_mask)


# %% [markdown]
# # Create Model

# %% [code]
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model


model = build_vgg19_unet((256, 256, 3))
model.summary()

# %% [markdown]
# ## Training  Model

# %% [code]
# Define a custom loss function for Vgg19 UNet model

epsilon = 1e-5
smooth = 1


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


# %% [code]
# compling model and callbacks functions
adam = tf.keras.optimizers.Adam(lr=0.05, epsilon=0.1)
model.compile(optimizer=adam,
              loss=focal_tversky,
              metrics=[tversky]
              )
# callbacks
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=30
                              )
# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="seg_model.h5",
                               verbose=1,
                               save_best_only=True
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                              )

# %% [code]
history = model.fit(train_data,
                    epochs=60,
                    validation_data=val_data,
                    callbacks=[checkpointer, earlystopping, reduce_lr]
                    )

# %% [code]
model = load_model("seg_model.h5",
                   custom_objects={"focal_tversky": focal_tversky, "tversky": tversky, "tversky_loss": tversky_loss})

# %% [markdown]
# ## Model Evaluation

# %% [code]
# history.history.keys()

# %% [code]
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss']);
# plt.plot(history.history['val_loss']);
# plt.title("SEG Model focal tversky Loss");
# plt.ylabel("focal tversky loss");
# plt.xlabel("Epochs");
# plt.legend(['train', 'val']);
#
# plt.subplot(1, 2, 2)
# plt.plot(history.history['tversky']);
# plt.plot(history.history['val_tversky']);
# plt.title("SEG Model tversky score");
# plt.ylabel("tversky Accuracy");
# plt.xlabel("Epochs");
# plt.legend(['train', 'val']);


test_ids = list(X_test.image_path)
test_mask = list(X_test.mask_path)

# Segmentation Model Performance

def prediction(test, model_seg):
    # empty list to store results
    mask, image_id, has_mask = [], [], []

    # itetrating through each image in test data
    for i in test.image_path:

        # Creating a empty array of shape 1,256,256,1
        X = np.empty((1, 256, 256, 3))
        # read the image
        img = io.imread(i)
        # resizing the image and coverting them to array of type float64
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        # standardising the image
        img -= img.mean()
        img /= img.std()
        # converting the shape of image from 256,256,3 to 1,256,256,3
        X[0,] = img

        # make prediction of mask
        predict = model_seg.predict(X)

        # if sum of predicted mask is 0 then there is not tumour
        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask :)')
        else:
            # if the sum of pixel values are more than 0, then there is tumour
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return pd.DataFrame({'image_path': image_id, 'predicted_mask': mask, 'has_mask': has_mask})


# %% [code]
# making prediction
df_pred = prediction(X_test, model)
# print(df_pred)

# %% [code]
# merging original and prediction df
df_pred = X_test.merge(df_pred, on='image_path')
# print(df_pred.head(10))

# %% [code]
# visualizing prediction
count = 0
# fig, axs = plt.subplots(15, 5, figsize=(30, 70))
#
# for i in range(len(df_pred)):
#     if df_pred.has_mask[i] == 1 and count < 15:
#         # read mri images
#         img = io.imread(df_pred.image_path[i])
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         axs[count][0].imshow(img)
#         axs[count][0].title.set_text('Brain MRI')
#
#         # read original mask
#         mask = io.imread(df_pred.mask_path[i])
#         axs[count][1].imshow(mask)
#         axs[count][1].title.set_text('Original Mask')
#
#         # read predicted mask
#         pred = np.array(df_pred.predicted_mask[i]).squeeze().round()
#         axs[count][2].imshow(pred)
#         axs[count][2].title.set_text('AI predicted mask')
#
#         # overlay original mask with MRI
#         img[mask == 255] = (255, 0, 0)
#         axs[count][3].imshow(img)
#         axs[count][3].title.set_text('Brain MRI with original mask (Ground Truth)')
#
#         # overlay predicted mask and MRI
#         img_ = io.imread(df_pred.image_path[i])
#         img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
#         img_[pred == 1] = (0, 255, 150)
#         axs[count][4].imshow(img_)
#         axs[count][4].title.set_text('MRI with AI PREDICTED MASK')
#
#         count += 1
#     if (count == 15):
#         break
#
# fig.tight_layout()

# %% [code]
