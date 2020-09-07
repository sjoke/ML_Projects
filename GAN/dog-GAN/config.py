import os
import os.path as osp
import platform
import sys
import shutil
import argparse


sess = '0'
if len(sys.argv) >= 2:
    sess = sys.argv[1]
sess = 'output_session_' + sess

clean = False
if len(sys.argv) >= 3 and sys.argv[2].lower() == 'true':
    clean = True


def remakedirs(path):
    global clean
    if osp.exists(path) and clean:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


if platform.system() == 'Windows':
    DATA_HOME = osp.join('/workspace', 'datasets', 'dogs')
    BUFFER_SIZE = 3000
    BATCH_SIZE = 64
    EPOCHS = 5
    NUM_WORKERS = 0
elif platform.system() == 'Linux':
    DATA_HOME = osp.join('/data1', 'laowang', 'dogs')
    BUFFER_SIZE = 60000
    BATCH_SIZE = 64
    EPOCHS = 500
    NUM_WORKERS = 4
else:
    raise ValueError('platform not support')


MODEL_PATH = osp.join(sess, 'models')
remakedirs(MODEL_PATH)


OUTPUT_PATH = osp.join(sess, 'images')
remakedirs(OUTPUT_PATH)

LOG_IMAGE_PATH = osp.join(sess, 'log_images')
remakedirs(LOG_IMAGE_PATH)

LOG_DIR = osp.join('logs', sess)
remakedirs(LOG_DIR)
print('output dir: ', sess)
print('log dir: ', LOG_DIR)


PREPROCESS_HOME = osp.join(DATA_HOME, 'preprocess')
if not osp.exists(PREPROCESS_HOME):
    os.makedirs(PREPROCESS_HOME)


noise_dim = 128
num_examples_to_generate = 16


c_lr_G = 0.001
c_lr_D = 0.001
c_beta1_G = 0.5
c_beta1_D = 0.5

c_real_label = 0.5
c_fake_label = 0
