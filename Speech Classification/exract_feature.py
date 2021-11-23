import json
import os
import math
import librosa
import numpy as np
from scipy.io import wavfile 
from playsound import playsound
from python_speech_features import mfcc, logfbank
from pickle import dump
from pickle import load
import tensorflow as tf
import matplotlib.pyplot as plt
DATASET_PATH = "recordings"
pickle_path = "feature.pkl"

def sample_featurex(DATASET_PATH, pickle_path): 
    data = {
      "labels": [],
      "feauture_mfcc": [],
      "mapping": []    
    }
    for filename in os.listdir(DATASET_PATH):
        filepath = os.path.join(DATASET_PATH, filename)
        sampling_freq, audio = wavfile.read(filepath)
        x = mfcc(audio, sampling_freq)
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = (x - means) / stddevs
        audio_len = tf.shape(x)[0]
        # padding
        pad_len = 55
        paddings = tf.constant([[0, pad_len], [0, 0]])
        mfcc_features = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
        mfcc_features = mfcc_features.numpy()
        label = filename.split("_")[0]
        label = int(label)
        data["feauture_mfcc"].append(mfcc_features.tolist())
        data["labels"].append(label)
        if label == 0:
            word = 'zero'
        if label == 1:
            word = 'one'
        if label == 2:
            word = 'two'
        if label == 3:
            word = 'three'
        if label == 4:
            word = 'four'
        if label == 5:
            word = 'five'
        if label == 6:
            word = 'six'
        if label == 7:
            word = 'seven'
        if label == 8:
            word = 'eight'
        if label == 9:
            word = 'nine'
        data["mapping"].append(word)
        # save MFCCs to pickle file
    dump(data, open(pickle_path , 'wb'))
if __name__ == "__main__":
	sample_featurex(DATASET_PATH, pickle_path)
