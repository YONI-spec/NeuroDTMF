import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

def buil_model(cfg:dict) -> tf.keras.Model :
    model_confg = cfg['model']
    train_confg = cfg['train']

    inp = Input(shape=tuple(model_confg['input_shape']))
    
