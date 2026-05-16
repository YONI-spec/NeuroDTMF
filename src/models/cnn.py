import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

def build_model(cfg:dict) -> tf.keras.Model :
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    inp = Input(shape=tuple(model_cfg['input_shape']))
    x = inp 
    for block in model_cfg['conv_blocks']:
        filters = block['filters']
        x=layers.Conv2D(filters,(3,3),activation='relu', padding = 'same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.MaxPooling2D((2,2))(x)
    x=layers.Flatten()(x)
    x=layers.Dense(model_cfg['dense_units'],activation='relu', kernel_regularizer=tf.keras.regularizers.l2(model_cfg['l2_reg']))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(model_cfg['dropout_rate'])(x)
    x=layers.Dense(model_cfg['num_classes'],activation='softmax' )(x) 

    model = Model(inputs=inp, outputs=x, name= model_cfg['architecture'])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate = train_cfg['learning_rate'],
        clipnorm= train_cfg['clipnorm']
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model 


