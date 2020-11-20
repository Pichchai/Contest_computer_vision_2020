from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2


BATCH_SIZE = 5
IMAGE_SIZE = (64,64)

dataframe = pd.DataFrame(columns=['filename', 'meat', 'veggie', 'noodle'])

for i in range(1, 301):
    dataframe = dataframe.append({'filename': "{}.jpg".format(i), 'meat': 0.0, 'veggie': 0.0, 'noodle': 0.0}, ignore_index=True)

datagen = ImageDataGenerator(rescale=1./3)

test_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:300],
    directory='rechallenge_round/images',
    x_col='filename',
    y_col=['meat','veggie','noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

model = load_model('contest_best.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)


test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('prediction:\n',predict)

for index in dataframe.index:
    dataframe.loc[index, 'meat'] = int(predict[index][0] * 300)
    dataframe.loc[index, 'veggie'] = int(predict[index][1] * 300)
    dataframe.loc[index, 'noodle'] = int(predict[index][2] * 300)

dataframe.to_csv(r'fired_noodles_predict.csv',index = False, header=True)

print(dataframe)