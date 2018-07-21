#딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

#필요한 라이브러리를 불러옵니다.
import numpy
import tensorflow as tf

#실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

#준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

#환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:, 0:17] #속성(attribute) -> arguments
Y = Data_set[:, 17] #클래스 (class) -> result

#딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#activation: 다음 층으로 어떻게 값을 넘길지 결정하는 부분
#loss: 한 번 신경만이 실행될 때마다 오차 값을 추적하는 함수
#optimizer: 오차를 어떻게 줄여 나갈지 정하는 함수

#딥러닝을 실행합니다.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=30, batch_size=10)

#결과값을 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))