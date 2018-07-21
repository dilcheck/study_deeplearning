#mean squared error
import numpy as np
ab = [3, 76] #[기울기, y절편]

data =[[2,81], [4,93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

def rmse(p, a):
    return np.sqrt(((p-a) **2).mean())

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

#예측 값이 들어길 빈 리스트를 만든다.
predict_result = []

#모든 x값을 한번씩 대입하여
for i in range(len(x)):
    #그 결과 predict_result 리스트를 완성한다.
    predict_result.append(predict(x[i]))
    print( "공부한 시간=%.f, 실제 점수=%.f, 얘측 점수=%.f" % (x[i], y[i], predict(x[i])))

#최종 RMSE 출력
print("rmse 최종값: " + str(rmse_val(predict_result,y)))