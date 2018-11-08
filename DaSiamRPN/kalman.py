import numpy as np
import cv2


class SimpleKalman2D():
    def __init__(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
        self.kalman.measurementNoiseCov = np.array(
            [[1, 0], [0, 1]], np.float32) * 1
        #self.kalman.statePre =  np.array([[x],[y]],np.float32)

    def correct(self, x, y):
        mes = np.array([[x], [y]], np.float32)
        return self.kalman.correct(mes)

    def predict(self):
        return self.kalman.predict()


if __name__ == "__main__":

    frame = np.zeros((800, 800, 3), np.uint8)
    last_mes = current_mes = np.array((2, 1), np.float32)
    last_pre = current_pre = np.array((2, 1), np.float32)
    simple_kalman = SimpleKalman2D(0, 0)

    def mousemove(event, x, y, s, p):
        global frame, current_mes, mes, last_mes, current_pre, last_pre
        last_pre = current_pre
        last_mes = current_mes
        simple_kalman.correct(x, y)
        current_pre = simple_kalman.predict()
        lmx, lmy = last_mes[0], last_mes[1]
        lpx, lpy = last_pre[0], last_pre[1]
        cmx, cmy = current_mes[0], current_mes[1]
        cpx, cpy = current_pre[0], current_pre[1]
        cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 200, 0))
        cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))

    cv2.namedWindow("Kalman")
    cv2.setMouseCallback("Kalman", mousemove)
    while(True):
        cv2.imshow('Kalman', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
