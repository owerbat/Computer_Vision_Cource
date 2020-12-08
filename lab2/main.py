import cv2
import numpy as np


def relu(x):
    return np.maximum(0, x)


def max_pooling(input):
    n = 2
    new_width = input.shape[-2]//n
    new_height = input.shape[-1]//n
    M = input.shape[0]
    output = np.empty((M, new_width, new_height), dtype=np.float32)

    for m in range(M):
        for i in range(new_width):
            for j in range(new_height):
                output[m, i, j] = np.max(input[m, i*n:(i+1)*n, j*n:(j+1)*n])

    return output


def main():
    input = cv2.imread('lenna.png')

    M = 5
    R = 3
    S = 3
    C = 3
    W = input.shape[0]
    H = input.shape[1]
    F = W
    E = H

    filters = np.random.uniform(size=(M, C, R, S))
    B = np.zeros(M)
    output = np.empty((M, F, E), dtype=np.float32)

    for m in range(M):
        for x in range(F):
            for y in range(E):
                output[m, x, y] = B[m]
                for i in range(R):
                    for j in range(S):
                        for k in range(C):
                            x_ = x+i if x+i < input.shape[1] else input.shape[1]-1
                            y_ = y+j if y+j < input.shape[2] else input.shape[2]-1
                            output[m, x, y] += input[k, x_, y_] * filters[m, k, i, j]
                output[m, x, y] = relu(output[m, x, y])

    result = max_pooling(output)
    for m in range(M):
        cv2.normalize(result[m], result[m], 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'results/layer{m}.png', result[m])


if __name__ == "__main__":
    main()
