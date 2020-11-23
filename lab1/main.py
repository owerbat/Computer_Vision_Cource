import cv2
import numpy as np


def black_and_white(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('results/black_and_white.png', result)


def high_contrast(image):
    result = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.imwrite('results/high_contrast.png', result)


def canny(image):
    result = cv2.Canny(image, 100, 200)
    cv2.imwrite('results/canny.png', result)


def corners(image):
    harris = cv2.cornerHarris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 2, 1, .05)

    harris_norm = np.empty(harris.shape, dtype=np.float32)
    cv2.normalize(harris, harris_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    result = cv2.convertScaleAbs(harris_norm)

    for i in range(harris_norm.shape[0]):
        for j in range(harris_norm.shape[1]):
            if int(harris_norm[i,j]) > 130:
                cv2.circle(result, (j,i), 2, (255), 2)

    cv2.imwrite('results/corners.png', result)


def main():
    image = cv2.imread('lenna.png')

    # black_and_white(image)
    # high_contrast(image)
    # canny(image)
    corners(image)


if __name__ == "__main__":
    main()
