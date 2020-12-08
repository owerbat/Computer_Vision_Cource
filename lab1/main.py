import cv2
import numpy as np


def black_and_white(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('results/black_and_white.png', result)

    return result


def high_contrast(image):
    result = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.imwrite('results/high_contrast.png', result)

    return result


def canny(image):
    result = cv2.Canny(image, 100, 200)
    cv2.imwrite('results/canny.png', result)

    return result


def corners(image):
    harris = cv2.cornerHarris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 2, 1, .05)

    harris_norm = np.empty(harris.shape, dtype=np.float32)
    cv2.normalize(harris, harris_norm, 0, 255, cv2.NORM_MINMAX)
    result = cv2.convertScaleAbs(harris_norm)

    for i in range(harris_norm.shape[0]):
        for j in range(harris_norm.shape[1]):
            if int(harris_norm[i,j]) > 130:
                cv2.circle(result, (j,i), 2, (255), 2)

    cv2.imwrite('results/corners.png', result)

    return result


def distance_map(image):
    result = cv2.distanceTransform(cv2.bitwise_not(image), cv2.DIST_L2, 3)
    result_norm = np.empty(result.shape, dtype=np.float32)
    cv2.normalize(result, result_norm, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('results/distance_map.png', result_norm)

    return result


def averaging(image, distances):
    k = 2
    hist = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    result = np.empty(hist.shape, dtype=np.float32)
    integral = cv2.integral(hist)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            radius = k*distances[i, j]//2
            if radius == 0:
                result[i, j] = hist[i, j]
            else:
                def clamp(value, a, b):
                    if value < a:
                        return a
                    elif value > b:
                        return b
                    else:
                        return value

                x1 = int(clamp(i-radius, 0, integral.shape[0]-1))
                x2 = int(clamp(i+radius, 0, integral.shape[0]-1))
                y1 = int(clamp(j-radius, 0, integral.shape[1]-1))
                y2 = int(clamp(j+radius, 0, integral.shape[1]-1))
                color = int((integral[x2, y2]-integral[x2, y1]-integral[x1, y2]+integral[x1, y1])/(x2-x1)/(y2-y1))
                result[i, j] = color if color > 0 else 1

    cv2.imwrite('results/averaging.png', result)

    return result


def main():
    image = cv2.imread('lenna.png')

    black_and_white(image)
    high_contrast(image)
    canny_img = canny(image)
    corners(image)
    distances = distance_map(canny_img)
    averaging(image, distances)


if __name__ == "__main__":
    main()
