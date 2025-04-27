import cv2 as cv

def show_image(image, window_name='image', timeout=0):
    """
    :param timeout. How many seconds to wait untill it close the window.
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 100, 100)  # Adjust the width and height as needed
    cv.imshow(window_name, cv.resize(image, None, fx=0.6, fy=0.6))
    cv.waitKey(timeout)
    cv.destroyAllWindows()