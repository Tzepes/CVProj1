import cv2 as cv2

def ShowImage(image, window_name='image', timeout=0):
    """
    :param timeout. How many seconds to wait untill it close the window.
    """
    img = image.copy()
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 100, 100)  # Adjust the width and height as needed
    # cv2.imshow(window_name, cv2.resize(img, None, fx=0.6, fy=0.6))
    # cv2.waitKey(timeout)
    # cv2.destroyAllWindows()
    
def ShowKeypoints(image, keypoints, title="Keypoints"):
    """
    Show the keypoints on the image.
    :param image: The image to show.
    :param keypoints: The keypoints to show.
    :param title: The title of the window.
    """
    img = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img)
    # plt.title(title)
    # plt.axis('off')
    # plt.show()