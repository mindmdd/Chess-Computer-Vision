import cv2

def check_circle(c):
    # initialize the shape name and approximate the contour
    result = False
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) >= 5:
        shape = "circle"
        result = True
    # return the name of the shape
    return result, len(approx)  

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def field_contour(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5))
    gray = clahe.apply(gray) 
    gray_cp = gray.copy() 
    gray_blur = cv2.medianBlur(gray_cp,7)
    gray = cv2.medianBlur(gray_cp,3)
    cv2.imwrite(name, gray_blur)