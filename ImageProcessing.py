import cv2
import numpy as np

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

def add_contrast(input_img, brightness = 50, contrast = 64):
        
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

def warp_corner(final_coor, img, w, h):
    src_pts = np.array(final_coor, dtype="float32")
    dst_pts = np.array([[0, h],
                        [0, 0],
                        [w, 0],
                        [w, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, w))

    return warped

def floodfill(img):
    # Fill hole
    im_floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    img = img | im_floodfill_inv

    return img