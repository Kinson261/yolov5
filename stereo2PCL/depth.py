import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
depth = (baseline * focal length) / disparity

baseline = distance between the cameras
disparity = difference in image location of the same 3D object under perspective to different cameras
"""

videoL = cv2.VideoCapture("C:\\Users\\Clark\\Desktop\\shot1\\left_out.avi")
videoR = cv2.VideoCapture("C:\\Users\\Clark\\Desktop\\shot1\\right_out.avi")

a1 = int(videoL.get(cv2.CAP_PROP_FRAME_WIDTH))
a2 = int(videoL.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Left video \t\t\t\tWidth :', a1, '\t\tHeight :', a2)

b1 = int(videoR.get(cv2.CAP_PROP_FRAME_WIDTH))
b2 = int(videoR.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Right video \t\t\tWidth :', b1, '\t\tHeight :', b2)


def nothing(x):
    pass

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)

cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)


while(videoL.isOpened() & videoR.isOpened()):
    retL, frameL = videoL.read()
    retR, frameR = videoR.read()

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frameL', grayL)
    cv2.imshow('frameR', grayR)

    stereo = cv2.StereoBM_create()
    """
    disparity = stereo.compute(grayL, grayR)
    plt.imshow(disparity, 'gray')
    plt.show()
    """

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    disparity = stereo.compute(grayL, grayR)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.
    disparity = disparity.astype(np.float32)
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
    cv2.imshow("picture", disparity)

    cv2.waitKey(3000)

    if cv2.waitKey(100) & 0xFF == ord('c'):
        break

videoL.release()
videoR.release()

cv2.destroyAllWindows()

