import numpy as np
import cv2
import glob
import os


def checkerboard_calibrate(image_folder, save_folder=None, square_size=25, max_images=10000, show=False):
    if save_folder is None:
        save_folder = image_folder

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # chessboard dims
    n = 8
    m = 6
    # square_size = 25 # mm
    print(square_size)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m, 3), np.float32)
    objp[:,:2] = np.mgrid[0:n,0:m].T.reshape(-1,2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Get image names
    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    for i, fname in enumerate(images):
        if i >= max_images:
            continue

        print(f"Processing image {i+1}/{min(len(images), max_images)}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n,m), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            if show:
                cv2.drawChessboardCorners(img, (n,m), corners2, ret)
                cv2.imshow('img', cv2.resize(img, (1000, 750)))
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Get calibration
    print(f"Computing matrix...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save calibration
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    np.save(os.path.join(save_folder, "intrinsic_matrix.npy"), mtx)
    np.save(os.path.join(save_folder, "distortion.npy"), dist)

    print(f"Intrinsic camera matrix:\n{mtx}")
    print(f"Distortion parameters:\n{dist}")

    # Check calibration accuracy
    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print(f"Total error per image as a proportion of image size: {mean_error/len(objpoints)/max(gray.shape)}")

    return mtx, dist

def undistort_image(file_name, mtx, dist):
    img = cv2.imread(file_name)
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('./temp/undistortion_result.png', dst)  


if __name__ == "__main__":
    mtx, dist = checkerboard_calibrate('./calibration/s10+_horizontal', square_size=25, show=False)

    # undistort_image("./calibration/s10+_horizontal/20240310_115154(0).jpg", mtx, dist)