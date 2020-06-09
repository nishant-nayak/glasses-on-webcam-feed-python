import dlib
import cv2
import numpy as np
#Importing ndimage to rotate filter
from scipy import ndimage

cap = cv2.VideoCapture(0)
glasses = cv2.imread("sunglasses.png", -1)

#Importing detector and predictor to detect face and facial landmarks
detector = dlib.get_frontal_face_detector()
#Predictor .dat file taken from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Function to resize an image to a given width
def resize(img, width):
    scale_factor = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * scale_factor))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

#Function to combine an image that has a transparency alpha channel
def blend_transparent(face_img, filter1):

    overlay_img = filter1[:,:,:3]
    overlay_mask = filter1[:,:,3:]
    
    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    res = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    return res

while True:
    #Reading the frames from the webcam feed
    _,img = cap.read()
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    try:
        #Detecting faces using dlib detector 
        faces = detector(gray, 1)

        #Looping over each face in the webcam feed
        for face in faces:
            #Finding the bounding box for each face
            x = face.left()
            y = face.top()
            w = face.right()
            h = face.bottom()

        dlib_rect = dlib.rectangle(x, y, w, h)
        
        #Detecting facial landmarks using dlib predictor
        detected_landmarks = predictor(gray, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        #Finding the angle of rotation of the eyes
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            if idx == 0:
                eye_left = pos
            elif idx == 16:
                eye_right = pos

            try:
                degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

            except:
                pass


        #Finding the center of the eye
        eye_center = (eye_left[1] + eye_right[1]) / 2

        #Traslation of glasses to align with the eyes
        glass_trans = int(.2 * (eye_center - y))

        face_width = w - x

        #Resizing glasses to match the face width
        glasses_resize = resize(glasses, face_width)

        #Rotate glasses based on angle between eyes
        yG, xG, cG = glasses_resize.shape
        glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree+90))
        glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree+90))


        #Blending the rotated glasses with the webcam feed
        h_rotate, w_rotate, _ = glass_rec_rotated.shape
        rec_resize = img_copy[y + glass_trans:y + h_rotate + glass_trans, x:x + w_rotate]
        blend_glasses = blend_transparent(rec_resize , glasses_resize_rotated)
        img_copy[y + glass_trans:y + h_rotate + glass_trans, x:x+w_rotate] = blend_glasses
        cv2.imshow('Output', img_copy)

    except:
        cv2.imshow('Output', img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
