import numpy as np
import cv2
from keras.models import load_model
import time

# load model
model = load_model("model.h5")

# load  live video
cap = cv2.VideoCapture(0)

# Video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initial previous time
pTime = 0

while True:
    # read frame
    ret, image = cap.read()
    # convert to gray and smoothen frame
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add gausian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # threshold
    ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw contours (optional)
    contour = cv2.drawContours(image, contours, -1, (255, 255, 255), 1)

    for contours in contours:
        x, y, w, h = cv2.boundingRect(contours)
        if 5 < w < 250 and 25 <= h <= 250:  # Detection range
            
            # draw rectangles on image
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (18, 18))  # Resized for padding
            roi = np.pad(roi, ((5, 5), (5, 5)), "constant", constant_values=0)  # Padded into 28x28 for proper detection
            
            # resize image
            roi = roi.reshape(-1, 28, 28, 1)
            roi = np.array(roi, dtype='float32')/255
            pred = np.argmax(model.predict(roi)[0])
            conf = round(max(model.predict(roi)[0])*100, 2)
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(image, f"{str(pred)}   {str(conf)}%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # calculate fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    # put fps
    cv2.putText(image, f"FPS: {str(int(fps))}", (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
    
    # show frame
    cv2.imshow("Result", image)
    if cv2.waitKey(1) & 0xFF == 27: # Esc key
        break

cap.release()
cv2.destroyAllWindows()