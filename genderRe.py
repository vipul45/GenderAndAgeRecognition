import cv2

#load pre trained moddels
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('deploy_gender2.prototxt', 'gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('deploy_age2.prototxt', 'age_net.caffemodel')

#define the list
age_groups = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32), (38-43)', '(43-53)', '(60-100)']

#Initiaalize the camera
video_capture = cv2.VideoCapture(0)

while True:
    #capture frame by frame
    ret, frame = video_capture.read()

    #convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecct faces in grayscale
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #Iteraate over the detected faces
    for(x, y, w, h) in faces:
        #extract face region of intrest (ROI)
        face_roi = frame[y:y+h, x:x+w]

        #Gender Classsification proccess
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        #forward pass through gender model
        gender_model.setInput(face_blob)
        gender_preds = gender_model.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        #processs face for age
        age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        #forward pass trough age model
        age_model.setInput(age_blob)
        age_preds = age_model.forward()
        age_idx = age_preds[0].argmax()
        age = age_groups[age_idx]

        #Display gender and age
        cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Display the result
    cv2.imshow('Face Recognition', frame)

    #exit loop
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

#Release Camera
video_capture.release()
cv2.destroyAllWindows()