from learn import init,predict,IMAGE_HEIGHT,IMAGE_WIDTH
import cv2
import PIL

model,class_names = init('datasets')
print(class_names)
vid = cv2.VideoCapture(0)
print(predict(model, PIL.Image.open('test.jpg').resize((IMAGE_HEIGHT,IMAGE_WIDTH))))
while True:
    ret,image = vid.read()
    cv2.imshow('detection',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    max_score = -1
    for (x, y, w, h) in faces:
        resized_img = cv2.resize(image[y:y+h,x:x+w],(IMAGE_HEIGHT,IMAGE_WIDTH))
        max_score,confidence = predict(model, resized_img)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if max_score != -1:
            if confidence >80 :
                text = class_names[max_score] + ' ' + str(confidence) + '%'
            else:
                text = "Unidentified person"
            cv2.putText(image,  text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()