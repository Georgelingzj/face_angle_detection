import numpy as np
import os
import cv2
import math
import face_recognition

class face_detector():
    def __init__(self):

        self.cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))

        self.haar_model1 = os.path.join(self.cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

        self.haar_model2 = os.path.join(self.cv2_base_dir, 'data/haarcascade_frontalface_alt.xml')

        self.haar_model3 = os.path.join(self.cv2_base_dir, 'data/haarcascade_frontalface_alt2.xml')

        self.haar_model4 = os.path.join(self.cv2_base_dir, 'data/haarcascade_frontalface_alt_tree.xml')


        self.face_cascade = cv2.CascadeClassifier(self.haar_model2)

        self.cap = cv2.VideoCapture(0)

        self.total = 0

        self.detected_num = 0


    def capture(self):

        while(1):
            # get a frame
            ret, frame = self.cap.read()
            # show a frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            faces = self.face_cascade.detectMultiScale(gray, 1.3,5)


            if(len(faces)>0):
                self.detected_num += 1
                self.total += 1
                percent = self.detected_num/self.total
                percent = str(percent)
                print("Detected " + percent + "\n" + "total number is " + str(self.total) + "\n" +
                    "detected nunber is " + str(self.detected_num))
                #from frame to detect angle
                is_angle,angle,direction = self.Horizontal_angle(frame)
                if(is_angle == 1):
                    print("angle is " + str(angle) + "direction is " + str(direction))
                print()
                for(x,y,w,h) in faces:
                    self.x = x
                    self.y = y
                    cv2.putText(frame,str(angle),(400,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)

                    #cv2.addText(frame,str(angle),(10,10),cv2.FONT_HERSHEY_SIMPLEX )
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

            else:
                self.total += 1

                print("No face detected")
                print()


            cv2.imshow("Detecting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

    def Horizontal_angle(self,input_frame):
        face_landmark = face_recognition.face_landmarks(input_frame,model="large")

        if len(face_landmark) == 0:
            #no face landmark has been detected
            return 0,-1,"no"
        face_landmark_dict = face_landmark[0]

        self.landmark = face_landmark_dict

        right_eye_list = face_landmark_dict['right_eye']
        left_eye_list = face_landmark_dict['left_eye']

        right_eye_centre_x = 0
        right_eye_centre_y = 0
        left_eye_centre_x = 0
        left_eye_centre_y = 0

        for i in range(len(right_eye_list)):
            right_eye_centre_x += right_eye_list[i][0]
            right_eye_centre_y += right_eye_list[i][1]

        for j in range(len(left_eye_list)):
            left_eye_centre_x += left_eye_list[i][0]
            left_eye_centre_y += left_eye_list[i][1]

        right_eye_centre_x,right_eye_centre_y = right_eye_centre_x/float(len(right_eye_list)),right_eye_centre_y/float(len(right_eye_list))
        left_eye_centre_x, left_eye_centre_y = left_eye_centre_x/float(len(left_eye_list)), left_eye_centre_y/float(len(left_eye_list))

        #print("right is {}".format((right_eye_centre_x,right_eye_centre_y)))
        #print("left is {}".format((left_eye_centre_x,left_eye_centre_y)))

        direction = 'clockwise'
        if right_eye_centre_y>left_eye_centre_y:
            #C_x,C_y = 0,right_eye_centre_y
            direction = 'counterclockwise'



        AB_length = math.sqrt(math.pow((right_eye_centre_x-left_eye_centre_x),2)+math.pow((right_eye_centre_y-left_eye_centre_y),2))
        AC_length = right_eye_centre_x
        BC_length = math.sqrt(math.pow(left_eye_centre_x,2)+math.pow((left_eye_centre_y-right_eye_centre_y),2))

        cosA = float((math.pow(AB_length,2)+math.pow(AC_length,2)-math.pow(BC_length,2)))/float(2*AB_length*AC_length)
        #print(cosA)
        #human head angle
        if cosA > 1.0 or cosA < -1.0:
            return 0

        angle = (math.acos(cosA))*(180/float(math.pi))
        #print(angle,direction)

        #self.angle = angle
        #self.direction = direction
        return 1,angle,direction




if __name__ == '__main__':
    work = face_detector()
    work.capture()
