#Importuojamos reikalingos bibliotekos
import numpy as np
import cv2 as cv

#Sukuriamas vaizdo įrašo skaitymo objektas
cap = cv.VideoCapture('video.mp4')

#Sukuriami fono atmetimo algoritmų objektai
fgbgMog = cv.bgsegm.createBackgroundSubtractorMOG()
fgbgMog2 = cv.createBackgroundSubtractorMOG2()
fgbgGmg = cv.bgsegm.createBackgroundSubtractorGMG()
fgbgKnn = cv.createBackgroundSubtractorKNN()
fgbgCnt = cv.bgsegm.createBackgroundSubtractorCNT()
fgbgGsoc = cv.bgsegm.createBackgroundSubtractorGSOC()
fgbgLsbp = cv.bgsegm.createBackgroundSubtractorLSBP()

#Kintamieji rezultatams saugoti
num_frames = 0

time_mog = 0
time_mog2 = 0
time_gmg = 0
time_knn = 0
time_cnt = 0
time_gsoc = 0
time_lsbp = 0

#Skaitomas pirmas kadras
ret, frame = cap.read()
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
prev_gray = cv.resize(prev_gray, (0,0), fx=0.5, fy=0.5)

#Vykdomi skaičiavimai kol yra kadrų
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        num_frames += 1
        print(num_frames)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (0,0), fx=0.5, fy=0.5)
        
        e1 = cv.getTickCount()
        fgmask = fgbgMog.apply(gray)
        e2 = cv.getTickCount()
        time_mog += (e2 - e1)/ cv.getTickFrequency()
        
        e1 = cv.getTickCount()
        fgmask = fgbgMog2.apply(gray)
        e2 = cv.getTickCount()
        time_mog2 += (e2 - e1)/ cv.getTickFrequency()

        e1 = cv.getTickCount()
        fgmask = fgbgGmg.apply(gray)
        e2 = cv.getTickCount()
        time_gmg += (e2 - e1)/ cv.getTickFrequency()

        e1 = cv.getTickCount()
        fgmask = fgbgKnn.apply(gray)
        e2 = cv.getTickCount()
        time_knn += (e2 - e1)/ cv.getTickFrequency()

        e1 = cv.getTickCount()
        fgmask = fgbgCnt.apply(gray)
        e2 = cv.getTickCount()
        time_cnt += (e2 - e1)/ cv.getTickFrequency()

        e1 = cv.getTickCount()
        fgmask = fgbgGsoc.apply(gray)
        e2 = cv.getTickCount()
        time_gsoc += (e2 - e1)/ cv.getTickFrequency()

        e1 = cv.getTickCount()
        fgmask = fgbgLsbp.apply(gray)
        e2 = cv.getTickCount()
        time_lsbp += (e2 - e1)/ cv.getTickFrequency()
        
    else:
        break

#Išvedami rezultatai
print("Average one frame processing time:" + \
      "\n\tMOG: " + str(time_mog/num_frames) + " fps = " + str(1/(time_mog/num_frames)) + \
      "\n\tMOG2: " + str(time_mog2/num_frames) + " fps = " + str(1/(time_mog2/num_frames)) + \
      "\n\tGMG: " + str(time_gmg/num_frames) + " fps = " + str(1/(time_gmg/num_frames)) + \
      "\n\tKNN: " + str(time_knn/num_frames) + " fps = " + str(1/(time_knn/num_frames)) + \
      "\n\tCNT: " + str(time_cnt/num_frames) + " fps = " + str(1/(time_cnt/num_frames)) + \
      "\n\tGSOC: " + str(time_gsoc/num_frames) + " fps = " + str(1/(time_gsoc/num_frames)) + \
      "\n\tLSBP: " + str(time_lsbp/num_frames) + " fps = " + str(1/(time_lsbp/num_frames)))

cap.release()
cv.destroyAllWindows()
