# Importuojamos reikalingos bibliotekos
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Sukuriami fono atmetimo algoritmų objektai
fgbgCnt = cv.bgsegm.createBackgroundSubtractorCNT()

# Kintamieji rezultatams saugoti
num_frames = 0

time_cnt = 0

# Skaitomas pirmas kadras
ret, frame = cap.read()
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
prev_gray = cv.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)

# Vykdomi skaičiavimai kol yra kadrų
while (True):
    ret, frame = cap.read()
    if ret:
        num_frames += 1
        print(num_frames)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (0, 0), fx=0.5, fy=0.5)

        e1 = cv.getTickCount()
        fgmask = fgbgCnt.apply(gray)
        e2 = cv.getTickCount()
        time_cnt += (e2 - e1) / cv.getTickFrequency()
        cv.imshow('fg_mask',fgmask)
    else:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()