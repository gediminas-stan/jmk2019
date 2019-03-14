#Importuojamos reikalingos bibliotekos
import numpy as np
import cv2 as cv

#Sukuriamas vaizdo įrašo skaitymo objektas
cap = cv.VideoCapture('video.mp4')

#Sukuriami optinio srauto algoritmų objektai
ofDis = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_FAST)
ofFb = cv.FarnebackOpticalFlow_create()
ofTvl = cv.optflow.DualTVL1OpticalFlow_create()
ofDf = cv.optflow.createOptFlow_DeepFlow()
ofPca = cv.optflow.createOptFlow_PCAFlow()
ofSf = cv.optflow.createOptFlow_SimpleFlow()
ofSd = cv.optflow.createOptFlow_SparseToDense()

#Kintamieji rezultatams saugoti
num_frames = 0

time_of_dis = 0
time_of_fb = 0
time_of_tvl = 0
time_of_df = 0
time_of_pca = 0
time_of_sf = 0
time_of_sd = 0

#Skaitomas pirmas kadras
ret, frame = cap.read()
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
prev_gray = cv.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)

#Vykdomi skaičiavimai kol yra kadrų
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        num_frames += 1
        print(num_frames)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (0, 0), fx=0.5, fy=0.5)

        e1 = cv.getTickCount()
        flow = ofDis.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_dis += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofFb.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_fb += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofTvl.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_tvl += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofDf.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_df += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofPca.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_pca += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofSf.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_sf += (e2 - e1) / cv.getTickFrequency()

        e1 = cv.getTickCount()
        flow = ofSd.calc(prev_gray, gray, None)
        e2 = cv.getTickCount()
        time_of_sd += (e2 - e1) / cv.getTickFrequency()

    else:
        break

#Išvedami rezultatai
print("Average one frame processing time:" + \
      "\n\tOF_FB: " + str(time_of_fb / num_frames) + " fps = " + str(1 / (time_of_fb / num_frames)) + \
      "\n\tOF_TVL: " + str(time_of_tvl / num_frames) + " fps = " + str(1 / (time_of_tvl / num_frames)) + \
      "\n\tOF_DF: " + str(time_of_df / num_frames) + " fps = " + str(1 / (time_of_df / num_frames)) + \
      "\n\tOF_PCA: " + str(time_of_pca / num_frames) + " fps = " + str(1 / (time_of_pca / num_frames)) + \
      "\n\tOF_SF: " + str(time_of_sf / num_frames) + " fps = " + str(1 / (time_of_sf / num_frames)) + \
      "\n\tOF_SD: " + str(time_of_sd / num_frames) + " fps = " + str(1 / (time_of_sd / num_frames)) + \
      "\n\tOF_DIS: " + str(time_of_dis / num_frames) + " fps = " + str(1 / (time_of_dis / num_frames)))

cap.release()
cv.destroyAllWindows()