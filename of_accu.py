#Importuojamos reikalingos bibliotekos
import numpy as np
import cv2 as cv
import flowlib as fl
import matplotlib.pyplot as plt

#Tyrimo pavyzdžiai
datasets = ("Dimetrodon","Grove2","Grove3","Hydrangea","RubberWhale","Urban2","Urban3","Venus")

#Sukuriami optinio srauto algoritmų objektai
ofDis = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
ofFb = cv.FarnebackOpticalFlow_create()
ofTvl = cv.optflow.DualTVL1OpticalFlow_create()
ofDf = cv.optflow.createOptFlow_DeepFlow()
ofPca = cv.optflow.createOptFlow_PCAFlow()
ofSf = cv.optflow.createOptFlow_SimpleFlow()
ofSd = cv.optflow.createOptFlow_SparseToDense()

#Funkcija optinio srauto algoritmų tikslumo vertinimui
def evaluate_of(of_alg, name):
    print("\nEvaluating: " + name + " algorithm:")
    sum_mepe = 0
    for dataset in datasets:
        first = cv.imread('./data/'+dataset+"/frame10.png",cv.IMREAD_GRAYSCALE)
        second = cv.imread('./data/' + dataset + "/frame11.png", cv.IMREAD_GRAYSCALE)
        flow_gt = cv.readOpticalFlow('./flow/' + dataset + '/flow10.flo')
        #flow_gt_img = fl.flow_to_image(flow_gt)
        #plt.imshow(flow_gt_img)
        #plt.show()
        flow = of_alg.calc(first,second, None)
        #flow_img = fl.flow_to_image(flow_gt)
        #plt.imshow(flow_img)
        #plt.show()
        mepe = fl.evaluate_flow(flow_gt,flow)
        print("\tResult for " + dataset + " dataset: EPE= %f" % (mepe))
        sum_mepe += mepe
    print("\tAverage EPE of " + name + " algorithm is %f" % (sum_mepe/8))

#Vertinami algoritmai
evaluate_of(ofDis,"OF_DIS")
evaluate_of(ofFb,"OF_FB")
evaluate_of(ofTvl,"OF_TVL")
evaluate_of(ofDf,"OF_DF")
evaluate_of(ofPca,"OF_PCA")
evaluate_of(ofSf,"OF_SF")
evaluate_of(ofSd,"OF_SD")

