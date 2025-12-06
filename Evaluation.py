#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:36:03 2019

@author: wzk
"""
import numpy as np
import math

#MAE
def Evaluation_AbsE( t_p, t_r ):
#    global location_num

    tripsNum = np.size(t_r,0)
    
    absError_list = []
    sumAbs_list = []
    absError_station_list = []
    for ti in range(tripsNum):

        t_r_matrix = t_r[ti,:,:]
        t_p_matrix = t_p[ti,:,:]
#        (abs_errors[1,ti], abs_errors[2,ti], abs_errors[3,ti]) = get_absError(t_p_matrix, t_r_matrix)
        (absError, sumAbs, absError_station) = get_absError(t_p_matrix, t_r_matrix)
        absError_list.append(absError)
        sumAbs_list.append(sumAbs)
        absError_station_list.append(absError_station)
        
        
    absError = np.array(absError_list)
    sumAbs = np.array(sumAbs_list)
    absError_station = np.array(absError_station_list)

#    temp = np.hstack( abs_errors[2,:,:])
    temp = sumAbs

    nanNum =  np.sum(np.isnan(temp))
#    test = np.where(np.isnan(temp)==False)
#    t1 = temp[test]
#    t2=np.mean(t1)
    aveAbsE_line  = np.mean(temp[np.where(np.isnan(temp)==False)])
    
    return aveAbsE_line,absError

def get_absError( t_p_matrix, t_r_matrix ):
    location_num = np.size(t_r_matrix,1)
    absError = np.abs(t_r_matrix - t_p_matrix)
    absError_station = sum(absError)/(np.arange(location_num,0,-1))     #(location_num:-1:1)%从不同站点预测的平均误差
    sumAbs = sum(absError_station)/location_num               #%整条轨迹的平均误差
    return absError, sumAbs, absError_station





def Evaluation_RmsE( t_p, t_r ):

    tripsNum = np.size(t_r,0)
    absError_list = []
    sumRms_list = []
    rmsError_station_list = []
    for ti in range(tripsNum):
        t_r_matrix = t_r[ti,:,:]
        t_p_matrix = t_p[ti,:,:]
        (absError, sumRms, rmsError_station) = get_rmsError(t_p_matrix, t_r_matrix)
        absError_list.append(absError)
        sumRms_list.append(sumRms)
        rmsError_station_list.append(rmsError_station)

    
    
    rms_errors = np.array(absError_list)
    sumAbs = np.array(sumRms_list)
    rmsError_station = np.array(rmsError_station_list)

#    temp = np.hstack( abs_errors[2,:,:])
    temp = sumAbs
    nanNum =  np.sum(np.isnan(temp))
    aveRmsE_line  = np.mean(temp[np.where(np.isnan(temp)==False)])  

    return aveRmsE_line, rms_errors



def get_rmsError( t_p_matrix, t_r_matrix ):
    location_num = np.size(t_r_matrix,1)
    absError = np.abs(t_r_matrix - t_p_matrix) #%min 
    
#    rmsError_station = sqrt(sum(absError.^2, 1)./(np.arange(location_num,0,-1))
    rmsError_station = np.sqrt(sum((absError)**2)/(np.arange(location_num,0,-1)))
    sumRms = sum(rmsError_station)/location_num

    return absError, sumRms, rmsError_station
    




    

if __name__ == "__main__":
    t_p_lstm = np.zeros((82,173,172))
    t_r_lstm = np.ones((82,173,172))
    stationFlag_test = np.zeros((82,173))
    
#    t_r_lstm_real_list = []
#    t_p_lstm_real_list = []
#    for n in range(testTripsNum):
#        test = t_r_lstm[n][stationFlag_test[n, :],stationFlag_test[n, 0:-1]]
#        t_r_lstm_real_list.append(t_r_lstm[n][stationFlag_test[n, :],stationFlag_test[n, 0:-1]])
#        t_p_lstm_real_list.append(t_p_lstm[n][stationFlag_test[n, :],stationFlag_test[n, 0:-1]])
#    
#    if sum(t_r_lstm_real[n][:,-1]) == 0:
#        t_r_lstm_real[n] = t_r_lstm_real[n][:,0:-1];
#        t_p_lstm_real[n] = t_p_lstm_real[n][:,0:-1];
    
#    Evaluation_AbsE( t_p_lstm, t_r_lstm )
    Evaluation_RmsE( t_p_lstm, t_r_lstm )