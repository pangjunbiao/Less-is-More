#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:33:38 2020

@author: wzk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import time
import scipy.io as scio
import math
from numpy.matlib import repmat 
import Evaluation

def testModel_PolicyGradient(net,trainTripNum_,data_inter,num_inter,num_feature,N_SELECT,stationFlag_test,historyTrip_11,testTrip_11,testTrip_21,Line):
    step = 100        
    trainTripNum = random.sample(range(1,trainTripNum_-1),200 )
    j = trainTripNum[2]
#        j=3
    
    s_a1 = data_inter[j,:,:]   #全局特征
    s_a1 = s_a1[np.newaxis,:,:]
    s_a1 = torch.Tensor(s_a1)
    
    s_b = create_label_first(num_inter,N_SELECT)   #选取的局部特征的标签
#        s_b = np.array(np.loadtxt('s_b14_test').tolist()).reshape(-1,1)  
    s_b = s_b[np.newaxis,:,:]
    s_b = torch.Tensor(s_b).cuda()            
    
    s_a2 = trans(s_a1,s_b,N_SELECT,num_feature)    #根据全局特征和局部特征的标签选取的局部特征
    s_a = torch.cat((s_a1, s_a2), 1).cuda()
    
#        p = net(s_a,s_b)
#        a = np.argmax(p.cpu().data.numpy(), axis=2)    #1*1*8 int
#        select_station = np.where(a[0, 0:-1]==1)
#        select_station = np.array(select_station).tolist()[0]
#        print(np.size(select_station))
    
    
#        np.savetxt('epoch_test_LrSingle_'+ test_epoch + '.txt', select_station) 
#        np.savetxt('epoch_test_Lr_'+ test + '.txt', select_station) 
    ave1_list = []
    ave2_list = []
    step_list = []
    for step_num in range(step):

#                action_prob = net(s_a,s_b)  #model rl网络 1*1*8*3
#                a = np.argmax(action_prob.cpu().data.numpy(), axis=2)    #1*1*8 int
        a_list = []
        p_test_list = []
        
#            for i in range(num_inter):
#                p=np.array(net(s_a,s_b)[0].cpu().data.numpy())[i]
#                a = np.argmax(p) 
#                a_list.append(a)
            
        p = net(s_a,s_b,num_inter,N_SELECT,num_feature)
        a = np.argmax(p.cpu().data.numpy(), axis=2)    #1*1*8 int
#                print(np.array(p_test_list))
        #take action
        a = np.array(a).reshape((1,-1))
        
#            a = np.ones((1,108))
        
        test = np.where(a[0, 0::]==1)[0]
#        print(np.size(test))

        s_b = take_action(a,s_b,num_inter).cuda() 
#        s_b = take_action_bound_paper(a,s_b.cpu().data,num_inter).cuda() 

        s_a2 = trans(s_a1, s_b,N_SELECT,num_feature)        #根据执行的动作更新选取的局部特征
        #get_action
        action = get_action(a)
        #记录setp中间数据
        s_a = torch.cat((s_a1, s_a2), 1).cuda()    #更新sa
        
        
        if step_num  % 2 == 0:
            trainTripNum = 512
            testTripsNum = 60   #Epoch499:1.5996691803
        #    testTripsNum = 82   #Epoch499:1.6259340646442557
            #随机生成100个选择站点的数据
            select_station = np.where(stationFlag_test[0, 0:-1]==1)
            select_station = np.array(select_station).reshape(-1,1)
            
            select_inter = np.where(stationFlag_test[0, 0:-1]==0)
            select_inter = np.array(select_inter)

            seleted_perm_inter = np.squeeze(s_b.cpu().data.numpy()).reshape((1,-1))
        
            seleted_perm_inter = np.where(seleted_perm_inter[0, 0::]==1)[0]
            seleted_perm_inter = np.array(seleted_perm_inter)
            
            
            list1 = seleted_perm_inter
            lista = []
        
            for i in list1:
                 i =int(i)
                 lista.append(i)
            
            select_inter_ = select_inter[:,lista].reshape(-1,1)
            aa = np.vstack((select_station,select_inter_))
            s_index = np.sort(aa,axis=0).tolist()
            
            s_station_num = np.size(s_index)
            
            historyTrip_11_ = np.squeeze(historyTrip_11[s_index,:])
            testTrip_11_ = np.squeeze(testTrip_11[s_index,:]    )         
            
            #Lr
            
            perm_train = np.loadtxt('perm_'+ str(Line) + '.txt').tolist()
            listp = perm_train
            listb = []
            for i in listp:
                i =int(i)
                listb.append(i)
            perm_train = listb
        
        #    perm_size = np.size(historyTrip_11,1)
        #    perm_train = random.sample(range(0,perm_size),perm_size)
          
        
            trainTripsId = perm_train[0:trainTripNum]
            testTripsId = perm_train[-testTripsNum::]
            
            t_h = historyTrip_11_[:,trainTripsId]
            t_r = testTrip_11_
            t_r_houridx = testTrip_21
            t_u = np.mean(t_h,1)
            location_num = np.size(t_r,0)
            
            
            t_p_lr_list = []
            t_r_lr_list = []
            
            
            for i in range(testTripsNum):
#                    if type_hour == 1:
#                        houridx = t_r_houridx[i]
        
                t_r_lr_list.append(np.tril(repmat(t_r[:,i].reshape((-1,1),order="F"),1,location_num-1) , -1)*0.5)
                t_p_lr_list.append(linear_regr(t_h, t_r[:, i])*0.5)
                        
            t_p_lr = np.array(t_p_lr_list)
            t_r_lr = np.array(t_r_lr_list)
            
            station_flag_lr = stationFlag_test[0, s_index]
                
#                sta = np.where(station_flag_lr==1)
            sta_h = np.where(station_flag_lr==1)
            sta_z = np.where(station_flag_lr==1)
            sta_z = np.array(sta_z).tolist()[0]           
            sta_h = np.array(sta_h).tolist()[0]

            sta_h.pop()
            
            t_r_lr_real = t_r_lr[:,sta_z,:]
            t_r_lr_real = t_r_lr_real[:,:,sta_h]
            
            t_p_lr_real = t_p_lr[:,sta_z,:]
            t_p_lr_real = t_p_lr_real[:,:,sta_h]
            
                              
            ( aveAbsE_line1,absError1 ) = Evaluation.Evaluation_AbsE( t_p_lr, t_r_lr )
            ( aveAbsE_line2,absError2 ) = Evaluation.Evaluation_AbsE( t_p_lr_real, t_r_lr_real )

        #    ( aveAbsE_line2,absError2 ) = Evaluation.Evaluation_AbsE( t_p_lstm_real, t_r_lstm_real )
        
#            print('Saving......')
#            print('Step:', step_num)
#            print('Station_num:', s_station_num)                
#            print('    aveAbsE_line_test(all): ', aveAbsE_line1);
#            print('\n')
        
            ave1_list.append(aveAbsE_line1)
            ave2_list.append(aveAbsE_line2)
            step_list.append(step_num)

    return ave1_list,ave2_list,step_list


def discount(rewards):   #标准化奖励  问题函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    discount_rewards = np.zeros_like(rewards)
    running_add = 0
    GAMMA = 0.99
    for t in reversed(range(len(rewards))):
#        print(rewards[t])
        running_add = running_add * GAMMA + rewards[t]
#        print(running_add)
        discount_rewards[t] = running_add
        
    return (discount_rewards-np.mean(discount_rewards))/(np.std(discount_rewards)+1e-7)


def create_label_first(inter_num,N_SELECT):              
    perm_train = np.array(random.sample(range(0,inter_num),inter_num))
    
    s_label = np.zeros(inter_num)
    num = np.where(perm_train<N_SELECT)
    num = np.array(num).tolist()[0]

    s_label[num] = 1
    s_label = s_label.reshape(-1,1)
    return s_label

def action_deal(N_SELECT,action_index,probility):
    a_list = []
    for i in N_SELECT:
        a = action_index[i]
        p = probility[i]
        if p == 0:
            a_ = a-1
        if p == 1:
            a_ = a
        if p == 2:
            a_ = a+1
        a_list.append(a_)
        return a_list
        

def take_action(action,sb,num_inter):   #执行动作，改变sb
    s_b = torch.zeros((1,num_inter,1))
    num = 0
    for i in range(len(action[0,:])):
        if sb[0,i,0] == 1:
            num =num+1
            if action[0,i] == 0 :    #左移
                if i != 0 :        #判断是否到左边界
                    s_b[0,i-1,0] =1
                else:
                    s_b[0,i,0] = 1
#            elif action[0,i] == 1 :  #不动
#                    s_b[0,i,0] =1
            elif action[0,i] == 1 :  #右移
                if i != (num_inter-1) :        #判断是否到右边界
                    s_b[0,i+1,0] =1
                else:
                    s_b[0,i,0] = 1
                    
#    print(num)
    return s_b


def take_action_bound(action,sb,num_inter):   # 将移动步长设定为0和1 但是中间三个相临的站点如何处理？？  增加移动区间 执行动作，改变sb
    s_b = torch.zeros((1,num_inter,1))
    num = 0
    for i in range(len(action[0,:])):
        if sb[0,i,0] == 1:
            num =num+1
            if i == 0 and i == 1:                                        #判断是否到左边界
                if sb[0,i+2,0] == 1 and action[0,i+2] == 0 and action[0,i] == 1:
                    L = 0
            elif i == (num_inter-1) and  i == (num_inter-2):                         #判断是否到右边界
                if sb[0,i-2,0] == 1 and action[0,i-2] == 1 and action[0,i] == 0 :
                    L = 0
            else:
                if sb[0,i-2,0] == 1 and action[0,i-2] == 1 and action[0,i] == 0 :
                    L = 0
                elif sb[0,i+2,0] == 1 and action[0,i+2] == 0 and action[0,i] == 1  :
                    L = 0
                else:
                    L = 1
                
                
            if action[0,i] == 0 :    #左移
                if i != 0 :             #判断是否到左边界
                    s_b[0,i-L,0] =1
                else:
                    s_b[0,i,0] = 1
#            elif action[0,i] == 1 :  #不动
#                    s_b[0,i,0] =1
            elif action[0,i] == 1 :  #右移
                if i != (num_inter-1) :        #判断是否到右边界
                    s_b[0,i+L,0] =1
                else:
                    s_b[0,i,0] = 1
                    
#    print(num)
    return s_b

def take_action_bound2(action,sb,num_inter):   # 在移动时 如果前一个已经为1 则不发生移动 增加移动区间 执行动作，改变sb
    s_b = torch.zeros((1,num_inter,1))
    num = 0
    for i in range(len(action[0,:])):
        if sb[0,i,0] == 1:
            num =num+1
             
            if action[0,i] == 0 :    #左移
                if i != 0 :             #判断是否到左边界
                    if s_b[0,i-1,0] == 1:
                        L = 0
                    else:
                        L=1
                        
                    s_b[0,i-L,0] =1
                else:
                    s_b[0,i,0] = 1
#            elif action[0,i] == 1 :  #不动
#                    s_b[0,i,0] =1
            elif action[0,i] == 1 :  #右移
                if i != (num_inter-1) :        #判断是否到右边界
                    if s_b[0,i+1,0] == 1:
                        L = 0
                    else:
                        L=1
                        
                    s_b[0,i+L,0] =1
                else:
                    s_b[0,i,0] = 1
                    
#    print(num)
    return s_b

def take_action_bound_paper(action,sb,num_inter):   # 使用paper上的移动区间方法 增加移动区间 执行动作，改变sb
    s_b = torch.zeros((1,num_inter,1))
    num = 0
    sb_select_index = np.where(sb[0,:,0]==1)
    sb_select_index = np.array(sb_select_index).tolist()[0]
    m = len(sb_select_index)

    for i in range(m):
        num =num+1
        if i != m-1:         #上界
            Mi = sb_select_index[i]
            Miadd1 = sb_select_index[i+1]
            upper_bound = np.ceil((Mi + Miadd1)/2)
            
        elif i == m-1 :
            upper_bound = (num_inter-1)
            
        if i != 0:                      #下界
            Mi = sb_select_index[i]
            Misub1 = sb_select_index[i-1]
            lower_bound = np.ceil((Misub1 + Mi)/2)
            
        elif i == 0 :
            lower_bound = 0
             
        if action[0,Mi] == 0 :    #左移
            sigma = -min(1,Mi-lower_bound)
            s_b[0,int(Mi+sigma),0] =1

#            elif action[0,i] == 1 :  #不动
#                    s_b[0,i,0] =1
        elif action[0,Mi] == 1 :  #右移
            sigma = min(1,upper_bound-Mi-1)
            s_b[0,int(Mi+sigma),0] =1

#    print(num)
    return s_b


def trans(sa1,sb,N_SELECT,num_feature):     #sa2随着sa1 和 sb 改变
    sa1 = sa1.cpu().data.numpy()
    sb = sb.cpu().data.numpy()
    sa2 = np.zeros((1,N_SELECT,num_feature))
    index = 0
    for i in range(len(sa1[0,:,0])):
        if sb[0,i,0] ==1:
            sa2[0,index,:] =sa1[0,i,:]
            index  += 1
    sa2 = torch.from_numpy(sa2).float()
    return sa2


def cal_reward(data_all,target_all,select_station,select_inter,s_b,meanSq_target,busAllMaxTime):    # LSTM作为奖励值 计算相同id之间选取的局部特征的距离 
    s_inter = np.where(s_b[0,:,0] == 1)
    s_inter = np.array(s_inter).tolist()[0]
    select_inter = np.array(select_inter).reshape(-1,1)
    select_station = np.array(select_station).reshape(-1,1)
    
    s_inter_ = select_inter[s_inter]
    test = np.vstack((select_station,s_inter_))
    
    s_index = np.sort(test,axis=0).tolist()
    
    data = np.squeeze(data_all[s_index,:])
    target = np.squeeze(target_all[s_index,:]).reshape(-1,1)
    
    data =  data[np.newaxis,:,:]
    data = torch.Tensor(data)
    data = Variable(data).cuda()
    
    
#    lstm=RNN()
    lstm=torch.load('params110_forPG_9999.pkl')
#    lstm.eval()
#    lstm = nn.DataParallel(lstm.cuda())
    
    output,hprev, Scprev  = lstm(data)
    out = output.cpu().data.numpy().reshape((-1,1))
    abx = np.abs(out-target)
#    abx = (abx + meanSq_target[:,0])*busAllMaxTime[0]
    abx = np.mean(abx)
    abx = 0.0015
    loss= 1/abx/1000

#    loss = (1/(1+np.exp(-0.03*loss))-0.5)*20     #sigmiod

    return loss


def cal_reward_min(data_all,target_all,select_station,select_inter,s_b,meanSq_target,busAllMaxTime):    #min loss
    s_inter = np.where(s_b[0,:,0] == 1)
    s_inter = np.array(s_inter).tolist()[0]
    select_inter = np.array(select_inter).reshape(-1,1)
    select_station = np.array(select_station).reshape(-1,1)
    
    s_inter_ = select_inter[s_inter]
    test = np.vstack((select_station,s_inter_))
    
    s_index = np.sort(test,axis=0).tolist()
    
    data = np.squeeze(data_all[s_index,:])
    target = np.squeeze(target_all[s_index,:]).reshape(-1,1)
    
    data =  data[np.newaxis,:,:]
    data = torch.Tensor(data)
    data = Variable(data).cuda()
      
#    lstm=RNN()
    lstm=torch.load('params110_forPG_9999.pkl')
#    lstm.eval()
#    lstm = nn.DataParallel(lstm.cuda())
    
    output,hprev, Scprev  = lstm(data)
    out = output.cpu().data.numpy().reshape((-1,1))
    abx = np.abs(out-target)
#    abx = (abx + meanSq_target[:,0])*busAllMaxTime[0]
    loss = np.mean(abx)  
    
    loss_list.append(loss) 
    

    return loss

def cal_reward_lr(t_h,t_r,t_u,select_station,select_inter,s_b):
    s_inter = np.where(s_b[0,:,0] == 1)
    s_inter = np.array(s_inter).tolist()[0]
    select_inter = np.array(select_inter).reshape(-1,1)
    select_station = np.array(select_station).reshape(-1,1)
    
    s_inter_ = select_inter[s_inter]
    test = np.vstack((select_station,s_inter_))
    
    s_index = np.sort(test,axis=0).tolist()
    
    t_h = np.squeeze(t_h[s_index,:])
    t_r = np.squeeze(t_r[s_index,:]    )
    location_num = np.size(t_r,0)

    
    t_p_lr_list = []
    t_r_lr_list = []
#    testTripsNum_lr = np.size(t_r,1)
    testTripsNum_lr = 15
    
    for i in range(testTripsNum_lr):       #之前lr形成173*172

        t_r_lr_list.append(np.tril(repmat(t_r[:,i].reshape((-1,1),order="F"),1,location_num-1) , -1)*0.5)
        t_p_lr_list.append(linear_regr(t_h, t_r[:, i])*0.5)
        

    t_p_lr = np.array(t_p_lr_list)
    t_r_lr = np.array(t_r_lr_list)

    #D-value
    abx = np.abs(t_p_lr - t_r_lr)
    abx = np.mean(abx)
    loss = 1/abx*10


    #Evaluation as loss 
#    ( aveAbsE_line1,absError1 ) = Evaluation.Evaluation_AbsE( t_p_lr, t_r_lr )
#    abx = aveAbsE_line1
##    abx = (abx + meanSq_target[:,0])*busAllMaxTime[0]
#    loss = 1/abx*10
    
    loss_list.append(loss) 
    

    return loss


def cal_reward_lr_single(t_h,t_r,t_u,select_station,select_inter,s_b):
    s_inter = np.where(s_b[0,:,0] == 1)
    s_inter = np.array(s_inter).tolist()[0]
    select_inter = np.array(select_inter).reshape(-1,1)
    select_station = np.array(select_station).reshape(-1,1)
    
    s_inter_ = select_inter[s_inter]
    test = np.vstack((select_station,s_inter_))
    
    s_index = np.sort(test,axis=0).tolist()
    
    t_h = np.squeeze(t_h[s_index,:])
    t_r = np.squeeze(t_r[s_index,:]    )
    location_num = np.size(t_r,0)

    
    t_p_lr_list = []
    t_r_lr_list = []
#    testTripsNum_lr = np.size(t_r,1)
    testTripsNum_lr = 64
    
    for i in range(testTripsNum_lr):        #LR由一站预测之后所有站的时间

        t_r_lr_list.append(t_r[:,i].reshape((-1,1),order="F") *0.5)
        t_p_lr_list.append(linear_regr_single(t_h, t_r[:, i])*0.5)
 
                
    t_p_lr = np.array(t_p_lr_list)
    t_r_lr = np.array(t_r_lr_list)

    #D-value
    abx = np.abs(t_p_lr - t_r_lr)
    abx = np.mean(abx)
    loss = 1/abx*10


    #Evaluation as loss 
#    ( aveAbsE_line1,absError1 ) = Evaluation.Evaluation_AbsE( t_p_lr, t_r_lr )
#    abx = aveAbsE_line1
##    abx = (abx + meanSq_target[:,0])*busAllMaxTime[0]
#    loss = 1/abx*10
    
    loss_list.append(loss) 
    

    return loss


#def get_action(a):   #得到计算用的action矩阵  !!!三个动作
#    action = torch.FloatTensor([])
#    for i in a[0,:]:
#        if i == 0:
#            action = torch.cat((action,torch.FloatTensor([[1,0,0]])))
#        elif i == 1:
#            action = torch.cat((action, torch.FloatTensor([[0,1,0]])))
#        elif i == 2:
#            action = torch.cat((action, torch.FloatTensor([[0,0,1]])))
#    return  action

def get_action(a):   #得到计算用的action矩阵 
    action = torch.FloatTensor([])
    for i in a[0,:]:
        if i == 0:
            action = torch.cat((action,torch.FloatTensor([[1,0]])))
        elif i == 1:
            action = torch.cat((action, torch.FloatTensor([[0,1]])))
#        elif i == 2:
#            action = torch.cat((action, torch.FloatTensor([[0,0,1]])))
    return  action


def station_fanal(s_b,select_inter,select_station):
    s_inter = np.where(s_b[0,:,0] == 1)
    s_inter = np.array(s_inter).tolist()[0]
    select_inter = np.array(select_inter).reshape(-1,1)
    select_station = np.array(select_station).reshape(-1,1)
    
    s_inter_ = select_inter[s_inter]
    test = np.vstack((select_station,s_inter_))
    
    s_index = np.sort(test,axis=0).tolist()
    return s_index
    

def linear_regr(history_time, t_r):
    t_u = np.mean(history_time,1)
    
    location_num = len(t_r)
    t_p_linear = np.zeros((location_num,location_num-1))
    
    sigma_h = np.cov(history_time, bias = 'Ture')
    sigma_h = sigma_h + np.diag(0.0001*np.ones(len(t_u)))
    
    for l in range(location_num-1):
        sigma_ll = sigma_h[0:l+1,0:l+1]
        sigma_lh = sigma_h[0:l+1,l+1::].T
        
#        test = (t_r[0:l+1] - t_u[0:l+1])
#        test1= np.dot(sigma_lh,np.linalg.inv(sigma_ll))
#        t_p_linear[l+1::,l] = t_u[l+1::] + sigma_lh/sigma_ll*(t_r[0:l] - t_u[0:l])
        t_p_linear[l+1::,l] = t_u[l+1::] + np.dot(np.dot(sigma_lh,np.linalg.inv(sigma_ll)),(t_r[0:l+1] - t_u[0:l+1]))
        
    return t_p_linear


def linear_regr_single(history_time, t_r):     #输出单列时间
    t_u = np.mean(history_time,1)
    
    location_num = len(t_r)
    t_p_linear = np.zeros((location_num,1))
    
    sigma_h = np.cov(history_time, bias = 'Ture')
    sigma_h = sigma_h + np.diag(0.0001*np.ones(len(t_u)))
    
    for l in range(1):
        sigma_ll = sigma_h[0:l+1,0:l+1]
        sigma_lh = sigma_h[0:l+1,l+1::].T
        
#        test = (t_r[0:l+1] - t_u[0:l+1])
#        test1= np.dot(sigma_lh,np.linalg.inv(sigma_ll))
#        t_p_linear[l+1::,l] = t_u[l+1::] + sigma_lh/sigma_ll*(t_r[0:l] - t_u[0:l])
        t_p_linear[l+1::,l] = t_u[l+1::] + np.dot(np.dot(sigma_lh,np.linalg.inv(sigma_ll)),(t_r[0:l+1] - t_u[0:l+1]))
        
    return t_p_linear

