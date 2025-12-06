#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:55:51 2020

@author: wzk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:31:04 2020

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


class RNN(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.rnn=torch.nn.LSTM(
#            input_size=10,
#            hidden_size=64,
#            num_layers=1,
#            batch_first=True
#        )
#        self.out=torch.nn.Linear(in_features=10,out_features=1)
#
#    def forward(self,x):
#        # 一下关于shape的注释只针对单项
#        # output: [batch_size, time_step, hidden_size]
#        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
#        # c_n: 同h_n
##        h0 = torch.randn(2, 3, 20)
##        c0 = torch.randn(2, 3, 20)
#        output,(h_n,c_n)=self.rnn(x)
##        output,(h_n,c_n)=self.rnn(x,(h0,c0))
#
#        print(output.size())
#        # output_in_last_timestep=output[:,-1,:] # 也是可以的
#        output_in_last_timestep=h_n[-1,:,:]
#        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
#        x=self.out(output_in_last_timestep)
#        return x
    
#========================================================================================    
    
    def __init__(self,input_size=10,hidden_size=64, output_size=1,num_layers=1):         #HJ
#    def __init__(self,input_size=10,hidden_size=64, output_size=115,num_layers=1):      #multiful-output

        super().__init__()
 
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers,bias=False)#,batch_first=True)
        self.reg = torch.nn.Linear(hidden_size,output_size)
 
    def forward(self,x):
#        x, (hn,cn) = self.rnn(x,(h,c))
        x, (hn,cn) = self.rnn(x)

        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x,hn,cn


class Net(nn.Module):             # no OD

    def __init__(self,num_inter,num_select,num_feature):
        super(Net, self).__init__()
#        num_inter = 212                  #
        self.conv = nn.Conv2d(num_inter+num_select, num_inter, kernel_size=1, stride=1, bias=False)
#        self.conv = nn.Conv2d(352, num_inter, kernel_size=1, stride=1, bias=False)   #!!!!!!!!!!!!!!!别的站点需要调参数
        self.fc1 = nn.Linear(num_feature, 5)
#        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(1, 5)
#        self.fc3 = nn.Linear(10, 3)
        
        self.out = nn.Linear(10, 2)
#    def forward(self, x,y, num_inter, num_select):
    def forward(self, x,y,num_inter,num_select,num_feature):

        x = self.conv(x.view(-1,num_inter+num_select,num_feature,1))
#        x = self.conv(x.view(-1,352,8,1))   #别的站点需要调参数
#        print(x.size())
#        x = x.view(-1,num_inter,8)     #!!!!!!!!!!!!!!!!!别的站点需要调参数
        x = x.view(-1,num_inter,num_feature)     #!!!!!!!!!!!!!!!!!别的站点需要调参数
#        print(x.size())
        
        x = self.fc1(x)
#        print(x.size())

        y = self.fc2(y)
#        print(y.size())

        x_y = torch.cat((x,y),-1)
#        print(x_y.size())

#        x_y = F.relu(x_y)
#        print(x_y.size())
        
        actions_value = self.out(x_y)
#        print(actions_value)
#        print(actions_value.size())
        
        actions_value = F.softmax(actions_value,dim=2)
#        print(actions_value)

        return actions_value


def discount(rewards):   #标准化奖励  问题函数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    discount_rewards = np.zeros_like(rewards)
    running_add = 0
    GAMMA = 0.99
    for t in reversed(range(len(rewards))):
#        print(rewards[t])
        running_add = running_add * GAMMA + rewards[t]
#        print(running_add)
        discount_rewards[t] = running_add

    return (discount_rewards-np.mean(discount_rewards))/ (np.std(discount_rewards)+1e-7)


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


def cal_reward(data_all,target_all,select_station,select_inter,s_b,meanSq_target,busAllMaxTime):    #计算相同id之间选取的局部特征的距离
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
    abx = (abx + meanSq_target[:,0])*busAllMaxTime[0]
    loss = np.mean(abx)
    

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
    
    test = history_time.T
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

         
       





            
if __name__ == "__main__":
    # 1. 加载数据
    Line = 14


    data_path="./data/data_" + str(Line) + "/data_target.mat"
    test_path="./data/data_" + str(Line) + "/test_data.mat"
#    testData = './data_2/ test_date11.mat'
    historyTime_path = "./data/data_" + str(Line) + "/historyTime_" + str(Line) + ".mat"
    
    
    data_train = scio.loadmat(data_path)
    data_test = scio.loadmat(test_path)
    historyTime = scio.loadmat(historyTime_path)

    target=data_train.get('target')#取出字典里的label    
    data = data_train.get('data')#取出字典里的data
#    opt = data_train.get('opt')
#    oldGradient = data_train.get('oldGradient')
    

    data_train = scio.loadmat(test_path)
    
    perm_train = data_train.get('perm')
    tripTime = data_train.get('tripTime_test')
    IdList = data_train.get('IdList_test') 
    meanSq_target = data_train.get('meanSq') 
    busAllMaxTime = data_train.get('AllMaxTime') 
#    mostOMode = data_train.get('mostOMode') 
#    mostDMode = data_train.get('mostDMode')
#    bacB_o = data_train.get('bacB_o')
#    bacB_d = data_train.get('bacB_d')
#    index_vals = data_train.get('index_vals')
#    index_vals_Loc = data_train.get('index_vals_Loc')
    stationFlag_test = data_train.get('stationFlag_test')


    data_train = scio.loadmat(test_path)
    
    historyTrip_11 = historyTime.get('history_11')
    historyTrip_21 = historyTime.get('history_21')    
    testTrip_11 = historyTime.get('testTrip_11')    
    testTrip_21 = historyTime.get('testTrip_21')

    type_hour = 0

#    testTripsNum = 82   #Epoch499:1.6259340646442557
    


    trainTripNum = np.size(data,0)  #1792 2304   np.size(data,0)
    testTripsNum = np.size(tripTime,0)
    predictionLength = np.size(IdList,2) #172 115
    data_feature = np.size(data,1)
    loss = []

    num_feature = np.size(data,1)
    data = (data[perm_train,:,:]).reshape((-1,data_feature,predictionLength) ,order="F")
    target = (target[perm_train,:,:]).reshape((-1,1,predictionLength), order="F")
    
    data = data.transpose(0,2,1)
    target = target.transpose(0,2,1)
    
    
#    net = Net()
#    net = net.cuda
#  Policy gradient

    

    # Hyper Parameters
    LR = 0.01                   # learning rate
    #env = gym.make('CartPole-v0')
    #env = env.unwrapped

    
    
    
    loss_list =[]
    step_list = []
    test_reward_list = []
    action_list = [0,1]
    FloatTensor = torch.FloatTensor
    
    select_inter = np.where(stationFlag_test[0, 0:-1]==0)
    select_inter = np.array(select_inter).tolist()[0]
    
    select_station = np.where(stationFlag_test[0, 0:-1]==1)
    select_station = np.array(select_station).tolist()[0]
    
    data_inter = data[:,select_inter,:]
    inter_num = np.size(data_inter,1)  #the numnber of inter
    target_inter = target[:,select_inter,:]
    
    num_inter = np.size(data_inter,1)
      
    N_ACTIONS = 3
    N_STATES = 212   #4
    N_SELECT = math.ceil(num_inter/3*2)
#    N_SELECT = 30
    N_step = 5
    bound = 10
    trainTripNum_ =np.size(data,0)
    
    bound_num = math.floor(num_inter/bound)
    
    #按照id提取特征维度for
    
    step = 1
    trainTripNum =np.size(data,0)
    
    
    net = Net(num_inter,N_SELECT,num_feature).cuda()
    optimizer_rl = torch.optim.SGD(net.parameters(), lr = 0.0001, momentum=0.95)


#    test_epoch = ['49','99','149','199','249','299','349','399','449','499']
    test_epoch = ['99','199','299','399','499']
    step = 200
#    test_epoch = ['9','19','29','39','49','59','69','79','89','99','109','119','129','139','149','159','169','179','189','199']
    
    for test in test_epoch:
#    net=torch.load('params110_PGmodel_499.pkl')
#    net=torch.load('params110_PGmodel_499.pkl')
        path = './params/params14_PGmodel_LrSingle_499_nforpaper.pkl'
#        path = 'params563_PGmodel_LrSingle_0_nforpaper.pkl'

#        path = 'params110_PGmodel_'+ test + '.pkl'
#        path = 'params'+ str(Line) + '_PGmodel_LrSingle_'+ test + '.pkl'
#        path = 'params'+ str(Line) + '_PGmodel_LrSingle_'+ test + '_bound' + '.pkl'

#        path = 'params110_PGmodel_Lr_'+ test + '.pkl'
        
        net=torch.load(path)
        
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
        ave_list = []
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
            print(np.size(test))
#
#            s_b = take_action(a,s_b,num_inter).cuda() 
            s_b = take_action_bound_paper(a,s_b.cpu().data,num_inter).cuda() 

            s_a2 = trans(s_a1,s_b,N_SELECT,num_feature)        #根据执行的动作更新选取的局部特征
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
                    if type_hour == 1:
                        houridx = t_r_houridx[i]
            
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
                test = t_r_lr[:,sta_z,:]
                t_r_lr_real = t_r_lr_real[:,:,sta_h]
                
                t_p_lr_real = t_p_lr[:,sta_z,:]
                t_p_lr_real = t_p_lr_real[:,:,sta_h]
                
                   
                ( aveAbsE_line1,absError1 ) = Evaluation.Evaluation_AbsE( t_p_lr, t_r_lr )
                ( aveAbsE_line2,absError2 ) = Evaluation.Evaluation_AbsE( t_p_lr_real, t_r_lr_real )
            #    ( aveAbsE_line2,absError2 ) = Evaluation.Evaluation_AbsE( t_p_lstm_real, t_r_lstm_real )
                print('Saving......')
                print('Step:', step_num)
                print('Station_num:', s_station_num)                
                print('    aveAbsE_line_test(all): ', aveAbsE_line1);
                print('    aveAbsE_line_test(stations): ', aveAbsE_line2)
                print('\n')
                ave_list.append(aveAbsE_line1)
                step_list.append(step_num)
            
    plt.plot(step_list,ave_list)
    plt.xlabel('Step')
    plt.ylabel('AbsE')
    
#    plt.plot(test_reward_list)  
    
    plt.show()
    

#=============================================================================================================


#    for i_episode in range(400):
#             
#        for j in range(trainTripNum):
#            #定义s_a和s_b
#            s_point = data[j,:,:]                            
#            if i_episode==0 & j==0:
#                s_label = create_label_first(inter_num,N_SELECT)
##            else:
##                s_label = s_label_model
#
#            for k in range(N_step):
#            
#                
#                action_tensor=FloatTensor([])
#                reward_tensor=FloatTensor([])
#                state_tensor=FloatTensor([])
#                
##                score = 0
##                step = 0
##                mean_step = 0
#                
#                s_point = torch.FloatTensor(s_point[np.newaxis,:])
#                s_label = torch.FloatTensor(s_label[np.newaxis,:])
#                s_point = Variable(s_point,requires_grad=True)
#                s_label = Variable(s_label,requires_grad=True)
#                
#                action_prob = net(s_point,s_label)
#                a = 0 if random.random() < action_prob.data[0][0] else 1
#        
#                # take action
#                s_, r, done, info = env.step(a)     #?????????????????????????????????
#                
#                
#                action = FloatTensor([[1, 0]] if a == 0 else [[0, 1]])
#                score += r
#        
#                action_tensor = torch.cat([action_tensor,action])
#                reward_tensor = torch.cat([reward_tensor,FloatTensor([[r]])])
#                state_tensor = torch.cat([state_tensor,s])
#        
#                s_label = s_label_model
#                
#            dis_reward = discount(reward_tensor)
#            action = net(state_tensor)#.gather(1, action_tensor)
#            action_tensor = Variable(action_tensor, requires_grad=True)
#            dis_reward = Variable(FloatTensor([dis_reward]))
#            log_lik = -action_tensor * torch.log(action)
#            log_lik_adv = log_lik * dis_reward
#            loss = torch.sum(log_lik_adv, 1).mean()
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            loss_list.append(loss.data)
            

