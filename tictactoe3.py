# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:22:33 2021

@author: sayori

关于模型的保存，本来用json的，结果发现只能使用pickle，json不支持类的存储
pickle只能以二进制读写

"""
import tkinter
import numpy as np
import time
import random
import pickle
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as torch_data




def get_matrix_index(a,value):
    ''' 
    非主要内容
    不想自己写的，但是搜了一圈numpy的where真是太不好用了，还不如自己写
    Parameters
    ----------
    a : np.array
    value : float
    Returns
    -------
    tuple
    '''
    u,v=a.shape
    for i in range(u):
        for j in range(v):
            if a[i,j]==value:
                return (i,j)

def generate_conv_kernel(cn,cross=True):
    '''
    输入cn代表多少个点连起来可以赢
    cross表示斜着是否算赢
    '''
    ret=[]
    for i in range(cn):
        kernel=np.zeros((cn,cn))
        kernel[i]=1
        ret.append(kernel)
        kernel=np.zeros((cn,cn))
        kernel[...,i]=1
        ret.append(kernel)
    
    if cross:
        kernel=np.zeros((cn,cn))
        for i in range(cn):
            kernel[i][i]=1
        ret.append(np.array(kernel,copy=True))
        ret.append(np.array(np.rot90(kernel,1),copy=True))
    return ret

class Game:
    chain_num=3 #默认三个子 连
    conv_kernel_list=generate_conv_kernel(chain_num)
    def __init__(self,N=3):
        self.N=N
        self.player=np.zeros((N,N),dtype=int)  
        self.computer=np.zeros((N,N),dtype=int)
        self.computer_history=[np.zeros((N,N),dtype=int)]
        self.player_history=[np.zeros((N,N),dtype=int)]
        self.algorithm=0
        self.MCTS=None
       
        
    def load(self,width=900,height=900):
        self.width=width
        self.height=height
        
        win=tkinter.Tk()
        win.title("TicTacToe")
        win.geometry("{}x{}".format(str(width),str(height)))
        
        m=tkinter.Menu(win)
        m.add_command(label='悔棋',command=self.regret)
        m.add_command(label='格数',command=self.adjust_grid)
        m.add_command(label='算法',command=self.set_algorithm)
        win['menu']=m
        
        canvas_size=min(width,height)
        cv=tkinter.Canvas(win,width=canvas_size,bg="blue",height=canvas_size)  
        cv.bind('<Button-1>',self.call)        
        
        self.win=win
        self.cv=cv
        self.canvas_size=canvas_size
        self.canvas_grid()
        self.cv.pack()
        
    def canvas_grid(self):
        cv=self.cv
        N=self.N
        l=self.canvas_size
        for i in range(1,N):
            cv.create_line(0,l//N*i,l,l//N*i)
            cv.create_line(l//N*i,0,l//N*i,l)
            
    def adjust_grid(self):
        new_n=tkinter.simpledialog.askinteger("S", "重新设置棋盘格数\n修改后会清空棋盘！！！",initialvalue=3)
        if new_n==self.N:
            return
        else:
            self.N=new_n
            self.reset()
            
    def reset(self):
        N=self.N
        self.player=np.zeros((N,N),dtype=int)  
        self.computer=np.zeros((N,N),dtype=int)
        self.computer_history=[np.zeros((N,N),dtype=int)]
        self.player_history=[np.zeros((N,N),dtype=int)]
        self.MCTS=None
        
        self.cv.delete("all")
        self.canvas_grid()
        self.refresh()
        print("reseted")
        
    def set_algorithm(self):
        
        def algorithm_confirm():
            for i in lb.curselection():
                print(lb.get(i))
                print(i)
                self.algorithm=int(i)
                self.reset()
                set_win.destroy()
        
        
        set_win=tkinter.Tk()
        set_win.geometry("600x400")
        set_win.title("算法设置")
        
        pr=tkinter.Text(set_win,width=40,height=4)
        pr.insert(tkinter.INSERT,"maxmin:\n基于穷举法的决策，运行时间长\nalphago:基于深度强化学习的策略\n需要预训练")
        pr.pack()
        
        m=tkinter.StringVar()
        lb=tkinter.Listbox(set_win,listvariable=m,height=2)
        for item in ["maxmin","alpha-zero"]:
            lb.insert("end", item)
        lb.pack()
        b1=tkinter.Button(set_win,text="确认",command=algorithm_confirm,width=30)
        b1.pack()
        
      
        
    def run(self):
        self.win.mainloop()
               
    def regret(self):
        if len(self.computer_history)==1:
           tkinter.messagebox.showinfo("提示","无子可悔！") 
           return
        self.computer_history.pop()
        self.player_history.pop()
        
        print(self.computer_history)
        self.computer=self.computer_history[-1]
        self.player=self.player_history[-1]
        if len(self.computer_history)==1:#不知道为啥点撤点撤出bug先改了
            self.computer_history=[np.zeros((self.N,self.N),dtype=int)]
            self.player_history=[np.zeros((self.N,self.N),dtype=int)]
            
        self.cv.delete("all")
        self.canvas_grid()
        self.refresh()
            
    def call(self,event):
        x=int(event.x/self.canvas_size*self.N)
        y=int(event.y/self.canvas_size*self.N)
        if self.player[x][y]==1:
            tkinter.messagebox.showinfo("提示","不能选择自己已落子的区域")
        elif self.computer[x][y]==1:
            tkinter.messagebox.showinfo("提示","不能选择对方已落子的区域")
        else:
            self.player[x][y]=1
            self.player_history.append(np.array(self.player,copy=True))
            self.refresh()
            if self.check_win():
                return
            time.sleep(0.5)
            self.auto_move()
            self.computer_history.append(np.array(self.computer,copy=True))#cccccc
            print(self.computer_history)
            self.refresh()
            self.check_win()           
    
    def refresh(self):
        unit=self.canvas_size//self.N
        for i in range(self.N):
            for j in range(self.N):
                if self.computer[i][j]==1:
                    self.cv.create_oval(i*unit,j*unit,(i+1)*unit,(j+1)*unit,fill="black")
                if self.player[i][j]==1:
                    self.cv.create_oval(i*unit,j*unit,(i+1)*unit,(j+1)*unit,fill="white")
                    
    def check_win(self):
        reset=False
        if check_win_single(self.player):
            tkinter.messagebox.showinfo("提示","己方获胜")
            reset=True
        elif check_win_single(self.computer):
            tkinter.messagebox.showinfo("提示","己方落败")
            reset=True
        elif check_even(self.computer, self.player):
            tkinter.messagebox.showinfo("提示","平局")
            
            reset=True
        if reset:
            self.reset()         
        return reset
    
    def auto_move(self):
        if self.algorithm==0:
            temp_x=0
            temp_y=0
            choosable=[]
            value=-1
            for i in range(self.N):
                for j in range(self.N):
                    if self.computer[i][j]==0 and self.player[i][j]==0:
                        choosable.append((i,j))
                        temp_computer=np.array(self.computer,copy=True)
                        temp_player=np.array(self.player,copy=True)
                        temp_computer[i][j]=1
                        temp_value=minsearch(temp_computer, temp_player)
                        if temp_value>value:
                            temp_x=i
                            temp_y=j
                            value=temp_value
                        if value==1:
                            break
            if value==-1:
                temp_x,temp_y=random.choice(choosable)#如果没希望了就随机走          
            self.computer[temp_x][temp_y]=1
        elif self.algorithm==1:
            if self.MCTS==None:      
                try:
                    with open("MCTS{}.sayoriaaa".format(self.N),"rb") as f:
                        self.MCTS=pickle.load(f)
                except IOError:
                    self.train_MCTS(train_time=1000)
                    
            self.MCTS.simulation(self.player,self.computer)
            _,next_move=self.MCTS.current_node.get_pi
            self.computer[next_move]=1
                
    def train_MCTS(self,train_time=1000,batch_size=1):  
        train_network=SNN(2,self.N)
        training_MCTS=MCTS(board_size=self.N,nn=train_network)
        N=self.N
        Z=[]
        
        for go in range(train_time):
            pi=[]
            player=np.zeros((N,N),dtype=int)  
            computer=np.zeros((N,N),dtype=int)
            computer_history=[np.zeros((N,N),dtype=int)]            
            player_history=[np.zeros((N,N),dtype=int)]
            training_MCTS.simulation(player, computer)
            
            while(True):
                z0,z1,z2=check_even(player, computer),check_win_single(player),check_win_single(computer)
                if not (z0 and z1 and z2):
                    if z0:
                        z=0
                    elif z1:
                        z=1
                    else:
                        z=-1
                    Z.append(z)
                    break
                ret,next_move=training_MCTS.current_node.get_pi
                pi.append(ret)
                if training_MCTS.current_player==1:
                    player[next_move]=1
                    player_history.append(player)
                else:
                    computer[next_move]=1
                    computer_history.append(computer)
            
##################################################
#一局训练好打包done
                state=[]
                Z_matrix=[]
                z_matrix=fill_matrix(size=N,num=z)
                z_matrix2=fill_matrix(size=N,num=-z)
                for i in len(player_history):
                    if i%2==0:
                        data=np.stack([player_history[i],computer_history[i]])
                        Z_matrix.append(z_matrix)
                    else:
                        data=np.stack([computer_history[i],player_history[i]])
                        Z_matrix.append(z_matrix2)
                    state.append(data)
                    
                tensor_states = torch.stack(tuple([torch.from_numpy(s) for s in state]))
                tensor_pi = torch.stack(tuple([torch.from_numpy(p) for p in pi]))
                tensor_z = torch.stack(tuple([torch.from_numpy(z) for z in Z_matrix]))
                dataset = torch_data.TensorDataset(tensor_states, tensor_pi, tensor_z)
                my_loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=True)
                       
                self.neural_network.train(my_loader) 
                print("已训练{}次对局".format(go))
            self.MCTS=training_MCTS
                
                        
                        
                        
                    
                    
            
            
            
            
        
        
                   
    def destroy(self):
        self.win.destroy()
def fill_matrix(size=0,num=1):
    ret=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            ret[i,j]=num
    return ret

def check_win_single(player):
    n,_=player.shape
    kernel_size,_=Game.conv_kernel_list[0].shape
      
    for i in range(n-kernel_size+1):#cccc
        for j in range(n-kernel_size+1):
            cut_area=player[i:i+kernel_size,j:j+kernel_size]
            for kernel in Game.conv_kernel_list:
                if (kernel*cut_area).sum()==kernel_size:
                    return True
    return False
        

def check_even(a,b):
    n,_=a.shape
    for i in range(n):
        for j in range(n):
            if a[i][j]==0 and b[i][j]==0:
                return False
    return True

def maxsearch(t_computer,t_player):
    '''
    value为最终的结果
    人赢了返回-1
    电脑赢了返回1
    平局返回0
    人想要value小，电脑想要value大
    '''
    temp_value=-1
    if check_win_single(t_computer):
        return 1
    if check_win_single(t_player):
        return -1
    if check_even(t_computer, t_player):
        return 0
    
    for i in range(3):
        for j in range(3):
            if t_computer[i][j]==0 and t_player[i][j]==0:
                a=np.array(t_computer,copy=True)
                b=np.array(t_player,copy=True)
                a[i][j]=1
                temp_value=max(temp_value,minsearch(a,b))
    return temp_value
               
def minsearch(t_computer,t_player):
    temp_value=1
    if check_win_single(t_computer):
        return 1
    if check_win_single(t_player):
        return -1
    if check_even(t_computer, t_player):
        return 0
    
    for i in range(3):
        for j in range(3):
            if t_computer[i][j]==0 and t_player[i][j]==0:
                a=np.array(t_computer,copy=True)
                b=np.array(t_player,copy=True)
                b[i][j]=1
                temp_value=min(temp_value,maxsearch(a,b))
    return temp_value

########################################################print

class Node:
    
    Cpuct=0.1
    
    def __init__(self,parent=None,P=None,player_label=1):
        self.parent=parent
        self.child={}#key用坐标的元组,直接在P里索引就好，我好聪明欸
        self.player_label=player_label
        
        self.N=1
        self.W=0
        
        self.Q=None
        self.P=P
        self.U=None
        self.UCB=None#本节点的ucb计算出来是给父节点做备选的
        
    def backup(self,v,stop=None):
        self.N+=1
        self.W+=v
        if self==stop:
            return
        if self.parent!=None:
            self.parent.backup(-v)
        
    def get_ucb(self):
        '''
        使用时要保证已经添加了子节点
        否则需要先调用self.expand
        simulation时使用ucb,实际走子时使用mcts分布
        '''
        max_UCB=-np.inf
        ret_move=(0,0)
        for move,node in self.child.items():
            node.Q=node.W/node.N
            node.U=Node.Cpuct*self.P[move]*np.sqrt(self.N)/(1+node.N)
            node.UCB=node.U+node.Q
            if node.UCB>max_UCB:
                max_UCB=node.UCB
                ret_move=move
        return ret_move
            
    def add_child(self,move):#先验概率在UCB中会赋值
        self.child[move]=Node(self,player_label=-self.player_label)
    
    def lack_child(self):
        return not bool(self.child)
    
    def get_pi(self,train=True):
        '''
        mcts分布
        根据当前父节点之下所有子节点的访问次数计算MCTS上的访问频率分布
        Returns
        -------
        棋盘size的分布矩阵，以及运行或训练下的选择(tuple)
        '''
        ret=np.zeros(self.P.shape)
        re_move=None
        for pos,node in self.child:
            ret[pos]=node.N
        ret=ret/np.sum(ret)
        if train:
            ret_value=np.random.choice(ret.flatten(),p=0.8*ret.flatten()+0.2*np.random.dirichlet(np.ones(ret.shape[0]**2)))
            re_move=get_matrix_index(ret, ret_value)
        else:
            re_move=divmod(np.argmax(ret), ret.shape[1]) 
            
        return ret ,re_move
    
        
            
            
            

class MCTS:
    def __init__(self,board_size=5,simulation_time=400,nn=None):
        self.board_size=board_size
        self.simulation_time=simulation_time
        self.neural_network=nn      
        self.root=Node(player_label=1)
        self.current_node=self.root
        self.current_player=1
        

        
    def simulation(self,player,computer):
        eval_counter=0
        for step in range(self.simulation_time):
            player_for_sim=np.array(player,copy=True)
            computer_for_sim=np.array(computer,copy=True)
            while self.check_prceed(player_for_sim, computer_for_sim):
                if self.current_node.lack_child():
                    P,v=self.neural_network.eva(self.get_state_for_eval(player,computer))
                    self.current_node.P=np.array(P.cpu()).reshape((self.board_size,self.board_size))
                    eval_counter+=1
                    self.current_node.backup(v)
                    self.expand_treenode(player, computer)
                next_move=self.current_node.get_ucb()
                self.sim_move(player_for_sim, computer_for_sim, next_move)
                self.current_node=self.current_node.child[next_move]
                self.current_node.parent={}
                self.current_node.N+=1
        return eval_counter/self.simulation_time#返回平均每次搜索的深度
    
    def get_state_for_eval(self,player,computer):
        if self.current_player==1:
            s=np.stack([player,computer])
        else:
            s=np.stack([computer,player])    
        tensor_states = torch.from_numpy(s)
        tensor_states=torch.unsqueeze(tensor_states, dim=0)
        return tensor_states
                
            
                
                
    def check_prceed(self,player,computer):
        if check_even(player, computer):
            return False
        if self.current_player==1:
            if check_win_single(computer):
                self.current_node=self.current_node.parent
                self.current_node.child={}
                self.current_node.backup(-1)
                return False
        if self.current_player==-1:
            if check_win_single(player):
                self.current_node=self.current_node.parent
                self.current_node.child={}
                self.current_node.backup(1)
                return False
        return True
    
    def expand_treenode(self,player,computer):
        size=player.shape[0]
        for i in range(size):
            for j in range(size):
                if player[i,j]==0 and computer[i,j]==0:
                    self.current_node.add_child((i,j))
    def sim_move(self,player,computer,move):
        if self.current_player==1:
            player[move]=1
        else:
            computer[move]=1
                    
                    
        
                
                

class NN1(nn.Module):   
    def __init__(self,input_layer,board_size=15):
        super(NN1,self).__init__()
        self.conv1=nn.Conv2d(input_layer, 32, 3)
        self.conv2=nn.Conv2d(32, 64, 3)
        self.conv3=nn.Conv2d(64, 128, 3)
        
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(128)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        
        self.board_size=board_size
        self.flatten_size=16*(board_size-6)*(board_size-6)#16是降采样后的通道数，见self.p_conv1=nn.Conv2d(128,16, 1)
        
        ###############################
        
        self.v_conv1=nn.Conv2d(128,16, 1)
        self.v_bn1=nn.BatchNorm2d(16)
        self.v_fc1=nn.Linear(256, 256)
        self.v_fc2=nn.Linear(256, 1)
        
        self.p_conv1=nn.Conv2d(128,16, 1)
        self.p_bn1=nn.BatchNorm2d(16)
        self.p_fc1=nn.Linear(self.flatten_size, out_features=board_size*board_size)
        
        
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.relu(self.bn3(self.conv3(x)))
        
        v=self.relu(self.v_bn1(self.v_conv1(x))).view(-1,self.flatten_size)
        v=self.relu(self.v_fc1(v))
        v=self.tanh(self.v_fc2(v))
        
        p=self.relu(self.p_bn1(self.p_conv1(x))).view(-1,self.flatten_size)
        p=self.relu(self.p_fc1(p))

        
        return p,v
    
class SNN:
    def __init__(self,input_layer,board_size,lr=0.1):
        self.model=NN1(input_layer,board_size)
        self.cuda=torch.cuda.is_available()
        if self.cuda:
            self.model=self.model.cuda().double()
        self.opt=torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=lr, weight_decay=1e-4)
        self.mse=nn.MSELoss()
        self.cross_entropy=nn.CrossEntropyLoss()
        
    def train(self,dataloader):
        loss_record=[]
        self.model.train()
        for idx,(S,pi,Z) in dataloader:
            
            pi=Variable(pi).double()
            Z=Variable(Z).double()
            if self.cuda:
                S=S.cuda()
                pi=pi.cuda()
                Z=Z.cuda()
            
            self.opt.zero_grad()
            p,v=self.model(S)
            output=F.log_softmax(p, dim=1)
            cross_entropy = -torch.mean(torch.sum(pi*output, 1))
            mse=F.mse_loss(v, Z)
            loss=cross_entropy+mse
            loss.backward()           
            self.opt.step()
            
            print("已训练{}局，mse{}，cross entropy:{}".format(idx,mse.data,cross_entropy.data))
            loss_record.append((mse.data,cross_entropy.data))
        return loss_record
    
    def eva(self,S):
        S=Variable(S).double()
        if self.cuda:
            S=S.cuda()
        with torch.no_grad():
            p,v=self.model(S)
        p=F.log_softmax(p, dim=1)
        return p,v
    
                
        
    
            
            
        
    
    
        
        
    
        
        

            
        
        
        
                        
                  
if __name__=="__main__":
    ga=Game()
    ga.load()
    ga.run()
    ga.destroy()
    
    
    
