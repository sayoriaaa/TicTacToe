# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:22:33 2021
@author: sayori
去掉了之前漫天飞的全局变量，改成类的数据
感觉有点下意识地写工厂模式了(面向对象真香)
实现了之前想的一个小idea:套用卷积的方法来实现模式的匹配
当然速度又慢了一点，但是实现了N字棋与棋盘sizeM的MN自由(m=5就不行了,m等于4勉强能玩)
但m=4的情况是有问题的，因为
井字棋计算机后手是一定不会输的
所以没有考虑可能会输的选择
但是m=4的情况计算机基本上算出来会输，这时候要选择加上即使是失败也要最努力的方向而不是直接随机躺平
不然就看不出智能了(怎么感觉努力后的躺平也是一种智能行为 哭)
暂时的思路是如果全部返回为-1的话就把min改为随机
"""
import tkinter
import numpy as np
import time
import random
import tkinter.messagebox
import tkinter.simpledialog

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
       
        

        
    def load(self,width=500,height=500):
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
        
        self.cv.delete("all")
        self.canvas_grid()
        self.refresh() 
        
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
                        alpha,beta=-np.inf,np.inf
                        temp_value=minsearch(temp_computer, temp_player,alpha,beta)[0]
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
            pass
        
    def destroy(self):
        self.win.destroy()

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

def maxsearch(t_computer,t_player,pre_alpha,pre_beta):
    '''
    value为最终的结果
    人赢了返回-1
    电脑赢了返回1
    平局返回0
    人想要value小，电脑想要value大
    '''
    temp_value=-1
    alpha,beta=pre_alpha,pre_beta
    if check_win_single(t_computer):
        return [1,alpha,beta]
    if check_win_single(t_player):
        return [-1,alpha,beta]
    if check_even(t_computer, t_player):
        return [0,alpha,beta]
    
    for i in range(t_computer.shape[0]):
        for j in range(t_computer.shape[0]):
            if t_computer[i][j]==0 and t_player[i][j]==0:
                a=np.array(t_computer,copy=True)
                b=np.array(t_player,copy=True)
                a[i][j]=1

                if(alpha>=beta):
                    break

                t_value,t_alpha,t_beta=minsearch(a,b,alpha,beta)
                #在max层最大化更新alpha
                temp_value=max(temp_value,t_value)
                alpha=max(t_alpha,alpha)

    return [temp_value,t_alpha,t_beta]
               
def minsearch(t_computer,t_player,pre_alpha,pre_beta):
    temp_value=1
    alpha,beta=pre_alpha,pre_beta
    if check_win_single(t_computer):
        return [1,alpha,beta]
    if check_win_single(t_player):
        return [-1,alpha,beta]
    if check_even(t_computer, t_player):
        return [0,alpha,beta]
    
    for i in range(t_computer.shape[0]):
        for j in range(t_computer.shape[0]):
            if t_computer[i][j]==0 and t_player[i][j]==0:
                a=np.array(t_computer,copy=True)
                b=np.array(t_player,copy=True)
                b[i][j]=1
                if(alpha>=beta):
                    break
                t_value,t_alpha,t_beta=maxsearch(a,b,alpha,beta)
                #在min层最小化更新beta
                temp_value=min(temp_value,t_value)
                beta=min(beta,t_beta)
    return [temp_value,alpha,beta]

########################################################

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
        
    def backup(self,v):
        self.N+=1
        self.W+=v
        if self.parent!=None:
            self.parent.backup(-v)
        
    def compute(self):
        max_UCB=-np.inf
        ret_move=(0,0)
        for move,node in self.child.items():
            node.Q=node.W/node.N
            node.U=Node.Cpuct*self.P[move]*np.sqrt(self.N)/(1+node.N)
            node.UCB=node.U+node.Q
            if node.UCB>max_UCB:
                max_UCB=node.UCB
                ret_move=move
        self.ret_move=ret_move
            
    def add_child(self,move,P):
        self.child[move]=Node(self,P)
    
    def lack_child(self):
        return not bool(self.child)

class MCTS:
    def __init__(self,board_size=5,simulation_time=400,nn=None):
        self.board_size=board_size
        self.simulation_time=simulation_time
        self.neural_network=nn
        
        self.root=Node(player_label=1)
        
        
        
        
        
        
                        
                  
if __name__=="__main__":
    ga=Game()
    ga.load()
    ga.run()
    ga.destroy()
    