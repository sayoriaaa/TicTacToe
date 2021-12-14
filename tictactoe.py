# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:22:33 2021

@author: sayori
"""
import tkinter
import numpy as np
import time

player=np.zeros((3,3),dtype=int)
computer=np.zeros((3,3),dtype=int)
canvas_size=0
cv=None

computer_history=[np.array(computer,copy=True)]
player_history=[np.array(player,copy=True)]

def load(width=900,height=900):
    global canvas_size
    global cv
    win=tkinter.Tk()
    win.title("TicTacToe")
    win.geometry("{}x{}".format(str(width),str(height)))
    win['menu']=initize_menu(win)
    canvas_size=min(width,height)
    cv=tkinter.Canvas(win,width=canvas_size,bg="blue",height=canvas_size)
    for i in range(1,3):
        cv.create_line(0,canvas_size//3*i,canvas_size,canvas_size//3*i)
        cv.create_line(canvas_size//3*i,0,canvas_size//3*i,canvas_size)
    cv.bind('<Button-1>',call)
    cv.pack()
    win.mainloop()

    
def initize_menu(win):
    win.title("TicTacToe")
    m=tkinter.Menu(win)
    m.add_command(label='悔棋',command=regret)
    return m

def regret():
    global computer_history
    global player_history
    global computer
    global player
    global cv
    if len(computer_history)==1:
       tkinter.messagebox.showinfo("提示","无子可悔！") 
       return
    computer_history.pop()
    player_history.pop()
    
    print(computer_history)
    computer=computer_history[-1]
    player=player_history[-1]
    if len(computer_history)==1:#不知道为啥点撤点撤出bug先改了
        computer_history=[np.zeros((3,3),dtype=int)]
        player_history=[np.zeros((3,3),dtype=int)]
        
    cv.delete("all")
    for i in range(1,3):
        cv.create_line(0,canvas_size//3*i,canvas_size,canvas_size//3*i)
        cv.create_line(canvas_size//3*i,0,canvas_size//3*i,canvas_size)
    refresh()
            
def call(event):
    x=int(event.x/canvas_size*3)
    y=int(event.y/canvas_size*3)
    global computer_history
    global player_history
    global player
    global computer
    if player[x][y]==1:
        tkinter.messagebox.showinfo("提示","不能选择自己已落子的区域")
    elif computer[x][y]==1:
        tkinter.messagebox.showinfo("提示","不能选择对方已落子的区域")
    else:
        player[x][y]=1
        player_history.append(np.array(player,copy=True))
        refresh()
        if check_win():
            return
        time.sleep(0.5)
        auto_move()
        computer_history.append(np.array(computer,copy=True))#cccccc
        print(computer_history)
        refresh()
        check_win()
            
    
def refresh():
    global computer
    global player
    global canvas_size
    unit=canvas_size//3
    for i in range(3):
        for j in range(3):
            if computer[i][j]==1:
                cv.create_oval(i*unit,j*unit,(i+1)*unit,(j+1)*unit,fill="black")
            if player[i][j]==1:
                cv.create_oval(i*unit,j*unit,(i+1)*unit,(j+1)*unit,fill="white")

def check_win_single(player):
    for i in range(3):
        if player[i][0]==1 and player[i][1]==1 and player[i][2]==1:
            return True
        if player[0][i]==1 and player[1][i]==1 and player[2][i]==1:
            return True
    if player[0][0]==1 and player[1][1]==1 and player[2][2]==1:
        return True
    if player[0][2]==1 and player[1][1]==1 and player[2][0]==1:
        return True
    return False
def check_even(a,b):
    for i in range(3):
        for j in range(3):
            if a[i][j]==0 and b[i][j]==0:
                return False
    return True
    
        
               
def check_win():
    global computer
    global player
    global computer_history
    global player_history
    reset=False
    if check_win_single(player):
        tkinter.messagebox.showinfo("提示","己方获胜")
        reset=True
    elif check_win_single(computer):
        tkinter.messagebox.showinfo("提示","己方落败")
        reset=True
    elif check_even(computer, player):
        tkinter.messagebox.showinfo("提示","平局")
        
        reset=True
    if reset:
        player=np.zeros((3,3),dtype=int)
        computer=np.zeros((3,3),dtype=int)
        cv.delete("all")
        for i in range(1,3):
            cv.create_line(0,canvas_size//3*i,canvas_size,canvas_size//3*i)
            cv.create_line(canvas_size//3*i,0,canvas_size//3*i,canvas_size)
        refresh()
        computer_history=[np.zeros((3,3),dtype=int)]
        player_history=[np.zeros((3,3),dtype=int)]
    return reset

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
      
        
def auto_move():
    global computer
    global player
    temp_x=0
    temp_y=0
    value=-1
    for i in range(3):
        for j in range(3):
            if computer[i][j]==0 and player[i][j]==0:
                temp_computer=np.array(computer,copy=True)
                temp_player=np.array(player,copy=True)
                temp_computer[i][j]=1
                temp_value=minsearch(temp_computer, temp_player)
                if temp_value>value:
                    temp_x=i
                    temp_y=j
                    value=temp_value
                if value==1:
                    break
    computer[temp_x][temp_y]=1
                    
                  
        
                 

load(1000,900)
