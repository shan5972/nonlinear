import threading,multiprocessing
import time
import math
import queue
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta,datetime


tasks =  pd.read_excel("data0310.xlsx")

# 初始化全局变量
V = 10  # 假设的总核心数
C = 1   # 核心效率
R = 32000  # 可用资源
Q_max = 100  # 任务队列最大容量
timestep = 0.3  # 资源检查间隔
timestep2 = 0.5  # 任务释放间隔
miu=len(tasks)/(V*C)
lamda=1/miu
Kt=0


start_time=time.time() #程序开始运行的时间
item_start_time=pd.Timestamp(datetime(2023,1,1,0,0,0)) #数据集开始时间



task_queue = []  # 任务队列
TimeLoss_queue = [] # 时间损失任务队列

lock = threading.Lock()

time_history = []
queue_size_history = []

def calculate_task_schedule(task,R,V,C,max_iteration):
    global TimeLoss,real_scheduled_time,real_deletion_time
    C_req=task['cpu_milli']
    num_iteration=0

    
    with lock:
        while num_iteration<=max_iteration:  #可以自主设置最大循环次数
#1----------------------------------------------------------------------------
    # 计算时间间隔（以秒为单位）
           time_delta_seconds = (1/lamda) * np.log(C_req/(C_req-R))

    # 将时间间隔转换为 Timedelta
           time_delta = pd.Timedelta(seconds=time_delta_seconds)

    # 将时间戳与时间间隔相加
           real_scheduled_time = task['creation_time'] + time_delta
  
    #print(real_scheduled_time)
           real_deletion_time=real_scheduled_time+pd.Timedelta(seconds=miu)
           TimeLoss=real_scheduled_time-task['creation_time']
#2------------------------------------------------------------------------------
           Kt=(len(tasks)-len(task_queue)-len(TimeLoss_queue))/(C*(TimeLoss.total_seconds()))
           Kt=math.ceil(Kt)  #向上取整
  
           limit = Kt - sum(item['C_req'] for item in task_queue) - sum(item['C_req'] for item in TimeLoss_queue) #分配额限制

           if limit>=0:
             C_req=min(limit,task['cpu_milli']) #所需资源和限制值之间的最小值
           else:
             C_req=task['cpu_milli']

           num_iteration+=1
           


        TimeLoss_queue.append(
        {'name':task['name'],
         'C_req':C_req,
         'creation_time':task['creation_time'],
         'scheduled_time':real_scheduled_time,
         'deletion_time':real_deletion_time,
         'status':'False',
        'TimeLoss':TimeLoss.total_seconds(),
         'left_cpu':R})  #更新之后的记录放到TimeLoss_queue里去

  #  return real_scheduled_time, real_deletion_time, TimeLoss

def resource_checker():
    global R,C_req
    while True:
        #print("扫描timeloss_queue了")
        time.sleep(timestep)
        with lock:
            if TimeLoss_queue:
                item=TimeLoss_queue[0]  # 取队列首任务但不弹出
                if R >= item['C_req']:
                    item=TimeLoss_queue.pop(0)
                    task_queue.append(item)
                    R -= C_req  # 更新可用资源
                


def task_releaser():
    global R, V
    while True:
        #print("扫描queue了")
        
        time.sleep(timestep)
        with lock:
            
            for item in task_queue:
                
               
                if item['status']=="False" and item['deletion_time']<item_start_time+pd.Timedelta(seconds=(time.time()-start_time)):
                  
                  item['status']="True"
                  R += item['C_req']  # 释放资源
                  V = V * Q_max / (len(task_queue) + 1)  # 调整V
            
                

def plot_queue_size():
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, queue_size_history, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Queue Size")
    plt.title("Task Queue Size Over Time")
    plt.grid()
    plt.show()

def main():
    global R
    threading.Thread(target=resource_checker, daemon=True).start()
    threading.Thread(target=task_releaser, daemon=True).start()
    
    
  
    for index,task in tasks.iterrows():
      if task['cpu_milli']<=R:
        task_queue.append({
            'name':task['name'],
            'C_req':task['cpu_milli'],
             'creation_time':task['creation_time'],
             'scheduled_time':task['scheduled_time'],
             'deletion_time':task['scheduled_time']+pd.Timedelta(seconds=1),
             'status':'False',
             'TimeLoss':0 ,
            'left_cpu':R-task['cpu_milli']             
        })
        R=R-task['cpu_milli']
      else:
        calculate_task_schedule(task,R,V,C,3)  #计算scheduled_time
        TimeLoss_queue.sort(key=lambda x: x['TimeLoss'], reverse=True) #把它排序
        time.sleep(1)
    
    #print(task_queue)
    output_df = pd.DataFrame(task_queue, columns=['name', 'C_req', 'creation_time', 'scheduled_time', 'deletion_time','status','TimeLoss','left_cpu'])
    output_df.to_excel("0310.xlsx", index=False,engine="openpyxl")

    output_df2 = pd.DataFrame(TimeLoss_queue, columns=['name', 'C_req', 'creation_time', 'scheduled_time', 'deletion_time','status','TimeLoss','left_cpu'])
    output_df2.to_excel("0310_2.xlsx", index=False,engine="openpyxl")


 





    time.sleep(20)  # 当主程序结束运行后，留10秒给其他线程用来处理任务
    #plot_queue_size()
    


if __name__ == "__main__":
    main()
