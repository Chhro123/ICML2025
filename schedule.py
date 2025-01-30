import plotresult
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import analysistodb
import setcasefromdb
import setcase
import re
import synsocket
import time


def main():
    TDIV = 6  # 6 slots per hour
    dt = pd.Timedelta(minutes=10)
    
    # Load EV CS Task data from local dataset
    Tasks = setcase.gen_task()
    EVs = setcase.gen_EV(Tasks)
    CSLBs = setcase.gen_CSLB()

    # Call the scheduling function with the local data
    EVs = schedule([], EVs, CSLBs, None)

    remove_vertialCS(EVs)
    analysistodb.sendinfotodb(EVs, CSLBs)


def remove_vertialCS(EVs):
    for ev in EVs:
        removeset = []
        for n,row in enumerate(ev.task):
            if row[1]>9000:
                ev.task[n-1][4]=ev.task[n-1][4]*2
                ev.task[n-1][5]=str(pd.Timedelta(ev.task[n][5])+pd.Timedelta(ev.task[n-1][5]))
                ev.task[n-1][6]=ev.task[n-1][6]+ev.task[n][6]
                removeset.append(n)
        for n in removeset[::-1]:
            del(ev.task[n])

def get_time():
    return time.time()

def schedule(scheduled_EV, updated_EV, CSLBs, now):
    Runingbegin = get_time()
    TDIV = 6  # 4 slots per hour
    dt = pd.Timedelta(minutes=10)
    N_site = len(CSLBs)
    CSidlist = [cs.id for cs in CSLBs if cs.type == 'CS']
    CSLBidlist = [v.id for v in CSLBs]

    for ev in scheduled_EV + updated_EV:
        ev.reset_extendflag(N_site)

    if now is not None:
        print("####now####", now)
        for ev in scheduled_EV + updated_EV:
            print("EV: ", ev.id)
            ev.task = updatenowtask(ev.task, now)
            ev.task_narrow = retrievetask(ev.task_narrow, ev.task)

    updated_EV = sorted(updated_EV, key=lambda x: x.wight, reverse=True)

    Totalloop = 0

    unbalance_flag = True
    while unbalance_flag:
        Totalloop += 1
        for priority, ev in enumerate(updated_EV):
            print("#########optimize EV ", ev.id, " priority ", priority, "#########")
            t0 = pd.to_datetime(str(ev.task_narrow[0][3][0:10] + " 00:00:00"))
            slack = pd.Timedelta("0:30:00")
            relaxtask(ev, slack, ev.wight)
            unscheu_flag = True
            unnarrowed_flag = True
            while unscheu_flag:
                cargotimeset = []
                for i in range(len(ev.task_narrow) - 1):
                    if i != 0 and ev.task_narrow[i][0] == 'LB':
                        for cargoslot in range(
                                int((pd.to_datetime(ev.task_narrow[i][3]) - pd.to_datetime(ev.task_narrow[i][2])) / dt)):
                            cargotimeset.append((ev.task[i][1], int((pd.to_datetime(ev.task[i][2]) - t0) / dt + cargoslot)))

                m = gp.Model('logistic')
                m.Params.OutputFlag = 0
                x = m.addVars(N_site, (24 * 2) * TDIV, vtype=GRB.BINARY, name='x')

                dwellset = []
                for count, j in enumerate(ev.task):
                    if count != 0 and count != len(ev.task) - 1:
                        for t in range(int((pd.to_datetime(j[2]) - t0) / dt), int((pd.to_datetime(j[3]) - t0) / dt)):
                            dwellset.append((CSLBidlist.index(j[1]), t))
                m.addConstrs((x[j, t] == 0 for j in range(N_site)
                              for t in range(24 * 2 * TDIV)
                              if ((j, t) not in dwellset) or ((j, t) in ev.conflictslots)))

                N_segs = (len(ev.task) - 2) * 2 + 1
                soc_segs = []
                for i in range(N_segs):
                    if i % 2 == 0:
                        soc_segs.append(-ev.task[int(i / 2)][4] * ev.consum / (100 * ev.cap))
                    else:
                        j = int((i + 1) / 2)
                        temp = sum(
                            [(20 / 6) / ev.cap * x[CSLBidlist.index(ev.task[j][1]), t] for t in
                             range(int((pd.to_datetime(ev.task[j][2]) - t0) / dt),
                                   int((pd.to_datetime(ev.task[j][3]) - t0) / dt))])
                        soc_segs.append(temp)
                soc_sum = []
                for NN in range(N_segs):
                    soc_sum.append(sum([soc_segs[i] for i in range(NN + 1)]))
                for i in range(N_segs):
                    m.addConstr((x[0, 0] + ev.soc + soc_sum[i]) >= ev.soc_min)
                    m.addConstr((x[0, 0] + ev.soc + soc_sum[i]) <= ev.soc_max)

                obj0 = sum([(0.999 + 0.0002 * t / (24 * TDIV)) * x[j, t] for j in range(N_site) for t in
                            range(24 * 2 * TDIV) if (j, t) not in cargotimeset])
                obj0 += sum(
                    [0.1 * (x[0, t] + x[1, t] + x[2, t] + x[3, t] + x[4, t]) for t in range(24 * 2 * TDIV) if
                     (0, t) not in cargotimeset])
                m.setObjective(obj0, GRB.MAXIMIZE)
                m.optimize()

                task_update = []
                for j in range(len(ev.task_narrow) - 1):
                    tempdate = ev.task_narrow[j][2]
                    tempdelta = pd.Timedelta((ev.task_narrow[j][3] - ev.task_narrow[j][2]).seconds * 1000000000)
                    temp = []
                    for t in range(24 * 2 * TDIV):
                        if x[CSLBidlist.index(ev.task_narrow[j][1]), t].x > 0.5:
                            if t % TDIV == 0:
                                temp.append(
                                    [tempdate, tempdate + pd.Timedelta(t * 60 * 10 ** 9),
                                     tempdate + pd.Timedelta((t + 1) * 60 * 10 ** 9)])
                    task_update.append(temp)

                ev.task = removeitinerary(ev.task)
                task_update = removeitinerary(task_update)
                ev.task = ev.task + task_update
                ev.task = sorted(ev.task, key=lambda x: x[2])

                unscheu_flag = False
                unscheu_flag_temp = False
                for count, task in enumerate(ev.task):
                    if task[0] == 'LB' and count == 0:
                        print("start of trip")
                    if task[0] == 'LB' and count != 0 and count != len(ev.task) - 1:
                        print("Unshedule: charge at ", task[1], "Duration:", task[3] - task[2])
                        unscheu_flag_temp = True
                    if task[0] == 'LB' and count == len(ev.task) - 1:
                        print("end of trip")
                        if unscheu_flag_temp:
                            unscheu_flag = True
                        print("Unshedule: charge at ", task[1], "Duration:", task[3] - task[2])
        unbalance_flag = False
    print("Total loop:", Totalloop)

def simple_schedule(EVs,CSLBs):
    CSLBidlist = [v.id for v in CSLBs]
    TDIV = 6  # 6 slots per hour
    dt = pd.Timedelta(minutes=10)
    N_site = len(CSLBs)
    for ev in EVs:
        t0 = pd.to_datetime(str(ev.task_narrow[0][3][0:10] + " 00:00:00"))
        slots_occu = [[] for i in range(N_site)]
        ev.task=ev.task_narrow
        soc=ev.soc
        for row in ev.task:
            if row[5]==None or row[2]==None:
                pass
            else:
                soc-=row[4]* ev.consum / (100 * ev.cap)
                if row[0]=='LB'and soc<0.5:
                    for t in range(int((pd.to_datetime(row[2])-t0).seconds/600),int((pd.to_datetime(row[3])-t0).seconds/600)):
                        if soc >0.8:
                            break
                        slots_occu[CSLBidlist.index(row[1])].append(t)
                        soc+=(20/6)/ev.cap
        ev.chargslots=list(slots_occu)

    #     x=np.zeros((N_site,24*6))
    #     for j in range(N_site):
    #         for t in slots_occu[j]:
    #             x[j,t]=1
    #     price = [s.price for s in CSLBs]
    #     obj1 = sum([(1/TDIV) * 20 * (0.999 + 0.0002 * t / (24 * TDIV)) * x[j, t] * price[j][t] for j in range(N_site) for t in range(24 * TDIV)])
    #     rates = [s.rate for s in CSLBs]
    #     obj2 = sum([(1/TDIV) * 20 * (0.99 + 0.0002 * t / (24 * TDIV)) * x[j, t] * rates[j][t][2] for j in range(N_site) for t in range(24  * TDIV)])
    #     ev.objresult = [0, obj1, obj2]
    #     print("obj0,1,2: ", ev.objresult)
    # plotresult.plot_result(EVs)
    plotresult.plot_result_fordb(EVs, CSLBidlist)
    return EVs


def relaxtask(ev,slack_given,weight):
    '''增加每个站点的停留时间，包括CS和LB，但时间优先情况下不增加CS时间'''
    t0=pd.to_datetime(str(ev.task_narrow[0][3][0:10]+" 00:00:00"))
    tasknew = []
    slack_sum=pd.Timedelta(minutes=0)
    for i, row in enumerate(ev.task_narrow):
        '''这里重复两遍是为了检测pd.to_datetime(ev.task[-1][2])-t0<pd.Timedelta("1 days 00:00:00")
        当时间优先情况下 物流点都relax到1天了还需要relax时，选择外部充电点进行relax
        非时间优先情况下，外部充点电在一开始就会relax'''
        if ev.task==None: #first time come here
            if  weight[0] > 0.5:
                if row[0] == "LB":
                    slack = slack_given
                else:
                    slack = pd.Timedelta(minutes=0)
            else:
                slack = slack_given
        else:# latter come here
            if pd.to_datetime(ev.task[-1][2])-t0<pd.Timedelta("1 days 00:00:00") and weight[0]>0.5: #24 is manually selected here
                if row[0]=="LB":
                    slack=slack_given
                else:
                    slack=pd.Timedelta(minutes=0)
            else:
                slack=slack_given


        if i == 0:
            tasknew.append(list(row))
        elif i == len(ev.task_narrow) - 1:
            tasknew.append([row[0], row[1], str(pd.to_datetime(row[2]) + slack_sum), None, None, None,None])
        else:
            tasknew.append([row[0], row[1], str(pd.to_datetime(row[2]) + slack_sum), str(pd.to_datetime(row[3]) + slack_sum+slack), row[4], row[5],row[6]])
            slack_sum += slack

    if pd.to_datetime(row[2])-t0 + slack_sum <pd.Timedelta("1 days 12:00:00"):
        ev.task = list(tasknew)
    else:
        raise Exception(print("taskassign overtime"))

    print("Relaxed",ev.task)

def narrowtask(ev,slots_occu,TDIV,CSLBidlist):
    '''缩短每个站点的停留时间，除去除充电和装卸货时间外的额外relax时间'''
    TDIV = 6  # 6 slots per hour
    dt = pd.Timedelta(minutes=10)

    for i, site in enumerate(ev.task):
        if i != 0 and i != len(ev.task) - 1:
            loadtime = pd.to_datetime(ev.task_narrow[i][3]) - pd.to_datetime(ev.task_narrow[i][2])
            chargtime = len(slots_occu[CSLBidlist.index(site[1]) ])*dt
            totaltime = max(loadtime, chargtime)
            reducetime = pd.to_datetime(site[3]) - pd.to_datetime(site[2]) - totaltime
            for j in range(i, len(ev.task)):
                if j == i:
                    ev.task[j][3] = str(pd.to_datetime(ev.task[j][3])-reducetime)
                elif j == len(ev.task) - 1:
                    ev.task[j][2] = str(pd.to_datetime(ev.task[j][2])-reducetime)
                else:
                    ev.task[j][2] = str(pd.to_datetime(ev.task[j][2])-reducetime)
                    ev.task[j][3] = str(pd.to_datetime(ev.task[j][3])-reducetime)
    print("Narrow",ev.task)

def retrievetask(task_narrow, tasknow):
    '''实时更新时，用于计算now时刻的最新narrow task'''
    N1=len(task_narrow)
    N2=len(tasknow)

    task=[]
    for i,v in enumerate(tasknow):
        if i==0:
            task.append(v)
        elif i==N2-1:
            summ = pd.Timedelta(task[i - 1][5]) + pd.to_datetime(task[i - 1][3])
            task.append([v[0], v[1], str(summ), None, None, None])
        else:
            summ=pd.Timedelta(task[i-1][5])+ pd.to_datetime(task[i-1][3])
            task.append([v[0],v[1],str(summ),str(summ+(pd.to_datetime(task_narrow[N1-N2+i][3])-pd.to_datetime(task_narrow[N1-N2+i][2]))),v[4],v[5]])
    for v in task:
        print(v)
    return task

def updatenowtask(task,now):
    '''实时更新时，用于计算now时刻的最新task'''
    ##确保now<task[-2][3],#确保now >task[0][3]
    #判断当前时间在规划的路上还是在站里
    N = len(task)
    if N < 3:
        raise Exception(print("Task number wrong   now=", str(now)))
    tasknow = []
    for i in range(N):
        if i == N - 1:  # 确保now<task[-2][3]
            tasknow.append(task[i])
        else:  # 确保now >task[0][3]
            if i != 0 and pd.to_datetime(task[i][2]) <= now < pd.to_datetime(task[i][3]):  # on site
                tasknow.append(['RD', -1, None, str(now), 0, 0])
                tasknow.append([task[i][0], task[i][1], str(now), task[i][3], task[i][4], task[i][5]])
            elif pd.to_datetime(task[i][3]) <= now< pd.to_datetime(task[i + 1][2]):  # on road
                tasknow.append(['RD', -1, None, str(now) , 10, str(pd.to_datetime(task[i + 1][2]) - now)])
            elif now > pd.to_datetime(task[i][3]):
                pass
            elif i != 0 and now< pd.to_datetime(task[i][2]):
                tasknow.append(task[i])
    for v in tasknow:
        print(v)
    return tasknow

def process(task,x1,x2,x3,x4,p1,p2,newid,N):
    for n, row in enumerate(task):
        if n == N:
            task[n - 1][4] = x1  # 到CS距离
            dt = pd.Timedelta(x2) - pd.Timedelta(task[n - 1][5])  # 因为到CS额外增加的时间
            task[n - 1][5] = str(x2)  # 到CS时间
            task[n - 1][6]=p1
            task[n][2] = str(pd.to_datetime(task[n][2]) + dt)
            task[n][3] = str(pd.to_datetime(task[n][3]) + dt)
            task[n][4] = x3  # CS到LB距离
            task[n][5] = x4  # CS到LB时间
            task[n][6] = p2
            task[n][1]=newid
        if n == len(task) - 1:
            task[n][2] = str(pd.to_datetime(task[n - 1][3]) + pd.Timedelta(task[n - 1][5]))
        elif n > N:
            deltatemp = pd.to_datetime(task[n][3]) - pd.to_datetime(task[n][2])
            task[n][2] = str(pd.to_datetime(task[n - 1][3]) + pd.Timedelta(task[n - 1][5]))
            task[n][3] = str(deltatemp + pd.to_datetime(task[n][2]))
    return task

def extendtask(task,task_narrow,siteid,task_raw,viceid=0):
    for n, row in enumerate(task):
        if row[1] == siteid:
            N = n
    print("N= ", N)

    x1 = task_raw[N - 1][7][viceid][1]  # distance
    x2 = task_raw[N - 1][7][viceid][2]  # timedelta
    p1 = task_raw[N - 1][7][viceid][3]
    x3 = task_raw[N][7][viceid][1]  # distance
    x4 = task_raw[N][7][viceid][2]  # timedelta
    p2 = task_raw[N][7][viceid][3]
    newid = task_raw[N][7][viceid][0]

    task = process(task, x1, x2, x3, x4, p1, p2, newid, N)
    task_narrow = process(task_narrow, x1, x2, x3, x4, p1, p2, newid, N)
    print("extend here#####")
    for row in task:
        print(row)
    return task, task_narrow, newid

def detendtask (task,task_narrow,siteid,task_raw):
    for n, row in enumerate(task):
        if row[1] == siteid:
            N = n
    print("N= ", N)

    x1 = task_raw[N - 1][4]  # distance
    x2 = task_raw[N - 1][5]  # timedelta
    p1 = task_raw[N - 1][6]
    x3 = task_raw[N][4]
    x4 = task_raw[N][5]
    p2 = task_raw[N][6]
    newid = task_raw[N][1]

    task = process(task, x1, x2, x3, x4, p1, p2, newid, N)
    task_narrow = process(task_narrow, x1, x2, x3, x4, p1, p2, newid, N)
    print("extend here#####")
    for row in task:
        print(row)
    return task, task_narrow, newid



if __name__ == '__main__':
    main()