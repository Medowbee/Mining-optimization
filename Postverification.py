import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import numba
from tqdm import tqdm
#1. Одна точка сбора
#2. Две цели не могут быть соединены
#3. Кол-во целей с платформой не должно превышать максимального количества (PADS)
#4. LIMI

#-------------------------------------------CHECKS-------------------------------------------#

def check_all_blocks(blocks): #(0) все ли блоки встретились в тексте
    if not all(blocks):
        print('Не все блоки присутствуют в файле')
        print(blocks)

def check_all_targets(TARG_df, PTL_DOT_df): #(1) проверка на достижение всех целей
    for i in range(len(TARG_df)):
        not_targ = True
        for j in range(len(PTL_DOT_df)):
            if(TARG_df['x'].iloc[i] == PTL_DOT_df['x'].iloc[j] and TARG_df['y'].iloc[i] == PTL_DOT_df['y'].iloc[j]):
                #PTL_DOT_df['type'].iloc[j] = 'target' Кто знает к чему это приведет...
                not_targ = False
                break
        if (not_targ):
            print('Цель:\n', TARG_df.iloc[i], '\nне была достигнута.')

def check_connectivity(PTL_DOT_df, PTL_TUBE_df): #(2) проверка на связность графа
    for i in range(1, len(PTL_DOT_df) + 1):
        if ((i in PTL_TUBE_df['one end'].unique()) or (i in PTL_TUBE_df['two end'].unique())):
            pass
        else:
            print('Изолированныя цели или прощадка: \n', PTL_DOT_df.iloc[i - 1])

def check_targets_and_site_location(LIMI_df, PTL_DOT_df): #(3) проверка на наличие целей и площадок в допустимых территориях
    for i in range(len(PTL_DOT_df)):
        object_not_in_LIMI = True
        for k in range(len(LIMI_df)):
            if (LIMI_df['x'].iloc[k] - 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2])) <= PTL_DOT_df['x'].iloc[i] <= LIMI_df['x'].iloc[k] + 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2]))):
                if (LIMI_df['y'].iloc[k] - 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2])) <= PTL_DOT_df['y'].iloc[i] <= LIMI_df['y'].iloc[k] + 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2]))):
                    object_not_in_LIMI = False
        if (object_not_in_LIMI):
            print('Объект: \n', PTL_DOT_df.iloc[i], '\nвне допустимой зоны')

def check_targets_radius(PTL_UNDER_df): #(4) проверка достижимости целей из соответствующих площадок
    for i in range(len(PTL_UNDER_df)):
        l2_dist = np.sqrt((PTL_UNDER_df['x1'].iloc[i] - PTL_UNDER_df['x2'].iloc[i]) ** 2 + (PTL_UNDER_df['y1'].iloc[i] - PTL_UNDER_df['y2'].iloc[i]) ** 2)
        if (l2_dist > PTL_UNDER_df['max deviation'].iloc[i]):
            print('Некорректная длина трубы до цели: \n', PTL_UNDER_df.iloc[i])

def check_underground_cost(PTL_UNDER_df): #(5) проверка на соответствие стоимости для подземных труб
    for i in range(len(PTL_UNDER_df)):
        l2_dist = np.linalg.norm((PTL_UNDER_df['x1'].iloc[i] - PTL_UNDER_df['x2'].iloc[i], PTL_UNDER_df['y1'].iloc[i] - PTL_UNDER_df['y2'].iloc[i]))
        if (1.05*PTL_UNDER_df['cost'].iloc[i]  < l2_dist * PTL_UNDER_df['price per meter'].iloc[i] or l2_dist * PTL_UNDER_df['price per meter'].iloc[i] < 0.95*PTL_UNDER_df['cost'].iloc[i]):
            print('Реальная стоимость : ', l2_dist * PTL_UNDER_df['price per meter'].iloc[i])
            print('Некорректная стоимость трубы для цели: \n', PTL_UNDER_df.iloc[i])

def check_aboveground_cost(PTL_ABOVE_df, TUBE_df):
    for i in range(len(PTL_ABOVE_df)):
        l2_dist = np.sqrt((PTL_ABOVE_df['x1'].iloc[i] - PTL_ABOVE_df['x2'].iloc[i]) ** 2 + (PTL_ABOVE_df['y1'].iloc[i] - PTL_ABOVE_df['y2'].iloc[i]) ** 2)
        if(PTL_ABOVE_df['cost'].iloc[i] - 0.05 * PTL_ABOVE_df['cost'].iloc[i] >= l2_dist * TUBE_df['the cost of a running meter for tube'].iloc[0] or PTL_ABOVE_df['cost'].iloc[i] + 0.05 * PTL_ABOVE_df['cost'].iloc[i] <= l2_dist * TUBE_df['the cost of a running meter for tube'].iloc[0]):
            print('Реальная стоимость : ', l2_dist * TUBE_df['the cost of a running meter for tube'].iloc[0])
            print('Некорректная стоимость трубы соединяющей площадки: \n', PTL_ABOVE_df.iloc[i]) #(6) проверка на соответствие стоимости для надземных труб

def check_link_in_constrain(PTL_TUBE_df, data1):
    if max(max(PTL_TUBE_df['one end']), max(PTL_TUBE_df['two end'])) > data1:                    #Проверка на максимальные значения индексов связей
       print('Неккоректные связи в PTL между целями: индекс связи больше, чем количество целей')    #между точками не превышают количество точек
       print(f"Максимальный конец ребра: {max(max(PTL_TUBE_df['one end']), max(PTL_TUBE_df['two end']))} > Количество вершин: {data1[0]}") #(7) проверка корректности введенных данных в PTL

def check_area_connection(PADS_df, PTL_TUBE_df, PTL_DOT_df): #(8) проверка количества соединенных с площадкой целей
    above_dot_df = PTL_DOT_df[PTL_DOT_df['type'] == 'area']
    underground_df = PTL_TUBE_df[PTL_TUBE_df['type'] == 'underground']
    for k in range(len(above_dot_df)):
        left  = underground_df[underground_df['one end'] == above_dot_df.index[k]]
        right = underground_df[underground_df['two end'] == above_dot_df.index[k]]
        if (len(left) + len(right) > float(PADS_df['max number of sites'].iloc[0])):
            print('Превышено число целей, соединенных с площадкой: ', above_dot_df.iloc[k])

def check_duplicate(PTL_DOT_df, PTL_TUBE_df):
    if any(PTL_DOT_df.duplicated()):
        print(f'Обнаружены дупликаты в таблице PTL_DOT_df в количестве: {sum(PTL_DOT_df.duplicated())}')
    if any(PTL_TUBE_df.duplicated()):
        print(f'Обнаружены дупликаты в таблице PTL_TUBE_df в количестве: {sum(PTL_TUBE_df.duplicated())}')

def check_tube_dot_count(tube_count, dot_count, PTL_TUBE_df,PTL_DOT_df):
    if tube_count != len(PTL_TUBE_df):
        print('Декларируемое количество строк не совпадает с фактическим:')
        print(f'tube_count: {tube_count} != len(PTL_TUBE_df): {len(PTL_TUBE_df)}')
    if dot_count != len(PTL_DOT_df):
        print('Декларируемое количество строк не совпадает с фактическим:')
        print(f'tube_count: {dot_count} != len(PTL_TUBE_df): {len(PTL_DOT_df)}')

@numba.njit
def check_LIMI_numba(PTL_ABOVE_df, LIMI_df):
    thresh = 0.5 * abs(float(LIMI_df[1,1]) - float(LIMI_df[2,1]))
    for i in range(len(PTL_ABOVE_df)):
        for t in np.arange(0, 1, 1e-3):
            not_in_LIMI = True
            for k in range(len(LIMI_df)):
                if (max(abs(PTL_ABOVE_df[i,0] * t + PTL_ABOVE_df[i,2] * (1 - t) - LIMI_df[k,0]), abs(PTL_ABOVE_df[i,1] * t + PTL_ABOVE_df[i,3] * (1 - t) - LIMI_df[k,1])) <= thresh):#0.5 * abs(float(LIMI_df['y']) - float(LIMI_df['y'].iloc[2]))):
                    not_in_LIMI = False
            if (not_in_LIMI):
                print('Труба за пределами допустимой области!')

def check_LIMI(PTL_ABOVE_df, LIMI_df):
    for i in range(len(PTL_ABOVE_df)):
        for t in np.arange(0, 1, 1e-2):
            not_in_LIMI = True
            for k in range(len(LIMI_df)):
                if (max(abs(PTL_ABOVE_df['x1'].iloc[i] * t + PTL_ABOVE_df['x2'].iloc[i] * (1 - t) - LIMI_df['x'].iloc[k]), abs(PTL_ABOVE_df['y1'].iloc[i] * t + PTL_ABOVE_df['y2'].iloc[i] * (1 - t) - LIMI_df['y'].iloc[k])) <= 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2]))):
                    not_in_LIMI = False
            if (not_in_LIMI):
                print('Труба за пределами допустимой области!')
                break #(9) Проверка на корректность в блоке LIMI

def fill_PTL(TARG_df, PTL_DOT_df, PTL_TUBE_df): #вспомогательная фугкция для заполнения информации блока PTL
    #Убрал лишний цикл и привел one end и two end к int()
    PTL_TUBE_df['one end'] = list(map(int, PTL_TUBE_df['one end']))
    PTL_TUBE_df['two end'] = list(map(int, PTL_TUBE_df['two end']))
    PTL_DOT_df['type'] = 'area'
    for i in range(len(TARG_df)):
        for j in range(len(PTL_DOT_df)):
            if (TARG_df['x'].iloc[i] == PTL_DOT_df['x'].iloc[j] and TARG_df['y'].iloc[i] == PTL_DOT_df['y'].iloc[j]):
                PTL_DOT_df['type'].iloc[j] = 'target'
    for i in range(len(PTL_DOT_df)):
        if PTL_DOT_df['type'].iloc[i] =='target' and (len(PTL_TUBE_df[PTL_TUBE_df['one end'] == i+1]) + len(PTL_TUBE_df[PTL_TUBE_df['two end'] == i+1])) > 1:
            PTL_DOT_df['type'].iloc[i] = 'area'


    for j in range(len(PTL_TUBE_df)):  # определение типа трубы
        if ((PTL_DOT_df['type'].loc[PTL_TUBE_df['one end'].iloc[j]] == 'area' and PTL_DOT_df['type'].loc[PTL_TUBE_df['two end'].iloc[j]] == 'area')):
            PTL_TUBE_df['type'].iloc[j] = 'aboveground'
        else:
            PTL_TUBE_df['type'].iloc[j] = 'underground'

    # for i in range(len(TARG_df)):
    #     for j in range(len(PTL_DOT_df)):
    #         if(TARG_df['x'].iloc[i] == PTL_DOT_df['x'].iloc[j] and TARG_df['y'].iloc[i] == PTL_DOT_df['y'].iloc[j]):
    #             PTL_DOT_df['type'].iloc[j] = 'target'
    #
    # for i in range(len(PTL_DOT_df)):    #определение типа точки (площадка)
    #     if (pd.isnull(PTL_DOT_df['type'].iloc[i])):
    #         PTL_DOT_df['type'].iloc[i] = 'area'
    #
    # for j in range(len(PTL_TUBE_df)):   #определение типа трубы
    #     if ((PTL_DOT_df['type'].iloc[int(PTL_TUBE_df['one end'].iloc[j]) - 1] == 'area' and PTL_DOT_df['type'].iloc[int(PTL_TUBE_df['two end'].iloc[j]) - 1] == 'area')):
    #         PTL_TUBE_df['type'].iloc[j] = 'aboveground'
    #     else:
    #         PTL_TUBE_df['type'].iloc[j] = 'underground'

def create_PTL_TUBE(TARG_df, PTL_DOT_df, PTL_TUBE_df, LIMI_df):
    PTL_UNDER_df = pd.DataFrame(columns = ['x1', 'y1', 'x2', 'y2', 'max deviation', 'price per meter', 'cost']) #сводная таблица информации о подземных трубах
    PTL_ABOVE_df = pd.DataFrame(columns = ['x1', 'y1', 'x2', 'y2', 'cost']) #сводная таблица информации о подземных трубах

    underground_df = PTL_TUBE_df[PTL_TUBE_df['type'] == 'underground']
    for i in range(len(underground_df)):
        if (PTL_DOT_df['type'].iloc[int(underground_df['one end'].iloc[i]) - 1] == 'target'):
            targ_frame = TARG_df[TARG_df['x'] == PTL_DOT_df['x'].iloc[int(underground_df['one end'].iloc[i]) - 1]][TARG_df['y'] == PTL_DOT_df['y'].iloc[int(underground_df['one end'].iloc[i]) - 1]]
        else:
            targ_frame = TARG_df[TARG_df['x'] == PTL_DOT_df['x'].iloc[int(underground_df['two end'].iloc[i]) - 1]][TARG_df['y'] == PTL_DOT_df['y'].iloc[int(underground_df['two end'].iloc[i]) - 1]]

        PTL_UNDER_df = PTL_UNDER_df.append({'x1':PTL_DOT_df['x'].iloc[int(underground_df['one end'].iloc[i]) - 1], 'y1':PTL_DOT_df['y'].iloc[int(underground_df['one end'].iloc[i]) - 1],
        'x2':PTL_DOT_df['x'].iloc[int(underground_df['two end'].iloc[i]) - 1], 'y2':PTL_DOT_df['y'].iloc[int(underground_df['two end'].iloc[i]) - 1],
        'max deviation':targ_frame['max deviation'].iloc[0] , 'price per meter':targ_frame['price per meter'].iloc[0] ,'cost': underground_df['cost'].iloc[i]}, ignore_index = True)

    aboveground_df = PTL_TUBE_df[PTL_TUBE_df['type'] == 'aboveground']
    for i in range(len(aboveground_df)):
        PTL_ABOVE_df = PTL_ABOVE_df.append({'x1':PTL_DOT_df['x'].iloc[int(aboveground_df['one end'].iloc[i]) - 1], 'y1':PTL_DOT_df['y'].iloc[int(aboveground_df['one end'].iloc[i]) - 1],
        'x2' : PTL_DOT_df['x'].iloc[int(aboveground_df['two end'].iloc[i]) - 1], 'y2':PTL_DOT_df['y'].iloc[int(aboveground_df['two end'].iloc[i]) - 1], 'cost':aboveground_df['cost'].iloc[i]}, ignore_index = True)

    return PTL_UNDER_df, PTL_ABOVE_df

def plot_map(LIMI_df, PTL_DOT_df, PTL_TUBE_df, max_cost):
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.title('Карта месторождений')
    ax.grid()
    for k in range(len(LIMI_df)): #построение ограничений LIMI
        if LIMI_df['price limit'].iloc[k] <= max_cost:
            ax.add_patch(Rectangle((float(LIMI_df['x'].iloc[k]) - 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2])), float(LIMI_df['y'].iloc[k]) - 0.5 * abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2]))), abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2])), abs(float(LIMI_df['y'].iloc[1]) - float(LIMI_df['y'].iloc[2]))))

    for j in range(len(PTL_TUBE_df)):   #построение труб
        if (PTL_TUBE_df['type'].iloc[j] == 'underground'):
            plt.plot([float(PTL_DOT_df['x'].iloc[int(PTL_TUBE_df['one end'].iloc[j]) - 1]), float(PTL_DOT_df['x'].iloc[int(PTL_TUBE_df['two end'].iloc[j]) - 1])], [float(PTL_DOT_df['y'].iloc[int(PTL_TUBE_df['one end'].iloc[j]) - 1]), float(PTL_DOT_df['y'].iloc[int(PTL_TUBE_df['two end'].iloc[j]) - 1])], 'c')
        else:
            plt.plot([float(PTL_DOT_df['x'].iloc[int(PTL_TUBE_df['one end'].iloc[j]) - 1]), float(PTL_DOT_df['x'].iloc[int(PTL_TUBE_df['two end'].iloc[j]) - 1])], [float(PTL_DOT_df['y'].iloc[int(PTL_TUBE_df['one end'].iloc[j]) - 1]), float(PTL_DOT_df['y'].iloc[int(PTL_TUBE_df['two end'].iloc[j]) - 1])], 'r')

    for i in range(len(PTL_DOT_df)):    #построение целей и площадок
        if (PTL_DOT_df['type'].iloc[i] == 'area'):
            ax.scatter(float(PTL_DOT_df['x'].iloc[i]),float(PTL_DOT_df['y'].iloc[i]), marker='^', c = 'deeppink', s = 100)
        else:
            ax.scatter(float(PTL_DOT_df['x'].iloc[i]),float(PTL_DOT_df['y'].iloc[i]), marker='o', c = 'c', s = 100) #строим карту месторождений

def cost_of_graph(PTL_TUBE_df, PTL_DOT_df, PADS_df):
    return sum(PTL_TUBE_df['cost']) + PADS_df['cost of accommodation'].iloc[0]*len(PTL_DOT_df[PTL_DOT_df['type'] == 'area'])
#-------------------------------------------PARSER-------------------------------------------#
def Postverification(input_path):
    TARG_df     = pd.DataFrame(columns = ['x','y','max deviation','price per meter'])       #Датафрейм для геологических целей
    #LIMI_df     = pd.DataFrame(columns = ['x','y','price limit'])                           #Датафрейм для ограницений на прокладку труб (по поверхности)
    PADS_df     = pd.DataFrame(columns = ['max number of sites', 'cost of accommodation'])  #Датафрейм для данных о площадках
    TUBE_df     = pd.DataFrame(columns = ['the cost of a running meter for tube'])          #Датафрейм для информации о стоимости прокладки труб
    PTL_DOT_df  = pd.DataFrame(columns = ['x', 'y', 'type'])                                #Точки геологических целей и площадок в полученном решении
    PTL_TUBE_df = pd.DataFrame(columns = ['one end', 'two end', 'cost', 'type'])            #Трубы соединяющие геологические цели и площадки

    main_df = pd.read_csv(input_path, index_col = None, header = None)
    blocks = [False, False, False, False, False]
    LIMI_flag = False
    LIMI_not_ok = True
    flag_PTL_DOT_count = False
    flag_PTL_TUBE_count = False
    count_dot = 0

    for i in tqdm(range(len(main_df))):

        if(main_df[0].iloc[i] == 'TARG'):
            blocks[0] = True
            flag = 'TARG'
            continue
        elif(main_df[0].iloc[i] == 'PADS'):
            blocks[1] = True
            flag = 'PADS'
            continue
        elif(main_df[0].iloc[i] == 'TUBE'):
            blocks[2] = True
            flag = 'TUBE'
            continue
        elif(main_df[0].iloc[i] == 'LIMI'):
            blocks[3] = True
            flag = 'LIMI'
            continue
        elif (main_df[0].iloc[i] == 'PTL'):
            blocks[4] = True
            flag = 'PTL'
            continue

        if(flag == 'TARG'):
            data = main_df[0].iloc[i].split()
            if (data[0] != '/'):
                TARG_df = TARG_df.append({'x':float(data[0]), 'y':float(data[1]), 'max deviation':float(data[2]), 'price per meter': float(data[3])}, ignore_index = True)
            else:
                flag = ''

        elif(flag == 'PADS'):
            data = main_df[0].iloc[i].split()
            if (data[0] != '/'):
                PADS_df = PADS_df.append({'max number of sites':int(data[0]), 'cost of accommodation':float(data[1])}, ignore_index = True)
            else:
                flag = ''

        elif(flag == 'TUBE'):
            data = main_df[0].iloc[i].split()
            if (data[0] != '/'):
                TUBE_df = TUBE_df.append({'the cost of a running meter for tube':float(data[0])}, ignore_index = True)
            else:
                flag = ''

        elif(flag == 'LIMI'):
            if (main_df[0].iloc[i].split()[0] != '/' and LIMI_flag and LIMI_not_ok):
                columns = ['x', 'y', 'price limit']
                LIMI_df = pd.DataFrame(main_df[0].iloc[i : xcount*ycount + i].str.split(expand=True))
                LIMI_df.columns = ['x', 'y', 'price limit']
                LIMI_df['x'] = LIMI_df['x'].astype(float)
                LIMI_df['y'] = LIMI_df['y'].astype(float)
                LIMI_df['price limit'] = LIMI_df['price limit'].astype(float)
                LIMI_df = LIMI_df.reset_index(drop = True)
                LIMI_not_ok = False
            elif(main_df[0].iloc[i].split()[0] != '/' and LIMI_not_ok):
                xcount   = int(main_df[0].iloc[i].split()[0])                          #количество ячеек по x
                ycount   = int(main_df[0].iloc[i].split()[1])                         #количество ячеек по y
                max_cost = int(main_df[0].iloc[i].split()[2])
                LIMI_flag = True
            else:
                flag = ''
        elif(flag == 'PTL'):
            data = main_df[0].iloc[i].split()
            if (data[0] == '/'):
                flag = ''
                continue
            
            if(not flag_PTL_DOT_count):
                data1 = int(data[0])
                flag_PTL_DOT_count = True
                continue
            elif(flag_PTL_DOT_count and count_dot < data1):
                PTL_DOT_df = PTL_DOT_df.append({'x':float(data[0]), 'y':float(data[1]), 'type':None}, ignore_index = True)
                count_dot += 1
            elif not flag_PTL_TUBE_count:
                
                data2 = int(data[0])
                flag_PTL_TUBE_count = True
                continue
            if(flag_PTL_TUBE_count):
                PTL_TUBE_df = PTL_TUBE_df.append({'one end':int(data[0]), 'two end':int(data[1]), 'cost':float(data[2]), 'type':None}, ignore_index = True)
                   
    # with open(input_path, 'r') as file: #Открываем файл
    #
    #     lines =list(filter(lambda x: x!='',list(map(lambda x: x.strip().split('--')[0],file.readlines())))) #Считываем все строки из файла, убираем комментарии, убираем пустые строки
    #     blocks = [False, False, False, False, False]
    #
    #     main_df = pd.re
    #
    #     for i in range(len(lines)): #итерируемся по строкам
    #          if   lines[i] == 'TARG': #читаем блок TARG
    #             i += 1
    #             blocks[0] = True
    #             while lines[i] != '/':
    #                 data = list(map(float,lines[i].split()))
    #                 if (len(data) != 4):
    #                     print('В блоке TARG не верное количество данных в строке: ', data)
    #                     i+=1
    #                 else:
    #                     TARG_df = TARG_df.append({'x':data[0], 'y':data[1], 'max deviation':data[2], 'price per meter': data[3]}, ignore_index = True)
    #                     i+=1
    #
    #          elif lines[i] == 'PADS': #читаем блок PADS
    #             i += 1
    #             blocks[1] = True
    #             while lines[i] != '/':
    #                 data = list(map(float,lines[i].split()))
    #                 if (len(data) != 2):
    #                     print('В блоке PADS не верное количество данных в строке: ', data)
    #                     i+=1
    #                 else:
    #                     PADS_df = PADS_df.append({'max number of sites':data[0], 'cost of accommodation':data[1]}, ignore_index = True)
    #                     i+=1
    #
    #          elif lines[i] == 'TUBE': #читаем блок TUBE
    #             i += 1
    #             blocks[2] = True
    #             while lines[i] != '/':
    #                 data = list(map(float,lines[i].split()))
    #                 if (len(data) != 1):
    #                     print('В блоке TUBE не верное количество данных в строке: ', data)
    #                     i+=1
    #                 else:
    #                     TUBE_df = TUBE_df.append({'the cost of a running meter for tube':data[0]}, ignore_index = True)
    #                     i+=1
    #
    #          elif lines[i] == 'LIMI': #читаем блок LIMI
    #             i += 1
    #             blocks[3] = True
    #             data = list(map(float,lines[i].split()))    #информация о последующих данных в блоке LIMI
    #             if (len(data)!=3):
    #                 print('Не хватает данных в заглавной строке блока LIMI')
    #             else:
    #                 xcount   = data[0]                          #количество ячеек по x
    #                 ycount   = data[1]                          #количество ячеек по y
    #                 max_cost = data[2]                          #максимальная стоимость
    #             i += 1
    #             while lines[i] != '/':
    #                 data = list(map(float,lines[i].split()))
    #                 if (len(data) != 3):
    #                     print('В блоке LIMI не верное количество данных в строке: ', data)
    #                     i+=1
    #                 else:
    #                     LIMI_df = LIMI_df.append({'x':data[0], 'y':data[1], 'price limit':data[2]}, ignore_index = True)
    #                     i+=1
    #             if (len(LIMI_df['x'].unique()) != xcount):     print('Не соответсвие количества ячеек по x с заданным в блоке LIMI')
    #             if (len(LIMI_df['y'].unique()) != ycount):     print('Не соответсвие количества ячеек по y с заданным в блоке LIMI')
    #
    #          elif (lines[i] == 'PLT' or lines[i] == 'PTL') : #читаем блок PTL
    #             i += 1
    #             blocks[4] = True
    #             data1      = list(map(float,lines[i].split()))    #информация о последующих данных в блоке PLT
    #             dot_flag  = False                                #чтобы отследить считывались ли цели и прощадки
    #             tube_flag = False                                #чтобы отследить считывались ли трубы
    #             while lines[i] != '/':
    #                 data = list(map(float,lines[i].split()))
    #                 if (len(data) == 1):
    #                     if (dot_flag):
    #                         tube_count = data[0]
    #                         tube_flag = True
    #                     else:
    #                         dot_count = data[0]
    #                         dot_flag = True
    #                 elif (len(data) == 2 and dot_flag and not tube_flag):
    #                     PTL_DOT_df = PTL_DOT_df.append({'x':data[0], 'y':data[1], 'type':None}, ignore_index = True)
    #                 elif (len(data) == 3 and tube_flag):
    #                     PTL_TUBE_df = PTL_TUBE_df.append({'one end':int(data[0]), 'two end':int(data[1]), 'cost':data[2], 'type':None}, ignore_index = True)
    #                 else:
    #                     print('В блоке PLT не верное количество данных в строке: ', data)
    #                 i+=1
    PTL_DOT_df.index  += 1
    PTL_TUBE_df.index += 1       
    check_all_blocks(blocks)
 
    print(f"Минимальное количество площадок: {int(np.ceil(len(TARG_df)/PADS_df['max number of sites'].iloc[0]))}, Максимальное количество площадлок: {len(TARG_df)}")
    if blocks[4]:
        #check_link_in_constrain(PTL_TUBE_df, data1)

        fill_PTL(TARG_df, PTL_DOT_df, PTL_TUBE_df) #заполняем недостающие данные о блоке PTL

        PTL_UNDER_df, PTL_ABOVE_df = create_PTL_TUBE(TARG_df, PTL_DOT_df, PTL_TUBE_df, LIMI_df)

        #Проверки после считывания
        check_all_blocks(blocks) #(0) все ли блоки встретились в тексте

        check_all_targets(TARG_df, PTL_DOT_df) #(1) проверка на достижение всех целей

        check_connectivity(PTL_DOT_df, PTL_TUBE_df) #(2) проверка на связность графа

        check_targets_and_site_location(LIMI_df[LIMI_df['price limit'] <=max_cost] , PTL_DOT_df) #(3) проверка на наличие целей и площадок в допустимых территориях

        check_tube_dot_count(data2, data1, PTL_TUBE_df, PTL_DOT_df)#(4) Проверка на соответствие количества строк с целями и трубами

        check_duplicate(PTL_DOT_df, PTL_TUBE_df)  #(4) проверка на наличие дупликатов целей и труб

        check_targets_radius(PTL_UNDER_df) #(5) проверка достижимости целей из соответствующих площадок

        check_underground_cost(PTL_UNDER_df) #(6) проверка на соответствие стоимости для подземных труб

        check_aboveground_cost(PTL_ABOVE_df, TUBE_df) #(7) проверка на соответствие стоимости для наземных труб


        check_area_connection(PADS_df, PTL_TUBE_df, PTL_DOT_df) #(8) проверка количества соединенных с площадкой целей

       # start_time = time.time()
        #check_LIMI(PTL_ABOVE_df, LIMI_df)
        #print(f'Время на check_limi: {time.time()- start_time}')


        #check_LIMI_numba(PTL_ABOVE_df.to_numpy(), LIMI_df[LIMI_df['price limit'] <=max_cost].to_numpy())
        print(f'Стоимость: {cost_of_graph(PTL_TUBE_df, PTL_DOT_df, PADS_df)}')

        #Формирование и вывод картинки:
        #plot_map(LIMI_df, PTL_DOT_df, PTL_TUBE_df, max_cost)


    return TARG_df, LIMI_df,max_cost,PADS_df, TUBE_df, PTL_DOT_df, PTL_TUBE_df#, PTL_UNDER_df, PTL_ABOVE_df
