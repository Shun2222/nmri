import numpy as np
import os
import os.path as osp
import tqdm

#target = 'dtidx:40888, latidx:1266, lonidx:5031'
def get_ais3_2detail(
        path = r'E:\shunsukeE\data\ais\150901-1log\log',
        filename = r'japan_20150901000000-20150901015959-2.ais3_2detail',
        target = 'dtidx:40887'
    ):

    full_path = osp.join(path, filename)
    line_data = """""" 
    insec_data = """"""
    with open(full_path, "r") as sr:
        lines = sr.readlines()
        in_line = False
        for i, line in enumerate(lines):
            if target in line:
                in_line = True
                print(f'Found target header')
                print(f'Reading lines now...')
            if in_line and 'curN' in line:
                cur = line
                print(f'Found target end line')
                break

            if in_line:
                if 'dtidx' in line:
                    res = line.split(', ')
                    res = [r.split(':') for r in res]
                    ress = ""
                    for i in range(len(res)):
                        ress += res[i][0]+res[i][1]
                    data_info = ress 
                    continue
                elif 'elem' in line:
                    line_data += line
                elif 'insec' in line:
                    insec_data += line

    print(f"""
          target:{data_info}
          cur:{cur}
          """
        )
    return data_info, line_data, insec_data, cur

def get_all_ais3_2detail(
        path = r'E:\shunsukeE\data\ais\150901-1log\log',
        filename = r'japan_20150901000000-20150901015959(10).ais3_2_debug',
        max_data_num = 100
    ):

    full_path = osp.join(path, filename)
    data_infos = []
    curs = []
    line_datas = []
    insec_datas = []
    with open(full_path, "r", encoding='utf-8') as sr:
        lines = sr.readlines()
        in_line = False
        bad_mmsi_count = 0
        for i, line in enumerate(lines):
            if 'dtidx' in line:
                in_line = True
                line_data = """""" 
                insec_data = """"""
                cur_data = []
                cur_data_mmsi = []
                bad_mmsi = []
                added_black_mmsi = []

            if in_line:
                if 'dtidx' in line:
                    if "dtidx,latidx,lonidx,curN,curE,curLambda1,curLambda2,lambda1,lambda2,psi1,psi2" in line:
                        continue
                    res = line.split(', ')
                    res = [r.split(':') for r in res]
                    ress = ""
                    for i in range(len(res)):
                        print(res)
                        ress += res[i][0]+res[i][1]
                    data_infos.append(ress[:-1])
                    continue
                elif 'elem' in line:
                    line_data += line
                elif 'insec' in line:
                    insec_data += line
                elif 'curN' in line:
                    if 'Min:' in line:
                        cur_data.append(line[4:])
                    elif 'NoBrokenLSM:' in line:
                        cur_data.append(line[12:])
                    elif "HdgLSM: " in line:
                        cur_data.append(line[7:])
                    elif "LambdaLSM: " in line:
                        cur_data.append(line[10:])
                    elif "Mmsi" in line: 
                        cur_data_mmsi.append(line)
                elif 'Bad' in line:
                    if 'All' in line and len(bad_mmsi)==0:
                        bad_mmsi.append('')
                        bad_mmsi.append(line)
                    else:
                        bad_mmsi.append(line)
                    bad_mmsi_count += 1
                elif 'Added' in line:
                    added_black_mmsi.append(line[15:])
                print(f'\r {i}/{len(lines)}, num_data: {len(data_infos)}', end='')

            if in_line and 'Lambda' in line:
                assert len(cur_data)>=4, f"{cur_data}"
                curs.append([cur_data, cur_data_mmsi, bad_mmsi, added_black_mmsi])
                line_datas.append(line_data)
                insec_datas.append(insec_data)
                if max_data_num<=len(insec_datas):
                    print(f'data num is over max data num: {max_data_num}')
                    break

    print(f"""
          num target:{len(data_infos)}
          bad mmsi count:{bad_mmsi_count}
          """
        )
    return data_infos, line_datas, insec_datas, curs

            