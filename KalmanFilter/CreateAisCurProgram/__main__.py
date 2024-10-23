#from KalmanFilterProgram.Ais4ToCur import * 
from KalmanFilterProgram.Ais4v2ToCur import * 

if __name__ == "__main__":
    # Create ais4s pkl files
    if True:
        # pm.printline('checking AISLoader class')
        # 引数で年、月、出力先フォルダ、pklの読み込み先を設定
        year = 2015
        month = 9
        out_folder = path_ais 
        #out_folder = 'test-ais' 
        al = AISLoader(year, month, out_folder, pkl_path=out_folder)
        
        # cur1: 固有値方向の偏流１, cur2：固有値方向の偏流２, lambda1:固有値１, lambda2:固有値２, phi1:固有ベクトルの角度１, phi2:固有ベクトルの角度２, n:北方向の偏流, e:東方向の偏流
        ais_keys = ['cur1', 'cur2', 'cur1_2', 'cur2_2', 'lambda1', 'lambda2', 'phi1', 'phi2', 'n', 'e', 'n2', 'e2']
        al.set_keys(ais_keys) # 使用するkeyの設定
        
        n_day = nday_month(month) 
        Settei.init(0, 0, out_folder)
        ais4_to_cur.save_nanmap()
        #check_diff_nanmap()

        for i in range(1, n_day+1):
            print(f'day: {i}')
            #al.load_test(i)
            al.create_ais4s_file(i)
            al.load_ais_day(i)
