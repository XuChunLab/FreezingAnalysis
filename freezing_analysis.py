from mylab.Cvideo import Video 
from mylab.exps.cfc import video_to_csv as v2c
from mylab.Ccsv import Csv
import glob,os,sys,platform
import subprocess
import csv
import time
#import concurrent.futures # for parallel computing

videolists = glob.glob(r'\\10.10.46.135\share\zhangna\4_Miniscope\behavior\analysis\217001-06\*.mp4')
coordinates = os.path.join(os.path.dirname(videolists[0]),'xy.txt')

freezing_stat = {}
frame_interval=7 ##7 # 产生_freezing_csv文件时，coulbourn system每秒有4个数据，所以建议调整产生数据的帧间隔（frame_interval）至每秒4-8个数据左右
diff_gray_value=30 ##30 #前后两帧同样像素点位置是否变化的阈值，一般不变，但是当曝光很暗，比如低于10lux时可以适当降低这个值
threshold =0.07#至少有多少比例的像素点变化了时，我们认为小鼠时运动着的，这里表示0.52%
#另外还有一个参数并没有写出来用于修改，即小鼠不动的时间要不小于1s,才会认为是freezing，这个值一般不动.

#判断是否选中视频
if len(videolists)==0:
    print("there are no video choosed")
    sys.exit()
#判断是否画了老鼠活动区域，没有则弹出frame
if not os.path.exists(coordinates):
    print("please draw the travel region.")
    _,_ = Video(videolists[0]).draw_roi()
#判断*_ts.txt文件是否存在，不存在则产生
for video in videolists:
    freeze_video = Video(video)





    
    if not os.path.exists(freeze_video.videots_path):
        freeze_video.generate_ts_txt()
    else:
        print(video,"*_ts.txt file already exists")
#判断*_freezing.csv是否存在，不存在则产生
    if not os.path.exists(freeze_video.videofreezing_path):
        v2c(video,Interval_number=frame_interval,diff_gray_value=diff_gray_value,show = True)
    else:
        print(video,"*_freezing.csv file already exists")
    freezing_stat[os.path.basename(video)]=Csv(freeze_video.videofreezing_path).freezing_percentage(threshold=threshold,start=0,stop=182,show_detail=True,save_epoch = True)


#将结果存储到同目录下的freezing_stat.csv中
current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
with open(os.path.join(os.path.dirname(videolists[0]),'freezing_stat.csv'),'w',newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Time',current_time])
    writer.writerow(['frame_interval',frame_interval])
    writer.writerow(['diff_gray_value',diff_gray_value])
    writer.writerow(['threshold',threshold])
    writer.writerow(['video_id','freezing%'])
    for row in freezing_stat.items():
        writer.writerow(row)
        print("threshold is:",threshold,row)
        









