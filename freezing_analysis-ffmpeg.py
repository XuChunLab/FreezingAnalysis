from freezing import CFCvideo
import glob,sys

# 请下载ffmpeg(www.ffmpeg.org)并添加至环境路径
# Please download ffmpeg and append it into your Enviornment

timestamp_method="ffmpeg" # we use ffmpeg here. # timestamps 的文件，注意在ffmpeg,datetime,miniscope中进行选择
Interval_number=7 # analyze every N frame. # 选第1+7n帧用于计算。产生_freezing_csv文件时，参考coulbourn system每秒有4个数据，所以建议调整产生数据的帧间隔（frame_interval）至每秒4-8个数据左右
diff_gray_value=30 # threshold for frame pixel differneces #前后两帧同样像素点位置是否变化的阈值，一般不变，但是当曝光很暗，比如低于10lux时可以适当降低这个值
show = True # show the video of first 100 frames #显示前100帧视频
threshold = 0.07 # threshold for mobile moves #当总共至少有多少比例的像素点变化了时，我们认为小鼠是运动着的，这里表示0.07%
start = 0 # start of behavioral analysis (sec) #分析行为学起始时间, in seconds
stop = 360 # stop of behavioral analysis (sec) #分析行为学结束时间 in seconds
show_detail=True # display on the screen #将结果,比如freezing的epoch 输出到屏幕上
percent =True # shown in percentage #freezing 时间比列 用省略百分号的百分比表示
save_epoch=True # save the freezing episodes # 将freezing的epoch也存储下来

videolists=glob.glob(r"F:\20201218\*.AVI")
if not len(videolists)==0:
    [print(i) for i in videolists]
else:
    print("videolists 路径不对")
CFCvideo(videolists[0]).freezing_percentage(
    timestamp_method=timestamp_method
    ,Interval_number=Interval_number
    ,diff_gray_value=diff_gray_value
    ,show = show
    ,threshold = threshold
    ,start = start
    ,stop = stop
    ,show_detail=show_detail
    ,percent =percent
    ,save_epoch=save_epoch)

Batch = False ##True  or False

if Batch: #如果调参完毕，将Batch=False 改为Batch=True
    CFCvideo.freezing_percentages(
        timestamp_method=timestamp_method
        ,videolists=videolists
        ,Interval_number=Interval_number
        ,diff_gray_value=diff_gray_value
        ,show = show
        ,threshold = threshold
        ,start = start
        ,stop = stop
        ,show_detail=show_detail
        ,percent =percent
        ,save_epoch=save_epoch)
else:
    print("you didn't set batch mode, default to analysis the first video in the videolists")
