# FreezingAnalysis
Video frame-pixel-based approach to analyze rodent freezing

中文说明:
videolists改为视频文件保存的路径，格式一致，比如AVI
frame_interval为提取数据的帧间隔，cinelyzer产生的视频文件每秒有30个数据点，我一般用2，也就是2帧取1帧，每秒15个数据点
diff_gray_value用30，判断前后两帧每个像素点的像素变化的阈值
threshold，与变化像素点的比例作比较，低于阈值则判定僵直。阈值越高，得到的freeze水平越高。同一天同一个context中，所有动物用一个阈值，不同context可能阈值不同，经常修正

    freezing_stat[os.path.basename(video)]=Csv(freeze_video.videofreezing_path).freezing_percentage(threshold=threshold,start=0,stop=182,show_detail=True,save_epoch = True)
其中 start为统计的起始时间，stop为终止时间，秒为单位。
save_epoch可以保存freeze的时间段，一般不改

其余不修改，运行后选择待分析的Arena（比如方形），键盘点”s“保存xy.txt文件（一般对于一个视频只需要处理一次，后续再次分析自动识别已有的txt文件）
通过freeze_epoch和行为视频校准threshold。
