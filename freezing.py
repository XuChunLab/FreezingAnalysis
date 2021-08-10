import os,sys,glob
import cv2
import pandas as pd
import csv
import datetime
import numpy as np
import re
class File():
    def __init__ (self,file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        self.file_name_noextension = self.file_name.split(".")[0]
        self.extension = os.path.splitext(self.file_path)[-1]
        self.abs_prefix = os.path.splitext(self.file_path)[-2]
        self.dirname = os.path.dirname(self.file_path)
    def add_prefixAsuffix(self,prefix = "prefix", suffix = "suffix",keep_origin=True):
        '''
        会在suffix前或者prefix后自动添加“——”
        keep_origin = True，表示会复制原文件，否则是直接操作源文件
        '''
        if os.path.exists(self.file_path):
            newname = os.path.join(self.dirname,prefix+self.file_name_noextension+suffix+self.extension)
            if keep_origin:
                copyfile(self.file_path,newname)
                print("Rename file successfully with original file kept")
            else:
                os.rename(self.file_path, newname)
                print("Rename file successfully with original file deleted")
        else:
            print(f"{self.file_path} does not exists.")

    def copy2dst(self,dst):
        """
        将文件copy到指定的位置（文件夹，不是文件名）
        dst: path of directory
        """
        if os.path.exists(self.file_path):
            newname = os.path.join(dst,self.file_name)
            copyfile(self.file_path,newname)
            print(f"Transfer {self.file_path} successfully")
        else:
            print("{self.file_path} does not exists.")
class FreezingFile(File):
    def __init__(self,file_path):
        super().__init__(file_path)
        self.freezingEpochPath = os.path.join(self.dirname,'behave_video_'+self.file_name_noextension+'_epoch.csv')
    def _rlc(self,x):
        name=[]
        length=[]
        for i,c in enumerate(x,0):
            if i ==0:
                name.append(x[0])
                count=1
            elif i>0 and x[i] == name[-1]:
                count += 1
            elif i>0 and x[i] != name[-1]:
                name.append(x[i])
                length.append(count)
                count = 1
        length.append(count)
        return name,length

    def freezing_percentage(self,threshold = 0.005, start = 0, stop = 300,show_detail=False,percent =True,save_epoch=True): 
        data = pd.read_csv(self.file_path)
        print(len(data['0']),"time points ;",len(data['Frame_No']),"frames")

        data = data.dropna(axis=0)             
        #print(f"{self.file_path}")
        data = data.reset_index()
        # slice (start->stop)
        
        #start_index
        if start>stop:
            start,stop = stop,start
            warning.warn("start time is later than stop time")
        if start >=max(data['0']):
            warnings.warn("the selected period start is later than the end of experiment")
            sys.exit()
        elif start <=min(data['0']):            
            start_index = 0
        else:
            start_index = [i for i in range(len(data['0'])) if data['0'][i]<=start and  data['0'][i+1]>start][0]+1

        #stop_index
        if stop >= max(data['0']):
            stop_index = len(data['0'])-1
            print("the selected period exceed the record time, automatically change to the end time.")
        elif stop <=min(data['0']):
            print("the selected period stop is earlier than the start of experiment")
            sys.exit()
        else:            
            stop_index = [i for i in range(len(data['0'])) if data['0'][i]<=stop and  data['0'][i+1]>stop][0]

        selected_data = data.iloc[start_index:stop_index+1]

        values,lengthes = self._rlc(np.int64(np.array(selected_data['percentage'].tolist())<=threshold))


        
        sum_freezing_time = 0
        if save_epoch:
            with open(self.freezingEpochPath,'w+',newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["start","stop"])
        for i,value in enumerate(values,0):
            if value ==1:              
                begin = sum(lengthes[0:i])
                end = sum(lengthes[0:i+1])
                if end > len(selected_data['0'])-1:
                    end = len(selected_data['0'])-1
                condition = selected_data['0'].iat[end]-selected_data['0'].iat[begin]
                if condition >=1:
                    if show_detail:
                        print(f"{round(selected_data['0'].iat[begin],1)}s--{round(selected_data['0'].iat[end],1)}s,duration is {round(condition,1)}s".rjust(35,'-'))
                    if save_epoch:
                        with open(self.freezingEpochPath,'a+',newline="") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([selected_data['0'].iat[begin],selected_data['0'].iat[end]])
                    sum_freezing_time = sum_freezing_time + condition
                else:
                    sum_freezing_time = sum_freezing_time
        print(f'the freezing percentage during [{start}s --> {stop}s] is {round(sum_freezing_time*100/(stop-start),2)}% ')
        if save_epoch:
            with open(self.freezingEpochPath,'a+',newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["","",f"{round(sum_freezing_time*100/(stop-start),2)}%"])
        if  percent:
            return sum_freezing_time*100/(stop-start)
        else:
            return sum_freezing_time/(stop-start)
class TimestampsFile(File):
    def __init__(self,file_path,method="ffmpeg",camNum=0):
        super().__init__(file_path)
        self.method = method
        self.camNum = camNum
        if not method in ["datetime","ffmpeg","miniscope"]:
            print("method are only available in 'ffmpeg','datetime',")
            sys.exit()

        self.ts=self.read_timestamp()
        # if self.ts.isnull().any():
        #     print(self.ts)
        #     print("ATTENTION: therea are 'NaN' in timestamps !!")

    def datetime2minisceconds(self,x,start):    
        # print(x,end = " " )
        delta_time = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')-start
        return int(delta_time.seconds*1000+delta_time.microseconds/1000)

    def read_timestamp(self):
        if self.method == "datetime":
            data = pd.read_csv(self.file_path,sep=",")
            start = datetime.datetime.strptime(data["0"][0], '%Y-%m-%d %H:%M:%S.%f')
            data["0"]=data["0"].apply(self.datetime2minisceconds,args=[start,])
            return data["0"]/1000
        if self.method  == "ffmpeg":
            try:
                return pd.read_csv(self.file_path,encoding="utf-16",header=None,sep=" ",names=["0"])
            except:
                print("default method is ffmpeg, try 'datetime'")
                sys.exit()
        if self.method == "miniscope":
            temp=pd.read_csv(self.file_path,sep = "\t", header = 0)
            temp = temp[temp["camNum"]==self.camNum] ## wjn的 case 是1， 其他的scope是0
            print("camNum in miniscope is %s"%self.camNum)
            # incase the first frame of timestamps is not common 比如这里会有一些case的第一帧会出现很大的正/负数
            if np.abs(temp['sysClock'][0])>temp['sysClock'][1]:
                value = temp['sysClock'][1]-13 # 用第2帧的时间减去13，13是大约的一个值
                if value < 0:
                    temp['sysClock'][0]=0
                else:
                    temp['sysClock'][0]=value

            ts = pd.DataFrame(temp['sysClock'].values)
            return ts
class Video():
    """
    """
    def __init__(self,video_path):
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)
        self.extension = os.path.splitext(self.video_path)[-1]
        self.abs_prefix = os.path.splitext(self.video_path)[-2]
        self.xy = os.path.join(os.path.dirname(self.video_path),'xy.txt')
        self.videots_path = self.abs_prefix + '_ts.txt'
        try:
            self.video_track_path = glob.glob(self.abs_prefix+"*.h5")[0]
        except:
            print("video havenp't been tracked")

        self.videosize = os.path.getsize(self.video_path)/1073741824 # video size is quantified by GB

    
    def timestamps(self,method="ffmpeg"):
        if not os.path.exists(self.videots_path):
            print("generating TimestampsFile by ffmpeg")
            self.generate_ts_txt()
        else:
            return TimestampsFile(self.videots_path,method=method).ts

    def play(self):
        """
        instructions:
            'q' for quit
            'f' for fast farward
            'b' for fast backward
        Args:

        """
        cap = cv2.VideoCapture(self.video_path)
        wait= 30
        step = 1
        while (1):
            ret,frame = cap.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow('video',gray)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
            if cv2.waitKey(wait) & 0xFF == ord('f'):
                if wait > 1:
                    wait = wait -step
                else:
                    print("it has played at the fast speed without dropping any frame")
            if cv2.waitKey(wait) & 0xFF == ord('b'):
                wait = wait + step
        cap.release()
        cv2.destroyAllWindows()

    def show_masks(self,aim="in_context"):
        masks = self.draw_rois(aim=aim)[0]
        for mask in masks:
            plt.imshow(mask)
            plt.show()

    def transcode(self,show_details=True):
        """
        for save larger size video  as very smaller one
        """
        if self.videosize>1:
            print("%s is as large as %.2fGB"%(self.video_path,self.videosize))
            if self.video_path.endswith(".avi"):
                newvideo=self.video_path.replace(".avi",".mp4")
                command = ["ffmpeg","-i",self.video_path,newvideo]

                child = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8')
                out = child.communicate()[1]
                if show_details:
                    print(out)
                child.wait()
                print("%s has been transcoded")

    def crop_video(self,show_details=False,multiprocess=False):
        '''
        ffmpeg -i $1 -vf crop=$2:$3:$4:$5 -loglevel quiet $6

        可以使用
        ret,frame = cv2.VideoCapture(videopath)
        cv2.selectRoi(frame)
        来返回 x,y,w,h
        '''

        croped_video = self.abs_prefix+"_crop.avi"
        crop_video_file = self.abs_prefix+"_crop.txt"

        if not os.path.exists(croped_video):
            if not os.path.exists(crop_video_file):
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES,1000)
                ret,frame = cap.read()
                x,y,w,h = cv2.selectROI("%s"%os.path.basename(self.video_path),frame)
                cv2.destroyAllWindows()
                coords = pd.DataFrame(data={"x":[x],"y":[y],"w":[w],"h":[h]})
                print(coords)
                coords.to_csv(crop_video_file,index=False)
            else:
                print("coords exists")
                coords = pd.read_table(crop_video_file,sep=",")

            command = [
            "ffmpeg",
            "-i",self.video_path,"-vf",
            "crop=%d:%d:%d:%d" % (coords["w"],coords["h"],coords["x"],coords["y"]),
            "-loglevel","quiet",croped_video]

            child = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8')
            out = child.communicate()[1]
            if show_details:
                print(out)
            if not multiprocess:
                child.wait()
            print("%s has been cropped"%video)
        else:
            print("%s was cropped."%video)


    
    def contrastbrightness(self,):
        print(self.video_path)
        command=[
            "ffmpeg",
            "-i",self.video_path,
            "-vf","eq=contrast=2:brightness=0.5",
            self.video_path.replace(".mp4",".avi")
        ]
        print("%s is adjusting contrast and brightness" % self.video_path)
        child = subprocess.Popen(command,stdout = subprocess.PIPE,stderr=subprocess.PIPE)
        out = child.communicate()[1].decode('utf-8')
    #     print(out)
        child.wait()
        print("%s done"%self.video_path)
    
    def _HMS2seconds(self,time_point):
        sum = int(time_point.split(":")[0])*3600+int(time_point.split(":")[1])*60+int(time_point.split(":")[2])*1
        return sum

    def _seconds2HMS(self,seconds):
        return time.strftime('%H:%M:%S',time.gmtime(seconds))

    def cut_video_seconds(self,start,end):
        '''
        this is for video cut in seconds
        ffmpeg -ss 00:00:00 -i video.mp4 -vcodec copy -acodec copy -t 00:00:31 output1.mp4
        starts and ends are in format 00:00:00
        '''
        i=1
        print(start,end)

        duration = self._seconds2HMS(self._HMS2seconds(end)-self._HMS2seconds(start))
        output_video_name = os.path.splitext(self.video_path)[0]+f"_cut_{i}"+os.path.splitext(self.video_path)[1]
        command = [
        "ffmpeg.exe",
        "-ss",start,
        "-i",self.video_path,
        "-vcodec","copy",
        "-acodec","copy",
        "-t",duration,
        output_video_name]
        print(f"{i}/{len(starts)} {self.video_path} is being cut")
        i = i+1
        child = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8')
        out = child.communicate()[1]
        #print(out)
        child.wait()

    def scale(self,distance):
        """
        Args:
            distance: the length in cm of a line that you draw
        """
        while True:
            _,coords_in_pixel = self.draw_rois(aim='scale')
            if len(coords_in_pixel[0]) ==2:
                break
            else:
                print("you should draw a line but not a polygon")

        print(coords_in_pixel[0][1],coords_in_pixel[0][0])
        distance_in_pixel = np.sqrt(np.sum(np.square(np.array(coords_in_pixel[0][1])-np.array(coords_in_pixel[0][0]))))
        distance_in_cm = int(distance) #int(input("直线长(in cm)： "))
        s = distance_in_cm/distance_in_pixel
        print(f"scale: {s} cm/pixel")
        return s

    @staticmethod
    def _angle(dx1,dy1,dx2,dy2):
        """
        dx1 = v1[2]-v1[0]
        dy1 = v1[3]-v1[1]
        dx2 = v2[2]-v2[0]
        dy2 = v2[3]-v2[1]
        """
        angle1 = math.atan2(dy1, dx1) * 180/math.pi
        if angle1 <0:
            angle1 = 360+angle1
        # print(angle1)
        angle2 = math.atan2(dy2, dx2) * 180/math.pi
        if angle2<0:
            angle2 = 360+angle2
        # print(angle2)
        return abs(angle1-angle2)

    @classmethod
    def speed(cls,X,Y,T,s,sigma=3):
        speeds=[0]
        speed_angles=[0]
        for delta_x,delta_y,delta_t in zip(np.diff(X),np.diff(Y),np.diff(T)):
            distance = np.sqrt(delta_x**2+delta_y**2)
            speeds.append(distance*s/delta_t)
            speed_angles.append(cls._angle(1,0,delta_x,delta_y))
        return pd.Series(speeds),pd.Series(speed_angles) # in cm/s

    def play_with_track(self,show = "Body",scale=40,latest=300):
        """
        instructions:
            'q' for quit
            'f' for fast farward
            'b' for fast backward
        Args:
            show. "Head",Body" or "Tail". default to be "Body"
        """
        if not os.path.exists(self.videots_path):
            try:
                print("generating timestamps by ffmpeg")
                self.generate_ts_txt()
            except:
                print("fail to generate timestamps by ffprobe")
                sys.exit()
        else:
            ts = pd.read_table(self.videots_path,sep='\n',header=None,encoding="utf-16")

        if not os.path.exists(self.video_track_path):
            print("you haven't done deeplabcut tracking")
            sys.exit()
        else:
            track = pd.read_hdf(self.video_track_path)


        s = self.scale(40)
        try:
            behaveblock=pd.DataFrame(track[track.columns[0:9]].values,columns=['Head_x','Head_y','Head_lh','Body_x','Body_y','Body_lh','Tail_x','Tail_y','Tail_lh'])
            print("get track of head, body and tail")
        except:
            behaveblock=pd.DataFrame(track[track.columns[0:6]].values,columns=['Head_x','Head_y','Head_lh','Body_x','Body_y','Body_lh'])
            print("get track of head and body")
        behaveblock['be_ts'] = ts[0]
        behaveblock['Headspeeds'],behaveblock['Headspeed_angles'] = self.speed(behaveblock['Head_x'],behaveblock['Head_y'],behaveblock['be_ts'],s)
        behaveblock['Bodyspeeds'],behaveblock['Bodyspeed_angles'] = self.speed(behaveblock['Body_x'],behaveblock['Body_y'],behaveblock['be_ts'],s)
        # behaveblock['Tailspeeds'],behaveblock['Tailspeed_angles'] = self.speed(behaveblock['Tail_x'],behaveblock['Tail_y'],behaveblock['be_ts'],s)
        if show ==  "Body":
            x = [int(i) for i in behaveblock["Body_x"]]
            y = [int(i) for i in behaveblock["Body_y"]]
            speed = behaveblock["Bodyspeeds"]
        elif show == "Head":
            x = [int(i) for i in behaveblock["Head_x"]]
            y = [int(i) for i in behaveblock["Head_y"]]
            speed = behaveblock["Headspeeds"]
        else:
            print("please choose from 'Body' and 'Head'")
        t = [i for i in behaveblock["be_ts"]]

        font = cv2.FONT_ITALIC
        cap = cv2.VideoCapture(self.video_path)
        wait=30
        step = 1
        frame_No = 0
        while True:
            ret,frame = cap.read()
            if ret:
                # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                cv2.circle(frame,(x[frame_No],y[frame_No]),3,(0,255,0),-1)
                cv2.putText(frame,f'{round(speed[frame_No],2)}cm/s',(x[frame_No]+5,y[frame_No]), font, 0.5, (100,100,255))
                for i in range(frame_No,0,-1):
                    if (t[frame_No]-t[i])<latest:
                        pts1=(x[i],y[i]);pts2=(x[i-1],y[i-1])
                        thickness=1
                        color = (0,0,255)
                        if (t[frame_No]-t[i])<5:
                            thickness=2
                            color = (0,255,0)
                        cv2.line(frame, pts1, pts2, color, thickness)
                cv2.imshow(self.video_name,frame)
                frame_No = frame_No + 1
                if cv2.waitKey(wait) & 0xFF == ord('q'):
                    break

                if cv2.waitKey(wait) & 0xFF == ord('f'):
                    if wait > 1:
                        wait = wait -step
                    else:
                        print("it has played at the fast speed without dropping any frame")
                    print("fps: %d"%round(1000/wait,1))
                if cv2.waitKey(wait) & 0xFF == ord('b'):
                    wait = wait + step
                    print("fps: %d"%round(1000/wait,1))
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


    def generate_ts_txt(self):
        if not os.path.exists(self.videots_path):
            print("generating timestamps...")
            if (platform.system()=="Linux"):
                command = r'ffprobe -i %s -show_frames -select_streams v -loglevel quiet| grep pkt_pts_time= | cut -c 14-24 > %s' % (self.video_path,self.videots_path)
                child = subprocess.Popen(command,shell=True)
            if (platform.system()=="Windows"):
                try:
                    powershell=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
                except:
                    print("your windows system doesn't have powershell")
                    sys.exit()
                # command below relys on powershell so we open powershell with a process named child and input command through stdin way.
                child = subprocess.Popen(powershell,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                command = r'ffprobe.exe -i "%s" -show_frames -loglevel quiet |Select-String media_type=video -context 0,4 |foreach{$_.context.PostContext[3] -replace {.*=}} |Out-File "%s"' % (self.video_path,self.videots_path)
                child.stdin.write(command.encode('utf-8'))
                out = child.communicate()[1].decode('gbk') # has to be 'gbk'
                #print(out)
                child.wait()
                print(f"{self.video_path} has generated _ts files")
        else:
            print("%s is aready there."%self.videots_path)

    def _extract_coord (self,file,aim):
        '''
        for reading txt file generated by draw_rois
        '''
        f = open (file)
        temp = f.readlines()
        coords = []
        for eachline in temp:
            eachline = str(eachline)
            if aim+' ' in str(eachline):
                #print (eachline)
                coord = []
                pattern_x = re.compile('\[(\d+),')
                coord_x = pattern_x.findall(str(eachline))
                pattern_y = re.compile('(\d+)\]')
                coord_y = pattern_y.findall(str(eachline))
                for x,y in zip(coord_x,coord_y):
                    coord.append([int(x),int(y)])
                coords.append(coord)
        f.close()
        return coords

    def draw_rois(self,aim,count = 1):
        '''
        count means how many arenas to draw, for each arena:
            double clicks of left mouse button to make sure
            click of right mouse button to move
            click of left mouse button to choose point
        '''
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,1000)
        ret,frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()

        origin = []
        coord = []
        coord_current = [] # used when move
        masks = []
        coords = []
        font = cv2.FONT_HERSHEY_COMPLEX
        state = "go"
        if os.path.exists(self.xy):
            existed_coords = self._extract_coord(self.xy,aim)
            for existed_coord in existed_coords:
                if len(existed_coord) >0:
                    existed_coord = np.array(existed_coord,np.int32)
                    coords.append(existed_coord)
                    mask = 255*np.ones_like(frame)
                    cv2.fillPoly(mask,[existed_coord],0)
                    masks.append(mask)
                    mask = 255*np.ones_like(frame)
                else:
                    print("there is blank coord record ")
                    continue
            if len(existed_coords) > count:
                print(f"there are more coords than you want, take the first {count}: ")
                print(coords[0:count])
                return masks[0:count],coords[0:count]
            if len(existed_coords) == count:
                print("you have drawn rois of '%s'"%aim)
                return masks,coords
            if len(existed_coords) < count:
                print("please draw left rois of '%s'"%aim)
        else:
            print("please draw rois of %s"%aim)

        def draw_polygon(event,x,y,flags,param):
            nonlocal state, origin,coord,coord_current,mask,frame
            try:
                rows,cols,channels= param['img'].shape
            except:
                print("Your video is broken,please check that if it could be opened with potplayer?")
                sys.exit()
            black_bg = np.zeros((rows,cols,channels),np.uint8)
            if os.path.exists(self.xy):
                for i,existed_coord in enumerate(existed_coords,1):
                    if len(existed_coord)>0:
                        existed_coord = np.array(existed_coord,np.int32)
                        cv2.fillPoly(black_bg,[existed_coord],(127,255,100))
                        cv2.putText(black_bg,f'{i}',tuple(np.trunc(existed_coord.mean(axis=0)).astype(np.int32)), font, 1, (0,0,255))
            if state == "go" and event == cv2.EVENT_LBUTTONDOWN:
                coord.append([x,y])
            if event == cv2.EVENT_MOUSEMOVE:
                if state == "go":
                    if len(coord) ==1:
                        cv2.line(black_bg,tuple(coord[0]),(x,y),(127,255,100),2)
                    if len(coord) >1:
                        pts = np.append(coord,[[x,y]],axis = 0)
                        cv2.fillPoly(black_bg,[pts],(127,255,100))
                    frame = cv2.addWeighted(param['img'],1,black_bg,0.3,0)
                    cv2.imshow("draw_roi",frame)
                if state == "stop":
                    pts = np.array(coord,np.int32)
                    cv2.fillPoly(black_bg,[pts],(127,255,100))
                    frame = cv2.addWeighted(param['img'],1,black_bg,0.3,0)
                    cv2.imshow("draw_roi",frame)
                if state == "move":
                    coord_current = np.array(coord,np.int32) +(np.array([x,y])-np.array(origin) )
                    pts = np.array(coord_current,np.int32)
                    cv2.fillPoly(black_bg,[pts],(127,255,100))
                    cv2.fillPoly(mask,[pts],0)
                    frame = cv2.addWeighted(param['img'],1,black_bg,0.3,0)
                    cv2.imshow("draw_roi",frame)
            if event == cv2.EVENT_RBUTTONDOWN:
                origin =  [x,y]
                state = "move"
            if event == cv2.EVENT_LBUTTONDBLCLK:
                if state == "move":
                    coord = coord_current.tolist()
                state = "end of this arena"
                print("stop")
                mask = 255*np.ones_like(frame)
                pts = np.array(coord,np.int32)
                cv2.fillPoly(mask,[pts],0)

        cv2.namedWindow("draw_roi")
        cv2.setMouseCallback("draw_roi",draw_polygon,{"img":frame})
        while(1):
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                if len(coord) >0:
                    masks.append(mask)
                    coords.append(coord)
                    f = open(self.xy,'a+')
                    f.write(f'{aim} {coord}\n')
                    f.close()
                print(f'{self.xy} is saved')
                cv2.destroyAllWindows()
                break
            if key == ord('q'):
                print("selected points are aborted")
                cv2.destroyAllWindows()
                return self.draw_rois()
            if key == ord('a'):
                f = open(self.xy,'a+')
                f.write(f'{aim} {coord}\n')
                f.close()
                print('please draw another aread')
                cv2.destroyAllWindows()
                return self.draw_rois(aim,count = count)
            if key==27:
                print("exit")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
        
        return masks,coords

    def check_frames(self,*args,location = "rightup",time_point=False):
        '''
        'a':后退一帧
        'd':前进一帧
        'w':前进一百帧
        's':后退一百帧
        'n':下一个指定帧
        '''
        if location == "leftup":
            location_coords = (10,15)
        if location == "rightup":
            location_coords = (400,15)
        font = cv2.FONT_ITALIC
        cap = cv2.VideoCapture(self.video_path)
        
        def nothing(x):  
            pass
            
        # cv2.namedWindow("check_frames")
        total_frame = int(cap.get(7))
        # cv2.createTrackbar('frame_No','check_frames',1,int(total_frame),nothing)
        print(f"there are {int(total_frame)} frames in total")
        
        frame_No=1
        
        if time_point:
            #转换成帧数，start from 1,因为在后面播放的时候，有-1的操作。
            args=[find_close_fast[self.timestamps.to_numpy(),i]+1 for i in args]

        specific_frames = args
        if len(specific_frames)==0:
            specific_frames=[0]
        else:
            print(specific_frames,"frames to check")
        marked_frames=[]
        
        for i in specific_frames:
            cv2.namedWindow("check_frames")
            cv2.createTrackbar('frame_No','check_frames',1,int(total_frame),nothing)
            if i < 1:
                frame_No = 1
                print(f"there is before the first frame")
            elif i > total_frame:
                frame_No = total_frame
                print(f"{i} is after the last frame")
            else:
                frame_No = i
                
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_No-1)
            cv2.setTrackbarPos("frame_No","check_frames",frame_No)
            
            ret,frame = cap.read()
            cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
            cv2.imshow('check_frames',frame)
            while 1:                
                key = cv2.waitKey(1) & 0xFF
                frame_No = cv2.getTrackbarPos('frame_No','check_frames')                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                ret,frame = cap.read()
                cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                cv2.imshow('check_frames',frame)
                
                if key == ord('m'):
                    marked_frames.append(frame_No)
                    print(f"the {frame_No} frame is marked")
                if key == ord('d'):
                    frame_No = frame_No +1
                    if frame_No >= total_frame:
                        frame_No = total_frame
                        print(f"you have reached the final frame {total_frame}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('a'):
                    frame_No = frame_No - 1
                    if frame_No <=1:
                        frame_No = 1
                        print(f"you have reached the first frame")
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('w'):
                    frame_No=frame_No +100
                    if frame_No >= total_frame:
                        frame_No = total_frame
                        print(f"you have reached the final frame {total_frame}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('c'):
                    frame_No=frame_No +10
                    if frame_No >= total_frame:
                        frame_No = total_frame
                        print(f"you have reached the final frame {total_frame}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('s'):
                    frame_No=frame_No -100
                    if frame_No <= 1:
                        frame_No = 1
                        print(f"you have reached the first frame")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('z'):
                    frame_No=frame_No -10
                    if frame_No <= 1:
                        frame_No = 1
                        print(f"you have reached the first frame")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_No-1)
                    cv2.setTrackbarPos("frame_No","check_frames",frame_No)
                    ret,frame = cap.read()
                    cv2.putText(frame,f'frame_No:{frame_No} ',location_coords, font, 0.5, (255,255,255))
                    cv2.imshow('check_frames',frame)
                if key == ord('n'):
                    #led_ons.pop(i-1)
                    print('end of this round checking')
                    cv2.destroyAllWindows()
                    break
                if key == ord('q'):
                    print('break out checking of this round')
                    cv2.destroyAllWindows()
                    break
                if key == 27:
                    print("quit checking")
                    cv2.destroyAllWindows()
                    sys.exit()                    
        print("finish checking")
        
        if len(marked_frames) !=0:
            print(marked_frames)
            return marked_frames
class CFCvideo(Video):
    def __init__(self,video_path):
        super().__init__(video_path)
        self.videofreezing_path = self.abs_prefix + '_freezing.csv'

    def show_masks(self):
        mask = self.draw_rois(aim="freezing",count=1)[0][0]
        cv2.imshow("mask",mask)

    def __video2csv(self,Interval_number=1,show = True):

        mask= self.draw_rois(aim="freezing",count=1)[0][0]

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cap = cv2.VideoCapture(self.video_path)
        frame_count =0
        font = cv2.FONT_HERSHEY_COMPLEX
        while(1):
            frame_count += 1
            ret,frame = cap.read()        
            if ret == True:
    ##            print(frame_count)
                if (frame_count-1)%Interval_number == 0:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_gray = cv2.add(mask,frame_gray)
                    if show:
                        if frame_count <=100:
                            cv2.putText(frame_gray,f'frame_No:{frame_count}',(10,15), font, 0.5, (0,0,0))
                            cv2.imshow("video",frame_gray)                    
                            cv2.waitKey(30)     
                    yield (frame_count,frame_gray)
            else :
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def video2csv(self,timestamp_method = "ffmpeg",Interval_number=3,diff_gray_value=30,show = True):
        """calculate the change in fixed roi"""
        if not os.path.exists(self.videots_path):
            try:
                self.generate_ts_txt()
            except:
                print("fail to extract tiemstamps of %s by ffprobe"%self.video_name)
                sys.exit()
        ts = pd.DataFrame(TimestampsFile(self.videots_path,method=timestamp_method).ts)
        # print(type(ts))
        print("==timestamps==")
        ts['Frame_No'] = list(range(1,len(ts)+1)) # Frame_No start from 1
        print(ts)
        print("==============")
        frame_grays = self.__video2csv(Interval_number = Interval_number,show = show)
        
        print(self.video_name+' Frame Number & timestamps are loaded successfully \nvideo is processing frame by frame...')
        changed_pixel_percentages = []
        Frame_No = []

        for item in frame_grays:
            Frame_No.append(item[0])
            if item[0]==1:
                width, height = item[1].shape
                total_pixel = width*height
                changed_pixel_percentages.append(0)
                frame1 = item[1];
            else:
                frame2 = item[1];judge = cv2.absdiff(frame2,frame1) > diff_gray_value
                changed_pixel_percentage = sum(sum(judge))/total_pixel*100
                changed_pixel_percentages.append(changed_pixel_percentage)
                frame1=frame2

        df= pd.DataFrame({'Frame_No':Frame_No,'percentage':changed_pixel_percentages},index=None)
        df = pd.merge(ts,df,on = 'Frame_No',how="outer").sort_values(by="Frame_No",ascending = True)    
        df.to_csv(self.videofreezing_path,index = False,sep = ',')

        print(self.video_path+' finish processing.')
        return df

    def freezing_percentage(self,timestamp_method = "ffmpeg",Interval_number=3,diff_gray_value=30,show = True
        ,threshold = 0.005, start = 0, stop = 300,show_detail=False,percent =True,save_epoch=True):
        """
        Syntax:
            CFCvideo(video).freezing_percentage(*args)
        Args:
            Interval_number=3,
            diff_gray_value=30,
            show = True,
            threshold = 0.005,
            start = 0,
            stop = 300,
            show_detail=True,
            percent =True,
            save_epoch=True
        """
        if not os.path.exists(self.videofreezing_path):
            self.video2csv(timestamp_method = timestamp_method,Interval_number=Interval_number
                ,diff_gray_value=diff_gray_value,show = show)

        return FreezingFile(self.videofreezing_path).freezing_percentage(threshold=threshold, start = start, stop = stop
            ,show_detail=show_detail,percent =percent,save_epoch=save_epoch)

    
    @classmethod
    def freezing_percentages(cls,videolists,timestamp_method="ffmpeg",Interval_number=3,diff_gray_value=30,show = True,
        threshold = 0.5, start = 0, stop = 300,show_detail=True,percent =True,save_epoch=True):
        """
        syntax:
            confirm the *args with `CFCvideo(video).freezing_percentage(*args)`, then,
            CFCvideo.freezing_percentage(videolists,*args).
            result will be saved named as `freezing_stat.csv` at the same directory
        Args:
            videolists: a list of video to calculate freezing 
            Interval_number=3,
            diff_gray_value=30,
            show = True,
            threshold = 0.005,
            start = 0,
            stop = 300,
            show_detail=True,
            percent =True,
            save_epoch=True
        """
        freezing_stat_path = os.path.join(os.path.dirname(videolists[0]),'freezing_stat.csv')

        with open(freezing_stat_path,'w',newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['diff_gray_value',diff_gray_value])
            writer.writerow(['threshold',threshold])
            writer.writerow(['video','freezing%'])
            for video in videolists:
                freezing = cls(video).freezing_percentage(timestamp_method=timestamp_method,Interval_number=Interval_number,diff_gray_value=diff_gray_value,show = show,threshold=threshold, start = start, stop = stop,show_detail=show_detail,percent =percent,save_epoch=save_epoch)
                writer.writerow([video,freezing])
if __name__ == "__main__":
    # help(CFCvideo.freezing_percentage)
    videolists=glob.glob(r"C:\Users\qiushou\Desktop\CFC\*.avi")
    # for video in videolists:
    CFCvideo.freezing_percentages(videolists)
        # CFCvideo(video).freezing_percentage(Interval_number=1,diff_gray_value=30,show = True
        #   ,threshold = 0.005, start = 0, stop = 100,show_detail=True,percent =True,save_epoch=True)
