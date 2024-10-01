import threading
from multiprocessing import Queue, Value, Array, Process
import time
import cv2
import signal
import os
import copy



class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

class usb_camera:
    def __init__(self, sensor_id, postprocess=None):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.sensorID = sensor_id
        self.isOpened = False
        self.sensorIsReady = False
        self.postprocess = None
        self.logObj = None
        self.writerObj = None
    
    def open(self):
        try:
            self.video_capture = cv2.VideoCapture(self.sensorID)
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                self.sensorIsReady = ret
        except:
            self.video_capture = None
        else:
            self.sensorIsReady = True
        finally:
            return self.sensorIsReady
    
    def start(self):
        if self.running:
            return None
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCapture)
            self.read_thread.start()
            self.isOpened = self.video_capture.isOpened()
        return self
    
    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        return True
      
    def updateCapture(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                pass
    
    def read(self):
        with self.read_lock:
            time_stamp = time.time()
            frame = self.frame
            grabbed = self.grabbed
            self.frame = None
            self.grabbed = 0
        return grabbed, frame, time_stamp


class core:
    def __init__(self, configFileName):
        print('[INFO] Loading configuration...')
        self.configFileName = configFileName
        self.config = [line.rstrip('\r\n') for line in open(self.configFileName)]
        self.configParameters = {}
        self.sampling = None
        self.maxWait = None
        self.is_running = False
        self.sensorList = []
        self.multiProcessorList = []
        self.index = []

    class objCreate:
        def __init__(self, time, index, data, data_is_ok=0):
            self.time = time
            self.index = index
            self.data = data
            self.data_is_ok = data_is_ok
            self.isSaved = False
            self.isProcessed = False

    class addProcessor:
        def __init__(self, name=None, target_function=None, args=None, num_of_cores=2):
            self.name           = name
            self.running        = Value('i', 0)
            self.buffer         = Queue()
            self.target         = target_function
            self.args           = args
            self.num_of_cores   = num_of_cores
            self.jobs           = []

        def start(self):
            self.running.value = 1
            for i in range(self.num_of_cores):
                process = Process(target=self.target, args=self.args)
                process.daemon = True
                self.jobs.append(process)
            for job in self.jobs:
                job.start()
            print(f'[INFO] Parallel {self.name} status: {bool(self.running)}')

        def stop(self):
            self.running.value = 0
            for job in self.jobs:
                if job.is_alive():
                    job.join()

    def addCamera(self, config):
        if int(config[1]):
            sensor_id = f'camera{int(config[2])}'
            sensor_obj = usb_camera(sensor_id)
            sensor_obj.open()
            return sensor_obj
        else:
            return None
    
    def loopCycleControl(self, start_time, sleep_enable=1):
        dT = time.time()-start_time
        if sleep_enable:
            if dT < self.maxWait:
                time.sleep(self.maxWait-dT)
            buffer_size = sum([p.args[0].qsize() for p in self.multiProcessorList])
            print(f"\r[INFO] Actual rate:{min([1/dT, self.sampling]):5.2f} - Buffer length: {buffer_size:03d}", end="")
            if buffer_size > 200:
                raise LargeBuffer
        else:
            if time.time()-start_time < 1/self.sampling:
                cv2.waitKey((1/self.sampling + start_time - time.time()) * 1000)
                print("\r[INFO] Actual rate:%d" % int(1/(time.time()-start_time)), end="")
            elif time.time()-start_time > 1/self.sampling:
                print("[WARNING] High sampling rate- Actual rate:%d" % int(1/(time.time()-start_time)), end="")
                time.sleep(0.001)
                print("", end='\r')
                cv2.waitKey(1)
        
    def setup(self, demo_visualizer=False):
        # refresh nvargus port
        try:
            os.system('sudo service nvargus-daemon stop')
            os.system('sudo service nvargus-daemon start')
        except:
            pass
        # register sensors
        for i in self.config:
            splitWords = i.split(" ")
            if "camera" in splitWords[0]:
                temp = self.addCamera(splitWords)
                if temp:
                    self.sensorList.append(temp)
                    self.blankImg = cv2.resize(
                        self.blankImg,
                        (int(splitWords[6]),
                         int(splitWords[7])),
                        interpolation=cv2.INTER_AREA
                    )
                    self.blankImgShape = (self.blankImg.shape[1], self.blankImg.shape[0])
                    temp.blankImg = self.blankImg
    
    def closeAll(self):
        for sensor in self.sensorList:
            sensor.stop()
        for processor in self.multiProcessorList:
            processor.stop()

    def readAll(self):
        for sensor in self.sensorList:
            grabbed, data, time_stamp = sensor.read()
            tempObj = self.objCreate(
                time_stamp,
                self.index,
                data,
                grabbed,
                sensor.sensorID)
            sensor.logObj.buffer.put(tempObj)
            if 'camera' in sensor.sensorID:
                sensor.writerObj.buffer.put(copy.copy(tempObj))
    
    def run(self):
        self.is_running = True
        if all([sensor.sensorIsReady for sensor in self.sensorList]):
            with DelayedKeyboardInterrupt():
                for processor in self.multiProcessorList:
                    processor.start()
            for sensor in self.sensorList:
                sensor.start()
                print(sensor, sensor.sensorID, sensor.isOpened)
            if self.sensorList \
                    and all([sensor.isOpened for sensor in self.sensorList]):
                print('[INFO] Running...')
                while self.is_running:
                    try:
                        start_time = time.time()
                        self.readAll()
                        self.loopCycleControl(start_time)
                        self.index += 1
                    except KeyboardInterrupt:
                        self.is_running = False
                        pass
                    except:
                        pass
                else:
                    print("\nCapture disrupted")
                    buffersize = sum([p.args[0].qsize() for p in self.multiProcessorList])
                    print(f"[INFO] Estimated buffer size :{buffersize}")
                    self.closeAll()
            else:
                print("[ERROR] Not all sensors have been opened correctly")
                self.closeAll()
        else:
            print("[ERROR] Not all sensors have been opened correctly")
            for sensor in self.sensorList:
                print(sensor, sensor.sensorID, sensor.sensorIsReady)

            

        
