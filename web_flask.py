from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import cv2
import threading
import queue
import os
import shutil
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
from fire_detect import detect_fire
from YOLOv4.helmet_detect import detect_helmet

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 添加一個密鑰來管理會話

caps = [cv2.VideoCapture(i) for i in range(2)]

# 影像帧數大小設置
frame_widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
frame_heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

# 使用絕對路徑設置靜態資料夾
static_folder = os.path.join(os.path.dirname(__file__), 'static')
app.static_folder = static_folder
video_directories = [os.path.join(app.static_folder, 'yolo_videos'), os.path.join(app.static_folder, 'fire_videos')]

# 確保路徑中有無影像資料夾，沒有就創建
for directory in video_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 創建空影像檔
def create_video_writer(index):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    filename = os.path.join(video_directories[index], datetime.now().strftime("%Y%m%d_%H%M") + '.mp4')
    return cv2.VideoWriter(filename, fourcc, 20.0, (frame_widths[index], frame_heights[index])), filename

# 初始化影像暫存queue
frame_queues = [queue.Queue(maxsize=50) for _ in range(2)]
processed_frame_queues = [queue.Queue(maxsize=50) for _ in range(2)]

# 用於同步啟動capture_frames的事件
app_started_event = threading.Event()

# 用來創建新影像檔及儲存影像
def capture_frames(index):
    global outs, video_filenames
    app_started_event.wait()  # 等待直到Flask server啟動
    out, video_filename = create_video_writer(index)
    outs[index] = out
    video_filenames[index] = video_filename
    next_switch_time = datetime.now() + timedelta(minutes=1)
    while True:
        try:
            frame = processed_frame_queues[index].get(timeout=1)
            if frame is None:
                break

            current_time = datetime.now()
            if current_time >= next_switch_time: #設置每60s儲存完畢一個影像
                out.release()
                convert_to_h264(video_filename, index)
                out, video_filename = create_video_writer(index)
                outs[index] = out
                video_filenames[index] = video_filename
                next_switch_time = current_time + timedelta(minutes=1)

            out.write(frame)
        except queue.Empty:
            continue

def process_frames(index):
    while True:
        try:
            frame = frame_queues[index].get(timeout=1)
            if frame is None:
                break

            if index == 0:
                processed_frame = detect_helmet(frame)  # 使用YOLOv4 detect函數
            else:
                processed_frame = detect_fire(frame)  # 使用detect_fire函數

            # 嘗試將處理後的幀放入隊列，如果隊列已滿，則丟棄舊的處理結果
            if not processed_frame_queues[index].full():
                processed_frame_queues[index].put_nowait(processed_frame)
            else:
                processed_frame_queues[index].get_nowait()  # 丟棄舊的處理結果
                processed_frame_queues[index].put_nowait(processed_frame)
        except queue.Empty:
            continue

#生成實時影像幀傳輸給客戶端
def generate_frames(index):
    while True:
        success, frame = caps[index].read()
        if not success:
            break

        try:
            frame_queues[index].put_nowait(frame)
        except queue.Full:
            pass  # 如果queue滿，就丟棄該幀

        try:
            processed_frame = processed_frame_queues[index].get_nowait()
            ret, buffer = cv2.imencode('.jpg', processed_frame)  # 編碼成'.jpg'檔後進行傳輸
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            continue

# 將input的影像檔轉成H.264編碼格式，輸出到原始資料夾中覆蓋原影像檔
def convert_to_h264(input_filename, index):
    input_path = os.path.join(video_directories[index], os.path.basename(input_filename))
    output_path = os.path.join(video_directories[index], 'temp_' + os.path.basename(input_filename))
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264')
    clip.close()
    shutil.move(output_path, input_path)

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    cam0_videos = [f for f in os.listdir(video_directories[0]) if f.endswith('.mp4')]
    cam1_videos = [f for f in os.listdir(video_directories[1]) if f.endswith('.mp4')]
    return render_template('design.html', cam0_videos=cam0_videos, cam1_videos=cam1_videos)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        #為公司內部應用，故不建立資料庫管理，簡易帳號/密碼設置為 admin / password
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/video_start/<int:cam_index>')
def video_start(cam_index):
    return Response(generate_frames(cam_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/list_videos/<int:cam_index>')
def list_videos(cam_index):
    video_files = [f for f in os.listdir(video_directories[cam_index]) if f.endswith('.mp4')]

    # 獲取當前時間戳加1分鐘的時間戳
    future_time_str = (datetime.now() - timedelta(minutes=1)).strftime("%Y%m%d_%H%M")
    # 獲取當前時間戳
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 過濾掉包含當前時間和未完成的影像檔
    filtered_videos = [f for f in video_files if future_time_str not in f and 'temp' not in f and current_time_str not in f]

    return jsonify(videos=filtered_videos)

if __name__ == "__main__":
    outs = [None, None]
    video_filenames = [None, None]

    for i in range(2):
        t_capture = threading.Thread(target=capture_frames, args=(i,))
        t_capture.daemon = True
        t_capture.start()

        t_process = threading.Thread(target=process_frames, args=(i,))
        t_process.daemon = True
        t_process.start()

    app_started_event.set()
    try:
        app.run(host='127.0.0.1', debug=True, port=5500)
    except KeyboardInterrupt:
        for thread in t_capture, t_process:
            thread.join()  # 等待capture_thread完全結束
        for cap in caps:
            cap.release()
        for out in outs:
            if out:
                out.release()
