import cv2
import time
import os
import base64
import pandas as pd
from ultralytics import YOLO
from zhipuai import ZhipuAI

# 读取智谱密钥环境变量
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
# 模型权重
model = YOLO("yolov8n.pt")
# 告警截图保存文件夹（人脸截图正常存在这里，不删除）
screenshot_dir = "alert_screenshot"
os.makedirs(screenshot_dir, exist_ok=True)
# 演示图片文件夹（用于仓库展示）
demo_dir = "demo_img"
os.makedirs(demo_dir, exist_ok=True)
# 日志文件
log_file = "告警日志.csv"
log_cols = ["时间", "图片路径", "告警类型", "类型说明"]
# 摄像头地址列表，优先本地电脑摄像头
cam_list = [
    "http://192.168.1.xxx:8080/video",
    0
]
# 告警类型文字映射
alert_text = {
    0: "环境误报（光影/杂物）",
    1: "真实行人闯入",
    2: "画面模糊，人工复核"
}

# 图片转base64，给智谱AI调用
def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 智谱AI判断画面是否为真实行人入侵
def judge_alert(img_path):
    b64_data = img_to_base64(img_path)
    resp = client.chat.completions.create(
        model="glm-4v",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "判断图中是否有人闯入厂区，只输出数字：0=没人/杂物，1=有人，2=画面看不清"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                ]
            }
        ]
    )
    res = resp.choices[0].message.content.strip()
    try:
        return int(res)
    except:
        return 2

# 自动切换可用摄像头
def get_camera():
    cap = None
    for url in cam_list:
        temp_cap = cv2.VideoCapture(url)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()
    return cap

if __name__ == "__main__":
    cap = get_camera()
    total_person = 0
    start_time = time.time()
    print("监控启动成功，检测到行人自动截图并调用智谱AI校验，按Q关闭窗口")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = get_camera()
            continue
        # verbose=False 关闭逐帧刷屏日志
        results = model(frame, classes=0, verbose=False)

        for res in results:
            boxes = res.boxes
            if len(boxes) > 0:
                total_person += 1
                now_stamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(screenshot_dir, f"{now_stamp}_告警截图.jpg")
                cv2.imwrite(save_path, frame)
                # 调用智谱AI识别
                alert_type = judge_alert(save_path)
                desc = alert_text[alert_type]
                # 写入日志
                row_data = {
                    "时间": now_stamp,
                    "图片路径": save_path,
                    "告警类型": alert_type,
                    "类型说明": desc
                }
                if os.path.exists(log_file):
                    df = pd.read_csv(log_file)
                    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                else:
                    df = pd.DataFrame([row_data], columns=log_cols)
                df.to_csv(log_file, index=False, encoding="utf-8-sig")
                # 仅打印一行告警，终端干净
                print(f"【告警】{now_stamp} | 识别结果：{desc}")
                cv2.imshow("厂区安防监控", frame)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    run_sec = round(time.time() - start_time, 2)
    print(f"\n监控结束，运行时长：{run_sec}秒，累计检测行人：{total_person}次")