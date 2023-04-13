# 使用Google Colab 進行 DeepFace測試以完成學校作業,此內容皆從網路上擷取
前製作業
-
import必要數據

    %matplotlib inline
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
安裝DeepFace套件

    !pip install deepface
將下載好的deepface import 進來

    from deepface import DeepFace as dp
   `輸出：`<br>`Directory  /root /.deepface created`<br>
   `Directory  /root /.deepface/weights created`
<br><br>底下這段代碼作用是使用matplotlib將圖片顯示顯示出來

    def show_image(*args):
    k = len(args)
    fig = plt.figure(figsize=(5*k, 5))
    for i, photo in enumerate(args):
        plt.subplot(1,k,i+1)
        plt.axis('off')
        plt.axis('equal')
        plt.imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
開始測試-比對兩張照片是否為同一人
-
定義圖片路徑

    im01_path = "/content/drive/MyDrive/IMG1.jpg"
    im01 = cv2.imread(im01_path)
    im02_path = "/content/drive/MyDrive/IMG2.jpg"
    im02 = cv2.imread(im02_path)
    ###注意！如果圖片位於google雲端硬碟其他地方記得修改路徑與檔名,建議直接複製檔案位置比較保險

秀出剛剛的圖片

    show_image(im01, im02)
    ###使用剛剛def的副程式 show出圖片，若只想show一張就只需要show_image(im01)

比較兩張照片是否為同一人

    ###查看result資料result為辨識後的各項數據
    result = dp.verify(im01, im02, model_name="DeepFace")
    result
   `{'verified': True,`<br>
    `'distance': 0.19951417603681987,
    'threshold': 0.23,`<br>
    `'model': 'DeepFace',
    'detector_backend': 'opencv',
    'similarity_metric': 'cosine',`<br>
    `'facial_areas': {'img1': {'x': 475, 'y': 1410, 'w': 1109, 'h': 1109},`<br>
    `'img2': {'x': 456, 'y': 1071, 'w': 1175, 'h': 1175}},
    'time': 11.36}`<br><br>
簡化後只顯示兩張照片是否為同一人

    if(result['verified'])==True:
        print("兩人為同一人")
    else:
        print("兩人不同人")

辨識人種、性別、年齡、情緒
-
底下程式是將辨識後的資料以簡潔的方式顯示出來

    ###labels將obj裡的資料，轉換成中文
    labels = {'angry':'生氣', 'disgust':'厭惡', 'fear':'恐懼',
          'happy':'開心', 'neutral':'普通', 
          'sad':'悲傷', 'surprise':'吃驚',
          'Man':'男', 'Woman':'女',
          'asian':'亞洲', 'black':'黑', 'indian':'印弟安',
          'latino hispanic':'拉丁美洲 (西班牙裔)', 
          'middle eastern':'中東', 'white':'白'}
    ###obj裡有不少數據，如同剛剛的result一樣，這段程式的作用就是將它可視化，不然輸出的數據會很雜亂
    def show_info(obj):
        age = obj[0]["age"]
        emotion = labels[obj[0]['dominant_emotion']]
        race = labels[obj[0]['dominant_race']]
        gender = labels[obj[0]['dominant_gender']]
        text = f"這是一位 {age} 歲的{race}人{gender}性, 他的表情是{emotion}的。"
        print(text)
        
開始辨識

    im03_path = "/content/drive/MyDrive/IMG1.jpg"
    im03 = cv2.imread(im03_path)
    obj = dp.analyze(img_path = im03_path, actions = ['age', 'gender', 'race', 'emotion'])
    show_image(im03)
    show_info(obj)
   `Action: emotion: 100%|██████████| 4/4 [00:01<00:00,  2.19it/s]`<br>
   `這是一位 34 歲的亞洲人男性, 他的表情是悲傷的。`<br>
    
    ###上面的輸出文字是經過簡化與可視化的結果，若直接顯示obj則:
    obj[0]
   `{'age': 27,
     'region': {'x': 537, 'y': 1455, 'w': 1107, 'h': 1107},`<br>
     `'gender': {'Woman': 0.18314786721020937, 'Man': 99.81684684753418},`<br>
    `'dominant_gender': 'Man',`<br>
    `'race': {'asian': 99.99927282333374,
    'indian': 1.5001997155650315e-05,
    'black': 1.701037199985933e-07,`<br>
    `'white': 0.0003110175157416961,
     'middle eastern': 1.1405072442016717e-07,
     'latino hispanic': 0.0004004386028100271},`<br>
    `'dominant_race': 'asian',`<br>
    `'emotion': {'angry': 0.04594368510879576,
      'disgust': 3.781720749884698e-05,
     'fear': 0.014255850692279637,`<br>
     `'happy': 0.00011057578603868023,
     'sad': 6.9861553609371185,
     'surprise': 5.13807272284339e-06,`<br>
     `'neutral': 92.9534912109375},
     'dominant_emotion': 'neutral'}`<br>
 這也可以測試不同人種,性別,情緒但要注意圖片一定要清晰,不然會無法辨識。
 
 參考網站
 -
[deepface GitHub本站](https://github.com/serengil/deepface)  <br>
[colab08 用 DeepFace 神速打造人臉辨識.ipynb](https://github.com/yenlung/Deep-Learning-Basics/blob/master/colab08%20%E7%94%A8%20DeepFace%20%E7%A5%9E%E9%80%9F%E6%89%93%E9%80%A0%E4%BA%BA%E8%87%89%E8%BE%A8%E8%AD%98.ipynb)
