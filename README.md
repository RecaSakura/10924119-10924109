# 使用Google Colab 進行 DeepFace測試
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
底下這段代碼作用是使用matplotlib將圖片顯示顯示出來

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

