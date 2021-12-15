import cv2 #opencv-python 모듈
import keyboard as keyboard
import mediapipe as mp #핸드트랙킹 위한 머신러닝 모듈
import time #시간모듈
import cvzone
import numpy as np

from PIL import ImageFont, ImageDraw, Image
import math
import tkinter

max_num_hands=1 #최대 감지 가능한 손 입력/수어 인식하려면 2/지문자 인식하려면 1로 해야한다.

cap = cv2.VideoCapture(1)
#웹캠 장치 번호 0은 노트북 1은 따로 연결한 웹캠
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils  #mphands, mpdraw 손가락 디텍션 모듈 초기화
hands = mpHands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

f=open('test.txt','w')
#제스쳐 인식 모델
file=np.genfromtxt('C:/images/dataSetplus.txt', delimiter=',') #dataSet.txt 학습데이터 각도들(0.33425634, 0.35231345,.0.135277 --- 0.0000) 마지막 수가 라벨
angleFile=file[:,:-1]
labelFile=file[:,-1]
angle=angleFile.astype(np.float32)
label=labelFile.astype(np.float32)
knn=cv2.ml.KNearest_create() #knn 최근접 알고리즘
knn.train(angle,cv2.ml.ROW_SAMPLE,label)

startTime = time.time()
prev_index=0
sentence=''
recognizeDelay=3
text =''

#우선 한국 지문자만.
#제스처 데이터들 관절 각도랑 각각 라벨 'ㄱ' 'ㄴ' ... 라벨
gesture = { 0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ', 7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ',
            13:'ㅎ', 14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅡ', 19:'ㅣ', 20:'ㅐ', 21:'ㅔ', 22:'ㅚ', 23:'ㅟ', 24:'ㅒ',
            25:'ㅖ', 26:'ㅢ', 27:'ㅗ', 28:'ㅛ', 29:'ㅜ ', 30:'ㅠ', 31:'삭제', 32:'띄어쓰기'} #27~30 은 시간 차이나 버튼으로 해결하면 괜찮을 듯.
#gesture_up={0:'된소리', 2:'ㄸ', 1:'ㄲ', 3:'ㅃ', 4:'ㅆ', 5:'ㅉ'}
#gestrue_eng={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}

setfont = ImageFont.truetype("C:/Users/ahn26/Downloads/SCDream6.otf", 40)
class Button:
    def __init__(self, pos, width,height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
    def draw_o(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0]+self.width, self.pos[1]+self.height),(165,208,229), cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),(81,94,115), 3)
        cv2.putText(img, self.value, (self.pos[0] + 10, self.pos[1] + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

button1 = Button((1000,160),80,80,'Space')
button2 = Button((1090,160),80,80,'Clear')
while True:
    success, img = cap.read() #열려있는 한 프레임씩 읽어온다.
    button1.draw_o(img)
    button2.draw_o(img)
    if not success:
        continue
    #읽어오는데 실패하면 다음 프레임
    #imgRGB = cv2.flip(img,1) #이미지 반전
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #^ opencv함수인 이미지 색상 표현 방식 함수, 선택된 이미지 RGB를 역전해서 BGR (이유:opencv rgb로 사용하는 수치를 bgr로 사용한다. 행렬 생성을 생각하면 의미파악할 수 있다.)
    results = hands.process(imgRGB)
    #이미지 값을 process에 저장. 저장되는 값 : 각 마디에 대한 위치 정보
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

    # 만약 result에서 multi_hand_landmarks 값 존재하면 밑에 진행
    if results.multi_hand_landmarks is not None:
        sc_point_x=0
        sc_point_y=0
        th_point_x, th_point_y =0,0
        for res in results.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):
                joint[j]=[lm.x,lm.y,lm.z]
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if j==8: #두번째 손가락
                    sc_point_x, sc_point_y = int(joint[j][0]*w),int(joint[j][1]*h)
                if j==12: #세번째 손가락
                    th_point_x, th_point_y =int(joint[j][0]*w),int(joint[j][1]*h)

            length = math.sqrt(abs(th_point_x-sc_point_x)+abs(th_point_y-th_point_y))

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,10,11, 0,13,14,15, 0,17,18,19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20],:]

            v = v2 - v1
            v = v / np.linalg.norm(v,axis=1)[:,np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9,10,12,13,14,16,17,8],:]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9,10,11,13,14,15,17,18,19,12],:]
            angle = np.arccos(np.einsum('nt,nt->n',compareV1,compareV2))

            angle = np.degrees(angle)
            if keyboard.is_pressed('a'): #누르면 각도 txt로 저장
                for num in angle:
                    num=round(num,6)
                    f.write(str(num))
                    f.write(',')
                f.write("29.000000")#데이터 저장할 제스쳐 라벨 번호
                f.write('\n')
                print("next")

            data=np.array([angle], dtype=np.float32)
            ret, results,neighbours,dist = knn.findNearest(data,3)
            index = int(results[0][0])
            if index in gesture.keys():
                if index != prev_index: #앞에 인덱스랑 다르면 초기화
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time()-startTime>recognizeDelay:
                        if (1000<sc_point_x<1080)&(160<sc_point_y<320):
                            sentence += '_'
                            cv2.circle(img, (1040,100), 10, (165,208,229), cv2.FILLED)
                            cv2.putText(img,'!',(1040,100),cv2.FONT_HERSHEY_PLAIN ,1, (50, 50, 50), 2) #고민!
                        elif (1090<sc_point_x<1170)&(160<sc_point_y<320):
                            sentence = ''
                            cv2.putText(img, 'wait3!', (1040, 100), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 50), 2) #고민!
                        else:
                            if index==15 or index==30:
                                if (sc_point_y>360):
                                    index==30
                                else:
                                    index==15
                            if index==6 or index ==28:
                                if(length<10):
                                    index==28
                                else:
                                    index==6
                            sentence += gesture[index]
                        startTime = time.time()
                #cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                draw.text((int(res.landmark[0].x*img.shape[1]-10),int(res.landmark[0].y*img.shape[0] + 40)),gesture[index].upper(), font=setfont, fill=(0, 0, 0))
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)


            cv2.rectangle(img, (th_point_x-50, th_point_y-80), (th_point_x+100, th_point_y-40),(165,208,229), cv2.FILLED) #거리 상자 그리기
            cv2.rectangle(img, (th_point_x-50, th_point_y-80), (th_point_x+100, th_point_y-40), (81,94,115), 2) #거리 상자 외곽 그리기
            cv2.putText(img, str(round(length, 6)), (th_point_x-40, th_point_y-50), cv2.FONT_HERSHEY_PLAIN,1.5 ,(70,72,77), 2) #중지-검지 거리 표시
            cv2.circle(img, (th_point_x, th_point_y),20, (165,208,229), cv2.FILLED) #중지 원 마크 그리기
            cv2.circle(img, (sc_point_x, sc_point_y), 20, (165,208,229), cv2.FILLED) #검지 원 마크 그리기
            cv2.line(img, (sc_point_x,sc_point_y), (th_point_x,th_point_y),(81,94,115),3) #중지-검지 거리 라인

            mpDraw.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS) #랜드마크 그리기

    #cv2.rectangle(img, (10, 10), (40,30), (81,94,115),2) #딜레이 시간 1 외곽
    cv2.rectangle(img, (0, 600), (1280, 720), (165, 208, 229), cv2.FILLED)  # 출력창
    cv2.line(img, (0,600), (1260, 600), (81,94,115), 2) #출력창 외곽선
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((30, 640), sentence, font=setfont, fill=(159,101,73)) #입력문장 출력하기
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

    logo =cv2.imread('C:/images/logo.png', cv2.IMREAD_UNCHANGED)
    logo=cv2.resize(logo,(0,0),None,0.3,0.3)

    hf,wf,cf = logo.shape
    hb,wb,cb=img.shape
    img = cvzone.overlayPNG(img,logo, [0, hb-hf])


    cv2.imshow("SONGUEL", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #q누르면 종료
        break

f.close();

