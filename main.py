import cv2 #opencv-python 모듈
import mediapipe as mp #핸드트랙킹 위한 머신러닝 모듈
import time #시간모듈

cap = cv2.VideoCapture(1)
#웹캠 장치 번호 0은 노트북 1은 따로 연결한 웹캠
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

max_num_hands=2 #최대 감지 가능한 손 입력

gesture = { 0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ', 7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ',
            13:'ㅎ', 14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅡ', 19:'ㅣ', 20:'ㅐ', 21:'ㅔ', 22:'ㅚ', 23:'ㅟ', 24:'ㅒ',
            25:'ㅖ', 26:'ㅢ', 27:'space', 28:'enter', 29:'backspace', 30:'clear'}
gesture_up={0:'된소리', 2:'ㄸ', 1:'ㄲ', 3:'ㅃ', 4:'ㅆ', 5:'ㅉ'}
gestrue_eng={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o',
             15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}

pTime = 0 #이전 시간
cTime = 0 #현재 시간

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #^ opencv함수인 이미지 색상 표현 방식 함수, 선택된 이미지 RGB를 역전해서 BGR (이유:opencv rgb로 사용하는 수치를 bgr로 사용한다. 행렬 생성을 생각하면 의미파악할 수 있다.)

    results = hands.process(imgRGB)
    #이미지 값을 process에 저장. 저장되는 값 : 각 마디에 대한 위치 정보

    #핸드를 인식하면 처리 되는 코드
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks: #만약 result에서 multi_hand_landmarks 값 존재하면 밑에 진행
        for handLandmarks in results.multi_hand_landmarks:
            #핸드의 각 관절 포인트의 ID와 좌표를 알아 내서 원하는 그림을 그려 넣을 수 있다. 
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #손목관절
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (180, 255, 255), cv2.FILLED)
                #중지 끝
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (90, 180, 0), cv2.FILLED)

            #인식된 핸드에 점과 선을 그려 넣는다. 초대박 간단하네...
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    #fps 출력
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

