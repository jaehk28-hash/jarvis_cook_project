import csv
import os
import sys

# 키 1~4를 누르면 그 동작을 기록하기 시작하고, 0을 누르면 기록을 멈춥니다. q는 종료.
LABELS = {'1': 'thumb', '2': 'v', '3': 'cook', '4': 'idle'}


def normalize(hand, is_left):
    """관절 21개 -> 숫자 63개 (손목 기준 상대 좌표, 손 크기로 나눔, 왼손은 좌우 반전)"""
    pts = [(p.x, p.y, p.z) for p in hand.landmark]
    wx, wy, wz = pts[0]                                  # 0번 = 손목
    pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]  # 손목을 원점으로
    mx, my, mz = pts[9]                                  # 9번 = 가운뎃손가락 뿌리
    size = (mx ** 2 + my ** 2 + mz ** 2) ** 0.5 or 1e-6  # 손 크기
    out = []
    for x, y, z in pts:
        if is_left:
            x = -x
        out += [x / size, y / size, z / size]
    return out


def main():
    import cv2
    import mediapipe as mp

    name = sys.argv[1] if len(sys.argv) > 1 else 'me'
    session = sys.argv[2] if len(sys.argv) > 2 else '1'
    session_id = f'{name}_{session}'

    os.makedirs('data', exist_ok=True)
    path = f'data/{session_id}.csv'
    new_file = not os.path.exists(path)
    f = open(path, 'a', newline='')
    writer = csv.writer(f)
    if new_file:
        writer.writerow(['label', 'session'] + [f'{a}{i}' for i in range(21) for a in 'xyz'])

    hands = mp.solutions.hands.Hands(max_num_hands=1)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    current = None
    counts = {v: 0 for v in LABELS.values()}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            if current:
                is_left = result.multi_handedness[0].classification[0].label == 'Left'
                row = [round(v, 5) for v in normalize(hand, is_left)]
                writer.writerow([current, session_id] + row)
                counts[current] += 1

        status = f'REC: {current}' if current else 'not recording'
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, ' '.join(f'{k}:{v}' for k, v in counts.items()), (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('collect', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            current = None
        elif chr(key) in LABELS:
            current = LABELS[chr(key)]

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print('저장됨:', path, counts)


if __name__ == '__main__':
    main()
