import argparse
import csv
import os
import time
from collections import Counter, deque
from datetime import datetime

# 동작 -> 키보드 키 (유튜브 기준: k = 재생/일시정지, 왼쪽 화살표 = 5초 뒤로)
KEYS = {'thumb': 'k', 'v': 'left'}   # 모델에 three/four가 있으면 main()에서 {'thumb':'k','three':'left','four':'right'}로 바뀜
CURSOR = 'point'  # 커서 이동 동작. model.pt에 point가 없으면(예전 모델) main()에서 'palm'으로 바꿉니다
CLICK = 'palm'   # 클릭 동작(보자기). 예전 model.pt(point 없음)면 main()에서 'fist'로 바꿉니다
COMMANDS = ['thumb', 'v', CURSOR, CLICK]
SCROLL = 'v'     # 커서 모드에서 브이(두 손가락)를 위아래로 움직이면 스크롤 (트랙패드 두 손가락처럼)
# 반응 시간 측정: 키보드 숫자 -> 잴 동작
REACTION_KEYS = {ord('1'): 'thumb', ord('2'): 'v', ord('3'): 'CURSOR', ord('4'): 'CLICK'}  # 'CURSOR'는 실행 시 커서 동작으로 바뀜
REACTION_TIMEOUT = 10.0  # 이 시간(초) 안에 실행되지 않으면 그 측정은 버립니다


class Controller:
    """프레임마다 (동작, 확률, 손 위치)를 받아서, 실행할 동작 목록을 돌려줍니다.
    - 신뢰도 게이트: 확률이 thresh 미만이면 명령으로 보지 않습니다.
    - 유지시간 게이트: 최근 window초 동안 같은 명령이 ratio 이상이어야 인정합니다.
    - 커서 모드: 손바닥을 유지하면 커서 모드가 켜지고, 커서 모드에서만 주먹이 클릭이 됩니다."""

    def __init__(self, thresh=0.8, move_thresh=0.5, window=0.5, ratio=0.7, min_frames=4,
                 cooldown=0.8, exit_after=1.5, click_lag=0.15):
        self.thresh, self.move_thresh = thresh, move_thresh
        self.window, self.ratio, self.min_frames = window, ratio, min_frames
        self.cooldown, self.exit_after, self.click_lag = cooldown, exit_after, click_lag
        self.hist = deque()        # (시간, 동작) 최근 기록
        self.pos_hist = deque()    # (시간, 손 위치) 손바닥일 때만 기록
        self.state = 'normal'      # 'normal' 또는 'cursor'
        self.last_fire = {}
        self.last_cursor_t = 0.0
        self.fist_start = None
        self.locked = None         # 한 번 실행된 동작은 손을 바꿀 때까지 다시 실행하지 않음
        self.unlock_frames = 0

    def candidate(self):
        if len(self.hist) < self.min_frames:
            return 'none'
        counts = Counter(lab for _, lab in self.hist if lab != 'none')
        if not counts:
            return 'none'
        lab, n = counts.most_common(1)[0]
        return lab if n / len(self.hist) >= self.ratio else 'none'

    def _trim(self, t):
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        while self.pos_hist and t - self.pos_hist[0][0] > 2.0:
            self.pos_hist.popleft()

    def update(self, t, label, conf, pos):
        lab = label if (label in COMMANDS and conf >= self.thresh) else 'none'
        move_lab = label if (label in COMMANDS and conf >= self.move_thresh) else 'none'
        self.hist.append((t, lab))
        self._trim(t)
        cand = self.candidate()
        actions = []

        if self.locked:
            self.unlock_frames = self.unlock_frames + 1 if lab != self.locked else 0
            if self.unlock_frames >= 5:
                self.locked, self.unlock_frames = None, 0

        if self.state == 'normal':
            if cand in KEYS and cand != self.locked and t - self.last_fire.get(cand, -99) >= self.cooldown:
                actions.append(('key', KEYS[cand]))
                self.last_fire[cand] = t
                self.locked, self.unlock_frames = cand, 0
                self.hist.clear()
            elif cand == CURSOR or (cand == SCROLL and SCROLL not in KEYS):   # 검지 또는 (브이가 키가 아니면) 브이로 커서 모드
                self.state, self.last_cursor_t, self.fist_start = 'cursor', t, None
                self.locked, self.unlock_frames = None, 0
                self.hist.clear()
        else:  # cursor
            if move_lab == CURSOR and pos is not None:
                self.pos_hist.append((t, pos))
                actions.append(('move', pos))
                self.last_cursor_t = t
                self.fist_start = None
            elif move_lab == SCROLL and pos is not None:
                actions.append(('scroll', pos))
                self.last_cursor_t = t
                self.fist_start = None
            elif lab == CLICK:
                self.last_cursor_t = t
                if self.fist_start is None:
                    self.fist_start = t
                if cand == CLICK and self.locked != CLICK:
                    target = None  # 주먹을 쥐기 직전의 커서 위치에서 클릭
                    for tt, p in reversed(self.pos_hist):
                        if tt <= self.fist_start - self.click_lag:
                            target = p
                            break
                    if target is None and self.pos_hist:
                        target = self.pos_hist[0][1]
                    actions.append(('click', target))
                    self.locked, self.unlock_frames = CLICK, 0
                    self.hist.clear()
                    self.fist_start = None
            if t - self.last_cursor_t > self.exit_after:
                self.state = 'normal'
                self.hist.clear()
        return actions


class NoGateController:
    """--gate-off용: 신뢰도, 유지시간, 재실행 대기, 커서 모드 없이 매 프레임 예측대로 실행합니다."""
    state = 'gate-off'

    def __init__(self):
        self.palm_pos = None  # 마지막으로 손바닥이 있던 위치 (주먹 클릭 위치)

    def update(self, t, label, conf, pos):
        if label in KEYS:
            return [('key', KEYS[label])]
        if label == SCROLL and pos is not None:
            return [('scroll', pos)]
        if label == CURSOR and pos is not None:
            self.palm_pos = pos
            return [('move', pos)]
        if label == CLICK:
            return [('click', self.palm_pos or pos)]
        return []


def executed_gesture(act, entered_cursor):
    """이번 실행이 어느 명령 동작인지 돌려줍니다. 손바닥은 커서 모드가 켜지는(게이트를 끄면 커서가 움직이는) 순간입니다."""
    if act is None:
        return CURSOR if entered_cursor else None
    if act[0] == 'key':
        return next((g for g, k in KEYS.items() if k == act[1]), None)
    return {'move': CURSOR, 'click': CLICK}.get(act[0])


class Meter:
    """--log용: 실행 기록(CSV), 누적 횟수, 분당 실행 횟수, 반응 시간을 관리합니다."""

    def __init__(self, start_t, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.fromtimestamp(start_t).strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'realtime_{stamp}.csv')
        self.file = open(self.path, 'w', newline='', encoding='utf-8-sig')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['시각', '경과초', '실행', '상세', '측정중', '반응동작', '반응시간초'])
        self.start_t = start_t
        self.total = 0               # 실행 시작 후 KEY + CLICK 횟수
        self.measure_start = None    # 측정 중이면 시작 시각
        self.measure_count = 0
        self.pending = None          # (동작, 숫자 키를 누른 시각)
        self.reactions = {}          # 동작 -> 반응 시간 목록
        print(f'기록 파일: {self.path}')

    def _write(self, t, action, detail, gesture='', reaction=None):
        self.writer.writerow([datetime.fromtimestamp(t).isoformat(sep=' ', timespec='milliseconds'),
                              f'{t - self.start_t:.3f}', action, detail,
                              int(self.measure_start is not None), gesture,
                              '' if reaction is None else f'{reaction:.3f}'])
        self.file.flush()

    def record(self, t, gesture, action=None, detail=''):
        """명령이 실행될 때 부릅니다. action이 'KEY'나 'CLICK'이면 횟수에 넣습니다."""
        reaction = None
        if self.pending and self.pending[0] == gesture:
            reaction = t - self.pending[1]
            self.pending = None
            self.reactions.setdefault(gesture, []).append(reaction)
        if action:
            self.total += 1
            if self.measure_start is not None:
                self.measure_count += 1
        elif reaction is None:
            return
        # 손바닥(커서 모드 켜기)은 KEY/CLICK이 아니므로 반응 시간을 잴 때만 CURSOR로 남깁니다.
        self._write(t, action or 'CURSOR', detail, gesture if reaction is not None else '', reaction)
        if reaction is not None:
            print(f'반응 시간 {gesture}: {reaction:.3f}초')
            self.print_average()

    def start_reaction(self, t, gesture, state):
        if gesture == CURSOR and state == 'cursor':
            print('이미 커서 모드입니다. 손을 내려 normal 모드가 된 뒤 3을 누르세요.')
            return
        if gesture == CLICK and state == 'normal':
            print(f'참고: {CLICK} 클릭은 커서 모드에서만 됩니다. 측정 시간에 {CURSOR}로 커서 모드를 켜는 시간이 포함됩니다.')
        self.pending = (gesture, t)
        print(f'반응 시간 측정 시작: {gesture}')

    def check_timeout(self, t):
        if self.pending and t - self.pending[1] > REACTION_TIMEOUT:
            print(f'{self.pending[0]}: {REACTION_TIMEOUT:.0f}초 안에 실행되지 않아 이번 측정은 버립니다.')
            self.pending = None

    def print_average(self):
        every = [r for rs in self.reactions.values() for r in rs]
        if not every:
            return
        parts = [f'{g} {sum(rs) / len(rs):.3f}초({len(rs)}회)' for g, rs in self.reactions.items()]
        print(f'평균 반응 시간: 전체 {sum(every) / len(every):.3f}초({len(every)}회) | ' + ', '.join(parts))

    def toggle(self, t):
        if self.measure_start is None:
            self.measure_start, self.measure_count = t, 0
            print('측정 시작')
        else:
            sec = max(t - self.measure_start, 1e-6)
            print(f'측정 정지: {sec:.1f}초 동안 {self.measure_count}회 실행, 분당 실행 횟수 {self.measure_count * 60 / sec:.1f}')
            self.measure_start = None

    def lines(self, t):
        text = [f'elapsed {t - self.start_t:.0f}s  count {self.total}']
        if self.measure_start is not None:
            text.append(f'MEASURING {t - self.measure_start:.0f}s  count {self.measure_count}')
        if self.pending:
            text.append(f'waiting: {self.pending[0]}')
        return text

    def close(self, t):
        if self.measure_start is not None:
            self.toggle(t)
        self.print_average()
        self.file.close()
        print(f'기록 저장: {self.path}')


def load_model(path):
    """model.pt에서 가중치를 읽어, 층 구성에 상관없이 동작하는 예측 함수를 만듭니다."""
    import torch
    ckpt = torch.load(path, map_location='cpu')
    sd, labels = ckpt['state_dict'], ckpt['labels']
    layers = [(sd[k], sd[k[:-len('weight')] + 'bias']) for k in sd if k.endswith('.weight')]
    n_out = layers[-1][0].shape[0]  # 출력 클래스 수 (model.pt는 6, model_cmd_only.pt는 4)
    if n_out != len(labels):
        raise SystemExit(f'{path}: 출력 {n_out}개와 동작 이름 {len(labels)}개가 맞지 않습니다.')
    print(f'모델: {path} (출력 {n_out}개: {", ".join(labels)})')

    def predict(vec):
        h = torch.tensor(vec, dtype=torch.float32)
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = torch.relu(h)
        return torch.softmax(h, dim=0).numpy()

    return labels, predict


def to_screen(pos, w, h, area=0.7, center_y=0.5):
    """카메라 화면의 area 비율 영역을 모니터 전체로 늘려서 대응시킵니다 (절대 모드).
    center_y를 0.5보다 작게 하면 영역이 위로 올라가서, 손을 카메라 아래 끝까지 내리지 않아도 화면 아래에 닿습니다."""
    lo_x, lo_y = 0.5 - area / 2, center_y - area / 2
    x = min(max((pos[0] - lo_x) / area, 0.0), 1.0) * w
    y = min(max((pos[1] - lo_y) / area, 0.0), 1.0) * h
    return min(max(x, 6), w - 6), min(max(y, 6), h - 6)  # 모서리(안전 정지)를 피함


class CursorMapper:
    """손 위치 -> 커서 위치.
    relative(기본): 마우스처럼 손이 움직인 만큼(× gain) 커서가 움직입니다. 빨리 움직이면 더 멀리 갑니다.
        손바닥을 잠깐(0.25초) 멈추거나 내리면 '마우스를 들어 올린' 것처럼 연결이 끊겨서,
        손을 다시 가운데로 가져와도 커서가 튀지 않습니다.
    absolute: 카메라 가운데 area 영역을 화면 전체에 대응시킵니다 (area가 작을수록 팔을 덜 움직임)."""

    def __init__(self, w, h, mode='relative', gain=1.2, area=0.4, start=None, gain_y=1.2, center_y=0.4):
        self.w, self.h, self.mode, self.gain, self.area = w, h, mode, gain, area
        self.gain_y, self.center_y = gain_y, center_y  # 세로는 손을 내릴 공간이 좁아서 더 민감하게
        self.cursor = start or (w / 2, h / 2)
        self.prev_hand = None   # 직전 손 위치 (부드럽게 만든 값)
        self.prev_t = None
        self.hist = []          # (손 위치 원본, 커서 위치): 클릭 위치를 되찾는 데 씀

    def _clip(self, x, y):
        return min(max(x, 6), self.w - 6), min(max(y, 6), self.h - 6)

    def move(self, t, pos):
        if self.mode == 'absolute':
            x, y = to_screen(pos, self.w, self.h, self.area, self.center_y)
            cx, cy = self.cursor
            self.cursor = (0.65 * cx + 0.35 * x, 0.65 * cy + 0.35 * y)
        else:
            if self.prev_t is None or t - self.prev_t > 0.25:
                self.prev_hand = pos            # 연결이 끊겼다가 다시 시작: 커서는 그대로
            hx = 0.65 * self.prev_hand[0] + 0.35 * pos[0]   # 손 떨림을 줄이는 평균 (검지 끝은 더 떨려서 더 강하게)
            hy = 0.65 * self.prev_hand[1] + 0.35 * pos[1]
            dx, dy = hx - self.prev_hand[0], hy - self.prev_hand[1]
            speed = (dx * dx + dy * dy) ** 0.5
            if speed > 0.002:                   # 아주 작은 떨림은 무시
                accel = 1.0 + min(speed / 0.02, 0.5)   # 빠르게 움직이면 최대 1.5배
                k = self.gain * accel
                self.cursor = (self.cursor[0] + dx * self.w * k, self.cursor[1] + dy * self.h * k * self.gain_y)
            self.prev_hand = (hx, hy)
        self.prev_t = t
        self.cursor = self._clip(*self.cursor)
        self.hist = (self.hist + [(pos, self.cursor)])[-90:]
        return self.cursor

    def click_point(self, target):
        """컨트롤러가 고른 '주먹 쥐기 직전'의 손 위치에 대응하던 커서 위치를 돌려줍니다."""
        for p, c in reversed(self.hist):
            if p is target or p == target:
                return c
        return self.cursor

    def jump(self, xy):
        self.cursor = self._clip(*xy)

    def scroll(self, t, pos, gain=150, invert=False):
        """브이 손을 위아래로 움직인 만큼 스크롤 칸 수를 돌려줍니다. 손을 내리면 페이지가 아래로 내려갑니다.
        0.25초 넘게 끊기면 새로 시작하므로 손을 다시 위로 가져와도 반대로 스크롤되지 않습니다."""
        if getattr(self, 's_prev_t', None) is None or t - self.s_prev_t > 0.25:
            self.s_prev_y, self.s_acc = pos[1], 0.0
        y = 0.65 * self.s_prev_y + 0.35 * pos[1]
        dy = y - self.s_prev_y
        self.s_prev_y, self.s_prev_t = y, t
        if abs(dy) < 0.002:
            return 0
        self.s_acc += -dy * gain * (-1 if invert else 1)
        n = int(self.s_acc)
        self.s_acc -= n
        return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='실제로 키보드/마우스를 조작 (없으면 화면에 글자만 표시)')
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--thresh', type=float, default=0.8, help='신뢰도 기준')
    parser.add_argument('--window', type=float, default=0.5, help='유지시간(초)')
    parser.add_argument('--log', action='store_true', help='실행 기록과 측정 기능 켜기 (r: 측정, 1~4: 반응 시간)')
    parser.add_argument('--gate-off', action='store_true', help='신뢰도/유지시간 게이트를 끄고 매 프레임 예측대로 실행')
    parser.add_argument('--cursor', choices=['relative', 'absolute'], default='relative',
                        help='relative: 마우스처럼 움직인 만큼 이동(기본) / absolute: 손 위치 = 화면 위치')
    parser.add_argument('--gain', type=float, default=1.2, help='relative 모드 감도 (클수록 조금만 움직여도 멀리 감)')
    parser.add_argument('--area', type=float, default=0.4, help='absolute 모드에서 화면 전체에 대응할 카메라 영역 비율 (예전 값 0.7)')
    parser.add_argument('--scroll-gain', type=float, default=150, help='스크롤 감도 (커서 모드에서 브이로 스크롤)')
    parser.add_argument('--scroll-invert', action='store_true', help='스크롤 방향 반대로')
    parser.add_argument('--gain-y', type=float, default=1.2, help='세로 감도 배수 (아래쪽이 잘 안 가면 키움)')
    parser.add_argument('--center-y', type=float, default=0.4, help='absolute 모드 영역의 세로 중심 (작을수록 위쪽)')
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    from collect import normalize  # 데이터 수집 때와 똑같은 전처리를 씁니다

    labels, predict = load_model(args.model)
    global CURSOR, CLICK, COMMANDS
    CURSOR = 'point' if 'point' in labels else 'palm'   # 예전 model.pt(6개 동작)는 손바닥 커서 그대로
    CLICK = 'palm' if 'point' in labels else 'fist'     # 새 모델: 검지 이동 + 보자기 클릭 / 예전: 손바닥 이동 + 주먹 클릭
    COMMANDS[:] = ['thumb', 'v', CURSOR, CLICK]
    if 'three' in labels and 'four' in labels:   # 손가락 3 = 5초 전, 4 = 5초 후, 브이 = 언제든 스크롤
        KEYS.clear(); KEYS.update({'thumb': 'k', 'three': 'left', 'four': 'right'})
        COMMANDS[:] = ['thumb', 'v', CURSOR, CLICK, 'three', 'four']
    print('키 동작:', KEYS)
    for k, g in list(REACTION_KEYS.items()):
        if g == 'CURSOR':
            REACTION_KEYS[k] = CURSOR
        elif g == 'CLICK':
            REACTION_KEYS[k] = CLICK
    print('커서 동작:', CURSOR, '/ 클릭 동작:', CLICK)
    ctl = NoGateController() if args.gate_off else Controller(thresh=args.thresh, window=args.window)
    pyautogui = None
    if args.live:
        import pyautogui
        pyautogui.PAUSE = 0
        screen_w, screen_h = pyautogui.size()
    else:
        screen_w, screen_h = 1440, 900

    hands = mp.solutions.hands.Hands(max_num_hands=1)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    start_xy = tuple(pyautogui.position()) if pyautogui else None
    mapper = CursorMapper(screen_w, screen_h, args.cursor, args.gain, args.area, start_xy, args.gain_y, args.center_y)
    last_action, last_action_t, prev_t = '', 0.0, time.time()
    meter = Meter(prev_t) if args.log else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, conf, pos = 'none', 1.0, None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            is_left = result.multi_handedness[0].classification[0].label == 'Left'
            probs = predict(normalize(hand, is_left))
            label, conf = labels[int(probs.argmax())], float(probs.max())
            tip = 8 if CURSOR == 'point' else 9   # 검지 커서면 검지 끝, 손바닥 커서면 가운뎃손가락 뿌리
            pos = (hand.landmark[tip].x, hand.landmark[tip].y)
        now = time.time()
        if meter:
            meter.check_timeout(now)
        prev_state = ctl.state
        actions = ctl.update(now, label, conf, pos)
        if meter and prev_state == 'normal' and ctl.state == 'cursor':
            meter.record(now, executed_gesture(None, True))
        for act in actions:
            kind = act[0]
            if meter:
                logged = {'key': ('KEY', act[1]), 'click': ('CLICK', '')}.get(kind, (None, ''))
                meter.record(time.time(), executed_gesture(act, False), *logged)
            if kind == 'key':
                last_action, last_action_t = f'KEY {act[1]}', now
                if pyautogui:
                    pyautogui.press(act[1])
            elif kind == 'move':
                x, y = mapper.move(now, act[1])
                if pyautogui:
                    pyautogui.moveTo(x, y)
            elif kind == 'scroll':
                n = mapper.scroll(now, act[1], args.scroll_gain, args.scroll_invert)
                if n:
                    last_action, last_action_t = f'SCROLL {n:+d}', now
                    if pyautogui:
                        pyautogui.scroll(n)
            elif kind == 'click':
                last_action, last_action_t = 'CLICK', now
                x, y = mapper.click_point(act[1])
                mapper.jump((x, y))
                if pyautogui:
                    pyautogui.click(x, y)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        mode = 'LIVE' if args.live else 'DRY (no control)'
        if args.gate_off:
            mode += '  GATE OFF'
        mode += f'  cursor {args.cursor}'
        if pos is not None:
            mode += f'  hand y {pos[1]:.2f}'
        lines = [f'{mode}  fps {fps:.0f}',
                 f'pred: {label} {conf:.2f}   mode: {ctl.state}',
                 f'last: {last_action}' if now - last_action_t < 1.5 else 'last: -']
        if meter:
            lines += meter.lines(now)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('realtime', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q 또는 Esc (한글 입력 상태에서는 q가 안 먹히므로 Esc를 쓰세요)
            break
        if meter and key == ord('r'):
            meter.toggle(time.time())
        elif meter and key in REACTION_KEYS:
            meter.start_reaction(time.time(), REACTION_KEYS[key], ctl.state)
    if meter:
        meter.close(time.time())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
