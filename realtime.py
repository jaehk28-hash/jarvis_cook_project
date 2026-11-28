import argparse
import csv
import os
import time
from collections import Counter, deque
from datetime import datetime

# 동작 -> 키보드 키 (유튜브 기준: k = 재생/일시정지, 왼쪽 화살표 = 5초 뒤로)
KEYS = {'thumb': 'k', 'v': 'left'}
COMMANDS = ['thumb', 'v', 'palm', 'fist']
# 반응 시간 측정: 키보드 숫자 -> 잴 동작
REACTION_KEYS = {ord('1'): 'thumb', ord('2'): 'v', ord('3'): 'palm', ord('4'): 'fist'}
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
            elif cand == 'palm':
                self.state, self.last_cursor_t, self.fist_start = 'cursor', t, None
                self.locked, self.unlock_frames = None, 0
                self.hist.clear()
        else:  # cursor
            if move_lab == 'palm' and pos is not None:
                self.pos_hist.append((t, pos))
                actions.append(('move', pos))
                self.last_cursor_t = t
                self.fist_start = None
            elif lab == 'fist':
                self.last_cursor_t = t
                if self.fist_start is None:
                    self.fist_start = t
                if cand == 'fist' and self.locked != 'fist':
                    target = None  # 주먹을 쥐기 직전의 커서 위치에서 클릭
                    for tt, p in reversed(self.pos_hist):
                        if tt <= self.fist_start - self.click_lag:
                            target = p
                            break
                    if target is None and self.pos_hist:
                        target = self.pos_hist[0][1]
                    actions.append(('click', target))
                    self.locked, self.unlock_frames = 'fist', 0
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
        if label == 'palm' and pos is not None:
            self.palm_pos = pos
            return [('move', pos)]
        if label == 'fist':
            return [('click', self.palm_pos or pos)]
        return []


def executed_gesture(act, entered_cursor):
    """이번 실행이 어느 명령 동작인지 돌려줍니다. 손바닥은 커서 모드가 켜지는(게이트를 끄면 커서가 움직이는) 순간입니다."""
    if act is None:
        return 'palm' if entered_cursor else None
    if act[0] == 'key':
        return next((g for g, k in KEYS.items() if k == act[1]), None)
    return {'move': 'palm', 'click': 'fist'}.get(act[0])


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
        if gesture == 'palm' and state == 'cursor':
            print('이미 커서 모드입니다. 손을 내려 normal 모드가 된 뒤 3을 누르세요.')
            return
        if gesture == 'fist' and state == 'normal':
            print('참고: 주먹 클릭은 커서 모드에서만 됩니다. 측정 시간에 손바닥으로 커서 모드를 켜는 시간이 포함됩니다.')
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


def to_screen(pos, w, h):
    """카메라 화면의 가운데 70% 영역을 모니터 전체로 늘려서 대응시킵니다."""
    x = min(max((pos[0] - 0.15) / 0.7, 0.0), 1.0) * w
    y = min(max((pos[1] - 0.15) / 0.7, 0.0), 1.0) * h
    return min(max(x, 6), w - 6), min(max(y, 6), h - 6)  # 모서리(안전 정지)를 피함


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='실제로 키보드/마우스를 조작 (없으면 화면에 글자만 표시)')
    parser.add_argument('--model', default='model.pt')
    parser.add_argument('--thresh', type=float, default=0.8, help='신뢰도 기준')
    parser.add_argument('--window', type=float, default=0.5, help='유지시간(초)')
    parser.add_argument('--log', action='store_true', help='실행 기록과 측정 기능 켜기 (r: 측정, 1~4: 반응 시간)')
    parser.add_argument('--gate-off', action='store_true', help='신뢰도/유지시간 게이트를 끄고 매 프레임 예측대로 실행')
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    from collect import normalize  # 데이터 수집 때와 똑같은 전처리를 씁니다

    labels, predict = load_model(args.model)
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
    smooth, last_action, last_action_t, prev_t = None, '', 0.0, time.time()
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
            pos = (hand.landmark[9].x, hand.landmark[9].y)
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
                x, y = to_screen(act[1], screen_w, screen_h)
                smooth = (x, y) if smooth is None else (0.65 * smooth[0] + 0.35 * x, 0.65 * smooth[1] + 0.35 * y)
                if pyautogui:
                    pyautogui.moveTo(smooth[0], smooth[1])
            elif kind == 'click':
                last_action, last_action_t = 'CLICK', now
                if pyautogui and act[1] is not None:
                    x, y = to_screen(act[1], screen_w, screen_h)
                    pyautogui.click(x, y)
                    smooth = (x, y)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        mode = 'LIVE' if args.live else 'DRY (no control)'
        if args.gate_off:
            mode += '  GATE OFF'
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
