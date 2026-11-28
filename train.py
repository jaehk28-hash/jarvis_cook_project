import argparse
import csv
import glob
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# collect.py의 동작 이름과 같아야 합니다. 모델 출력 번호 = 이 목록의 순서입니다.
LABELS = ['thumb', 'v', 'cook', 'idle', 'palm', 'fist', 'point']  # point = 검지만 펴기 (커서 이동)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
FEATURES = [f'{a}{i}' for i in range(21) for a in 'xyz']  # x0,y0,z0 ... x20,y20,z20 (63개)


def load_data():
    """data/*.csv를 전부 읽어서 하나로 합칩니다. 세션 이름 = 파일 이름(확장자 제외)입니다."""
    files = sorted(glob.glob('data/*.csv'))
    if not files:
        raise SystemExit('data/ 폴더에 CSV가 없습니다. 먼저 collect.py로 데이터를 모으세요.')
    # CSV 안의 session 값은 파일 이름을 바꾸기 전 이름일 수 있으므로 파일 이름으로 덮어씁니다.
    df = pd.concat([pd.read_csv(f).assign(session=os.path.splitext(os.path.basename(f))[0]) for f in files],
                   ignore_index=True)
    return df[df['label'].isin(LABELS)].reset_index(drop=True)


def split_data(df, val_sessions, test_sessions):
    """train / val / test로 나눕니다.
    세션 이름을 지정하면 세션 단위로, 지정하지 않으면 세션 안에서 시간 순서대로 70/15/15 나눕니다."""
    if val_sessions and not test_sessions:
        raise SystemExit('--val 만 지정할 수 없습니다. --test 를 같이 지정하세요.')
    known = set(df['session'])
    for s in val_sessions + test_sessions:
        if s not in known:
            raise SystemExit(f'없는 세션 이름: {s}  (있는 세션: {sorted(known)})')
    if test_sessions and not val_sessions:
        # --test 만 지정: 나머지 세션은 전부 학습에 쓰고, 각 세션의 마지막 15%(시간 순서)만 검증용으로 뗍니다.
        test = df[df['session'].isin(test_sessions)]
        parts = {'train': [], 'val': []}
        for _, g in df[~df['session'].isin(test_sessions)].groupby(['session', 'label'], sort=False):
            a = int(len(g) * 0.85)
            parts['train'].append(g.iloc[:a])
            parts['val'].append(g.iloc[a:])
        return pd.concat(parts['train']), pd.concat(parts['val']), test, '--test 세션만 시험용, 나머지는 학습 (각 세션 마지막 15%는 검증용)'
    if val_sessions or test_sessions:
        val = df[df['session'].isin(val_sessions)]
        test = df[df['session'].isin(test_sessions)]
        train = df[~df['session'].isin(val_sessions + test_sessions)]
        return train, val, test, '세션 단위 분할'
    parts = {'train': [], 'val': [], 'test': []}
    for _, g in df.groupby(['session', 'label'], sort=False):
        n = len(g)
        a, b = int(n * 0.70), int(n * 0.85)
        parts['train'].append(g.iloc[:a])
        parts['val'].append(g.iloc[a:b])
        parts['test'].append(g.iloc[b:])
    train, val, test = (pd.concat(parts[k]) for k in ('train', 'val', 'test'))
    return train, val, test, '시간 순서 분할 (임시 방식: 세션이 충분히 모이면 --val/--test 로 세션 단위 분할)'


def to_xy(d, label_to_id=LABEL_TO_ID):
    x = torch.tensor(d[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(d['label'].map(label_to_id).values, dtype=torch.long)
    return x, y


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def report(y_true, y_pred):
    cm = confusion(y_true, y_pred, len(LABELS))
    f1s = []
    print('\n혼동 행렬 (행 = 정답, 열 = 모델 예측)')
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print('\n동작별 성능')
    print(f'{"동작":<8}{"정밀도":>8}{"재현율":>8}{"F1":>8}{"개수":>8}')
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        print(f'{name:<8}{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}{cm[i, :].sum():>8d}')
    return np.trace(cm) / cm.sum(), float(np.mean(f1s))  # (6클래스 정확도, macro F1)


def group_report(y_true, y_pred):
    """cook과 idle을 '비명령(none)' 하나로 묶어서, 명령 기준으로 다시 평가합니다."""
    names = COMMANDS + ['none']
    none_id = len(COMMANDS)

    def to_group(ids):
        return np.array([names.index(LABELS[i]) if LABELS[i] in COMMANDS else none_id for i in ids])

    cm = confusion(to_group(y_true), to_group(y_pred), len(names))
    print('\n명령 기준 혼동 행렬 (cook + idle = none)')
    print(pd.DataFrame(cm, index=names, columns=names))
    print(f'명령 기준 정확도: {np.trace(cm) / cm.sum():.3f}')
    none_total = cm[none_id].sum()
    cmd_total = cm[:none_id].sum()
    if none_total:
        print(f'비명령 중 명령으로 실행된 비율(오작동): {cm[none_id, :none_id].sum() / none_total:.1%}')
    if cmd_total:
        print(f'명령 중 비명령으로 놓친 비율(누락): {cm[:none_id, none_id].sum() / cmd_total:.1%}')
        wrong = cm[:none_id, :none_id].sum() - np.trace(cm[:none_id, :none_id])
        print(f'명령 중 다른 명령으로 잘못 실행된 비율: {wrong / cmd_total:.1%}')
    return np.trace(cm) / cm.sum()  # 명령 기준 정확도


def evaluate(model, loss_fn, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        loss = loss_fn(out, y).item()
        pred = out.argmax(dim=1)
    return loss, (pred == y).float().mean().item(), pred


COMMANDS = ['thumb', 'v', 'point', 'fist']  # 실제로 키/마우스를 움직이는 동작 (palm은 이제 비명령)


def predict_probs(model, x):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).numpy()


def augment(x, noise, max_deg):
    """학습할 때만 좌표를 살짝 흔듭니다: 화면 평면에서 조금 돌리고, 작은 잡음을 더합니다."""
    b = x.shape[0]
    pts = x.view(b, 21, 3)
    theta = (torch.rand(b) * 2 - 1) * max_deg * math.pi / 180
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    nx = c * pts[:, :, 0] - s * pts[:, :, 1]
    ny = s * pts[:, :, 0] + c * pts[:, :, 1]
    pts = torch.stack([nx, ny, pts[:, :, 2]], dim=2)
    pts = pts + torch.randn_like(pts) * noise
    pts[:, 0, :] = 0.0  # 손목(0번)은 항상 원점이므로 그대로 둡니다
    return pts.reshape(b, 63)


def gate_report(probs, y_true, title='신뢰도 게이트 효과'):
    """신뢰도(확률)가 기준 이상일 때만 명령으로 인정하면 오작동이 얼마나 줄어드는지 봅니다."""
    cmd_ids = [LABEL_TO_ID[c] for c in COMMANDS]
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    is_cmd = np.isin(y_true, cmd_ids)
    is_cook = y_true == LABEL_TO_ID['cook']
    print(f'\n{title} (기준 이상일 때만 명령으로 인정)')
    print(f'{"기준":>6}{"cook 오작동":>14}{"비명령 오작동":>14}{"명령 인식률":>14}')
    rows = {}
    for th in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        fire = np.isin(pred, cmd_ids) & (conf >= th)
        cook_mis = fire[is_cook].mean() if is_cook.any() else 0.0
        non_mis = fire[~is_cmd].mean() if (~is_cmd).any() else 0.0
        ok = (fire & (pred == y_true))[is_cmd].mean() if is_cmd.any() else 0.0
        rows[th] = (cook_mis, non_mis, ok)
        print(f'{th:>6.2f}{cook_mis:>13.1%}{non_mis:>14.1%}{ok:>14.1%}')
    print('cook 오작동 = 요리 동작 프레임 중 명령으로 실행된 비율 (낮을수록 좋음)')
    print('비명령 오작동 = cook + idle 프레임 중 명령으로 실행된 비율 (낮을수록 좋음)')
    print('명령 인식률 = 진짜 명령 프레임 중 올바른 명령으로 실행된 비율 (높을수록 좋음)')
    return rows  # {기준: (cook 오작동, 비명령 오작동, 명령 인식률)}


RESULTS_PATH = 'results.csv'
RESULTS_HEADER = ['모델이름', 'test세션', '6클래스정확도', '명령기준정확도', 'macro F1',
                  '게이트0.0_cook오작동', '게이트0.8_cook오작동', '게이트0.8_명령인식률']


def append_result(name, test_sessions, acc6, cmd_acc, macro_f1, gates):
    """results.csv에 결과 한 줄을 덧붙입니다. 값은 0~1 사이 비율이고, 해당 없는 칸(None)은 비워 둡니다."""
    values = [acc6, cmd_acc, macro_f1, gates[0.0][0], gates[0.8][0], gates[0.8][2]]
    new_file = not os.path.exists(RESULTS_PATH) or os.path.getsize(RESULTS_PATH) == 0
    # utf-8-sig: 엑셀에서 한글 제목이 깨지지 않게 합니다.
    with open(RESULTS_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(RESULTS_HEADER)
        writer.writerow([name, '+'.join(test_sessions) or '시간순분할']
                        + ['' if v is None else f'{v:.4f}' for v in values])
    print(f'{RESULTS_PATH}에 결과 추가: {name}')


def command_only_report(probs, y_true):
    """명령 4개만 배운 모델을 6개 동작 전체로 시험합니다. y_true는 LABELS 번호입니다.
    확률을 LABELS 순서의 6칸으로 옮기면(cook, idle 칸은 0) 기존 평가 함수를 그대로 쓸 수 있습니다."""
    full = np.zeros((len(probs), len(LABELS)), dtype=probs.dtype)
    full[:, [LABEL_TO_ID[c] for c in COMMANDS]] = probs
    pred = full.argmax(axis=1)
    cm = confusion(y_true, pred, len(LABELS))
    print('\n혼동 행렬 (행 = 정답 6개 동작, 열 = 모델 예측 4개 명령)')
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS)[COMMANDS])
    return full, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', nargs='*', default=[], help='검증용 세션 이름들 (예: yunho_2)')
    parser.add_argument('--test', nargs='*', default=[], help='시험용 세션 이름들 (예: gaon_1)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.2, help='드롭아웃 비율 (예: 0.3)')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률 (예: 5e-4)')
    parser.add_argument('--wd', type=float, default=0.0, help='가중치 감쇠 (예: 1e-4)')
    parser.add_argument('--smooth', type=float, default=0.0, help='라벨 스무딩 (예: 0.1)')
    parser.add_argument('--aug', action='store_true', help='학습 데이터 증강 켜기')
    parser.add_argument('--cook-weight', type=float, default=1.0, help='cook 가중치 배수 (예: 1.5)')
    parser.add_argument('--only-commands', action='store_true',
                        help='cook, idle을 빼고 명령 4개만 학습 (model_cmd_only.pt로 저장)')
    args = parser.parse_args()

    torch.manual_seed(0)
    df = load_data()
    train, val, test, mode = split_data(df, args.val, args.test)
    print('분할 방식:', mode)
    print('\n동작별 개수')
    print(pd.DataFrame({'train': train['label'].value_counts(),
                        'val': val['label'].value_counts(),
                        'test': test['label'].value_counts()}).reindex(LABELS).fillna(0).astype(int))

    # 모델이 구분하는 동작 목록. --only-commands면 학습/검증에서 cook, idle을 뺍니다. 시험 세트는 항상 전체입니다.
    model_labels = COMMANDS if args.only_commands else LABELS
    model_label_to_id = {name: i for i, name in enumerate(model_labels)}
    if args.only_commands:
        train = train[train['label'].isin(COMMANDS)]
        val = val[val['label'].isin(COMMANDS)]
        print(f'\n--only-commands: 학습/검증에서 cook, idle 제외 (학습 {len(train)}개, 검증 {len(val)}개)')

    x_tr, y_tr = to_xy(train, model_label_to_id)
    x_va, y_va = to_xy(val, model_label_to_id)
    x_te, y_te = to_xy(test)  # 시험 정답은 항상 6개 동작 번호

    # 동작마다 개수가 다르므로, 적은 동작에 더 큰 가중치를 줍니다.
    n_out = len(model_labels)
    counts = np.bincount(y_tr.numpy(), minlength=n_out)
    weights = np.where(counts > 0, counts.sum() / (n_out * np.maximum(counts, 1)), 0.0)
    if 'cook' in model_label_to_id:
        weights[model_label_to_id['cook']] *= args.cook_weight
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32),
                                  label_smoothing=args.smooth)

    model = nn.Sequential(
        nn.Linear(63, 64), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(32, n_out),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f'\n설정: dropout={args.dropout} lr={args.lr} wd={args.wd} smooth={args.smooth} '
          f'aug={args.aug} cook-weight={args.cook_weight}')

    best_val, best_state, wait, patience = float('inf'), None, 0, 20
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 64):
            idx = order[i:i + 64]
            xb = x_tr[idx]
            if args.aug:
                xb = augment(xb, noise=0.02, max_deg=15)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), y_tr[idx])
            loss.backward()
            optimizer.step()
        tr_loss, tr_acc, _ = evaluate(model, loss_fn, x_tr, y_tr)
        va_loss, va_acc, _ = evaluate(model, loss_fn, x_va, y_va)
        if epoch % 10 == 0 or epoch == 1:
            print(f'epoch {epoch:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}')
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f'epoch {epoch}: 검증 손실이 {patience}번 연속 좋아지지 않아 멈춥니다.')
                break

    model.load_state_dict(best_state)
    if args.only_commands:
        name, path = 'mlp_cmd_only', 'model_cmd_only.pt'
        probs, pred = command_only_report(predict_probs(model, x_te), y_te.numpy())
        print(f'\n시험 {len(x_te)}개 (cook, idle 프레임의 정답 = 비명령. 이 모델은 비명령을 출력할 수 없습니다)')
        acc6, macro_f1 = None, None  # 4개만 출력하는 모델이라 6클래스 지표는 비교할 수 없습니다
        cmd_acc = group_report(y_te.numpy(), pred)
    else:
        name, path = 'mlp', 'model.pt'
        _, te_acc, pred = evaluate(model, loss_fn, x_te, y_te)
        print(f'\n최종 test 정확도: {te_acc:.3f}  (test {len(x_te)}개)')
        acc6, macro_f1 = report(y_te.numpy(), pred.numpy())
        cmd_acc = group_report(y_te.numpy(), pred.numpy())
        probs = predict_probs(model, x_te)
    gates = gate_report(probs, y_te.numpy(), '[시험 세트] 신뢰도 게이트 효과')
    # 설정을 비교할 때는 시험 세트가 아니라 이 검증 세트의 표를 보고 고릅니다.
    # (--only-commands는 검증 세트에 cook, idle이 없어 오작동을 잴 수 없으므로 생략)
    if args.val and not args.only_commands:
        gate_report(predict_probs(model, x_va), y_va.numpy(), '[검증 세트] 신뢰도 게이트 효과')

    torch.save({'state_dict': best_state, 'labels': model_labels}, path)
    print(f'\n{path} 저장 완료')
    append_result(name, args.test, acc6, cmd_acc, macro_f1, gates)


if __name__ == '__main__':
    main()
