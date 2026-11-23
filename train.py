import argparse
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# collect.py의 동작 이름과 같아야 합니다. 모델 출력 번호 = 이 목록의 순서입니다.
LABELS = ['thumb', 'v', 'cook', 'idle', 'palm', 'fist']
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
FEATURES = [f'{a}{i}' for i in range(21) for a in 'xyz']  # x0,y0,z0 ... x20,y20,z20 (63개)


def load_data():
    """data/*.csv를 전부 읽어서 하나로 합칩니다."""
    files = sorted(glob.glob('data/*.csv'))
    if not files:
        raise SystemExit('data/ 폴더에 CSV가 없습니다. 먼저 collect.py로 데이터를 모으세요.')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df['label'].isin(LABELS)].reset_index(drop=True)


def split_data(df, val_sessions, test_sessions):
    """train / val / test로 나눕니다.
    세션 이름을 지정하면 세션 단위로, 지정하지 않으면 세션 안에서 시간 순서대로 70/15/15 나눕니다."""
    if val_sessions or test_sessions:
        if not (val_sessions and test_sessions):
            raise SystemExit('--val 과 --test 를 둘 다 지정하세요.')
        known = set(df['session'])
        for s in val_sessions + test_sessions:
            if s not in known:
                raise SystemExit(f'없는 세션 이름: {s}  (있는 세션: {sorted(known)})')
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


def to_xy(d):
    x = torch.tensor(d[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(d['label'].map(LABEL_TO_ID).values, dtype=torch.long)
    return x, y


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def report(y_true, y_pred):
    cm = confusion(y_true, y_pred, len(LABELS))
    print('\n혼동 행렬 (행 = 정답, 열 = 모델 예측)')
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print('\n동작별 성능')
    print(f'{"동작":<8}{"정밀도":>8}{"재현율":>8}{"F1":>8}{"개수":>8}')
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f'{name:<8}{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}{cm[i, :].sum():>8d}')


def evaluate(model, loss_fn, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        loss = loss_fn(out, y).item()
        pred = out.argmax(dim=1)
    return loss, (pred == y).float().mean().item(), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', nargs='*', default=[], help='검증용 세션 이름들 (예: yunho_2)')
    parser.add_argument('--test', nargs='*', default=[], help='시험용 세션 이름들 (예: gaon_1)')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(0)
    df = load_data()
    train, val, test, mode = split_data(df, args.val, args.test)
    print('분할 방식:', mode)
    print('\n동작별 개수')
    print(pd.DataFrame({'train': train['label'].value_counts(),
                        'val': val['label'].value_counts(),
                        'test': test['label'].value_counts()}).reindex(LABELS).fillna(0).astype(int))

    x_tr, y_tr = to_xy(train)
    x_va, y_va = to_xy(val)
    x_te, y_te = to_xy(test)

    # 동작마다 개수가 다르므로, 적은 동작에 더 큰 가중치를 줍니다.
    counts = np.bincount(y_tr.numpy(), minlength=len(LABELS))
    weights = np.where(counts > 0, counts.sum() / (len(LABELS) * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = nn.Sequential(
        nn.Linear(63, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, len(LABELS)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state, wait, patience = float('inf'), None, 0, 20
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 64):
            idx = order[i:i + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
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
    _, te_acc, pred = evaluate(model, loss_fn, x_te, y_te)
    print(f'\n최종 test 정확도: {te_acc:.3f}  (test {len(x_te)}개)')
    report(y_te.numpy(), pred.numpy())

    torch.save({'state_dict': best_state, 'labels': LABELS}, 'model.pt')
    print('\nmodel.pt 저장 완료')


if __name__ == '__main__':
    main()
