"""train.py의 신경망과 비교할 기준 모델(로지스틱 회귀, 랜덤 포레스트)을 학습하고 평가합니다."""
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from train import (FEATURES, LABEL_TO_ID, LABELS, append_result, gate_report, group_report,
                   load_data, report, split_data)


def to_numpy(d):
    return d[FEATURES].values.astype(np.float32), d['label'].map(LABEL_TO_ID).values.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', nargs='*', default=[], help='검증용 세션 이름들 (train.py와 같은 뜻)')
    parser.add_argument('--test', nargs='*', default=[], help='시험용 세션 이름들 (예: test_1 test_2)')
    args = parser.parse_args()

    df = load_data()
    train, val, test, mode = split_data(df, args.val, args.test)
    # train.py의 신경망이 가중치를 배우는 것과 똑같은 train 부분만 씁니다 (val 부분은 쓰지 않음).
    x_tr, y_tr = to_numpy(train)
    x_te, y_te = to_numpy(test)
    print('분할 방식:', mode)
    print(f'학습 {len(x_tr)}개, 시험 {len(x_te)}개')

    models = [
        ('logreg', 'LogisticRegression', LogisticRegression(max_iter=3000)),
        ('random_forest', 'RandomForestClassifier', RandomForestClassifier(n_estimators=200, random_state=0)),
    ]
    for name, title, model in models:
        print(f'\n{"=" * 20} {title} {"=" * 20}')
        model.fit(x_tr, y_tr)
        # predict_proba의 열 순서(model.classes_)를 LABELS 순서의 6칸에 맞춥니다.
        probs = np.zeros((len(x_te), len(LABELS)))
        probs[:, model.classes_] = model.predict_proba(x_te)
        pred = probs.argmax(axis=1)

        print(f'\n6클래스 test 정확도: {(pred == y_te).mean():.3f}  (test {len(x_te)}개)')
        acc6, macro_f1 = report(y_te, pred)
        cmd_acc = group_report(y_te, pred)
        gates = gate_report(probs, y_te, f'[시험 세트] {title} 신뢰도 게이트 효과')
        append_result(name, args.test, acc6, cmd_acc, macro_f1, gates)


if __name__ == '__main__':
    main()
