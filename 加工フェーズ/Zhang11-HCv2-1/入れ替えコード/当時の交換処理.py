"""探索時の replay() を収録したもの。ここから探索・採点は起動しない。

出典: 実験記録/05_約15時間の探索/分析.py の replay()。
replay() の本文は当時のコードのまま。
original() は、当時の候補名・計画ファイルからの読み込みを、
指定したCSVからの読み込みに置き換えた共有用の補助関数。
"""
import pandas as pd


def original(base):
    return pd.read_csv(base, float_precision='round_trip')


def replay(base,ops):
    c=original(base).copy()
    for op in ops:
        attr,i,j=op['attribute'],op['i'],op['j']
        if attr=='time':
            assert tuple(c.loc[i,['onset','death']])==tuple(c.loc[j,['onset','death']])!=(0,0)
        c.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
    return c
