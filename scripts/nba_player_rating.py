"""Player-level game rating (RAPM-style): each player's impact per minute on the margin, fit by ridge regression on
past games (recent games weighted more), then a game's prediction = home players' ratings x their usual minutes
minus the away side's, plus home court. Walk-forward by month. Used as one input to scripts/nba_games_v3.py."""
import pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.linear_model import Ridge

def rapm(g, alpha=10.0):
    pg=pd.read_csv("data/nba/player_games.csv",dtype={"game_id":str,"player_id":str})
    for c in ("min","dnp","home"): pg[c]=pd.to_numeric(pg[c],errors="coerce").fillna(0)
    pg["date"]=pd.to_datetime(pg.date); pg=pg.sort_values(["player_id","date"])
    played=(pg.dnp==0)&(pg["min"]>0)
    pg["mp"]=pg["min"].where(played)
    pg["usual_min"]=pg.groupby("player_id").mp.transform(lambda x:x.shift(1).ewm(halflife=8,min_periods=1).mean())
    g=g.copy(); g["id"]=g["id"].astype(str); g["date"]=pd.to_datetime(g.date); g["month"]=g.date.dt.to_period("M")
    on=pg[played].copy()
    players=sorted(on.player_id.unique()); pidx={p:i for i,p in enumerate(players)}; n=len(players)
    gid=g.id.tolist(); gi={x:i for i,x in enumerate(gid)}
    on=on[on.game_id.isin(gi)]
    # actual minutes design (fit) and expected minutes design (predict)
    def design(col):
        rows=[];cols=[];vals=[]
        for r in on.itertuples():
            v=getattr(r,col)
            if not np.isfinite(v): continue
            rows.append(gi[r.game_id]); cols.append(pidx[r.player_id]); vals.append((1 if r.home==1 else -1)*v)
        M=sp.csr_matrix((vals,(rows,cols)),shape=(len(gid),n))
        tot=np.abs(M).sum(1).A1/2  # minutes per team approx
        return M
    A=design("min"); E=design("usual_min")
    # normalize each team's minutes to 240 (rows: home positive, away negative)
    def norm(M):
        M=M.tocoo(); df=pd.DataFrame({"r":M.row,"c":M.col,"v":M.data}); df["s"]=np.sign(df.v)
        tot=df.groupby(["r","s"]).v.transform(lambda x:x.abs().sum())
        df["v"]=df.v/tot*5  # 5 players on court
        return sp.csr_matrix((df.v,(df.r,df.c)),shape=M.shape)
    A=norm(A); E=norm(E)
    hca=np.ones((len(gid),1))
    y=g.margin.values.astype(float)
    out=np.full(len(gid),np.nan)
    for m in sorted(g.month.unique()):
        tr=(g.month<m).values; te=(g.month==m).values
        if tr.sum()<300: continue
        # weight recent games more (halflife ~ 120 days)
        age=(g.date[te].min()-g.date[tr]).dt.days.values
        w=0.5**(age/150)
        X=sp.hstack([A[tr],sp.csr_matrix(hca[tr])]).tocsr()
        r=Ridge(alpha=alpha,fit_intercept=False).fit(X,y[tr],sample_weight=w)
        out[te]=sp.hstack([E[te],sp.csr_matrix(hca[te])]).tocsr()@r.coef_
    return out
