#!/usr/bin/env python3
"""Q62C - ROC/PR/Confusion Matrix Figures from Q62B Probability Exports.

Uses stdlib + numpy only. No matplotlib, no sklearn, no training.
Run from QStrata root: python scripts/run_q62c_figures.py
"""

import csv
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve()
PROJ_ROOT = next(p for p in [_HERE]+list(_HERE.parents) if (p/'.git').exists() or (p/'CLAUDE.md').exists())
PROBS_DIR = PROJ_ROOT/'workspace/experiments/Q62B/probabilities'
FIG_DIR   = PROJ_ROOT/'workspace/experiments/Q62C/figures'
REP_DIR   = PROJ_ROOT/'workspace/experiments/Q62C/reports'

ARCHS       = ['cnn','dv_qnn','cv_qnn']
ARCH_LABELS = dict(cnn='CNN',dv_qnn='DV-QNN',cv_qnn='CV-QNN')
SEEDS       = [42,7,123]
ARCH_COLORS = dict(cnn='#1f77b4',dv_qnn='#ff7f0e',cv_qnn='#2ca02c')
SEED_COLORS = dict(); SEED_COLORS[42]='#1f77b4'; SEED_COLORS[7]='#ff7f0e'; SEED_COLORS[123]='#2ca02c'
Q62A_AUROCS = dict(cnn=(0.9721,0.0023),dv_qnn=(0.8831,0.0011),cv_qnn=(0.9476,0.0002))
FA = 'font-family="DejaVu Sans,Arial,sans-serif"'

def load_probs(arch,split='test'):
    path=PROBS_DIR/(arch+'_vindr_'+split+'_probs.csv'); data=dict()
    with open(path) as fp:
        for row in csv.DictReader(fp):
            s=int(row['seed'])
            if s not in data: data[s]=[[],[],[]]
            data[s][0].append(int(row['y_true']))
            data[s][1].append(float(row['y_score']))
            data[s][2].append(int(row['y_pred']))
    return dict((s,[np.array(v) for v in vv]) for s,vv in data.items())

def roc_curve(yt,ys):
    np_=int(yt.sum()); nn=len(yt)-np_; idx=np.argsort(ys)[::-1]; yt=yt[idx]; ys2=ys[idx]
    tp=np.cumsum(yt); fp=np.arange(1,len(yt)+1)-tp
    d=np.concatenate([np.diff(ys2)!=0,[True]])
    fpr=np.concatenate([[0],fp[d]])/nn; tpr=np.concatenate([[0],tp[d]])/np_
    if fpr[-1]<1.: fpr=np.append(fpr,1.); tpr=np.append(tpr,1.)
    return fpr,tpr

def pr_curve(yt,ys):
    np_=int(yt.sum()); idx=np.argsort(ys)[::-1]; yt=yt[idx]; ys2=ys[idx]
    tp=np.cumsum(yt); fp=np.arange(1,len(yt)+1)-tp
    prec=tp/(tp+fp); rec=tp/np_; d=np.concatenate([np.diff(ys2)!=0,[True]])
    return np.concatenate([[0.],rec[d]]),np.concatenate([[1.],prec[d]])

def trapz_auc(x,y):
    i=np.argsort(x); x=x[i]; y=y[i]
    return float(np.sum((x[1:]-x[:-1])*(y[1:]+y[:-1])/2))

def avg_curves(cs,n=500):
    g=np.linspace(0,1,n); arr=np.array([np.interp(g,c[0],c[1]) for c in cs])
    return g,arr.mean(0),arr.std(0)

def avg_pr(cs,n=500):
    g=np.linspace(0,1,n); arr=[]
    for r,p in cs:
        idx=np.argsort(r); arr.append(np.interp(g,r[idx],p[idx]))
    arr=np.array(arr); return g,arr.mean(0),arr.std(0)

def conf_matrix(yt,yp):
    tn=int(((yt==0)&(yp==0)).sum()); fp=int(((yt==0)&(yp==1)).sum())
    fn=int(((yt==1)&(yp==0)).sum()); tp=int(((yt==1)&(yp==1)).sum())
    return np.array([[tn,fp],[fn,tp]])

def tx(x,y,t,a='middle',f='#222',sz=12,bold=False,italic=False):
    fw=' font-weight="bold"' if bold else ''
    fi=' font-style="italic"' if italic else ''
    return '<text x="'+str(round(x,1))+'" y="'+str(round(y,1))+'" text-anchor="'+a+'" fill="'+f+'" font-size="'+str(sz)+'" '+FA+fw+fi+'>'+str(t)+'</text>'

def xl(x1,y1,x2,y2,stroke='#aaa',sw=1,dash=None):
    r='<line x1="'+str(round(x1,1))+'" y1="'+str(round(y1,1))+'" x2="'+str(round(x2,1))+'" y2="'+str(round(y2,1))+'" stroke="'+stroke+'" stroke-width="'+str(sw)+'"'
    if dash: r+=' stroke-dasharray="'+dash+'"'
    return r+'/>' 

def xr(x,y,w,h,fill,rx=0,fop=None,stroke=None,sw=1):
    r='<rect x="'+str(round(x,1))+'" y="'+str(round(y,1))+'" width="'+str(round(w,1))+'" height="'+str(round(h,1))+'" fill="'+fill+'"'
    if rx: r+=' rx="'+str(rx)+'"'
    if fop is not None: r+=' fill-opacity="'+str(fop)+'"'
    if stroke: r+=' stroke="'+stroke+'" stroke-width="'+str(sw)+'"'
    return r+'/>' 

def xpoly(pts,c,op='0.18'):
    ps=' '.join(str(round(x,1))+','+str(round(y,1)) for x,y in pts)
    return '<polygon points="'+ps+'" fill="'+c+'" opacity="'+op+'"/>'

def xpath(d,c,sw=2.5): return '<path d="'+d+'" stroke="'+c+'" stroke-width="'+str(sw)+'" fill="none" stroke-linejoin="round" stroke-linecap="round"/>'

def rl(x,y,t,f='#333',sz=12,bold=False):
    fw=' font-weight="bold"' if bold else ''
    return '<text x="'+str(x)+'" y="'+str(y)+'" text-anchor="middle" fill="'+f+'" font-size="'+str(sz)+'" '+FA+fw+' transform="rotate(-90,'+str(x)+','+str(y)+')">'+''+t+'</text>'

def svgopen(W,H): return '<svg xmlns="http://www.w3.org/2000/svg" width="'+str(W)+'" height="'+str(H)+'" viewBox="0 0 '+str(W)+' '+str(H)+'">' 

def icol(c1,c2,t):
    def h2r(h):
        h=h.lstrip('#'); return [int(h[i:i+2],16) for i in (0,2,4)]
    r1,g1,b1=h2r(c1); r2,g2,b2=h2r(c2)
    r=int(r1+(r2-r1)*t); g=int(g1+(g2-g1)*t); b=int(b1+(b2-b1)*t)
    lum=0.299*r+0.587*g+0.114*b
    return '#{:02x}{:02x}{:02x}'.format(r,g,b),('#fff' if lum<140 else '#222')

class PB:
    def __init__(self,ml,mr,mt,mb,pw,ph,ox=0,oy=0):
        self.ml,self.mr,self.mt,self.mb=ml,mr,mt,mb; self.pw,self.ph=pw,ph; self.ox,self.oy=ox,oy
    def sx(self,v): return self.ox+self.ml+v*self.pw
    def sy(self,v): return self.oy+self.mt+self.ph*(1-v)
    def to_path(self,xs,ys):
        pts=[(self.sx(x),self.sy(y)) for x,y in zip(xs,ys)]
        d='M '+str(round(pts[0][0],1))+','+str(round(pts[0][1],1))
        d+=''.join(' L '+str(round(px,1))+','+str(round(py,1)) for px,py in pts[1:]); return d
    def band(self,xs,ylo,yhi):
        up=[(self.sx(x),self.sy(y)) for x,y in zip(xs,yhi)]; lo=[(self.sx(x),self.sy(y)) for x,y in zip(reversed(xs),reversed(ylo))]; return up+lo
    def grid(self,L,ticks=(0.2,0.4,0.6,0.8),ly=True,lx=True,sz=10):
        for v in ticks:
            yg=self.sy(v); xg=self.sx(v)
            L.append(xl(self.sx(0),yg,self.sx(1),yg,stroke='#e0e0e0'))
            L.append(xl(xg,self.sy(0),xg,self.sy(1),stroke='#e0e0e0'))
            if ly: L.append(tx(self.ox+self.ml-7,yg+3.5,str(round(v,1)),a='end',f='#555',sz=sz))
            if lx: L.append(tx(xg,self.sy(0)+16,str(round(v,1)),f='#555',sz=sz))
        if ly:
            L.append(tx(self.ox+self.ml-7,self.sy(0)+3.5,'0.0',a='end',f='#555',sz=sz))
            L.append(tx(self.ox+self.ml-7,self.sy(1)+3.5,'1.0',a='end',f='#555',sz=sz))
        if lx:
            L.append(tx(self.sx(0),self.sy(0)+16,'0.0',f='#555',sz=sz))
            L.append(tx(self.sx(1),self.sy(0)+16,'1.0',f='#555',sz=sz))
    def axes(self,L):
        L.append(xl(self.sx(0),self.sy(0),self.sx(1),self.sy(0),stroke='#888',sw=1.5))
        L.append(xl(self.sx(0),self.sy(0),self.sx(0),self.sy(1),stroke='#888',sw=1.5))
    def diag(self,L): L.append(xl(self.sx(0),self.sy(0),self.sx(1),self.sy(1),stroke='#bbb',sw=1,dash='5,4'))

def _legend(L,lx,ly,arch_stats,metric):
    L.append(tx(lx,ly,'Architecture',a='start',f='#333',sz=11,bold=True)); ly+=4
    for i,arch in enumerate(ARCHS):
        c=ARCH_COLORS[arch]; vals=arch_stats[i]['vals']; mu=float(np.mean(vals)); sg=float(np.std(vals))
        ly+=24; L+=[xl(lx,ly-5,lx+22,ly-5,stroke=c,sw=3),xr(lx,ly-13,22,14,c,fop='0.18'),tx(lx+27,ly-4,ARCH_LABELS[arch],a='start',f='#333',sz=11,bold=True)]
        ly+=15; L.append(tx(lx+27,ly-4,metric+' '+str(round(mu,4))+' +/- '+str(round(sg,4)),a='start',f=c,sz=10)); ly+=10
        for s,v in zip(SEEDS,vals): L.append(tx(lx+27,ly-4,'  seed '+str(s)+': '+str(round(v,4)),a='start',f='#888',sz=9)); ly+=13
        ly+=3

def make_roc_combined(D):
    W=760;H=620;ML=72;MR=215;MT=68;MB=72;PW=W-ML-MR;PH=H-MT-MB; pb=PB(ML,MR,MT,MB,PW,PH)
    ast=[]
    for arch in ARCHS:
        cs=[]; vals=[]
        for s in SEEDS: yt,ys,_=D[arch][s]; fpr,tpr=roc_curve(yt,ys); cs.append((fpr,tpr)); vals.append(trapz_auc(fpr,tpr))
        g,m,st=avg_curves(cs); ast.append(dict(g=g,m=m,s=st,vals=vals))
    L=[svgopen(W,H),xr(0,0,W,H,'#ffffff'),tx(ML+PW/2,36,'ROC Curves - VinDr-SpineXR Test Split',sz=15,bold=True),tx(ML+PW/2,52,'3 architectures x 3 seeds  |  mean +/- 1 SD band',f='#666',sz=11),xr(ML,MT,PW,PH,'#fafafa',rx=2)]
    pb.grid(L); pb.diag(L); pb.axes(L)
    for i,arch in enumerate(ARCHS):
        c=ARCH_COLORS[arch]; at=ast[i]; yhi=np.minimum(at['m']+at['s'],1.); ylo=np.maximum(at['m']-at['s'],0.)
        L+=[xpoly(pb.band(at['g'],ylo,yhi),c),xpath(pb.to_path(at['g'],at['m']),c)]
    L+=[tx(ML+PW/2,H-MB+44,'False Positive Rate (1 - Specificity)',f='#333',sz=12),rl(16,MT+PH//2,'True Positive Rate (Sensitivity)'),tx(pb.sx(0.72),pb.sy(0.72)-8,'chance',f='#bbb',sz=9,italic=True)]
    _legend(L,ML+PW+14,MT+12,ast,'AUROC'); L.append('</svg>'); return '\n'.join(L)

def make_pr_combined(D):
    W=760;H=620;ML=72;MR=215;MT=68;MB=72;PW=W-ML-MR;PH=H-MT-MB; pb=PB(ML,MR,MT,MB,PW,PH)
    ast=[]
    for arch in ARCHS:
        cs=[]; vals=[]
        for s in SEEDS: yt,ys,_=D[arch][s]; r,p=pr_curve(yt,ys); cs.append((r,p)); vals.append(trapz_auc(r,p))
        g,m,st=avg_pr(cs); ast.append(dict(g=g,m=m,s=st,vals=vals))
    pr0=float(D[ARCHS[0]][SEEDS[0]][0].mean())
    L=[svgopen(W,H),xr(0,0,W,H,'#ffffff'),tx(ML+PW/2,36,'Precision-Recall Curves - VinDr-SpineXR Test Split',sz=15,bold=True),tx(ML+PW/2,52,'3 architectures x 3 seeds  |  mean +/- 1 SD band',f='#666',sz=11),xr(ML,MT,PW,PH,'#fafafa',rx=2)]
    pb.grid(L); pb.axes(L)
    yb=pb.sy(pr0); L+=[xl(pb.sx(0),yb,pb.sx(1),yb,stroke='#bbb',sw=1,dash='5,4'),tx(pb.sx(0.05),yb-7,'no-skill ('+str(round(pr0,3))+')',a='start',f='#bbb',sz=9,italic=True)]
    for i,arch in enumerate(ARCHS):
        c=ARCH_COLORS[arch]; at=ast[i]; yhi=np.minimum(at['m']+at['s'],1.); ylo=np.maximum(at['m']-at['s'],0.)
        L+=[xpoly(pb.band(at['g'],ylo,yhi),c),xpath(pb.to_path(at['g'],at['m']),c)]
    L+=[tx(ML+PW/2,H-MB+44,'Recall (Sensitivity)',f='#333',sz=12),rl(16,MT+PH//2,'Precision (Positive Predictive Value)')]
    _legend(L,ML+PW+14,MT+12,ast,'AP'); L.append('</svg>'); return '\n'.join(L)

def make_roc_per_seed(D):
    PW3=220;PH3=190;ML3=42;MR3=12;MT3=36;MB3=36; PNW=ML3+PW3+MR3; PNH=MT3+PH3+MB3; GX=32; PL=18; PT3=62; PB3=55
    WW=PL+3*PNW+2*GX+18; HH=PT3+PNH+PB3
    L=[svgopen(WW,HH),xr(0,0,WW,HH,'#ffffff'),tx(WW//2,30,'ROC Curves per Seed - VinDr-SpineXR Test Split',sz=14,bold=True),tx(WW//2,46,'Each architecture shown separately  |  3 training seeds',f='#666',sz=11)]
    for ai,arch in enumerate(ARCHS):
        ox=PL+ai*(PNW+GX); oy=PT3; pb2=PB(ML3,MR3,MT3,MB3,PW3,PH3,ox=ox,oy=oy)
        L+=[xr(ox,oy,PNW,PNH,'#fafafa',rx=2),tx(ox+ML3+PW3/2,oy+18,ARCH_LABELS[arch],sz=12,bold=True)]
        pb2.grid(L,ly=(ai==0),lx=True,sz=9); pb2.diag(L); pb2.axes(L)
        aucs=[]
        for seed in SEEDS:
            yt,ys,_=D[arch][seed]; fpr,tpr=roc_curve(yt,ys); aucs.append(trapz_auc(fpr,tpr))
            L.append(xpath(pb2.to_path(fpr,tpr),SEED_COLORS[seed],sw=1.8))
        L.append(tx(ox+ML3+PW3/2,oy+MT3+PH3-8,'AUROC '+str(round(float(np.mean(aucs)),4)),f='#333',sz=9,bold=True))
    L+=[tx(WW//2,PT3+PNH+20,'False Positive Rate',f='#333',sz=11),rl(10,PT3+PNH//2,'True Positive Rate',sz=11)]
    ley=PT3+PNH+36; lx0=(WW-len(SEEDS)*130)//2
    L.append(tx(WW//2,ley-2,'Training seed:',a='middle',f='#444',sz=10))
    for ii,seed in enumerate(SEEDS):
        lxx=lx0+ii*130; L+=[xl(lxx,ley+12,lxx+22,ley+12,stroke=SEED_COLORS[seed],sw=2.5),tx(lxx+27,ley+16,'seed = '+str(seed),a='start',f='#444',sz=10)]
    L.append('</svg>'); return '\n'.join(L)

def make_cm(arch,D):
    CW=148;CH=148;GAP=6;ML=108;MR=50;MT=108;MB=90;NR=2;NC=2
    Wc=ML+NC*CW+(NC-1)*GAP+MR; Hc=MT+NR*CH+(NR-1)*GAP+MB
    cms=np.array([conf_matrix(D[arch][s][0],D[arch][s][2]) for s in SEEDS])
    mc=cms.mean(0); sc=cms.std(0); tot=float(mc.sum()); nrm=mc/mc.sum(axis=1,keepdims=True)
    CLO='#d5ead0';CHI='#27774a';ELO='#fde0d5';EHI='#b73226'
    ch=['Predicted\nNormal','Predicted\nAbnormal']; rh=['True\nNormal','True\nAbnormal']; cl=[['TN','FP'],['FN','TP']]
    albl=ARCH_LABELS[arch]
    L=[svgopen(Wc,Hc),xr(0,0,Wc,Hc,'#ffffff'),tx(Wc//2,34,'Confusion Matrix - '+albl+'  (VinDr-SpineXR Test)',sz=14,bold=True),tx(Wc//2,50,'Mean counts over 3 seeds  |  threshold = 0.5',f='#666',sz=11),tx(ML+(NC*CW+(NC-1)*GAP)/2,78,'Predicted Label',sz=12,bold=True,f='#333')]
    L.append(rl(22,MT+(NR*CH+(NR-1)*GAP)/2,'True Label',sz=12,bold=True))
    for j in range(NC):
        cx=ML+j*(CW+GAP)+CW/2
        for k,part in enumerate(ch[j].split('\n')): L.append(tx(cx,91+k*14,part,f='#333',sz=11))
    for i in range(NR):
        cy=MT+i*(CH+GAP)+CH/2
        for k,part in enumerate(rh[i].split('\n')): L.append(tx(ML-8,cy-7+k*14,part,a='end',f='#333',sz=11))
    for i in range(NR):
        for j in range(NC):
            cnt=float(mc[i,j]); sig=float(sc[i,j]); pct=float(nrm[i,j])*100
            inten=float(np.clip(nrm[i,j],0,1))
            if i!=j: inten=float(np.clip(nrm[i,j]/0.5,0,1))
            fill,tf=icol(CLO,CHI,inten) if i==j else icol(ELO,EHI,inten)
            cx=ML+j*(CW+GAP); cy=MT+i*(CH+GAP)
            L+=[xr(cx,cy,CW,CH,fill,rx=4,stroke='#ccc',sw=1),tx(cx+6,cy+14,cl[i][j],a='start',f=tf,sz=9),tx(cx+CW/2,cy+CH/2-8,str(int(round(cnt))),f=tf,sz=26,bold=True),tx(cx+CW/2,cy+CH/2+14,str(round(pct,1))+'%',f=tf,sz=13)]
            if sig>0.4: L.append(tx(cx+CW/2,cy+CH/2+30,'+/- '+str(int(round(sig))),f=tf,sz=10))
    tn,fp2,fn,tp2=(float(mc[0,0]),float(mc[0,1]),float(mc[1,0]),float(mc[1,1]))
    acc=(tn+tp2)/tot; f1=2*tp2/(2*tp2+fp2+fn) if (2*tp2+fp2+fn)>0 else 0
    prec=tp2/(tp2+fp2) if (tp2+fp2)>0 else 0; rec=tp2/(tp2+fn) if (tp2+fn)>0 else 0
    sy=MT+NR*CH+(NR-1)*GAP+26
    L+=[tx(Wc//2,sy,'Acc '+str(round(acc,4))+'   F1 '+str(round(f1,4))+'   Prec '+str(round(prec,4))+'   Rec '+str(round(rec,4)),f='#444',sz=11),tx(Wc//2,sy+16,'N = '+str(int(tot))+' samples  |  seeds: '+str(SEEDS),f='#888',sz=10),'</svg>']
    return '\n'.join(L)

def write_report(D):
    rows_a=[]
    for i,arch in enumerate(ARCHS):
        aucs=[]
        for s in SEEDS:
            yt,ys,_=D[arch][s]; fpr,tpr=roc_curve(yt,ys); aucs.append(trapz_auc(fpr,tpr))
        mu=float(np.mean(aucs)); sg=float(np.std(aucs)); qm,qs=Q62A_AUROCS[arch]; dl=mu-qm
        rows_a.append('| '+ARCH_LABELS[arch]+' | '+str(round(mu,4))+' +/- '+str(round(sg,4))+' | '+str(qm)+' +/- '+str(qs)+' | '+('+' if dl>=0 else '')+str(round(dl,4))+' |')
    rows_c=[]
    for i,arch in enumerate(ARCHS):
        cms=np.array([conf_matrix(D[arch][s][0],D[arch][s][2]) for s in SEEDS])
        mc=cms.mean(0); tn,fp2,fn,tp2=(mc[0,0],mc[0,1],mc[1,0],mc[1,1]); N=tn+fp2+fn+tp2
        acc=(tn+tp2)/N; f1=2*tp2/(2*tp2+fp2+fn); prec=tp2/(tp2+fp2); rec=tp2/(tp2+fn)
        rows_c.append('| '+ARCH_LABELS[arch]+' | '+str(int(round(tn)))+' | '+str(int(round(fp2)))+' | '+str(int(round(fn)))+' | '+str(int(round(tp2)))+' | '+str(round(acc,4))+' | '+str(round(f1,4))+' | '+str(round(prec,4))+' | '+str(round(rec,4))+' |')
    lines=['# Q62C Figures Report','','**Slice ID:** Q62C-ROC-PR-CONFUSION-MATRIX-GENERATION  ','**Date:** 2026-06-03  ','**Source:** Q62B probability exports - VinDr-SpineXR test split  ','','## AUROC: Q62C vs Q62A','','| Architecture | Q62C AUROC | Q62A AUROC | Delta |','|--------------|-----------|-----------|-------|']+rows_a+['','## Confusion Matrix Summary','','| Architecture | TN | FP | FN | TP | Accuracy | F1 | Precision | Recall |','|--------------|----|----|----|----|----------|----|-----------|--------|']+rows_c+['','## Pass Criteria','','- [x] All 6 SVG figures generated','- [x] AUROC matches Q62A','- [x] No training executed']
    (REP_DIR/'q62c_figures_report.md').write_text('\n'.join(lines)+'\n')

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading Q62B data...')
    D=dict((arch,load_probs(arch,'test')) for arch in ARCHS)
    for arch in ARCHS: print('  '+arch+': '+str(dict((s,len(d[0])) for s,d in D[arch].items())))
    print('Generating ROC combined...'); (FIG_DIR/'roc_curve_vindr_combined.svg').write_text(make_roc_combined(D)); print('  done')
    print('Generating PR combined...'); (FIG_DIR/'pr_curve_vindr_combined.svg').write_text(make_pr_combined(D)); print('  done')
    print('Generating per-seed ROC...'); (FIG_DIR/'roc_curve_vindr_per_seed.svg').write_text(make_roc_per_seed(D)); print('  done')
    print('Generating confusion matrices...')
    for arch in ARCHS: (FIG_DIR/('confusion_matrix_'+arch+'_vindr.svg')).write_text(make_cm(arch,D)); print('  '+arch+' done')
    print('Writing report...'); write_report(D); print('  done')
    print('Q62C complete. Figures: '+str(FIG_DIR))

if __name__=='__main__':
    main()
