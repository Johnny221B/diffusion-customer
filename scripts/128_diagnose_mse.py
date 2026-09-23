#!/usr/bin/env python3
"""CPU diagnostics: prediction error decomposition, independent fits, MAP audit.

No online training or acquisition is changed. Lambda selection uses independent
validation seeds; the existing test seeds 5--9 never enter fitting or selection.
"""
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.cmts_sim import laplace_map,project_norm,sigma


def errors(pred,true):
    bias=float(pred.mean()-true.mean())
    sp,st=float(pred.std()),float(true.std())
    corr=float(np.corrcoef(pred,true)[0,1]) if sp>1e-12 else 0.
    mse=float(np.mean((pred-true)**2))
    scale=(sp-st)**2
    shape=2*sp*st*(1-corr)
    np.testing.assert_allclose(mse,bias*bias+scale+shape,atol=1e-10)
    return dict(mse=mse,variance_nmse=mse/(st*st),bias=bias,pred_mean=float(pred.mean()),
                true_mean=float(true.mean()),pred_std=sp,true_std=st,correlation=corr,
                bias_mse=bias*bias,scale_mse=scale,shape_mse=shape)


def fit(X,y,lam,intercept=False):
    F=np.column_stack([X,np.ones(len(X))]) if intercept else X
    penalty=np.full(F.shape[1],lam)
    if intercept:penalty[-1]=0.
    def fun(b):
        logits=F@b
        return float(np.sum(np.logaddexp(0,logits)-y*logits)+.5*np.sum(penalty*b*b))
    def jac(b):return F.T@(expit(F@b)-y)+penalty*b
    result=minimize(fun,np.zeros(F.shape[1]),jac=jac,method="L-BFGS-B",
                    options=dict(maxiter=3000,ftol=1e-14,gtol=1e-8,maxls=50))
    # Resolve L-BFGS function-tolerance termination with safeguarded Newton
    # polishing, especially for large datasets with an objective of thousands.
    b=result.x.copy()
    for _ in range(20):
        grad=jac(b)
        if np.linalg.norm(grad)<1e-7:break
        p=expit(F@b)
        H=F.T@(F*(p*(1-p))[:,None])+np.diag(penalty)
        step=np.linalg.solve(H,grad)
        factor=1.
        current=fun(b)
        for _ in range(30):
            candidate=b-factor*step
            if (fun(candidate)<=current+1e-12*max(1.,abs(current))
                    and np.linalg.norm(jac(candidate))<np.linalg.norm(grad)):
                b=candidate
                break
            factor*=.5
        else:break
    if np.linalg.norm(jac(b))/len(y)>1e-7:
        raise RuntimeError(f"Fit failed: {result.message}, gradient={np.linalg.norm(jac(result.x))}")
    return b,dict(objective=fun(b),gradient_norm=float(np.linalg.norm(jac(b))))


def predict(X,b,intercept=False):
    return expit(X@b[:-1]+b[-1]) if intercept else expit(X@b)


def main():
    root=PROJECT/'outputs/cmts_tau_sweep_20260922_a8_v05_lam100/tau1.25'
    cfg=json.loads((root/'config_partial0.json').read_text())
    geom=np.load(root/'geometry_partial0.npz')
    out=PROJECT/'outputs/continuous_diagnostics/continuous_mse_diagnosis'
    out.mkdir(parents=True,exist_ok=True)
    test=np.load(PROJECT/'outputs/continuous_diagnostics/continuous_tau1.25_sim000/test_set.npz')
    Xtest=test['z']-geom['z_comp']; ptest=test['true_probability']
    with (root/'sim000/_ckpt.pkl').open('rb') as f:online=pickle.load(f)
    history=pd.DataFrame(online['rows']);X,y=online['Phi'],online['y']
    T=online['t_done'];beta=None;decomp=[];predictions={};betas={}
    for t in range(T+1):
        n=cfg['n0']+t*cfg['B']
        beta,_=laplace_map(X[:n],y[:n],cfg['lam'],cfg['d'],beta0=beta)
        beta=project_norm(beta,cfg['S'])
        pred=sigma(Xtest@beta)
        decomp.append(dict(round=t,**errors(pred,ptest)))
        if t in [0,20,70,200,400,T]:
            predictions[f'online_{t}']=pred;betas[t]=beta.copy()
        if t:
            logged=history.iloc[n-cfg['B']:n]
            np.testing.assert_allclose(sigma(X[n-cfg['B']:n]@beta),logged.predicted_p,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(beta,online['beta_hat'],atol=1e-6,rtol=1e-5)
    decomposition=pd.DataFrame(decomp)
    decomposition.to_csv(out/'online_error_decomposition.csv',index=False)
    print('Online error decomposition complete',flush=True)

    # Independent source searches cover >200 nearest anchors each. Source seeds
    # 0--2 train, 3--4 select lambda; test seeds 5--9 are completely excluded.
    source=PROJECT/'outputs/cmts_lam50_v4.0_bbright_a15_d16_B8_T300_0617_1156'
    pool=np.load(PROJECT/cfg['pool_dir']/'embeddings.npz',allow_pickle=True)
    keep=pool['words'].astype(str)!=cfg['B_word']
    old=PCA(n_components=cfg['d'],random_state=0).fit(pool['embs'][keep].astype(np.float32))
    nn=NearestNeighbors(n_neighbors=cfg['k']).fit(geom['anchors'])
    xs=[];probs=[];frames=[]
    for seed in range(5):
        state=np.load(source/f'sim{seed:03d}/posterior.npz')
        df=pd.read_csv(source/f'sim{seed:03d}/trajectory.csv')
        np.testing.assert_array_equal(df.y,state['y'])
        z=state['Phi']+state['z_comp']
        z=(old.inverse_transform(z)-geom['pca_mean'])@geom['pca_components'].T
        dist,indices=nn.kneighbors(z)
        valid=dist[:,-1]<=cfg['tau_d']+1e-3
        p=sigma(cfg['alpha']*(cfg['D_B']-df.ds_to_R.to_numpy()))
        df['source_sim']=seed;df['source_row']=np.arange(len(df));df['nearest_anchor']=indices[:,0]
        xs.append(z[valid]-geom['z_comp']);probs.append(p[valid]);frames.append(df.loc[valid])
    Xpool=np.vstack(xs);ppool=np.concatenate(probs);table=pd.concat(frames,ignore_index=True)
    assert NearestNeighbors(n_neighbors=1).fit(Xpool).kneighbors(Xtest)[0].min()>1e-3
    table['probability_alpha8']=ppool
    rng=np.random.default_rng(20260923)
    # Cap each anchor's contribution separately in train and validation groups.
    balanced=np.zeros(len(table),dtype=bool)
    for seeds in [[0,1,2],[3,4]]:
        subset=table[table.source_sim.isin(seeds)]
        for _,group in subset.groupby('nearest_anchor'):
            chosen=rng.choice(group.index,size=min(10,len(group)),replace=False)
            balanced[chosen]=True
    table['balanced']=balanced
    table.to_csv(out/'independent_pool.csv',index=False)
    noisy=np.random.default_rng(20260924).binomial(1,ppool)
    validation=[];comparison=[]
    for dataset in ['warm_only','broad_balanced']:
        base=(table.phase=='warm').to_numpy() if dataset=='warm_only' else balanced
        train=base & table.source_sim.isin([0,1,2]).to_numpy()
        val=base & table.source_sim.isin([3,4]).to_numpy()
        for target in ['oracle_probability','simulated_bernoulli']:
            labels=ppool if target=='oracle_probability' else noisy
            for intercept in [False,True]:
                tag=f'{dataset}/{target}/'+('intercept' if intercept else 'original_model')
                choices=[]
                for lam in [.01,.1,1.,10.,100.,1000.]:
                    b,check=fit(Xpool[train],labels[train],lam,intercept)
                    vmse=float(np.mean((predict(Xpool[val],b,intercept)-ppool[val])**2))
                    row=dict(model=tag,lam=lam,validation_mse=vmse,**check)
                    validation.append(row);choices.append(row)
                best=min(choices,key=lambda r:r['validation_mse'])
                allfit=train|val
                b,check=fit(Xpool[allfit],labels[allfit],best['lam'],intercept)
                pred=predict(Xtest,b,intercept)
                row=dict(model=tag,train_count=int(train.sum()),validation_count=int(val.sum()),
                         refit_count=int(allfit.sum()),nearest_anchors=int(table.loc[allfit,'nearest_anchor'].nunique()),
                         selected_lambda=best['lam'],validation_mse=best['validation_mse'],
                         fitted_beta_norm=float(np.linalg.norm(b)),**errors(pred,ptest),**check)
                comparison.append(row);predictions[tag]=pred
                print(tag,'test variance NMSE',row['variance_nmse'],flush=True)
    pd.DataFrame(validation).to_csv(out/'validation_fits.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'independent_model_comparison.csv',index=False)

    # Hold the model and lambda fixed to separate these results from tuning.
    controls=[]
    for name,F,labels in [
        ('online_observed_binary',X,y),
        ('online_oracle_probability',X,history.true_p_soft.to_numpy()),
        ('broad_balanced_oracle_probability',Xpool[balanced],ppool[balanced]),
        ('broad_balanced_simulated_bernoulli',Xpool[balanced],noisy[balanced])]:
        b,check=fit(F,labels,cfg['lam'])
        pred=expit(Xtest@b)
        controls.append(dict(model=name,lambda_fixed=cfg['lam'],train_count=len(labels),
                             **errors(pred,ptest),**check))
    pd.DataFrame(controls).to_csv(out/'fixed_lambda_controls.csv',index=False)

    # Refit the exact same observed binary data with a safeguarded convex solver.
    audits=[]
    for seed in [0,1,2]:
        checkpoint=root/f'sim{seed:03d}/_ckpt.pkl'
        if not checkpoint.exists():continue
        with checkpoint.open('rb') as f:saved=pickle.load(f)
        F,labels=saved['Phi'],saved['y'];stored=saved['beta_hat']
        robust,check=fit(F,labels,cfg['lam'])
        def objective(b):return float(np.sum(np.logaddexp(0,F@b)-labels*(F@b))+.5*cfg['lam']*(b@b))
        grad=F.T@(expit(F@stored)-labels)+cfg['lam']*stored
        row=dict(sim=seed,rounds=saved['t_done'],stored_norm=float(np.linalg.norm(stored)),
                 robust_norm=float(np.linalg.norm(robust)),stored_gradient_norm=float(np.linalg.norm(grad)),
                 robust_gradient_norm=check['gradient_norm'],stored_objective=objective(stored),
                 robust_objective=objective(robust),beta_difference=float(np.linalg.norm(stored-robust)),
                 stored_test_nmse=errors(expit(Xtest@stored),ptest)['variance_nmse'],
                 robust_test_nmse=errors(expit(Xtest@robust),ptest)['variance_nmse'])
        audits.append(row)
    pd.DataFrame(audits).to_csv(out/'optimizer_audit.csv',index=False)
    np.savez_compressed(out/'test_predictions.npz',true_probability=ptest,**predictions)
    summary=dict(selected_online_rounds=T,selected_online_initial=decomp[0],selected_online_final=decomp[-1],
                 independent_fit_protocol='train seeds 0-2, validation seeds 3-4 choose lambda, refit 0-4; unchanged independent test seeds 5-9',
                 independent_source=str(source),test_count=len(ptest),
                 limitation='Offline finite-sample diagnostic, not a proved model-class ceiling. Oracle soft targets deliberately remove Bernoulli noise; original source alpha15 labels are NOT reused. New Bernoulli labels at alpha8 are simulated only for diagnostic fits. Historical PCA is approximately reconstructed. No acquisition or GPU experiment is modified.',
                 comparisons=comparison,fixed_lambda_controls=controls,optimizer_audit=audits)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    v=float(ptest.var())
    for col,label in [('bias_mse','Mean bias'),('scale_mse','Spread mismatch'),('shape_mse','Pattern mismatch')]:
        axes[0].plot(decomposition['round'],decomposition[col]/v,label=label)
    axes[0].plot(decomposition['round'],decomposition.variance_nmse,color='black',lw=1.6,label='Total')
    axes[0].set(xlabel='Round',ylabel='Error / true probability variance',title='Online prediction error decomposition')
    axes[0].legend(fontsize=8,frameon=False)
    for t in [0,70,T]:
        axes[1].scatter(ptest,predictions[f'online_{t}'],s=12,alpha=.65,label=f'Round {t}')
    axes[1].plot([0,1],[0,1],'k--',lw=1)
    axes[1].set(xlabel='True probability',ylabel='Predicted probability',title='Same independent test points',xlim=(0,1),ylim=(0,1))
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'mse_diagnosis.{ext}',dpi=230)
    plt.close(fig)
    print('FINAL ONLINE',json.dumps(decomp[-1]),flush=True)
    print('OPTIMIZER AUDIT',json.dumps(audits),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
