"""Redraw figures from reported aggregate counts; no simulated observations."""
from pathlib import Path
import math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#245B78','#B7782C','#943F46']
counts={'HumanEval':{'n':148,'successes':[91,65,54]},'MBPP':{'n':54,'successes':[27,14,13]}}
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return m-h,m+h
fig,axs=plt.subplots(1,2,figsize=(8.4,3.25),sharey=True)
for ax,(name,d) in zip(axs,counts.items()):
 rates=np.array(d['successes'])/d['n']*100
 ci=np.array([wilson(k,d['n']) for k in d['successes']])*100
 bars=ax.bar(range(3),rates,color=colors,width=.61,yerr=[rates-ci[:,0],ci[:,1]-rates],capsize=4,error_kw={'linewidth':1})
 for b,k,r in zip(bars,d['successes'],rates): ax.text(b.get_x()+b.get_width()/2,4,f'{k}/{d["n"]}',ha='center',color='white',fontsize=10,fontweight='bold')
 ax.set_xticks(range(3),['Task only','+ Failing code','+ Code + error']);ax.set_ylim(0,80);ax.set_title(name,fontweight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].set_ylabel('Failures resolved (%)')
fig.tight_layout();fig.savefig(ROOT/'figures/context_ablation.pdf',bbox_inches='tight');plt.close(fig)
data={'caps_pct':[25,40,50,75], 'risk':[7,9,14,30], 'random':[13.5,21.3,26.8,40.1], 'reversed':[24,36,40,47], 'length':[27,37,40,51]}
fig,ax=plt.subplots(figsize=(7.9,3.55))
for key,label,c,marker in [('risk','Ascending pass score (Smart)','#943F46','s'),('random','Random (mean)','#777777','o'),('reversed','Reversed score (post hoc)','#B7782C','^'),('length','Shortest program first','#245B78','D')]:
 ax.plot(data['caps_pct'],data[key],label=label,color=c,marker=marker,lw=1.6,markersize=5)
ax.set_xlabel('Repair-call cap (% of 368 calls)');ax.set_ylabel('Additional passing programs');ax.set_xticks(data['caps_pct']);ax.set_ylim(0,57);ax.grid(alpha=.17);ax.legend(loc='upper left',frameon=False,fontsize=9);fig.tight_layout();fig.savefig(ROOT/'figures/allocation_budget.pdf',bbox_inches='tight');plt.close(fig)
(ROOT/'figure_data.json').write_text(json.dumps({'context':counts,'allocation':data,'source':'Uploaded main_joint.tex, context ladder and length-gate tables','uncertainty':'Wilson intervals calculated from aggregate counts; paired tests are reported separately.'},indent=2))
