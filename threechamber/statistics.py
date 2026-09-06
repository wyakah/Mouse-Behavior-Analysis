"""Reproducible mouse-level summaries and explicitly selected group comparisons."""
from itertools import combinations, product
import math
import warnings
import numpy as np

METRICS = {
    'stranger_interaction_percent': 'Stranger Interaction %',
    'stranger_interaction_seconds': 'Stranger interaction time (s)',
    'target_nose_seconds': 'Social / novel cup time (s)',
    'other_nose_seconds': 'Other cup time (s)',
    'preference_index': 'Preference index',
    'left_chamber_seconds': 'Left chamber time (s)',
    'center_chamber_seconds': 'Center chamber time (s)',
    'right_chamber_seconds': 'Right chamber time (s)',
    'left_nose_seconds': 'Left cup time (s)',
    'right_nose_seconds': 'Right cup time (s)',
}
DEFAULT_METRICS = ['stranger_interaction_percent', 'target_nose_seconds', 'other_nose_seconds', 'preference_index']
MODES = ['descriptive', 'genotype', 'within_sex', 'factorial', 'all']


def settings(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict): raise ValueError('Invalid statistics settings.')
    mode = value.get('mode', 'descriptive')
    if mode not in MODES: raise ValueError('Choose a supported statistical comparison.')
    metrics = value.get('metrics', DEFAULT_METRICS)
    if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) or m not in METRICS for m in metrics):
        raise ValueError('Select at least one supported statistical outcome.')
    try: alpha = float(value.get('alpha', .05))
    except (ValueError, TypeError): raise ValueError('Choose a valid statistical alpha.')
    if alpha not in (.05, .01): raise ValueError('Choose alpha 0.05 or 0.01.')
    return dict(mode=mode, metrics=list(dict.fromkeys(metrics)), alpha=alpha,
                correction='holm', experimental_unit='one independent mouse per sample ID')


def sample_metadata(entry):
    sex = str(entry.get('sex') or 'unknown').strip().lower()
    if sex not in ('male', 'female', 'unknown'): raise ValueError('Sex must be male, female, or not recorded.')
    genotype = entry.get('genotype', '')
    if genotype is None: genotype = ''
    if not isinstance(genotype, str) or len(genotype.strip()) > 120:
        raise ValueError('Genotype must be text up to 120 characters.')
    return dict(sex=sex, genotype=genotype.strip())


def finite(value):
    return isinstance(value, (int,float,np.number)) and not isinstance(value, (bool,np.bool_)) and math.isfinite(float(value))


def metric_value(entry, metric):
    s = entry.get('summary') or {}
    if entry.get('status') != 'complete': return None, 'Recording did not complete.'
    if metric in ('stranger_interaction_percent','stranger_interaction_seconds'):
        from threechamber.social import stranger_metrics
        value=stranger_metrics(s)[metric]
        return (value,None) if value is not None else (None,'Stranger position, nose observations, or chamber time unavailable.')
    if metric in ('target_nose_seconds', 'other_nose_seconds', 'preference_index'):
        target = s.get('target_side')
        if target not in ('left','right'): return None, 'Social / novel cup side is unspecified.'
    if 'nose' in metric or metric == 'preference_index':
        if not finite(s.get('nose_scoreable_fraction')) or s['nose_scoreable_fraction'] <= 0:
            return None, 'No scoreable nose observations.'
    if 'chamber' in metric and (not finite(s.get('center_valid_fraction')) or s['center_valid_fraction'] <= 0):
        return None, 'No valid body-center observations.'
    if metric in ('target_nose_seconds', 'other_nose_seconds'):
        side = target if metric == 'target_nose_seconds' else ('left' if target == 'right' else 'right')
        value = s.get(side+'_nose_seconds')
    else: value = s.get(metric)
    if not finite(value): return None, 'Outcome is unavailable (including a zero preference denominator).'
    return float(value), None


def analyze_statistics(batch):
    # Imports are local: statistics cannot prevent video tracking from starting.
    import scipy
    from scipy import stats
    import statsmodels
    from statsmodels.stats.multitest import multipletests
    plan = settings(batch.get('statistics'))
    metrics = plan['metrics']; observations=[]; exclusions=[]; groups=[]; comparisons=[]; anova=[]
    for e in batch['entries']:
        meta=sample_metadata(e)
        observation=dict(id=e['id'],**meta,duration=e.get('summary',{}).get('analyzed_seconds'),values={})
        for metric in metrics:
            value,reason=metric_value(e,metric);observation['values'][metric]=value
            if reason:exclusions.append(dict(id=e['id'],metric=METRICS[metric],scope='All summaries and tests',reason=reason))
        if not meta['genotype']:exclusions.append(dict(id=e['id'],metric='All selected outcomes',scope='Genotype comparisons',reason='Genotype not recorded.'))
        if meta['sex']=='unknown':exclusions.append(dict(id=e['id'],metric='All selected outcomes',scope='Within-sex and factorial comparisons',reason='Sex not recorded.'))
        observations.append(observation)
    # Keep every observed cell, including missing metadata and groups with zero valid outcomes.
    cells=sorted({(o['genotype'],o['sex']) for o in observations})
    pooled=sorted({o['genotype'] for o in observations})
    for metric in metrics:
        for genotype,sex in [(g,'pooled') for g in pooled]+cells:
            members=[o for o in observations if o['genotype']==genotype and (sex=='pooled' or o['sex']==sex)]
            valid=[o for o in members if o['values'][metric] is not None]
            a=np.array([o['values'][metric] for o in valid]);n=len(a)
            mean=float(a.mean()) if n else None;sd=float(a.std(ddof=1)) if n>1 else None
            sem=sd/math.sqrt(n) if sd is not None else None
            margin=float(stats.t.ppf(.975,n-1))*sem if n>1 else None
            groups.append(dict(metric=metric,label=METRICS[metric],genotype=genotype or 'Not recorded',sex=sex,
                               total=len(members),n=n,missing=len(members)-n,mean=mean,sd=sd,sem=sem,
                               ci_low=mean-margin if margin is not None else None,ci_high=mean+margin if margin is not None else None,
                               sample_ids=[o['id'] for o in valid]))
    levels=sorted({o['genotype'] for o in observations if o['genotype']})
    scopes=[]
    if plan['mode'] in ('genotype','all'): scopes.append('pooled')
    if plan['mode'] in ('within_sex','all'): scopes.extend(['female','male'])
    for metric in metrics:
        for sex in scopes:
            if len(levels)<2:
                comparisons.append(dict(metric=metric,label=METRICS[metric],sex=sex,status='Not calculated',reason='At least two recorded genotypes are required.'))
            for ga,gb in combinations(levels,2):
                ma=[o for o in observations if o['genotype']==ga and (sex=='pooled' or o['sex']==sex) and o['values'][metric] is not None]
                mb=[o for o in observations if o['genotype']==gb and (sex=='pooled' or o['sex']==sex) and o['values'][metric] is not None]
                a=np.array([o['values'][metric] for o in ma]);b=np.array([o['values'][metric] for o in mb])
                result=dict(metric=metric,label=METRICS[metric],sex=sex,group_a=ga,group_b=gb,n_a=len(a),n_b=len(b),
                            mean_a=float(a.mean()) if len(a) else None,mean_b=float(b.mean()) if len(b) else None,
                            method='Two-sided Welch t-test',status='Not calculated',reason=None)
                if min(len(a),len(b))<2:result['reason']='At least two mice with valid outcomes are required in each group.'
                elif not _same_duration(ma+mb):result['reason']='Analyzed durations differ. Use a consistent scoring window before comparing mice.'
                else:
                    va=float(a.var(ddof=1)/len(a));vb=float(b.var(ddof=1)/len(b));se=math.sqrt(va+vb)
                    if se<=0:result['reason']='Both groups have zero variance; a finite Welch test cannot be estimated.'
                    else:
                        test=stats.ttest_ind(a,b,equal_var=False);df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1))
                        difference=float(a.mean()-b.mean());margin=float(stats.t.ppf(.975,df))*se
                        if finite(test.statistic) and finite(test.pvalue):
                            result.update(status='Calculated',difference=difference,ci_low=difference-margin,ci_high=difference+margin,
                                          statistic=float(test.statistic),df=df,p=float(test.pvalue))
                        else:result['reason']='The data did not produce a finite test statistic.'
                comparisons.append(result)
        if plan['mode'] in ('factorial','all'): anova.extend(_factorial(observations,metric,levels))
    estimable=[r for r in comparisons+anova if r.get('status')=='Calculated']
    if estimable:
        adjusted=multipletests([r['p'] for r in estimable],alpha=plan['alpha'],method='holm')[1]
        for r,p in zip(estimable,adjusted):r.update(p_adjusted=float(p),below_alpha=bool(p<plan['alpha']))
    return dict(settings=plan,groups=groups,comparisons=comparisons,anova=anova,exclusions=exclusions,
                observations=observations,test_count=len(estimable),versions={'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
                notes=[
                    'One independent mouse per sample ID. Video frames and bouts are not independent replicates.',
                    'Fixed statistical report for this batch. To change metadata, outcomes, or tests, update setup and run a new batch. Excel edits do not recompute inference.',
                    'Pairwise tests are two-sided Welch tests. Mean differences are group A minus group B. Confidence intervals are unadjusted 95% intervals.',
                    f'Holm adjustment covers all {len(estimable)} estimable pairwise tests and ANOVA terms across selected outcomes and comparison scopes in this export.',
                    'Within-sex comparisons do not themselves test whether genotype effects differ between sexes. Use the genotype × sex interaction for that question.',
                    'Factorial tests use OLS, sum contrasts, Type III ANOVA with interaction. Assumptions: independent mice, approximately normal residuals, constant residual variance. These assumptions are not established by exporting this report.',
                    'Inferential tests require matching analyzed durations and at least two observations in each tested group/cell. This minimum permits calculation, not a claim of adequate power.',
                    'Target-relative outcomes follow the social / novel cup side. Compare mice from the same test phase and experimental conditions.',
                    'No automatic outlier removal, imputation, or low-coverage exclusion threshold. Unavailable outcomes and missing grouping metadata are listed explicitly. Tracking uncertainty remains relevant to interpretation.',
                ],sources=[
                    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html',
                    'https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html',
                    'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html'])


def _same_duration(members):
    durations=[o['duration'] for o in members]
    return bool(durations) and all(finite(d) and d>0 for d in durations) and max(durations)-min(durations)<.01


def _factorial(observations,metric,levels):
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    valid=[o for o in observations if o['genotype'] and o['sex'] in ('male','female') and o['values'][metric] is not None]
    cells={(g,s):[o for o in valid if o['genotype']==g and o['sex']==s] for g,s in product(levels,['female','male'])}
    base=dict(metric=metric,label=METRICS[metric],n=len(valid),method='Type III two-way ANOVA; sum contrasts',status='Not calculated')
    if len(levels)<2 or not cells or min(map(len,cells.values()))<2:
        return [dict(base,reason='Requires at least two genotypes, both sexes, and two valid mice in every genotype × sex cell.')]
    if not _same_duration(valid):return [dict(base,reason='Analyzed durations differ. Use a consistent scoring window before comparing mice.')]
    data=pd.DataFrame([dict(value=o['values'][metric],genotype=o['genotype'],sex=o['sex']) for o in valid])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            model=ols('value ~ C(genotype, Sum) * C(sex, Sum)',data=data).fit()
            if model.df_resid<=0 or np.linalg.matrix_rank(model.model.exog)<model.model.exog.shape[1] or model.ssr<=np.finfo(float).eps*max(1.,float(np.square(data.value).sum())):
                return [dict(base,reason='Residual variance or model rank is insufficient for ANOVA.')]
            table=anova_lm(model,typ=3)
        rows=[]
        for term,label in [('C(genotype, Sum)','Genotype'),('C(sex, Sum)','Sex'),('C(genotype, Sum):C(sex, Sum)','Genotype × sex')]:
            t=table.loc[term];ss=float(t['sum_sq']);f=float(t['F']);p=float(t['PR(>F)'])
            if not finite(f) or not finite(p):return [dict(base,reason='ANOVA did not produce finite test results.')]
            rows.append(dict(base,status='Calculated',effect=label,statistic=f,df=float(t['df']),df_residual=float(model.df_resid),
                             sum_squares=ss,partial_eta_squared=ss/(ss+model.ssr),p=p,reason=None))
        return rows
    except (ValueError,RuntimeWarning,np.linalg.LinAlgError) as error:
        return [dict(base,reason=f'ANOVA could not be estimated ({type(error).__name__}).')]
