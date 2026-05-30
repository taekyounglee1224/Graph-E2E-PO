# Research Findings & Notes

## [2026-05-30] Architecture × Loss Interaction Effect

### 발견 내용

RQ2 (loss function 효과) 결과가 architecture에 따라 **반대 방향**으로 나타남:

| 비교 | Sharpe t-stat | 결론 |
|---|---|---|
| RQ2a: Graph-L1 vs Graph-L2 | +4.74 (p<0.001) | Graph에서는 **L1 > L2** |
| RQ2b: MLP-L1   vs MLP-L2   | −8.04 (p<0.001) | MLP에서는 **L2 > L1** |

### 해석

- **Graph + L1**: GCN이 상관관계 구조를 통해 공분산을 충분히 잘 추정 → L2 loss의 return signal이 오히려 노이즈로 작용
- **MLP + L2**: MLP는 graph 없이 공분산 추정이 약함 → return signal(L2)이 부족한 공분산 정보를 보완적으로 보강

→ **Graph 구조 자체가 L1 loss만으로도 return 정보를 암묵적으로 인코딩한다**는 가설 가능

### 논문 활용 방향

- 2×2 factorial ANOVA 틀에서 **Architecture × Loss 교호작용(interaction)** 으로 보고
- "The benefit of incorporating a return signal in the loss is architecture-dependent:
  it hurts GNN-based models (which already capture return-relevant structure through message passing)
  but benefits MLP-based models (which lack relational inductive bias)."
- Figure: 2×2 Sharpe 평균 테이블 또는 interaction plot (x=Loss, lines=Architecture)

### 데이터 기반 (N=50 seeds)

```
           Ann.Return   Ann.Vol   Sharpe   Sortino   Max DD    CVaR
Graph-L1   +0.230      +0.245    +0.939   +1.625    −0.401   +0.136
Graph-L2   +0.201      +0.240    +0.838   +1.416    −0.456   +0.137
MLP-L1     +0.085      +0.135    +0.629   +0.952    −0.230   +0.090
MLP-L2     +0.163      +0.204    +0.797   +1.302    −0.337   +0.126
```

Graph-L1이 전체 최고 성과. MLP-L2가 MLP-L1 대비 크게 개선되나 여전히 Graph 모델에 미치지 못함.
