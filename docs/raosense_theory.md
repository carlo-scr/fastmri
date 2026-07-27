# RaoSense mathematical contract and implementation audit

This note states the exact mathematical object implemented by
`src/samplers/rao_belief.py` and `src/samplers/joint_operator.py`.  It
separates facts proved for the finite-mixture surrogate from properties that
are only approximations, empirical hypotheses, or future work.  In
particular, none of the results below proves that the surrogate is calibrated
to a true diffusion posterior or that the resulting acquisition policy
improves reconstruction quality.

Status terms used throughout:

- **Proved** means a statement follows from the assumptions in this note.
- **Implemented** means the corresponding algebra is present in the named
  source file.
- **Tested** means a unit test checks a nontrivial finite-dimensional
  instance.  A unit test is not a proof.
- **Not certified** means the current code or experiments do not establish
  the claim.

## 1. Surrogate model and assumptions

At one frozen planning state, let the branch label be
\[
 Z\in\{1,\ldots,K\},\qquad \Pr(Z=k)=w_k,\qquad
 w_k\geq 0,\quad \sum_k w_k=1.
\]
For each positive-weight branch, draw an independent standard-normal
coefficient
\[
 \xi_k\mid Z=k\sim\mathcal N(0,I_{r_k}).
\]
The branch state and the bank of noiseless candidate measurements are
\[
 X=m_k^x+U_k^x\xi_k,\qquad
 S=m_k^s+U_k^s\xi_k.
\]
The equal column semantics of \(U_k^x\) and \(U_k^s\) are essential for the
analytic update.  The code permits a measurement-only belief, in which case
\(X\) is absent and all information statements concern \(S\).

An action \(a\) selects distinct rows through \(P_a\).  Its observation is
\[
 Y_a=P_aS+\epsilon_a,\qquad
 \epsilon_a\sim\mathcal N(0,\Gamma_a),
\]
where \(\Gamma_a\) is diagonal and strictly positive in the implementation.
Define
\[
 D_a=\Gamma_a^{-1/2},\quad
 \mu_{ak}=P_am_k^s,\quad
 V_{ak}=D_aP_aU_k^s.
\]

The exact results require the following assumptions.

1. **Finite local mixture.**  The displayed mixture is the operative belief.
   It need not equal the true inverse-problem posterior.
2. **Local affine response.**  Within branch \(k\), the complete candidate
   response is affine in the same coefficient that generates the state.
3. **Independent Gaussian noise.**  Measurement noise is independent of
   \(Z,\xi_1,\ldots,\xi_K\).  Sequential measurements use fresh independent
   noise unless already-conditioned measurements are explicitly represented.
4. **Frozen model during fantasy planning.**  Branch response factors are
   conditioned algebraically but are not relinearized with a denoiser inside
   horizon-two fantasies.
5. **Exact subspace for exact claims.**  The sufficient-subspace theorems use
   an exact basis for the signal span.  Numerical singular-value truncation
   makes the implemented projection an approximation when a discarded
   singular value is nonzero.
6. **Real coordinates.**  Complex data are first stacked into real and
   imaginary coordinates with the corresponding real noise variances.

If \(S=B X\) almost surely under every branch for one deterministic forward
bank \(B\), then the information about the latent branch coefficient below is
also information about \(X\).  Without that consistency condition, the code
optimizes information about the noiseless response \(S_a=P_aS\), not
necessarily about a semantic target image.

## 2. Exact total covariance factor

Let
\[
 \bar m^s=\sum_k w_km_k^s,\qquad \delta_k=m_k^s-\bar m^s
\]
and form
\[
 G=
 \left[
   \sqrt{w_1}U_1^s,\ldots,\sqrt{w_K}U_K^s,\,
   \sqrt{w_1}\delta_1,\ldots,\sqrt{w_K}\delta_K
 \right].
\]

**Proposition 1 (law-of-total-covariance factor).**
\[
 \operatorname{Cov}(S)=GG^\top
 =\sum_k w_kU_k^s(U_k^s)^\top
  +\sum_k w_k\delta_k\delta_k^\top.
\]
The identical statement holds for the state after replacing \(s\) by \(x\).

**Proof sketch.**  Condition on \(Z\), apply the law of total covariance, and
expand the matrix product \(GG^\top\).  Concatenated columns contribute a sum
of outer products and hence create no cross terms.

If \(K_+\) branches have positive weight, then
\[
 \operatorname{rank}(G)
 \leq \sum_{k:w_k>0}r_k+K_+-1,
\]
because \(\sum_k w_k\delta_k=0\) gives one dependence among the
positive-weight mean-offset columns.

**Implementation and evidence.**

- `measurement_moments` and `state_moments` implement this factorization.
- `maximum_sufficient_rank` reports the conservative
  \(\sum_{k=1}^K r_k+K-1\), including zero-weight branches.
- `test_total_covariance_factor_matches_finite_mixture_exactly` checks both
  state and measurement covariance factors.
- The factor is PSD by construction.  This does **not** establish that its
  columns accurately approximate a diffusion posterior covariance.

## 3. Mutual-information decomposition and estimator

Let the operative latent variable be
\[
 C=(Z,\xi_Z).
\]
Since \(Y_a\) depends on \(C\) only through \(S_a=P_aS\),
\[
 I(C;Y_a)=I(S_a;Y_a).
\]
If \(S_a\) is a deterministic function of the represented state \(X\), then
this also equals \(I(X;Y_a)\).

**Theorem 1 (Rao--Blackwellized information decomposition).**
\[
 I(C;Y_a)
 =
 \underbrace{\sum_k w_k\,
   \frac12\log\det\!\left(I+V_{ak}^\top V_{ak}\right)}
   _{L_a\text{, analytic within-branch information}}
 +
 \underbrace{I(Z;Y_a)}
   _{M_a\text{, mode information}}.
\]

**Proof sketch.**  Apply the chain rule
\[
 I(Z,\xi_Z;Y_a)=I(Z;Y_a)+I(\xi_Z;Y_a\mid Z).
\]
Conditioned on \(Z=k\), this is a linear Gaussian channel with standardized
response \(V_{ak}\), whose mutual information is the displayed log
determinant.

For a sample \((Z,Y_a)\) from the predictive mixture, define the mode
information density
\[
 \imath_Z(Y_a)
 =\log p(Y_a\mid Z)-\log\sum_jw_jp(Y_a\mid Z=j).
\]
Then
\[
 \mathbb E[\imath_Z(Y_a)]=I(Z;Y_a),
\]
so the sample mean of \(\imath_Z\), added to the analytic \(L_a\), is an
unbiased estimator of \(I(C;Y_a)\) under the frozen surrogate and exact
projection.

The pointwise upper bound
\[
 \imath_k(y)\leq-\log w_k
\]
follows from \(p(y)\geq w_kp(y\mid k)\).  There is no corresponding finite
pointwise lower bound in general; Gaussian-mixture information densities can
have negative tails.  Consequently, the reported sample standard error is a
descriptive Monte Carlo error estimate, not a finite-sample confidence
certificate.

**Implementation and evidence.**

- `information_bounds` computes \(L_a\).
- `estimate_information` samples only the mode term and evaluates every
  within-branch term analytically.
- For \(K=1\), it returns the exact Gaussian information without Monte Carlo.
- `test_information_sandwich_and_single_component_reduction` checks the
  single-Gaussian identity and checks a large-sample mixture estimate against
  the analytic sandwich below.
- Unbiasedness is **proved for the surrogate** but is not directly tested by
  a repeated-seed calibration test.

## 4. Lower bound, two upper bounds, and safe one-step pruning

Let \(G_a=P_aG\) and \(\widetilde G_a=D_aG_a\).  Define the moment-Gaussian
upper bound
\[
 U_a^{\mathrm G}
 =\frac12\log\det\!\left(I+\widetilde G_a^\top\widetilde G_a\right)
\]
and the categorical entropy
\[
 H(w)=-\sum_{k:w_k>0}w_k\log w_k.
\]

**Theorem 2 (analytic information sandwich).**
\[
 L_a
 \;\leq\;
 I(C;Y_a)
 \;\leq\;
 U_a
 :=
 \min\left\{
 U_a^{\mathrm G},\,L_a+H(w)
 \right\}.
\]

**Proof sketch.**

1. The lower bound follows from Theorem 1 and \(I(Z;Y_a)\geq0\).
2. Since \(I(Z;Y_a)\leq H(Z)=H(w)\), Theorem 1 gives
   \(I(C;Y_a)\leq L_a+H(w)\).
3. \(Y_a\) has covariance
   \(\Gamma_a+G_aG_a^\top\).  A Gaussian maximizes differential entropy at
   fixed covariance, while \(h(Y_a\mid C)=h(\epsilon_a)\).  Subtracting these
   entropies and applying the matrix determinant lemma gives
   \(I(C;Y_a)\leq U_a^{\mathrm G}\).
4. Taking the smaller valid upper bound gives \(U_a\).

Here “tighter upper bound” means the minimum of these two generic analytic
bounds.  It does not mean that \(U_a\) is the best possible upper bound for a
specified mixture.

Useful equality cases are:

- \(K=1\): \(L_a=I(C;Y_a)=U_a^{\mathrm G}\).
- \(M_a=0\): the lower bound is exact.
- Gaussian predictive \(Y_a\): the moment-Gaussian upper bound is exact.
- Perfectly distinguishable branch labels approach \(M_a=H(w)\), making the
  entropy cap tight.

Roundoff can produce a computed \(U_a^{\mathrm G}\) microscopically below
\(L_a\).  The implementation protects the sandwich by replacing the
computed Gaussian upper bound with
\(\max\{U_a^{\mathrm G},L_a\}\) before taking the entropy cap.

**Corollary 2.1 (safe pruning for one-step information).**  Let
\[
 L_\star=\max_b L_b.
\]
If \(U_a<L_\star\), then action \(a\) cannot maximize
\(I(C;Y_a)\).

**Proof.**  Choose \(b\) with \(L_b=L_\star\).  Then
\[
 I(C;Y_a)\leq U_a<L_b\leq I(C;Y_b).
\]

`screen_actions` implements the conservative floating-point rule
\[
 U_a+\tau<L_\star
\]
with default \(\tau=10^{-10}\).

The word “safe” has four important limits.

1. It is safe relative to the finite-mixture surrogate, not the true
   posterior or final reconstruction metric.
2. The theorem is for a **one-step raw-information objective**.
   Pruning roots by immediate information is not safe for horizon-two value:
   a low-immediate-information router can have the best continuation.
3. If action-dependent costs are used, a safe comparison must use
   \(U_a-c_a\) and \(\max_b(L_b-c_b)\).  The current screen ignores costs.
4. The optional runner-level `planner_candidate_cap` ranks by upper bound and
   keeps a fixed number of actions.  That is an explicitly heuristic cap,
   not Corollary 2.1.

**Implementation and evidence.**

- The sandwich and entropy cap are implemented in `information_bounds`.
- The one-step pruning rule is implemented in `screen_actions`.
- `test_safe_bound_screening_removes_zero_information_action` checks a
  strict dominated case and rejects a negative tolerance.
- `plan_horizon_two` deliberately does not use this screen at its root.  It
  reports immediate estimates for every supplied root before an explicitly
  heuristic beam truncation.  The runner-level candidate cap can still
  remove roots before the planner and is not certified for horizon two.

## 5. Operator-sufficient projection

For one action, use the mixture mean as reference and whiten:
\[
 \widetilde Y_a
 =D_a(Y_a-P_a\bar m^s).
\]
Let
\[
 \mathcal S_a
 =
 \operatorname{span}\left(
 \{D_aP_aU_k^s\}_{k:w_k>0}
 \cup
 \{D_aP_a(m_k^s-\bar m^s)\}_{k:w_k>0}
 \right),
\]
and let \(Q_a\) have orthonormal columns spanning \(\mathcal S_a\).  The
compressed statistic is \(T_a=Q_a^\top\widetilde Y_a\).

**Theorem 3 (exact action sufficiency and dimension).**
\[
 C\;\perp\!\!\!\perp\;\widetilde Y_a\mid T_a,
\qquad
 I(C;T_a)=I(C;Y_a),
\]
and
\[
 \dim(T_a)
 \leq
 \min\left\{
 |a|,\,
 \sum_{k:w_k>0}r_k+K_+-1
 \right\}.
\]

**Proof sketch.**  Complete \(Q_a\) to an orthogonal matrix
\([Q_a,Q_a^\perp]\).  Every component mean offset and factor column lies in
\(\mathcal S_a\).  Therefore
\[
 (Q_a^\perp)^\top\widetilde Y_a
 =(Q_a^\perp)^\top D_a\epsilon_a
 \sim\mathcal N(0,I)
\]
and is independent of \(C\), \(T_a\), and the signal-bearing noise
coordinates.  It can be discarded without changing likelihood ratios,
posteriors, or mutual information.  The dimension bound follows from
Proposition 1.

This theorem explains why the cost of mixture likelihoods and fantasy
updates depends on retained branch ranks and mode count rather than the
number of scalar measurements in an MRI line or LiDAR tile.

**Implementation and evidence.**

- `project_action` obtains \(Q_a\) from an SVD of the whitened total
  covariance factor.
- `ProjectedAction` evaluates likelihoods and generates fantasies only in
  these coordinates.
- `test_operator_sufficient_dimension_is_independent_of_action_size`
  compresses a 200-coordinate action to three coordinates.
- The default SVD threshold is
  \(\max(10^{-10},10^{-6}\sigma_{\max})\).  Thus the code implements an
  approximate sufficient statistic whenever a nonzero singular direction
  falls below this threshold.  No current bound converts discarded singular
  mass into posterior or information error.
- `numerical_rank_bound` stores the algebraic upper bound but the constructor
  does not assert it at runtime.

## 6. Analytic conditioning and PSD preservation

For branch \(k\), write the compressed channel as
\[
 T_a=\nu_{ak}+\widehat V_{ak}\xi_k+\eta,\qquad
 \eta\sim\mathcal N(0,I),
\]
where
\[
 \nu_{ak}=Q_a^\top D_a(\mu_{ak}-P_a\bar m^s),\qquad
 \widehat V_{ak}=Q_a^\top V_{ak}.
\]
Given \(T_a=t\), define
\[
 C_{ak}=(I+\widehat V_{ak}^\top\widehat V_{ak})^{-1},\qquad
 c_{ak}=C_{ak}\widehat V_{ak}^\top(t-\nu_{ak}).
\]

**Theorem 4 (mixture update and PSD closure).**  The exact posterior under
the frozen surrogate is a mixture with
\[
 w_k'
 =
 \frac{
   w_k\,\mathcal N(t;\nu_{ak},
   I+\widehat V_{ak}\widehat V_{ak}^\top)
 }{
   \sum_jw_j\,\mathcal N(t;\nu_{aj},
   I+\widehat V_{aj}\widehat V_{aj}^\top)
 },
\]
\[
 X\mid t,Z=k
 \sim
 \mathcal N\!\left(
   m_k^x+U_k^xc_{ak},\,
   U_k^xC_{ak}(U_k^x)^\top
 \right),
\]
and analogously for the full measurement bank using \(m_k^s,U_k^s\).
Moreover,
\[
 0\prec C_{ak}\preceq I,
\]
so every updated within-branch covariance is PSD and no larger than its
prior coefficient covariance in Loewner order.

**Proof sketch.**  Complete the square in the standard linear-Gaussian
likelihood.  Positive definiteness follows because
\(I+\widehat V^\top\widehat V\succ0\).  Multiplication by \(U\) preserves
positive semidefiniteness.  Bayes' rule gives the branch weights.

With the exact sufficient basis, conditioning on \(T_a\) equals conditioning
on the full action observation.  With numerical truncation, it is the exact
posterior only for the truncated surrogate.

**Implementation and evidence.**

- `condition_projected` implements \(C_{ak}\), \(c_{ak}\), the aligned state
  and measurement update, and the mixture likelihood update.
- `_covariance_shrink` symmetrizes the coefficient covariance and constructs
  its square root by eigendecomposition with a machine-epsilon eigenvalue
  floor.  The stored covariance factors therefore remain PSD.
- `test_projected_conditioning_matches_full_gaussian_mixture_update` checks
  posterior means, covariances, and weights against a direct update.
- In exact arithmetic, an exactly zero prior branch weight remains zero.
  The likelihood code clamps weights to the smallest positive floating-point
  value before taking logs, so a zero-weight branch can acquire a negligible
  nonzero posterior weight numerically.  Removing zero-weight branches is
  cleaner when exact support semantics matter.
- The protected square root uses floored eigenvalues while the posterior
  coefficient mean uses the unfloored inverse.  This is deliberately
  PSD-safe, but at numerical extremes it is not one exactly self-consistent
  Gaussian posterior.
- A `ProjectedAction` is not strongly identity-bound to its source belief;
  the update checks the component count but not all means, factors, or
  weights.  Callers must not reuse a projection with a different belief.

## 7. Why a single frozen Gaussian has no outcome-adaptive advantage

The distinction between one Gaussian and a multimodal belief is structural,
not merely empirical.

Consider a finite horizon, one Gaussian latent state, linear actions,
independent Gaussian noise, and feasible successor sets that depend only on
previously chosen actions.  Let \(A\) be the action history.  Its posterior
coefficient covariance has the form
\[
 C_A=
 \left(
 I+\sum_{a\in A}V_a^\top V_a
 \right)^{-1},
\]
with the obvious block form for vector actions.  It depends on selected
actions but not on observed values.

**Theorem 5 (Gaussian no-adaptivity under a frozen linear model).**  For the
above model, an optimal finite-horizon expected-information policy can be
chosen open loop.  Conditioning future action choices on realized
measurement values cannot improve expected information.

**Proof sketch.**  Given action history \(A\), the gain of candidate \(b\) is
\[
 g(b\mid A)
 =
 \frac12\log\det\!\left(
 I+V_bC_AV_b^\top
 \right),
\]
which is independent of the posterior mean and hence of realized
observations.  Backward induction gives a Bellman value whose state is only
the deterministic covariance/action history.  At every node, the maximizing
continuation can therefore be fixed before seeing the data.

This statement stops applying when any of the following is present:

- a mixture whose weights and branch-relative means change with outcomes;
- outcome-dependent feasible actions or costs;
- nonlinear response factors relinearized after an observation;
- a denoiser or diffusion state rerun inside the lookahead tree;
- a utility depending on the posterior mean or realized reconstruction
  rather than information alone.

**Implementation and evidence.**

- For \(K=1\), `condition_projected` has outcome-independent covariance, as
  the theorem predicts.
- `plan_horizon_two` deliberately freezes the factors and makes no denoiser
  calls inside fantasies.
- There is no dedicated no-adaptivity regression test.  With finite Monte
  Carlo, a \(K=1\) planner can exhibit small spurious outcome variation from
  estimator noise even though the exact objective is nonadaptive.

## 8. Horizon-two objective and a conditional Monte Carlo regret bound

Let the net immediate score be
\[
 g_a(b)=I_b(C;Y_a)-c_a
\]
for current belief \(b\).  For a root outcome \(y\), let \(b_{a,y}\) be the
analytically conditioned belief and define the optional-stop continuation
\[
 V_a(y)
 =
 \max\left\{
 0,\,
 \max_{d\in\mathcal A(a)}g_d(b_{a,y})
 \right\}.
\]
The exact frozen-surrogate horizon-two value is
\[
 Q(a)=g_a(b)+\gamma\,
 \mathbb E_{Y_a\sim p_b(\cdot\mid a)}[V_a(Y_a)].
\]

The implementation replaces the expectation with predictive fantasies and
each mixture information with the estimator from Theorem 1.  The inner
maximum makes the nested estimate upward biased in general, even though each
individual information estimate is unbiased.

Let \(\mathcal B\) be the root set that remains after every screen and beam
truncation.  Define its exact truncation loss
\[
 \Delta_{\mathcal B}
 =
 \max_{a\in\mathcal A}Q(a)-\max_{a\in\mathcal B}Q(a)\geq0.
\]

**Theorem 6 (event-conditional root regret).**  Suppose:

1. every future screen preserves the exact net-utility maximizer;
2. for every \(a\in\mathcal B\),
   \(\lvert\widehat g_a-g_a\rvert\leq\varepsilon_0\);
3. for every sampled root outcome and retained future action,
   \(\lvert\widehat g_d-g_d\rvert\leq\varepsilon_1\);
4. for every \(a\in\mathcal B\), the fantasy average of the exact
   \(V_a\) differs from its expectation by at most
   \(\varepsilon_F\).

If \(\widehat a=\arg\max_{a\in\mathcal B}\widehat Q(a)\), then
\[
 \max_{a\in\mathcal A}Q(a)-Q(\widehat a)
 \leq
 \Delta_{\mathcal B}
 +2\left[
   \varepsilon_0+\gamma(\varepsilon_1+\varepsilon_F)
 \right].
\]

**Proof sketch.**  Maximum and positive-part operators are
1-Lipschitz in the sup norm, so the inner score event changes each estimated
continuation by at most \(\varepsilon_1\).  Adding fantasy and immediate
errors yields
\[
 |\widehat Q(a)-Q(a)|
\leq\varepsilon_0+\gamma(\varepsilon_1+\varepsilon_F)
\]
uniformly on \(\mathcal B\).  The empirical maximizer pays twice this uniform
error.  Comparing the best retained root with the global best adds
\(\Delta_{\mathcal B}\).

One possible high-probability corollary requires additional tail assumptions
that the current implementation does not verify.  For example, suppose
centered root and inner information estimates are uniformly sub-Gaussian
with per-sample parameters \(\sigma_0,\sigma_1\), respectively, and
\(0\leq V_a(Y)\leq V_{\max}\).  Let:

- \(n_0\) be root information samples;
- \(n_1\) be inner information samples;
- \(M\) be fantasies per retained root;
- \(N_0=|\mathcal B|\);
- \(N_1\) be the total number of sampled-outcome/future-action estimates.

Then a union bound and Hoeffding concentration give, with probability at
least \(1-\delta\), the event in Theorem 6 with
\[
 \varepsilon_0
 =
 \sigma_0\sqrt{\frac{2\log(6N_0/\delta)}{n_0}},
\]
\[
 \varepsilon_1
 =
 \sigma_1\sqrt{\frac{2\log(6N_1/\delta)}{n_1}},
\]
\[
 \varepsilon_F
 =
 V_{\max}
 \sqrt{\frac{\log(6N_0/\delta)}{2M}}.
\]
Gaussian-mixture information densities are not pointwise bounded below and
are generally better treated as sub-exponential than assumed sub-Gaussian.
A publishable certificate therefore needs either proved uniform tail
parameters, robust mean estimates, or empirical confidence-sequence
calibration.

**Current certification gaps.**

- The beam is selected by noisy immediate estimates, not a confidence-safe
  horizon-two bound.  This also contributes to
  \(\Delta_{\mathcal B}\).
- The runner-level `planner_candidate_cap` ranks roots by an immediate upper
  bound before the planner.  It can discard the true router and contributes
  an unbounded term to \(\Delta_{\mathcal B}\).
- Future one-step screens are used only when candidate costs are equal.
  Unequal-cost leaves are all evaluated, but no cost-aware analytic pruning
  bound is implemented.
- `InformationEstimate.standard_error` is not propagated through fantasies,
  maxima, root comparisons, or beam selection.
- The planner reports no confidence interval and no empirical
  \(\Delta_{\mathcal B}\) certificate.

Therefore the regret result is **proved conditionally but not certified by
the current planner**.  The synthetic
`test_horizon_two_prefers_mode_router_over_myopic_ternary_action` is useful
functional evidence that outcome-dependent routing can occur in a mixture;
it is not a statistical regret guarantee.

## 9. Joint image/operator target information

The joint-operator diagnostic is a one-branch linear-Gaussian calculation.
Let
\[
 \xi\sim\mathcal N(0,I_r),\qquad
 X=U_x\xi,\qquad
 \Phi=U_\phi\xi,\qquad
 \widetilde Y=V\xi+\epsilon,\quad
 \epsilon\sim\mathcal N(0,I).
\]
Means do not affect these mutual informations.  Let \(N_x\) have orthonormal
columns spanning \(\ker U_x\), let \(N_\phi\) span
\(\ker U_\phi\), and let \(N_{x\mid\phi}\) span
\(\ker(U_xN_\phi)\).

**Theorem 7 (Gaussian target and known-nuisance information).**
\[
 I(\xi;\widetilde Y)
 =
 \frac12\log\det(I+V^\top V),
\]
\[
 I(X;\widetilde Y)
 =
 \frac12\left[
 \log\det(I+V^\top V)
 -
 \log\det\!\left(I+(VN_x)^\top VN_x\right)
 \right],
\]
and
\[
 I(X;\widetilde Y\mid\Phi)
 =
 \frac12\left[
 \log\det\!\left(I+(VN_\phi)^\top VN_\phi\right)
 -
 \log\det\!\left(
 I+(VN_\phi N_{x\mid\phi})^\top
      VN_\phi N_{x\mid\phi}
 \right)
 \right].
\]

**Proof sketch.**  Since \(\xi\) is isotropic Gaussian, conditioning on
\(X=U_x\xi\) leaves coefficient covariance equal to the orthogonal projector
\(N_xN_x^\top\).  Thus
\[
 \operatorname{Cov}(\widetilde Y\mid X)
 =I+VN_xN_x^\top V^\top.
\]
Subtract the corresponding Gaussian entropies to obtain the second formula.
Conditioning first on \(\Phi\) restricts the random coefficient to
\(\ker U_\phi\); repeating the same argument inside that subspace gives the
third formula.

The implementation reports
\[
 \texttt{known\_operator\_optimism}
 =
 I(X;\widetilde Y\mid\Phi)-I(X;\widetilde Y).
\]
This difference is guaranteed nonnegative when \(X\) and \(\Phi\) are
independent.  Indeed,
\[
 I(X;Y\mid\Phi)-I(X;Y)
 =
 I(X;\Phi\mid Y)-I(X;\Phi),
\]
and Gaussian independence is equivalent to
\[
 U_xU_\phi^\top=0.
\]
Under independence, the difference equals
\(I(X;\Phi\mid Y)\geq0\): an unknown nuisance can explain away a measurement
that would otherwise look informative about \(X\).

Without \(X\perp\Phi\), the reported difference can be negative.  For
example, if \(X=\Phi\) and \(Y\) observes that shared variable,
\(I(X;Y\mid\Phi)=0<I(X;Y)\).  The function does not check independence, so
the word “optimism” is semantically justified only when the caller supplies
an independent image/nuisance construction.

`augment_branch_with_nuisance` creates exactly such a block construction:
image and nuisance coefficient columns are disjoint, so image-state rows and
appended nuisance rows are prior-independent.  It also makes the following
modeling assumptions:

1. nuisance uncertainty is independent of the image within a branch;
2. the measurement mean is already evaluated at the nuisance mean;
3. the image and nuisance responses add to first order;
4. image--nuisance interaction and higher-order operator terms are ignored.

`finite_difference_nuisance_measurement_factor` computes the central
finite-difference approximation to \(J_\phi U_\phi\) at the fixed state and
nuisance means.  It uses forward-operator calls but no denoiser calls.

**Implementation and evidence.**

- `gaussian_target_information` implements all three log-determinant
  formulas through numerical null spaces.
- `test_target_information_exposes_known_operator_optimism` checks the
  independent scalar model
  \(Y=aX+b\Phi+\epsilon\) against analytic values.
- `test_nuisance_augmentation_builds_aligned_block_factor` checks the
  independent block factor.
- `test_nuisance_forward_factor_matches_analytic_jacobian` checks the finite
  difference against \(J_\phi U_\phi\).
- The diagnostic itself is for **one Gaussian branch**.  The mixture utility
  below combines its target term with branch-label information.
- `joint_information` is literally \(I(\xi;Y)\).  Calling it information
  about the complete represented state additionally requires that discarded
  coefficient directions cannot affect measurements; a sufficient condition
  is that the complete state factor has full column rank.
- `rank_tolerance` controls numerical null spaces and is validated as finite
  and nonnegative.  Near the threshold, target and nuisance information can
  still change discontinuously.

### 9.1 Mixture target information with nuisance marginalized

For branch \(k\), let \(X_k\) denote the selected target-state rows and let
all coefficient directions in the null space of its target factor remain
unobserved nuisance.  The mixture-level target is explicitly
\[
 T=(Z,X_Z).
\]
Including \(Z\) is a semantic choice: distinguishing two branch labels is
valuable under this objective even if their physical target-state
distributions overlap.

**Theorem 8 (mixture target decomposition and sandwich).**  Define
\[
 L_a^{\mathrm{target}}
 =
 \sum_k w_k I(X_k;Y_a\mid Z=k),
\]
where every within-branch term is given by Theorem 7.  Then
\[
 I(T;Y_a)
 =
 L_a^{\mathrm{target}}+I(Z;Y_a).
\]
If \(U_a^{\mathrm G}\) is the moment-Gaussian upper bound computed from the
complete predictive measurement mixture, then
\[
 L_a^{\mathrm{target}}
 \leq I(T;Y_a)
 \leq
 \min\left\{
 U_a^{\mathrm G},
 L_a^{\mathrm{target}}+H(w)
 \right\}.
\]

**Proof sketch.**  The equality is the chain rule
\[
 I(Z,X_Z;Y_a)=I(Z;Y_a)+I(X_Z;Y_a\mid Z).
\]
The second conditional term is the weight-averaged one-branch target
information.  Nonnegativity of \(I(Z;Y_a)\) gives the lower bound and
\(I(Z;Y_a)\leq H(w)\) gives the entropy upper bound.  Finally,
\((Z,X_Z)\) is a function of the complete branch latent
\((Z,\xi_Z)\), so data processing and Theorem 2 give
\[
 I(Z,X_Z;Y_a)
 \leq I(Z,\xi_Z;Y_a)
 \leq U_a^{\mathrm G}.
\]

The moment-Gaussian term contains all predictive variance, including
nuisance-only variance, and can therefore be loose for a target-specific
objective.  It remains valid.  For \(K=1\), \(H(w)=0\), so the entropy cap
makes both bounds equal the exact one-branch target information even when
the measurement contains substantial nuisance variance.

Sampling the projected information density from Theorem 1 estimates only
\(I(Z;Y_a)\).  Adding its sample mean to
\(L_a^{\mathrm{target}}\) gives an unbiased estimator of
\(I(T;Y_a)\) under the exact frozen surrogate and exact sufficient
projection.  Its reported standard error covers only this mode term.

**Corollary 8.1 (safe target-information H1 screening).**  If
\[
 U_a^{\mathrm{target}}
 <
 \max_b L_b^{\mathrm{target}},
\]
then action \(a\) cannot maximize the one-step target information
\(I(T;Y_a)\).  The proof is identical to Corollary 2.1.

**Implementation and evidence.**

- `mixture_target_information_bounds` returns the analytic lower bound and
  the minimum of the complete moment-Gaussian and entropy upper bounds.
- `estimate_mixture_target_information` reuses the ordinary Rao projected
  mixture estimator for \(I(Z;Y_a)\); no nuisance Monte Carlo is required.
- `screen_target_information_actions` implements Corollary 8.1 for H1.
- `test_mixture_target_information_reduces_to_single_branch_analytic` checks
  a scalar image-plus-nuisance channel and prunes a nuisance-only action.
- `test_mixture_target_information_adds_two_mode_information` checks that a
  separated two-mode mixture approaches the analytic within-target term plus
  categorical entropy.
- No target-specific H2 planner is implemented.  H1 screening must not be
  reused as a horizon-two root screen, and target-row semantics would need to
  remain fixed and meaningful after every fantasy update.
- If branch labels represent nuisance-only hypotheses rather than
  target-relevant modes, the explicit target \((Z,X_Z)\) overcounts the
  desired physical target information.  In that setting a label-marginalized
  \(I(X;Y)\) objective is different and is not implemented here.

## 10. Claim and evidence matrix

| Claim | Mathematical status | Code status | Test status | Safe paper wording |
|---|---|---|---|---|
| Exact total covariance of the finite mixture | Proved, Proposition 1 | Implemented | Direct state and measurement test | “Exact under the compressed mixture belief” |
| Within/mode MI decomposition | Proved, Theorem 1 | Implemented | Single-Gaussian exact case; mixture sandwich check | “Rao--Blackwellizes within-branch information and samples only mode information” |
| Lower and two-way upper sandwich | Proved, Theorem 2 | Implemented with roundoff guard | Tested on Gaussian and mixture examples | “Certified under the local mixture surrogate” |
| Safe action pruning | Proved for one-step raw MI | Implemented only at one-step leaves/H1 | Dominated-action test | Root beam/candidate-cap truncation is heuristic |
| Dimension at most \(\sum r_k+K-1\) | Proved for exact span | Numerical SVD implementation | 200-to-3 test | State numerical truncation explicitly |
| Exact projected mixture conditioning | Proved for exact span | Implemented with PSD floor | Direct full-update test | “Exact for the retained finite-mixture surrogate, up to numerical truncation” |
| PSD posterior factors | Proved | Implemented by factor form and eigenvalue floor | Implicit covariance comparison | Does not imply diffusion calibration |
| Gaussian no-adaptivity | Proved under frozen linear-Gaussian assumptions | Algebra is consistent | No dedicated test | Use as motivation for retaining modes, not as an empirical result |
| Horizon-two root regret | Conditional theorem only | No confidence-certified selection | Functional router test only | Do not claim a finite-sample regret guarantee yet |
| Joint Gaussian target information | Proved, Theorem 7 | Implemented | Scalar analytic test | One-branch diagnostic |
| Mixture target information \(I((Z,X);Y)\) | Proved, Theorem 8 | Implemented for H1 | Single-branch analytic and two-mode tests | Branch label must be target-relevant |
| Nonnegative known-operator optimism | Proved only for independent target/nuisance | Independence created by augmentation, not checked by diagnostic | Independent scalar test only | State the independence assumption beside the claim |
| Diffusion posterior calibration | Not proved | Factors are supplied by external sketch code | Requires experiments | Present as an empirical hypothesis |
| Better reconstruction or cross-domain generalization | Not implied by theory | Runners exist | Requires locked benchmark evidence | Do not infer from these algebraic tests |

## 11. Minimum additions needed for stronger theorem-facing claims

The current implementation supports a rigorous local-belief story, but the
following additions are required before stronger planning claims are safe.

1. Replace root immediate-information screening with a valid horizon-two
   upper bound, or report it as heuristic and measure
   \(\Delta_{\mathcal B}\) by exhaustive small-bank audits.
2. Make all screens cost-aware when unequal costs are enabled.
3. Add confidence-aware beam retention and nested Monte Carlo error
   propagation, or avoid a finite-sample regret claim.
4. Quantify information lost by numerical subspace truncation, for example
   through discarded whitened singular values.
5. Add repeated-seed calibration tests for mode-information bias, standard
   errors, fantasy value, and root-selection error.
6. Add a correlated image/nuisance counterexample test so that future edits
   cannot silently assume “optimism” is always nonnegative.
7. Before extending target information to H2, specify whether branch labels
   are part of the scientific target, keep target rows stable through
   conditioning, and add exhaustive small-bank fantasy-tree tests.

These gaps do not invalidate the exact finite-mixture algebra.  They delimit
which parts are presently theorem-backed and which remain experimental
research contributions.
