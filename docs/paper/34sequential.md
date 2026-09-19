\subsection{The Sequential Learning Problem}
\label{sec:learning_problem}

The preference model above turns the firm's design decision into a
sequential learning problem. The valid region is constructed before
the survey begins, but the consumer's preferences within that region
are unknown. Each trusted evaluation provides information about those
preferences at a cost, while a digital-twin evaluation provides a
cheaper assessment whose relationship to the consumer's actual choice
is unspecified. The firm must therefore decide where to collect
trusted observations as the study proceeds, using the information
already available to guide subsequent queries. The two objects it
seeks to learn are the preference parameter that explains the
consumer's choices and a valid design that performs well under those
preferences.

\paragraph{Preferences and the design target.}
We condition throughout on the offline construction of the design
space and treat the resulting valid region
$\mathcal M \subset \mathbb R^d$ as fixed, nonempty, and compact.
A design is a vector $\boldsymbol z \in \mathcal M$, and its augmented
feature vector is
\[
\boldsymbol\phi(\boldsymbol z)
:=
(1,\boldsymbol z^\top)^\top
\in \mathbb R^{d+1}.
\]
Let $\Theta \subseteq \mathbb R^{d+1}$ denote the parameter space.
For any candidate preference parameter
$\boldsymbol\theta \in \Theta$, define
\begin{equation}
p_{\boldsymbol\theta}(\boldsymbol z)
:=
\sigma\!\left(
\boldsymbol\theta^\top\boldsymbol\phi(\boldsymbol z)
\right),
\qquad
\sigma(u):=\frac{1}{1+e^{-u}}.
\label{eq:learning_win_probability}
\end{equation}
The target consumer has an unknown, fixed parameter
$\boldsymbol\beta^\star \in \Theta$, so the true win-probability
function is
$p(\boldsymbol z)=p_{\boldsymbol\beta^\star}(\boldsymbol z)$.
The parameter includes both the intercept, which incorporates the
fixed competitor's appeal, and the coefficients governing how design
coordinates affect the log-odds of winning. The online learning
problem takes this logistic specification as the model of trusted
choices; it does not assume that the digital twin follows the same
model.
Denote
\begin{equation}
p^\star
:=
\max_{\boldsymbol z\in\mathcal M}
p_{\boldsymbol\beta^\star}(\boldsymbol z),
\qquad
\mathcal Z^\star
:=
\operatorname*{arg\,max}_{\boldsymbol z\in\mathcal M}
p_{\boldsymbol\beta^\star}(\boldsymbol z),
\label{eq:learning_optimum}
\end{equation}
where the maximum is attained because $\mathcal M$ is nonempty and compact
and the win-probability function is continuous. We use
$\boldsymbol z^\star$ to denote any element of $\mathcal Z^\star$;
the optimal design need not be unique. Since the logistic link is
strictly increasing, we have $
\mathcal Z^\star
=
\operatorname*{arg\,max}_{\boldsymbol z\in\mathcal M}
\boldsymbol\beta^{\star\top}
\boldsymbol\phi(\boldsymbol z)$. 
The intercept is constant across designs, so it affects the level
of the win probability but not the maximizing design under this
specification. All design comparisons in the learning problem are
made relative to $\mathcal M$. They do not assert optimality over
coherent product images that the offline construction may have
excluded.

\paragraph{The two evaluation channels.}
A trusted query at $\boldsymbol z$ shows the rendered design against
the fixed competitor and returns a binary response: one if the
consumer chooses the design and zero otherwise. Its conditional
success probability is
$p_{\boldsymbol\beta^\star}(\boldsymbol z)$. This is the meaning of
trust in the formulation: the response follows the preference model
defining the firm's target, not a possibly different model supplied
by the digital twin.

A digital-twin query at the same design returns the score
\begin{equation}
A(\boldsymbol z)=g_{\mathrm{AI}}(\boldsymbol z),
\qquad
g_{\mathrm{AI}}:\mathcal M\longrightarrow\mathbb R.
\label{eq:learning_twin_score}
\end{equation}
We treat $g_{\mathrm{AI}}$ as a fixed but unknown function whose
values become available through queries. Its output may represent
a predicted win probability or another preference score, and its
scale need not match that of $p_{\boldsymbol\beta^\star}$. The
baseline problem places no calibration, ranking, or functional-form
restriction on the relationship between the two functions. The twin
may agree with the consumer in some regions and disagree in others,
and its errors need not average to zero. Thus a twin score is an
auxiliary assessment of a design rather than a trusted observation
of the consumer's choice. Any additional assumption connecting the
twin to the consumer must be stated separately.

\paragraph{The sequence of observations.}
We index the online study by trusted evaluations. Let $T$ be the
trusted-label budget, and let $\mathcal F_{t-1}$ denote the
information accumulated through trusted evaluation $t-1$, including
all digital-twin queries made up to that point. 

Before acquiring trusted response $t$, the firm may purchase a
finite number $B_t\geq 0$ of additional digital-twin evaluations.
Denote the resulting query--score record by
\begin{equation}
\mathcal A_t
:=
\left(
    \left(
    \boldsymbol z^A_{t,j},
    g_{\mathrm{AI}}(\boldsymbol z^A_{t,j})
    \right)
\right)_{j=1}^{B_t},
\qquad
\boldsymbol z^A_{t,j}\in\mathcal M.
\label{eq:learning_twin_record}
\end{equation}
Both the number of queries and their locations may depend on
previously available information. These queries may also be made
sequentially, so a later twin query can depend on scores already
returned in the same stage. Each decision must use only information
available when it is made.

Next, the firm selects a
design $\boldsymbol z_t\in\mathcal M$
and observes $
Y_t
\sim
\operatorname{Bernoulli}\!\left(
p_{\boldsymbol\beta^\star}(\boldsymbol z_t)
\right)$ for $
t=1,\ldots,T.
$
After receiving the response, the information is updated. In particular, a twin score may influence which design is shown to
the trusted evaluator, but it does not change the conditional
distribution of the trusted response at that design.

This information structure permits adaptive querying without
specifying an algorithm. The firm may revisit a design or query
a point that has never been evaluated, and a trusted query need not
be one of the designs scored by the twin. 

\paragraph{The budget and learning outputs.}
The primary budget is the number of trusted evaluations. If the
study uses $T$ trusted queries and
\[
N_A(T):=\sum_{t=1}^T B_t
\]
digital-twin queries, its evaluation cost under the cost notation
introduced above is
\begin{equation}
C_T=c_HT+c_AN_A(T).
\label{eq:learning_evaluation_cost}
\end{equation}
The inequality $c_A\ll c_H$ motivates treating $T$ as the binding
budget, but does not make twin evaluations literally free. We
retain $N_A(T)$ when reporting total evaluation cost. The
formulation requires only finitely many twin calls before each
trusted evaluation and does not otherwise fix their number or
prescribe their allocation.

At the end of the study, the firm returns an
$\mathcal F_T$-measurable parameter estimate
$\widehat{\boldsymbol\beta}_T\in\Theta$ and a recommended design
$\widehat{\boldsymbol z}_T\in\mathcal M$. The parameter estimate
targets $\boldsymbol\beta^\star$, while the recommendation targets
a design with win probability close to $p^\star$. For a tolerance
$\varepsilon>0$, the latter requirement is
\begin{equation}
p_{\boldsymbol\beta^\star}
\!\left(\widehat{\boldsymbol z}_T\right)
\geq
p^\star-\varepsilon.
\label{eq:learning_recommendation_target}
\end{equation}
This is the desired property of the recommendation, and it can be evaluated into the regret
\begin{equation}
R_T := \sum_{t=1}^T \big[\sigma(\bbeta^{\star\top}\bphi(\bz^\star)) - \sigma(\bbeta^{\star\top}\bphi(\bz_t))\big].
\end{equation}

The two learning targets are related but distinct. Learning the
preference parameter addresses the firm's interest in the structure
behind consumer choice, while learning a high-win-probability
design addresses its product decision. The formulation does not
prescribe an estimator, a rule for selecting trusted queries, or
a way of using the digital twin. It specifies the feasible designs,
the unknown preference model, the information available at each
decision, and the budget under which those two targets must be
pursued.