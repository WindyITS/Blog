---
tags: [statistics, python]
created: 2025-10-09
---

The big question: can we use regression causal inference? The answer is yes, but is not that simple. The idea is that we would like to interpret the parameters of our regression as *causal effects* of the regressor over the dependent variable $Y$, but how do we identify those parameters that, in regression terms, only catch correlation and not causality? 

Here the POF comes in help, because just the regression is not enough: correlation does not imply causality, therefore we can't just interpretate our $\beta$ as causal effects. We need, instead, to see the things the other way around: we use causal effects identified by the POF as parameters for our regression.

## The link between POF and Regression
To link POF and regression we can use the following trick: we can express the causal framework with a regression-like format, suggesting that we can adapt it to serve as the regression that we need:
$$
Y_{i} (D)= E[Y_{i}(D) + U_{i}(D)
$$
This is exactly a regression: *the value of the dependent variable for the i-th unit and for a certain value of $D$ depends on the average outcome with respect to that $D$ plus an error given by the individual properties of the i-th unit*.
Also, notice: 
$$
E[Y_{i}(D) |D] = Y_{i} \implies Y_{i} = Y_{i}
$$
That's because we expect our error, as in every regression, to have an expected value when conditioned on the regressor equal to zero:
$$
E[U_{i}|D_{i}=1] = E[U_{i} | D_{i} =0] = 0
$$

From here we can go further, because we can build from zero the POF by using regression. The first thing to do is to rewrite the **individual treatment effect**: 
$$
\begin{align*}  
\Delta_{i} &= Y_{i}(1) - Y_{i}(0) \\
&= (E[Y_{i}(1)] + U_{i}(1)) - (E[Y_i(0)] + U_i(0))\\ 
& = (E[Y_{i}(1)] - E[Y_{i}(0)]) + (U_{i}(1)-U_i(0))\\
& = \delta^{ATE} + (U_{i}(1)-U_i(0))  
\end{align*}
$$
This is very powerful. Notice: if error is the same in both potential case, then the individual treatment effect is exactly equal to the ATE.
The ITE for any unit in the population is given by the sum of two parts:
- **average treatment effect (ATE)**: this is the common effect of the treatment between all units
- **idiosyncratic effect for that unit**: this is the *unobserved heterogeneity* in the treatment effect across units, meaning that the effect for a single unit will be different from the ATE because of specific characteristics of that particular unit

Now, let's bridge the two worlds: the one of the potential outcomes that we just described and the one of realized outcomes, the data that we actually observe. We know from the POF that:
$$
Y_{i} = Y_{i}(1)\cdot D_{i} + (1-D_{i}) \cdot Y_{i}(0)
$$
So, we can substitute inside the observed outcome $Y_{i}$ the potential outcomes coming from $Y_{i}(D)$. Doing so, after some calculations, leads us here:
$$
\begin{align*}
Y_i &= E[Y_i(0)] + \{E[Y_i(1)] - E[Y_i(0)] + U_i(1) - U_i(0)\} \cdot D_i + U_i(0) \\
&= E[Y_i(0)] + \Delta_i \cdot D_i + U_i(0)
\end{align*}
$$
Look at how powerful this model is. The outcome that we observe is son of:
- $E[Y_{i}(0)]$ -> **expected value of not being treated**: it acts as our baseline from which we start building up the consequences of the treatment
- $\Delta_i \cdot D_i$ -> **treatment effect on the i-th unit**: this is the switch in the model, meaning that if the unit gets treated then the outcome will go up for the effect of the treatment on that individual
- $U_{i}(0)$ -> **heterogeneity on the starting line**: it tells us how much the baseline for our unit is distant from where we expect we start, value given by the expected value of not being treated

In regression terms we know what this is: a simple bivariate linear regression model where the coefficient on the treatment dummy $D_{i}$ is the *individual treatment effect* $\Delta_{i}$. This is also known as **random coefficient model**, and can be rewritten as follows:
$$
Y_{i} = \beta_{0}^i +\beta_{1}^iD_{i} +v_{i}
$$
where $\beta_0^i = E[Y_i(0)]$, the coefficient is $\beta_1^i = \Delta_i$ and the error term is $v_i = U_i(0)$, which has $E[v_i]=0$ as a consequence of $E[U_{i}|D]=0$.

We can develop this model further if we are interested in **ATE**, because we can express the regression in its terms. To do this, we just have to substitute $\Delta_{i}$ inside the model:
$$
\begin{align}
Y_{i} & = \beta_0^i \cdot [\delta^{ATE} + (U_{i}(1)-U_i(0))] \cdot D_{i} + v_{i} \\
&= E[Y_i(0)] + \delta^{ATE} \cdot D_i + \{U_i(0) + D_i \cdot [U_i(1) - U_i(0)]\} \\
\end{align}
$$
Which in the end becomes:
$$
Y_{i} = \beta_0^i + \beta_1 \cdot D_i + \varepsilon_i
$$
where we have something new:
- $\beta_{1}$ -> **average treatment effect**: if the unit gets treated, then its outcome gets "updated" with the average effect of the treatment over the all population
- $\varepsilon_i = U_i(0) + D_i \cdot [U_i(1) - U_i(0)]$ -> **error term**: we see that if the unit doesn't get treated then the only error taken onto account is the one that changes the race's starting line, but if it gets treated then the error also considers the different effect of the treatment over that unit

Notice: under **homogeneity assumption** ($U_{i}(1)=U_{i}(0)=U_{i}$) we have that the treatment effect is constant for all individuals because $\Delta_i = \delta^{ATE}$. In other words, we are assuming that the potential error is equal in both cases, so that it cancels out when it comes to impact on the treatment effect.

## Endogeneity and Selection Bias
The just discussed regression model seems to provide a direct path to estimate the ATE, because it can be viewed as a parameter of the regression coming from OLS. BUT, there is a fundamental problem: **the error term is correlated with the regressor**. Notice: $\varepsilon_{i}(D_{i})$, meaning that the error depends from the value that our regressor $D_i$ assumes. This is a big problem, because it violates our Zero Conditional Mean assumption: $E[\varepsilon_i|D_i]=0$.

This correlation comes from a *non-random selection*, and is also known as **endogeneity**. It implies that our OLS estimation will give us a *biased* estimate of ATE. Now, we can see this mathematically by using the *naive estimator* and its relative decomposition:
$$
NE = E[Y_{i}|D_{i}=1]-E[Y_{i}|D_{i}=0]
$$
Which we know can be decomposed into:
$$
NE = \delta^{ATE} + \text{ bias term} +\text{heterogeneity term}
$$
Now, if we substitute into the $\text{NE}$ formula everything we've discussed now and adjust a little the terms, we get to:
$$
\begin{align*}
\text{NE}= \delta^{ATE} + \underbrace{\{E[U_i(0)|D_i=1] - E[U_i(0)|D_i=0]\}}_{\text{selection term}} 
+ \underbrace{(1 - P(D_i=1)) \cdot \{E[\Delta_U|D_i=1] - E[\Delta_U|D_i=0]\}}_{\text{heterogeneity term}}
\end{align*}
$$
where $\Delta_U = U_i(1) - U_i(0)$. So, the comparison of the means is a biased estimator for the ATE unless both the terms are zero and, if we sum them up, they become: $E[U_i(1)|D_i=1] - E[U_i(0)|D_i=0]$. What is this telling us is the following: the naive estimator is a good estimator for the ATE, but this is true IF AND ONLY IF the *error term is not correlated with the regressor*, or, in other words, the *error term is the same across the two groups*. If that's the case, then the two groups I'm using to estimate the ATE are statistically equal, implying that $NE$ is an excellent estimator.

To go further, let's see what each term tells us:
- **selection term**: captures the differences in the starting line between the two groups caused by the error, which is basically unobserved regressors for our dependent variable
- **heterogeneity term**: captures the difference in the treatment effects given by unobserved characteristics that differ across the two groups

Sadly, in reality is really hard to have a case where we are sure that errors are completely not correlated with the regressor, so we have to find a solution.

### Solution to Selection Bias
There are to main solutions to allow the regression coefficient to be interpreted as the ATE:

#### Randomized design
A perfectly **randomized experiment** ensures that treatment assignment is statistically independent of potential outcomes, because I'm selecting units from the population without looking at their characteristics. This means that, on average, the two groups *before* the treatment are perfectly equal.

The implication is simple: the expected value of any unobserved component (error term) it's the same across the two groups because is zero for each of them! If randomly take units from the population, I expect the units to, on average, be equal one to the other, so that the expected error on each unit becomes zero:
$$
    \begin{cases}
    E[U_i(1)|D_i=1] = E[U_i(1)|D_i=0] = E[U_i(1)] = 0 \\
    E[U_i(0)|D_i=1] = E[U_i(0)|D_i=0] = E[U_i(0)] = 0 
    \end{cases} 
$$

The result of this randomization is having no error in the estimation of ATE, because both terms are zero: $E[U_i(1)|D_i=1] - E[U_i(0)|D_i=0] = 0$. So, here we have:
$$
\text{NE} =\beta_{1}= \delta^{ATE}
$$

#### Mean Independence Assumption in observational data
When we are in **observational settings** we don't have the luxury of randomization, thus the only thing we can do is to *assume* mean independence; in other words, we make the assumption that the unobserved characteristics are mean independent of treatment status (equal for the two groups):
$$
    \begin{cases}
    E[U_i(1)|D_i=1] = E[U_i(1)|D_i=0] \\
    E[U_i(0)|D_i=1] = E[U_i(0)|D_i=0]
\end{cases}
$$
This condition is much weaker than the one in the randomized case, because it doesn't imply by experiment design to have expected value equal zero; still, it does the job perfectly: the errors cancel out and our naive estimator estimates ATE perfectly.

This assumption is takes us to the sample place we would arrive by assuming the **ZCM condition** ($E[\varepsilon_i|D_i] = 0$) because the last one is a stronger condition the implies our mean independence assumption.

## Selection on Observables
In observational studies, the assumption of unconditional mean independence is often not credible. A more plausible approach is to assume that selection into treatment is not random overall, but becomes random after accounting for a set of observable covariates $X$. In other words, the systematic differences across the groups are due to *observable* characteristics. In this case, the *selecion on observables* approach can solve the problem.

#### Conditional Independence Assumption - CIA
The core of this approach is the **Conditional Independence Assumption (CIA)**, also known as **Unconfoundedness**, which states that *conditional on a set of observable covariates $X$, the treatment assignment $D$ is statistically independent of the potential outcomes $Y(0)$ and $Y(1)$*.
$$
D \perp \{Y(0), Y(1)\} | X
$$

The idea is that for any sub-population defined by a specific value of $X$, the treatment assignment is effectively random. Again, why is this independence so important? Because it allows us to have unbiased estimations of the parameters.

#### Conditional Mean Independence - CMI
The **conditional mean independence** is a weaker version of the CIA which states that, *conditional on $X$, the mean of the potential outcomes is the same regardless of treatment status*. What is the difference with the CIA? Well, unconfoundedness means that the entire distribution of potential outcomes is independent from the treatment status when conditioned on $X$, but instead the CMI only requires for the means to be independent. Mathematically:
$$
\begin{cases}
E[Y_i(0)|D_i=1, X_i] = E[Y_i(0)|D_i=0, X_i] \\
E[Y_i(1)|D_i=0, X_i] = E[Y_i(1)|D_i=1, X_i]
\end{cases}
$$
You see? We don't care if $D$ and the potential outcomes are independent given $X$, we want only to be sure that their means are, because that's what unlocks for our coefficients' estimator to not have bias and heterogeneity.

Also, CMI is useful because allows us to see the counterfactual: the average of the counterfactual given $X$ is exactly equal at the average of what I observe given $X$. So, when I notice an average difference between treated and untreated with the same $X$, I can attribute that difference to the treatment and not to pre-existing differences in the groups.

Now, if we play under conditioning on $X$ we must also consider that we cannot use anymore the average treatment effect over the all population, because we are more interested in the *conditional average treatment effect* (**CATE**), which is the average effect of the treatment only on the units with characteristics $X$. Mathematically:
$$
\delta^{CATE} = E[Y_{i}(1)|X_{i}] - E[Y_{i}(0)|X_{i}] = E[Y_{i}|X_{i},D_{i}= 1] -E[Y_{i}|X_{i},D_{i}= 0]
$$

#### Implementation with regression
Thanks to who designed this universe, regression provides an easy way to perform this conditioning over $X$. In particular, we can compare the mean outcomes across treatments *within* levels of $X$! Now, this depends on regression specification:
- **saturated model**: if $X$ is discrete and the sample size is large enough, we can estimate a fully saturated model. Suppose $X$ is a binary variable:
$$
E[Y_i|D_i, X_i] = \alpha + \beta D_i + \gamma X_i + \rho (D_i \cdot X_i)
$$
Here, the CATEs are directly estimated by the coefficients: for $X=0$ we have $\delta^{ATE}(X=0) = \beta$, while for $X=1$ we have $\delta^{ATE}(X=1) = \beta + \rho$. 
- **homogeneous ATE model**: if we have the courage to assume that the CATE is constant across all values of $X$, then we can use a different model that doesn't need the interaction term. Mathematically:
$$
E[Y_i|D_i, X_i] = \alpha + \beta D_i + \gamma X_i
$$
in this case $\beta$ is the constant ATE.

The last assumption is very strong, but is usually done when $X$ is high-dimensional (can take a lot of values), because the *curse of dimensionality* makes the saturated model impossible.

#### What $X$ should I pick?
The fundamental question: "which is the correct $X$ that makes the CMI assumption hold?". Well, in general the choice requires a theory about the selection process and how we want to make it, because in the end the best $X$ is the one that includes **all** confounders that are correlated with both the treatment status and the outcome such that CMI holds.

## Selection on Unobservables
When the systematic differences across groups are not due to observable characteristics but to *unobservable* ones, the solution is not that easy because we can't put in the regressor these features.

To better visualize this, let's model first the **selection process**. The core of the problem lies in *how* units are selected into treatment, because we know that if they were taken randomly then none of this problems would exist. In reality this very rarely happens, meaning that there is a systematic difference across groups because units that belong to a group and units that belong to another one belong to that group for specific observed and / or unobserved characteristics, creating an important bias when confronting.
We can model the treatment status as follows:
$$
D_{i} = \alpha_{0} + Q^T_{i}\theta+V_{i}
$$
where:
- $D_{i}$ -> **treatment status**: wether the i-th unit goes in the treated group or in the untreated one
- $\alpha_{0}$ -> **base propensity to treatment**: the tendency for the i-th unit to get selected in the treated group
- $Q_{i}$ -> **observable variables**: vector containing all the observable variables that influence wether a unit gets treated or not
- $\theta$ -> **coefficient vector**: contains the weight / effect of each observable variable on treatment selection
- $V_{i}$ -> **unobservable variables**: vector containing all the unobservable variables that influence the treatment status

Now, in the theoretical world we assume $E[V_{i}] = 0$, but in reality this is very rare to happen, meaning that usually there is a *selection bias* caused by some variables that we do not observe being correlated both with the *treatment status* and with the *potential outcome* of our model. 

### Model with homogeneity assumption
To start understanding how to play our cards when we have unobservables it's best to make the **homogeneity assumption**, stating that the individual treatment effect is equal on every unit, which means the error in both potential cases is equal so that ITE = ATE:
$$
U_{i}(0) = U_{i}(1) = U_{i} \implies \Delta_{i} = E[Y_{i}|D_{i}=1]-E[Y_{i}|D_{i}=0] = \delta^{ATE}
$$
This simplification is done to eliminate the *heterogeneity term* from the analysis, which complicate the discussion quite a bit. It will discussed later on.

So, with this assumption we can take our model which will be:
$$
Y_{i} = \beta_0 + \delta^{ATE} \cdot D_i + U_{i}
$$
Well, here we know that $U_{i}$ represents all unobserved factors affecting the outcome of $Y_{i}$ and, by construction of the regression, we know that $E[U_{i}]=0$ because their average effects enters $\beta_{0}$.

Now, *if* the unobserved factors are linked to the treatment status we have **endogeneity**: ZCM condition doesn't hold, therefore the covariance is not zero:
$$
\text{Cov}(D_{i},U_{i})≠0
$$
The problem is here: the OLS estimator requires *exogeneity*, because it acts as the condition was true when you minimize the FOCs in the derivation. Then, if the condition is not satisfied we will have biased estimators. Let's try go look at the mathematics:
$$
\begin{aligned}
\hat{\delta}^{OLS} &= \frac{\operatorname{Cov}(Y_i, D_i)}{\operatorname{Var}(D_i)} \\
\\
&= \frac{\operatorname{Cov}(\mu_0 + \delta^{ATE} D_i + U_i, D_i)}{\operatorname{Var}(D_i)} \\ \\
&= \frac{\operatorname{Cov}(\mu_0, D_i) + \delta^{ATE}\operatorname{Cov}(D_i, D_i) + \operatorname{Cov}(U_i, D_i)}{\operatorname{Var}(D_i)} \\ \\
&= \frac{0 + \delta^{ATE}\operatorname{Var}(D_i) + \operatorname{Cov}(U_i, D_i)}{\operatorname{Var}(D_i)} \\ \\
&= \delta^{ATE} + \underbrace{\frac{\operatorname{Cov}(U_i, D_i)}{\operatorname{Var}(D_i)}}_{\text{Bias Term}}
\end{aligned}
\quad
$$

We can see it here: the OLS captures the true causal effect plus a bias term that arises from the correlation between unobservables affecting the outcome and treatment status. Our goal is clear: find a way to isolate ATE even when this correlation do exists and we cannot do anything about it because is based on something we do not observe / quantify.

### Instrumental Variables - IV 
A solution to the endogeneity problem arising from correlation between the error and the regressor can be found using **instrumental variables** (**IV**). The core idea is that we can identify an *observable variable* $Z$, called "*instrument*" that can *explain a part of the variation of our regressor* $D$ that is uncorrelated with the error term $U$.

In other words, we are saying that we are in search of an instrument, so a variable, that is directly correlated with our main regressor but not with the outcome $Y$ nor with the error $U$, so that I can use the part of variance of the regressor explained by $Z$ to isolates its effect over the outcome.

For a variable to be an instrument, there are **three fundamental conditions**:
- **relevance condition**: the instrument $Z$ must be related to the regressor, which in our case is $D$. So:
$$
\text{Cov}(Z,D)≠0
$$
This has a very easy intuition: if we want Z to explain a part of $D$, then we need them to be linked by something. If $Z$ is completely unrelated, then it gives us no information about the variation of $D$ and so is useless. Now, very important: this covariance should not just come out from "mindless" statistical correlation, but it should come out from a **causal mechanism**, from a *why* $Z$ explains $D$. The reason is simple: we want to have stability in the relation, and to be sure of that we need some true explanation behind the scene. Also, notice: this condition is *provable* because we do observe both $Z$ and $D$
- **validity / exogeneity condition**: the instrument $Z$ must be uncorrelated with the unobservable factors $U$, because if it was then it would not be helpful in capturing the unexplained part of $D$ left. So:
$$
\text{Cov}(Z,U) = 0
$$
The intuition is simple: we want $Z$ to explain the part of  that is not explained by $U$, therefore we need a instrument that has nothing to do with $U$ at all, because if they were linked then it's link with the regressor would explain a part that is, at least surely, not entirely "clean" from the impact of what we cannot observe. Notice: this condition is *not provable* because we do not observe $U$; so we have to assume it.
- **exclusion restriction**: the instrument $Z$ must influence the outcome $Y$ only through $D$. So, we want to be sure that:
$$
Y= \mathcal{f} (D,U,\varepsilon)
$$
In other words, we want for $Z$ to not be a parameter of the outcome's function. Again, the intuition is not complex: if $Z$ ha da direct impact on $Y$, then it would mean that is either an omitted observable variable or an unobservable one in $U$. The second case can be excluded because just the fact that we are using $Z$ it means that we can observe it; instead, the first clearly means that $Z$ should be used in the model to avoid biases, so it can't be useful to explain the "unexplained" part of $X$.

### 2SLS - Two Stage Least Squares
How do we procede from here? Well, our objective is the following: estimating the real impact of $D$ over our dependent variable $Y$, but unobservables are in the way. We get rid of them by using an IV which explains the part of $D$ that is uncorrelated with $U$, meaning that if we compute that $D$ and regress $Y$ over it, we have the true impact of it on the outcomes.

To do this, we first of all have to get "cleaned" regressor and, to do this, we run a regression of $D$ over $Z$, saying: "we estimate the regressor with the instrumental variable because, by the exogeneity condition, we know that the estimated regressor won't be correlated with unobservables".
This first step is called "**first stage**":

#### First stage
Let's strip away the "bad" variation in $D$ by regressing it on something that explains only its clean part. When we do this, we must ensure we account for all other observable, exogenous factors, $W$, which act as **control variables**. These controls must be included to isolate the effect of $Z$ on $D$ after accounting for everything else that is "clean."

The regression uses the endogenous variable $D$ as the dependent variable, regressed on the instrument $Z$ and all control variables $W$:

$$
D = \gamma_{0}+\gamma_{1}Z+\mathbf{W}\boldsymbol{\alpha} +V
$$

We are asking: how much does $Z$ influence $D$, controlling for $W$? We also include $W$ because they might already explain a large part of $D$. By including $W$, we ensure $Z$ only captures the *remaining* variation in $D$.

Now we run the regression and calculate our predicted values of $D$:

$$
\hat{D}=\hat{\gamma}_{0}+\hat{\gamma}_{1}Z + \mathbf{W}\hat{\boldsymbol{\alpha}}
$$

Why is the error term $V$ not used? Because $V$ is the unobserved residual in the first stage. So we are  excluding it because $V$ might still be correlated with the structural error term $U$ but we only want the part of $D$ described by the exogenous factors.

So, now we have $\hat{D}$, which is, by construction, *exogenous* with respect to the original structural error term $U$. This is because $\hat{D}$ is built as a linear combination of $Z$ and $W$, and both $Z$ and $W$ satisfy the exogeneity condition ($\text{Cov}(Z, U) = 0$ and $\text{Cov}(W, U) = 0$).
In other words, $\hat{D}$ represents the part of $D$'s variation that is explained *only* by the instrument $Z$ and the control variables $W$. 

*Notice*: the simplest case is where there is no $W$.

#### Second stage
It is now time to estimate the causal effect of $D$ over $Y$. To do this we replace the original regressor $D$ with its "clean" version $\hat{D}$. As usual, we run a regression of the outcome $Y$ over our predicted values $\hat{D}$ also including the control variables $W$ from the first stage:

$$
Y =\beta_{0}+\beta_{1}\hat{D}+\mathbf{W}\boldsymbol{\theta}+\varepsilon
$$

What is the intuition here? We are estimating the effect of $D$ on $Y$, but we must hold $W$ constant. If we omitted $W$ here, the estimated $\hat{\beta}_1$ would absorb any remaining causal link between $W$ and $Y$ that happens to correlate with $\hat{D}$ so, by including $W$ in both stages, we fully isolate the causal effect of $D$ on $Y$. Also, notice that the parameter vector for $W$ has changed, it's now $\theta$: the impact of $W$ on $Y$ is not necessarily the same of the one on $D$ and it can be also null ($\theta=0$).

We have arrived: $\hat{\beta}_{1}$ in this case is the true estimation of the impact of $D$ over $Y$, because the regressor $\hat{D}$ is completely cleansed of any correlation with the unobserved error term $U$. To see it explicitly, remember our outcome model:
$$
Y = \beta_0 + \beta_{1} \cdot D + U
$$
By using the IV, we arrive to the point where: $\hat{\beta}_{1} = \hat{\delta}^{ATE}$, so:
$$
Y = \beta_0 + \hat{\delta}^{ATE} \cdot D + U
$$

### Ratio derivation and interpretation
There is also another and easier way to arrive to $\delta^{ATE}$, but with a big difference: it is simpler only in the base case where there are no control variables and the IVs are only one. Still, in this simplest case we can pass from the covariance of the outcome with our instrument:
$$
\operatorname{Cov}(Y, Z) = \operatorname{Cov}(\beta_0 + \delta^{ATE} D + U, Z)
$$
And by using the property of covariance:
$$
\operatorname{Cov}(Y, Z) = \operatorname{Cov}(\beta_0, Z) + \delta^{ATE} \operatorname{Cov}(D, Z) + \operatorname{Cov}(U, Z)
$$
$$
\operatorname{Cov}(Y, Z) = 0 + \delta^{ATE} \operatorname{Cov}(D, Z) + \operatorname{Cov}(U, Z)
$$
$$
\operatorname{Cov}(Y, Z) = \delta^{ATE} \operatorname{Cov}(D, Z) + \operatorname{Cov}(U, Z)
$$
Let's now isolate the ATE, action that we can do because we know by condition that $\text{Cov}(D,Z)≠0$:
$$
\delta^{ATE} = \frac{\operatorname{Cov}(Y, Z)}{\operatorname{Cov}(D, Z)} - \frac{\text{Cov}(U,Z)}{\text{Cov}(D,Z)} = \delta^{IV}
$$
Notice: the second term should not exists if IV is a perfect instrumental variable, because it gets taken out by the second condition. Still, given the fact that is not provable, it may be different from zero and introducing bias. So, we rewrite the following as:
$$
\delta^{ATE} = \frac{\operatorname{Cov}(Y, Z)}{\operatorname{Cov}(D, Z)} - \text{bias}
$$

This is also usually called the "**IV estimator**" and, as we can see, is defined by the ratio of these two covariances in the perfect case. From now on, unless specified, we'll consider the bias being zero.

How do we interpretate this? Well, we can rewrite the formula as:
$$
\delta^{ATE}=\frac{\operatorname{Cov}(Y, Z)/\operatorname{Var}(Z)}{\operatorname{Cov}(D, Z)/\operatorname{Var}(Z)} = \frac{\text{Effect of Z on Y}}{\text{Effect of Z on D}}
$$
Where, notice:
- *numerator*: is the slope coefficient from a regression of $Y$ on $Z$, and is called **reduced form**. It captures the total effect of the instrument on the outcome. It may sounds conflictual with the third condition (exclusion restriction), but it's actually not because it refers to the impact on $Y$ through $D$.
- *denominator*: is the slope coefficient from a regression of $D$ on $Z$, that we've already seen because it's $\gamma_{1}$ from the **first stage**. It captures the effect of the instrument on the treatment.

So, intuitively, why should I be able to rewrite my IV estimator like that? It all comes down to the IV logic, which answers the question: "for every unit change in treatment status $D$ that was *induced by the instrument* $Z$, how much did the outcome $Y$ change?". A qualitative way to see it is the following:
$$ Z \xrightarrow{\text{causes}} D \xrightarrow{\text{causes}} Y $$
So, the denominator measures the strength of the first link: "for a one unit change in $Z$, how many units does $D$ change in total?"; instead, the numerator measures the strength of all the chain: "for a one unit change in $Z$, how many units does $Y$ change in total?". Now, the logic becomes arithmetic: the change in $Y$ for every one unit change in $D$ must be the total change in $Y$ divided by the change in $D$ that caused it, which is in turn due to change in $Z$!

### IV as a Method of Moments (MM) Estimator
We can also see at instrumental variables as a *Method of Moments* estimator. Think about it: we can run our regression even if there is endogeneity in the model, because the way we estimate the OLS parameters is done by assuming exogeneity. In other words, nothing is different from what we've always done, but when we go in the derivation, if we want unbiased estimators, we know that we must have:
$$
E[D\cdot U] =0
$$
But the big problem is that we're discussing all of this exactly because this is not satisfied, given that:
$$
E[D\cdot U] ≠0
$$
In other words, our moment condition is "broken". Luckily, the IV comes in and fixes it by saying: "the rule doesn't work, but we have an instrument $Z$ which is correlated with $D$ and for which the rule works by construction". So:
$$
E[Z\cdot U] =0
$$
So, instead of estimating a biased OLS, by doing this we can actually get our IV estimator for the model:
$$
Y_i = \beta_0 + \delta D_i + \mathbf{W_i}\boldsymbol{\theta} +U_i
$$
From this, we just have to rearrange for the error and write down the the usual conditions, remembering that we also have to take onto account the exogenous variables $W$:
$$
\begin{align}
& E[1\cdot U_{i}] = 0 \to \text{intercept}  \\  \\
& E[W_{ik} \cdot U_i] = 0 \quad \text{ for each }k  \to \text{effect of the k-th variable}\\ \\ 
& E[Z_i \cdot U_i] = 0 \to \text{effect of the IV}
\end{align}
$$

This will give us a system of $2+K$ equations that, when solved, will give us $\hat{\beta}_0, \hat{\delta}, \hat{\boldsymbol{\theta}}$!
Seeing the framework from this point of view makes it crystal clear that we need **at least one instrument variable for each endogenous regressor**, because each endogenous variable will need its condition rewritten in terms of its relative instrument variable.

### The strength of an instrument
So, we have our nice and beautiful three conditions that allow us to identify an instrumental variable, but is that all? Not quite. The IV method requires on the instrument to have a **sufficiently strong relationship** with the endogenous regressor. This is very very important.

If the link (covariance) between $Z$ and $D$ is *weak*, we call it a "*weak instrument*", because the *first stage* will not be much accurate after all: the instrument doesn't have much power to explain the variation in $D$ after all. The intuition is simple: if our "clean" variation is just a little piece of the total variation, we're trying to learn about the whole effect from a very small piece of information which could be, without any problem, just noise.

Now, weak instruments have bad **consequences**:
- **bias in finite samples**: we know that our IV estimator is *consistent*, however if it's weak and we're in sample land, then it auto re-introduces bias. Why? Because may be it is not that good in explaining the variability in $D$ when we have so few data and such a low correlation: we might confuse causality / correlation with noise. The funny part is that if it's all noise, then using $Z$ gives us even more bias than just doing the OLS with the unobservables.
- **incorrect inference**: the standard errors that we can compute are very unreliable, because very small changes in $D$ given $Z$ are badly measured and they may also be caused by random noise. This will give us a small SE, so that when we use it to compute t-tests completely inflates the t-statistics, that leads to a small p-value, making us reject the null hypothesis even when it's true
- **magnification of tiny biases**: from the derivation of the IV estimator we can actually see the bias of it, which is:
$$
\text{Bias}(\delta^{IV}) = \frac{\operatorname{Cov}(U, Z)}{\operatorname{Cov}(D, Z)}
$$
So, if our condition on the numerator doesn't hold and the covariance between $D$ and $Z$ is very little, then a violation of validity (numerator) can be magnified into huge biases. Also, notice, we can see the first condition coming exactly out of this formula, given the fact that we have to assume the numerator because we can't be prove it.

#### How to Test for Weak Instruments
So, how do we know if we have this problem? We need to check the strength of our first stage. The standard practice is to look at the *F-statistic* for the joint significance of all the instruments in the first-stage regression. 

The classic **rule of thumb** is that if the *first-stage F-statistic is greater than 10*, we can have some confidence that our instruments are not weak. Still, in the last times the minimum value is very discussed but, in the end, the higher the better.

### Over-identification
Even if the big problem of IV is to actually identify them, there are some rare cases where we have *more* instruments $Z_1, Z_2,\dots,>Z_{k}$ than *endogenous* variables. This case is called "**over-identification**".

When we are in this situation a good thing to do is to test wether they roughly give the same estimation of our regressor, because if they don't then it's probable that at least one of them is not an IV at all. Careful: even in the case they all give the same values, it's still possible that all of them are invalid but biased in a similar way.

One last important thing to say is about *homogeneity*: the entire logic of this test rests on the idea that all instruments have the *same* causal effect over $D$. If $D$ changes, then different instruments might identify different average treatment effects caused by *heterogeneity*.
