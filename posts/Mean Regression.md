
# Mean Regression
Regression analysis aims to understand how the dependent variable $Y$ changes when other variables $X_{i}$ do. If it's true that we can be interested in a lot feature from the distribution of $P(Y|X)$, it's also true that the **mean regression** is by far the most popular choice for the analysis, especially in econometrics. Why is that? Two main reasons:
1. **when the goal is description or prediction**: the mean is a very powerful value, because it summarize the central location of a single variable. Now, $E[Y|X]$ tells us how does the average of $Y$ change along with $X$, showing us how the central location of the distribution of $Y$ evolves given the observation of another variable. Now, this goal is also very closed to minimizing the prediction error, because we know that if we choose $L(\cdot)=(\cdot)^2$, then $E[Y|X]$ is by far the best predictor for $Y$ given $X$. In this case, we call the conditional average the *Minimum Mean Squared Error* (MMSE) predictor, because it is the one that minimizes the loss.
2. **when the goal is causal inference**: the mean regression is highly compatible with the *potential outcomes framework* (POF), because all the causal parameters of interest (ATE, ATU, ATT) are all based on differences between means.

So, what we see is how much flexible and useful the mean can be in all the three possible objectives that we might have.

## CEF - Conditional Expectation Function
We can define the regression over the mean as the **conditional expectation function**, which depends on $X$. Now, this difference is key: the CEF describes the *true* relationship between the mean of $Y$ and $X$, therefore it does not belong in the sample land BUT in the population one. Mathematically:
$$
\text{CEF} = E[Y|X]
$$

The distinction between population land and sample land is key, because taking a look at CEF's properties we can understand the reason it is such an attractive target for modeling:
- **Law of Iterated Expectations**: given the fact that we work with means, we can use the LIE to compute the average of $Y$ by looking at the weighted average of the conditional averages $E[Y|X]$. Mathematically:
$$
E[Y] = E[E[Y|X]]
$$
- **CEF decomposition**: we can decompose the $i$-th value of $Y$ by a part that it's explained by $X$ and another, called the "error term", that is not. Mathematically:
$$
Y= E[Y|X] + \epsilon
$$
Now, the key part is that by starting from this condition and using LIE, we can easily arrive to a very important implication:
$$
E[\epsilon|X] = 0
$$
This can be read: "the expected value of the error conditioned on $X$ is zero", meaning that the average impact of the error term among all the $X$ is perfectly null. Why is this important? Because if it wasn't null then the error would have a systemic influence on $Y$, causing problems for the analysis. Also, this results implicate other two insights:
$$
E[{\epsilon}] = 0 \quad \quad Cov(X,\epsilon) = 0
$$
This property property is then very very powerful, because it has allowed us to divide the value we want to predict in two parts: the one we explain by $X$ and the one that, on average, is unrelated to $X$.

- **CEF best predictor**: we know for sure that when we have a squared-error loss, the CEF is the one that solves the following minimization problem:
$$
E[y_i|X_i] = \arg \min_{g(X_i)} E \{[y_i - g(X_i)]^2|X_i\}
$$
where we know $g(X)$ being any possible function of $X$ and $\text{arg}$ stands for "argument", meaning that we want that $g(X)$ which minimizes the expected value.

Let's now see what happens when we go back to the real world.

## Linear Regression Model
Let's get back to Earth. The CEF belongs to the population land, showing us the true distribution that sadly we can't observe in the real world. Now, the best thing we can do is find a somewhat way to approximate it and, the best way to do so, it's to use a **parametric model**, usually a **linear** one.

Now, starting from the decomposition property of the CEF:
$$
y_i = E(y_i|X_i) + \varepsilon_i = X_i^T\beta + \varepsilon_i
$$
which is the same thing as saying:
$$
y_i = \beta_0 + \beta_1 \cdot x_{i1} + ... + \beta_K \cdot x_{iK} + \varepsilon_i
$$
Let's break this all down:
- **model** **($E[y_{i}|X_{i}]=X^T\beta$)**: this is the core of the regression, because it's saying that we expect for the mean of $Y$ to change with respect to our explanatory variables "weighted" by the parameters of the model $\beta$.
- **error term**: it is a scalari that captures the gap between our model and reality, incorporating every determinant of $y_{i}$ that is not in the model
- **parameters ($\beta$)**: this is the vector which contains the intercept of the model with the $Y$-axis and all the other slope coefficients.

What is the main goal of a linear regression then? It's to estimate these parameters $\beta$ using the data AND *assumptions* about the error term.

### Estimation
To estimate our parameters we leverage the properties of the CEF, but given the fact that we are not actually working with the real one, instead of implication properties they become *assumptions*. The most important assumption of all is the **Zero Conditional Mean** (or Mean Independence), which is the first implication of CEF's decomposition:
$$
E[\epsilon|X] = 0
$$
As we said, this has two important implications:
- **unconditional mean of the error is null**: by applying the law of iterated expectation is clear that
$$
E[\epsilon]  =0
$$
- **error term is uncorrelated with regressors**: if $E[\epsilon]=0$, then it can be shown that also the covariance is null
$$
Cov(\epsilon,X) = 0
$$
This is a very important condition, because this implies
$$
Cov(\epsilon,X) = 0 \implies E[\epsilon \cdot X] = 0
$$

Arrived here, we can choose how to go forward: Method of Moments or Minimum Mean Squared Error? Which way? Well, luckily for us all roads lead to Rome.

#### Method of Moments
If we choose to use the **Method of Moments** (MoM), what we do is writing a system of linear equations (called *moment conditions*) to find our parameters which is based on the last condition implied by the ZCM: $E[\epsilon \cdot X] = 0$. By using this, and noticing that the error can be rewritten as:
$$
y_i = X_i^T\beta + \varepsilon_i \implies \epsilon_{i} = y_{i} - X_{i}^T\beta
$$
We can then write:
$$
E[X \cdot \epsilon ] = E[X_i \cdot (y_i - X_i'b)] = 0
$$
Which can be expanded in:
$$
\begin{align*}
E[1 \cdot \varepsilon_i] &= E[1 \cdot (y_i - X_i^T\beta)] = 0 \\
E[x_{i1} \cdot \varepsilon_i] &= E[x_{i1} \cdot (y_i - X_i^T\beta)] = 0 \\
E[x_{i2} \cdot \varepsilon_i] &= E[x_{i2} \cdot (y_i - X_i^T\beta)] = 0 \\
&\vdots \\
E[x_{iK} \cdot \varepsilon_i] &= E[x_{iK} \cdot (y_i - X_i^T\beta)] = 0
\end{align*}
$$

This system of $K+1$ conditions can be solved to find $\beta$, and given that we are in sample land we can use the *sample analogs* of these population moment conditions to achieve our goal.

#### MMSE - Minimum Mean Squared Error
If instead we like linear algebra a little bit more, the MMSE is the perfect alternative. The idea is that instead of using the implication of the ZCM to solve the problem, we minimize the squared loss using our model as predictor:
$$
\beta = \min_{\beta} E[(y_i - X_i^T\beta)^2]
$$
To solve this, we take the **first order conditions** (**FOCs**) by differentiating with respect to each parameter belonging to the vector $\beta$ and, after that, we set the result to zero (optimization). After some calculation, we end up in the exact same case of the MoM:
$$
E[X_i \cdot (y_i - X_i^Tb)] = 0
$$
And by solving this from here by inverting the the equation, we arrive to:
$$
\beta = E[X_i X_i^T]^{-1} E[X_i y_i] = (X^TX)^{-1}(X^Ty)
$$

The main difference between this method and MoM is the way we consider the ZCM condition. In other words, we will see that the condition is very very important in both cases but, while the MMSE allows you to compute the estimators of the parameters without any conditions because based on pure math logic, the MoM requires the ZCM in order to write the equations system.

#### OLS - Derivation and Notes
Independently from the way we decided to go by, we end up with the same result. Why? Because minimizing the sum of squared residuals in the sample is the same thing of minimizing the MMSE in the population.

Let's now see what the derivation really looks like in the case simplest case of all: one regressor $X$
$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$
Now, we said that it doesn't matter which path we take, we always arrive at a shared condition. In this case, the shared condition is:
1. $E[1 \cdot (y_i - \beta_0 - \beta_1 x_i)] = 0 \implies E[y_i - \beta_0 - \beta_1 x_i] = 0$
2. $E[x_i \cdot (y_i - \beta_0 - \beta_1 x_i)] = 0 \implies E[x_i(y_i - \beta_0 - \beta_1 x_i)] = 0$

Let's from the first equation, using the linearity of expectation, imply the following:
$$
\beta_0 = E(y_i) - \beta_1 E(x_i)
$$
Nice, now let's work on $\beta_{1}$ by substituting $\beta_{0}$ into it:
$$
E[x_i(y_i - (E(y_i) - \beta_1 E(x_i)) - \beta_1 x_i)] = 0
$$
$$
E[x_i((y_i - E(y_i)) - \beta_1(x_i - E(x_i)))] = 0
$$
$$
E[x_i(y_i - E(y_i))] - \beta_1 E[x_i(x_i - E(x_i))] = 0
$$
From here we can recognize variance and covariance, because:
$$
\text{Cov}(X,Y)=E[(X-E(X))(Y-E(Y))]=E[X(Y-E(Y))]
$$
$$
V(X)=E[(X-E(X))^2]=E[X(X-E(X))]
$$
Therefore:
$$
\text{Cov}(x_i, y_i) - \beta_1 V(x_i) = 0
$$
And so we arrive to:
$$
\beta_1 = \frac{\text{Cov}(y_i, x_i)}{V(x_i)}
$$

Now, do not forget: we are working with sample data, so
$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} 

\quad \text{ and } \quad

\hat{\beta}_1^{OLS} = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}
$$
Also, $\hat{\beta}_{1}$ can be re-arranged:
$$
\hat{\beta}_{1}=\sum_{i=1}^N w_i y_i \quad \text{where } w_i = \frac{(x_i - \bar{x})}{\sum_{j=1}^N (x_j - \bar{x})^2}
$$

This parameters take the name of **Ordinary Least Squares** (**OLS**), and understanding their nature is fundamental: they are not the "true" parameters of the mean regression, but they surely are the *estimator* of those by using sample data and, when computed, they become *estimates*. Also, consider the fact that they are *linear* in $Y$, but generally they are *nonlinear* in $X$, because they cannot be written as linear combination of the regressors (notice $w_{i}$).

Now, the most important part of this OLS estimators is that, *under certain assumptions*, they have very desirable statistical properties:
- **unbiasedness**: the expected value of the estimator (the average of its value of many different samples) is exactly equal to the true population parameter that we're trying to estimate. Mathematically:
$$
E[\hat{\beta}] = \beta
$$
- **consistency**: as the sample size $N$ increases up to infinity, the estimator converges in probability to the true parameter value. Mathematically:
$$
\hat{\beta} \xrightarrow{p} \beta \quad \text{when } N\to \infty
$$
In other words, the great is the sample that we consider, the lower the probability that our estimator are far away from the true values.
- **BLUE (Best Linear Unbiased Estimator)**: implicated by the Gauss-Markov Theorem, it is provable that among all the possible estimators that are linear in $Y$ and unbiased, OLS is the one who has the smallest variance. Mathematically:
$$
V(\hat{\beta}_k^{OLS}) \le V(\tilde{\beta}_k)
$$

More in depth:
[[Gauss-Markov and OLS properties]] 

#### Multivariate interpretation: Frisch-Waugh-Lovell Theorem
A big problem comes out when we ask ourselves how should we interpretate a parameter, say $\beta_{k}$, in a multivariate regression:
$$
y_i = \beta_0 + \beta_1 X_{i1} + ... + \beta_k X_{ik} + ... + \beta_K X_{iK} + \varepsilon_i
$$
Before diving into the solution proved by the theorem, is important to well understand what is going on. First of all, the big question: what does $\beta_{k}$ measures? Well, even by intuition we can think that the parameter represents *how much the predicted value $y_{i}$ changes when $x_{k}$ changes of one unit*. Careful now: is this sentence complete? No, it is not, because we actually have a problem: correlation.

Two or more regressors can be correlated between them, creating a problem of interpretation. If we just say that $\beta_{k}$ is the value for which $y_{i}$ changes when $X_{k}$ changes of one unit, we are just telling half the truth! Let's think about it: if $X_{k}$ and $X_{h}$ are correlated, then if $X_{k}$ changes of one unit also $x_{h}$ will change for a given entity. So, for our intuition to work we need to add an important concept: **ceteris paribus**, *all the other things remaining the same*.

So, what is $\beta_{k}$? *It's the change observed in $y_{i}$ when $X_{k}$ changes of one unite AND all the other regressors stay fixed.*

Now, how do we prove it? First of all, notice that we can divide $X$ in two parts:
- *correlated part*: this is the part of $X$ that is shared with other regressors
- *uncorrelated part*: this is the part of $X$ that is completely unrelated to other regressors

What the FWL Theorem states is that what the parameter $\beta_{k}$ actually measures is only the second part, the *uncorrelated* one. The process of isolating this "unique part" is called "**partialling out**".

How do we do this partialling out? The intuition is actually very simple: 
1. **regress the variable of interest on all the other regressors**: what we do is predicting $X_{k}$ with all the other regressors that are part of the main model. Mathematically:
$$
X_{ik} = x_{ik} = \alpha_0 + \alpha_1 X_{i1} + ... + \alpha_{k-1} X_{i(k-1)} + \alpha_{k+1} X_{i(k+1)} + ... + r_{ik}
$$
Why is this so useful? Because the predicted values of $X_{k}$ can only be given by the explanation (correlation) that the other regressors have on it! So, the real star here is $r_{ik}$, the *residual*, because it represents the part that the other regressors haven't been able to explain: the *uncorrelated part*.
2. **regress $Y$ on the residuals**: now that our residuals are actually the part of $X_{k}$ that truly is uncorrelated, we can find the impact that a change of one unit of $X_{k}$ has over $Y$ ($\beta_{k}$), without having the problem of the correlation with other regressors. So:
$$
y_{i} = \delta_{0}+\delta_{1}r_{ik} +  \varepsilon_i
$$
where $\beta_k = \delta_1 = \text{Cov}(y_i, r_{ik}) / Var(r_{ik})$ 

So, the reason why this procedure works is because it isolates the unique relationship between $Y$ and that part of $X_{k}$ not correlated / shared with other variables. This is the mathematical foundation for the "ceteris paribus" interpretation!

In the end, we see that *any slope coefficient* $\beta_{k}$ can be viewed as the slope coefficient of a simple mean linear regression, say $\delta_{1}$.

##### Estimating coefficient via FWL
Given that the $\hat{\beta}_{k}$ can be computed as the slope of a simple regression of $y_{i}$ on the residuals $\hat{r}_{ik}$ coming from an auxiliary regression on all the other regressors in $X$ but $x_{ik}$, we have:
$$
\hat{\beta}_{k} = \frac{\sum_{i}^N y_{i}\hat{r}_{ki}}{\sum_{i}^N \hat{r}^2_{ki}}
$$

### Flexibility of the Linear Model
A key fact to understand is that "linear regression" means **linear in parameters**, not necessarily in its variables! This is a crucial distinction, because:
- $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \varepsilon$ IS a linear model
- $y = (\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3)^{-1} + \varepsilon$ IS NOT a linear model

So, being able to still use the linear model for cases where it is not linear in its variable is insanely useful! This allows us to adjust the model to capture a wide variety of non-linear relationship with the same framework.

Now, common specification are:
- **logarithmic**: we can use natural logs for $Y$ and/or $X$ in order to model relationship in terms of percentages or elasticities
	- **log-log**: in this case, $\beta_{1}$ is an *elasticity*, the percentage change in $Y$ for a 1% change in $X$. Mathematically: $\beta_{1} ≈ \frac{\% \Delta Y}{\% \Delta X}$
	- **log-level**: in this case the logarithm is only applied to the $Y$. Thus we measure a *semi-elasticity*: the $(\delta_{1} \times 100)$ change in $Y$ for a 1% change in $X$
- **polynomials**: including higher-order terms allow us to capture relationships where the impact of one variable on $Y$ is not constant. Especially for *quadratic*: $Y = \beta_{0}+\beta_{1}X+\beta_{2}X^2$
	- *marginal effect*: the effect is not just given anymore from the slope, but instead also from the value of $X$ itself, showing that the impact changes as $X$ does. Mathematically: $\frac{\partial Y}{\partial X} = \beta_1 + 2\beta_{2}X$
- **binary variables**: also called "dummies", they are variable which can take only two values, 0 or 1. This specification be very useful because provides great flexibility by acting as on/off switches, allowing us to see different regression by conditioning on this dummies.

## Linear Regression with Binary Regressors 
Entering binary regressors, also known as "**dummy variables**" in a regression is very powerful, because having a tool that only can take two values (0,1) allow us to have a "switch" inside the model. In particular, they are very useful in three cases:
- **qualitative / ordinal informations**: with a switch we can model the effect of non-numerical categories, because by setting 1 and 0 on the dummies we can literally shut down or wake up different parts of the model
- **modeling non-linearities and non-constant effects**: with dummies we can allow the effect of a continuous variable to differ across groups
- **specifying saturated models**: these are highly flexible models that have a separate parameter for every possible value the regressors can take.

Now, let's dive deeper to see how, in practice, dummies actually act as on/off switches. 

### Main effects of dummy regressors
Let's consider a standard mean linear regression model:
$$
E[Y|X_{i}] = \beta_{0} + \delta_{0}D_{i} + \beta_{1}X_{i}
$$

Where $D_{i}$ is our dummy variable.
The idea here is the following: the dummy variable acts as switch for the model to consider or not the parameter $\delta_{0}$, which when considered **shifts the intercept** on the $Y$ axis. Let's see it more in detail:
- $D_{1}=0$ -> intercept of the model is $\beta_{0}$
- $D_{1}=1$ -> intercept of the model is $\beta_{0}+\delta_{0}$

What is the interpretation of these two?
- $\beta_{0}$ -> it's the mean effect of the group that has not been taken onto account, called the "**reference group**" because it's like it is represented by an always switched-on dummy variable $D_{\beta}=1$
- $\delta_{0}$ -> it's the difference between the mean effect of the group that has been taken onto account (the one represented by $D_{1}=1$) and the one of the reference group
- $\beta_{0}+\delta_{0}$ -> it's the mean effect of the group that been taken onto account

So, what is the result? With the dummy we can have a model that shows us *two* (or more) regression lines, the one that doesn't consider $\delta_{0}$ at all and the one that instead does. *Two perfectly parallel lines with different intercept.*

**(e.g.) - dummy regressor in a linear regression model about wages and education**
Let's create a model to predict an individual's wage based on their years of education and gender. We can define our model as:
$$
\text{Wage}_i = \beta_0 + \delta_0 \text{Female}_i + \beta_1 \text{Education}_i
$$
In this case, $\text{Female}_i$ is the dummy variable, where $\text{Female}_i = 1$ if the individual is female and $\text{Female}_i = 0$ if the individual is male. Now:
- The group where $\text{Female}_i = 0$ (males) is the reference one. Their wage is predicted by $\text{Wage}_i = \beta_0 + \beta_1 \text{Educ}_i$. The intercept for this group is $\beta_0$.
- For the group where $\text{Female}_i = 1$ (females), the intercept shifts. Their wage is predicted by $\text{Wage}_i = (\beta_0 + \delta_0) + \beta_1 \text{Education}_i$.


### The dummy trap
We have to be also careful when using dummies, because we can't have a dummy for each group in the model. What do I mean? I mean that if we have $K$ groups and we use $K$ dummies, then we encounter **perfect collinearity**.

Let's look at this. Imagine we have two groups, then:
$$
E[Y|X_{i}] = \beta_{0} +\delta_{0}D_{1}+\gamma_{0}D_{2}+\beta_{1}X_{i}
$$

The big problem arises if $D_{1}$ and $D_{2}$ refers one to the same medal but different face of the other, because: $D_{1} = 1-D_{2}$. It's like a switch:
- If $D_{1}=1 \implies D_{2}=0$
- If $D_{1}=0 \implies D_{2}=1$

Now, the problem arises when considering that we also have the *intercept* $\beta_{0}$. The intercept can be seen as a value correlated to an invisible dummy variable witch is always switched on: $D_{\beta} = 1$ BUT, if we have two categories, we can only have two "mean effect" parameters:
- $\beta_{0}$ -> represents the mean effect of one of the categories
- $\delta_{0}$ -> represents the difference between the mean effect of the other category and the one covered by the intercept

Any additional parameter here will be just *too much*, because it could be described by the other two. This can be summarized in the following:
$$
\beta_{0} = \delta_{0} + \gamma_{0}
$$
Or, in terms of their respective dummy variables:
$$
D_{\beta} = D_{1} + D_{2}
$$
We can now perfectly see the dependence in this way: $D_{\beta} = 1 \implies D_{1} = 1 \cap D_{2}=0$ or viceversa, but knowing $D_{1}$ or $D_{2}$ will also make me know the other one. 

Now, notice, if something like this is inside our model then the very logic of it is being disrupted: if $\beta_{0}$ represents the "reference group" and only two groups exist, then any dummy variable referencing to those in addition to $D_{1}$ will be *redundant information*! It's like counting once again an information that we already have! It makes no sense at all!

Now, why is this such a big problem? Because the mathematics falls to pieces! Having dependency between dummy variables inside the regressors' matrix is a disaster: the matrix will be *singular*, therefore *not invertible* and, if that's the case, it's **impossible to get the OLS** estimators.
Let's take a look at the mathematics. Consider the first column the intercept, the second one $D_{1}$ and the third one $D_{2}$:
$$
X = \begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}

$$
We see that we can write $D_{2} = \beta_{0}-D_{1}$. This means that $\Delta_{X} = 0$, so the equation for our estimators:
$$
\hat{\beta} = (X'X)^{-1} X'y

$$
Will be impossible to solve.

NOTICE: substituting $D_{2}=1-D_{1}$ inside the model actually gives you back the model without it! What this means is that a model with three parameters is observationally equivalent to one with just two of them, with the difference that the last one is estimable. 

The **solution** to the dummy trap is, if we want to keep the intercept, choosing exactly $K-1$ dummy variables. In this case, the group for which we omit the dummy will be the *reference group*.

**(e.g.) - too much dummy variables**
Suppose we try to model wage with both a dummy for being female and a dummy for being male, plus an intercept:
$$
\text{Wage}_i = \beta_0 + \delta_0\,\text{Female}_i + \gamma_0\,\text{Male}_i + \beta_1\,\text{Education}_i
$$
But for every individual, either $\text{Female}_i = 1$ and $\text{Male}_i = 0$, or $\text{Female}_i = 0$ and $\text{Male}_i = 1$. So:
$$
\text{Female}_i + \text{Male}_i = 1
$$
This means the three columns (intercept, $\text{Female}_i$, $\text{Male}_i$) are perfectly collinear. The design matrix is singular, and you can't estimate the model using OLS.

### Interaction effects with dummy regressors
Dummy variables can also be used to effect the *slope* of the regression model, not just the intersection. The so called **interaction term** is very useful, because it allows us to see the difference in how much greater or smaller the average effect of the regressors is. 

Consider the model:
$$
E[Y|X] = \beta_{0} +\delta_{0}D_{1}+\beta_{1}X_{i} + \delta_{1}D_{1}X_{i}
$$
The cases here are two:
- $D_{1}=0$ -> $E[Y|X] = \beta_{0}+\beta_{1}X_{i}$
- $D_{1}=1$ -> $E[Y|X] = (\beta_{0} +\delta_{0})+(\beta_{1} + \delta_{1})X_{i}$

So, we see that when $D_{1}=1$ the actual slope changes, telling us exactly the *differences* across groups in terms of "average effect" of $X_{i}$ over the two groups. In this case, $\delta_{1}$ is the key parameter, representing the **difference in slopes**.

Now we see the power of dummies: using the *interaction term* and the *difference in intercepts term* we can describe a profile of a given regression for two different groups, each one with a standalone starting point (intercepts) and growth rates (slopes).

### Using dummies for ordinal variables
Suppose that we are working with an ordinal variable $X$. The simplest model treats $X$ as numeric:
$$
E[Y|X] = \beta_{0} + \beta_{1} X
$$

This assumes the effect of moving from one value $x_i$ to another $x_j$ (where $i < j$) is **constant**, so that each step up in $X$ increases $Y$ by the same amount (the slope $\beta_1$). In reality, this is often too restrictive, since the effect of changing $X$ may not be uniform.

To relax this assumption, we can use **dummy variables** for each possible value of $X$ (except one, to avoid the dummy variable trap):
$$
D_{X_j} = 1 \text{ if } X = j \quad \text{where } j = 0,1,2,3,\dots
$$
For example, if $X = 3$, then $D_{X_3} = 1$, all other dummies are 0.
So, the model becomes:
$$
E[Y|X] = \beta_{0} + \delta_{1} D_{X_1} + \delta_{2} D_{X_2} + \delta_{3} D_{X_3} + \dots
$$

Here, $\beta_0$ is the mean of $Y$ for the **reference group** (e.g., $X = 0$), and each $\delta_j$ is the difference in mean between group $X = j$ and the reference group. 

The key point is the following: in this dummy variable model, $X$ itself does not appear as a numeric variable, and there is **no slope** for $X$. Instead, each category of $X$ gets its own mean value for $Y$. This allows the relationship between $Y$ and $X$ to be non-linear and flexible, capturing differences between groups without assuming equal spacing or effect size.


### Saturated regression models
What happens when we want maximum flexibility from our regression model? Enter **saturated regression models**, which are the most flexible linear models you can possibly build. These models include a separate parameter for every possible combination of values that your regressors can take, but there's a catch: this is only feasible when all regressors are *discrete*.

The key insight is this: saturated models use dummy variables and their interactions as building blocks to create a model that can capture *any* relationship between discrete predictors and the outcome, without imposing functional form restrictions.

For example, consider a saturated model for gender and education. Suppose to have this two dummy variables: gender is binary (female / non-female) and education also binary (high / low). This means there are $2\times 2=4$ possible combinations: $\Omega=\{ FH,FL,NH,NL \}$. So, we have to build a regression that has a specific parameter for every single possible combination:
$$
E[Y|\text{gender, education}] = \beta_{0} +\beta_{1}\cdot \text{gender} + \beta_{2}\cdot \text{education} + \beta_{3}(\text{gender}\cdot \text{education})
$$
where $\text{gender}$ and $\text{education}$ are our dummy variables. Let's see what this $\beta$ mean:
- $\beta_{0}$ -> is the average of $Y$ for the reference group, when both the dummies are null ($E[Y|\text{gender = 0, education = 0}]$)
- $\beta_{1}$ -> represents the difference between the case of the reference group and the one where we have female 
- $\beta_{2}$ -> represents the difference between the case of the reference group and the one where we have high education
- $\beta_{3}$ -> represents the compound effect of the case in which we have both females and high education, which may not just be the sum of their individual differences with $\beta_{0}$

Clearly this model is very powerful, but careful: it's **non-parametric**. If you notice, we are not assuming an exact shape or behavior for the relationship, instead we are asking it to show us how it's made by giving it every single parameter for every single case that might show up. This perfectly covers what the data are telling us, without any generalization. The result will clearly be a step-by-step function, as the average jumps from a case to another.

So, when estimating the OLS what are we actually doing? Well, in this case, computing OLS is the same as calculating the *sample mean* of the outcome variable $Y$ within each possible combination of regressors values.

Even if great, this model has a big problem: the **curse of dimensionality**. As the number of regressors or the number of categories within them grows, the number of possible combination explodes. This also means that cells will have very few or zero observation, making the estimates imprecise or impossible to compute. A good *compromise* is to use dummies only on few key variables of interest while modeling the other cases with assumption over the shape (e.g. linear terms).

## Linear probability model
A **linear probability model** is a model where the outcome variable $Y$ is *binary*. Now, consider a standard linear regression model:
$$
Y_{i} = E(Y_i|X_i) + \varepsilon_i
$$
If $Y_{i}$ is binary (0/1), its expectation is the probability of that variable to be equal to one:
$$
E(Y_i|X_i) = 0 \cdot P(Y_i=0|X_i) + 1 \cdot P(Y_i=1|X_i) = P(Y_i=1|X_i)
$$
So, when the the outcome is binary, the linear mean regression model becomes a model for the conditional probability of success:
$$
E[Y_{i}|X_{i}] = P(Y_{i} =1|X_{i}) = \beta_{0}+\beta_{1}X_{1i}+\dots+\beta_{K}X_{Ki}
$$
This is exactly the *linear probability model*, because it models the probability as a linear function of the regressors. In this case, we can interpretate $\beta_{K}$ as the change in the probability of $Y_{i} =1$ for a one unit change in $X_{Ki}$ given all the other regressors constant.

### Estimations for the LPM
When trying to estimate the OLS for the LPM we encounter a big problem: *heteroskedasticity*. The reason is the linear probability model is, by design, heteroskedastic. For a binary variable: $Y_{i}^2=Y_{i}$, so:
$$
\begin{align*}
V(Y_i|X_i) &= E(Y_i^2|X_i) - [E(Y_i|X_i)]^2 \\
&= E(Y_i|X_i) - [E(Y_i|X_i)]^2 \\
&= P(Y_i=1|X_i) \cdot [1 - P(Y_i=1|X_i)]
\end{align*}
$$
But $P(Y_{i} =1|X_{i})$ is equal to $E[Y_{i}|X_{i}]=X_{i}^T\beta$, therefore:
$$
V(Y_i|X_i) = (X_i^T\beta)(1 - X_i^T\beta)
$$
Because the variance of $Y_{i}|X_{i}$ depends on $X_{i}$, when $X_{i}$ changes the same does the variance, which is exactly the definition of heteroskedasticity given that, conditional on X, $V(y_i|X_i) = V(\varepsilon_i|X_i)$.

Now, we know that when a model is heteroskedastic, the standard OLS estimates are still *unbiased* and *consistent*, but the *standard errors* are not correct. Two solutions:
- **heteroskedasticity robust standard errors**: we can use formulas that are valid even if the exact form of heteroskedasticity is unknown
- **model the heteroskedasticity**: we could model the form of the variance and use estimation methods to obtain more efficient estimates, but is much less common and is done only when there is a specific reason 






