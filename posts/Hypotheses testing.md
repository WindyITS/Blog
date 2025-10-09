## T-Test for individual parameters

In order to perform hypothesis tests we first need to know the sampling distribution of the OLS estimator or, in other words, how the estimator changes along with the samples that I can take.

To do so, we need one last assumption.

### A6 - Normality of error terms
The last important assumption that we have to make if we want to test hypothesis is the following: the *error terms are i.i.d. as a Normal distribution*. Mathematically:
$$
\varepsilon_{i} \sim N(0, \sigma^2) \quad \text{independently}
$$
Now, notice:
- $Y_{i}|X$ **normally distributed**: remember that the model is made of two parts: the coefficient ($\beta$) part and the error one. When I condition on $X$, the coefficient part becomes a constant, therefore the only part the changes is the error, which follows a normal distribution under A6. Therefore, we will have:
$$
Y_{i}|X \sim N(E[Y_{i}|X], \sigma^2)
$$
- **Assumption A1-A6** imply that OLS has the smallest variance even among *non-linear* unbiased estimators
- **For large samples A6 can be dropped**: the CLT ensures that the OLS is normally distributed under A1-A5 alone, because the estimators mathematically involve sample averages of the underlying errors, and the CLT applies to these averages

### Normal sampling and t-statistic
Under all six assumptions, the OLS estimators $\hat{\beta}_{k}$ is normally distributed:
$$
\hat{\beta}_k^{OLS} \sim N(\beta_k, V(\hat{\beta}_k^{OLS}))
$$
This is because I can write the OLS estimators as perfect linear transformation of the error, which under A6 is distributed as normal; therefore, also the coefficients will be normal.
Now we can standardize the distribution:
$$
\frac{\hat{\beta}_k^{OLS} - \beta_k}{SD(\hat{\beta}_k^{OLS})} \sim N(0, 1)
$$
where $SD$ is the *standard deviation* of the estimation of the coefficient in the population world. Sadly, $SD$ must be estimated by $SE$, therefore the normal distribution is replaced by the **t-distribution**.

Therefore, we get:
$$
t_k = \frac{\hat{\beta}_k - \beta_k}{SE(\hat{\beta}_k)} \sim t_{N-K-1}
$$
where $N-K-1$ are the degrees of freedom.

### Testing hypothesis with individual parameters
With the computed $t_{k}$ we can test a null hypothesis about the population parameter. In particular, we have the following cases:

#### Testing against one-sided alternative: ">0"
Here we have:
- *hypothesis*: $H_{0}: \beta_{k}=0$; against $H_{1}: \beta_{k}>0$
- *rejection rule*: we reject the null hypothesis if the computed statistic exceeds a certain critical value -> $t_{k}>c$
- *critical value*: it is the value of $t$ such that the probability of observing a value larger than $c$ is equal to the chosen *significance level* ($\alpha$)

#### Testing against one-sided alternative: "<0"
Here we have:
- *hypothesis*: $H_{0}: \beta_{k}=0$; against $H_{1}: \beta_{k}<0$
- *rejection rule*: we reject the null hypothesis if the computed statistic falls below a certain critical value -> $t_{k}<c$
- *critical value*: it is the value of $t$ such that the probability of observing a value less than $-c$ is equal to the chosen *significance level* ($\alpha$)

#### Testing against two-sided alternatives: "≠0"
Here we have:
- *hypothesis*: $H_{0}: \beta_{k}=0$; against $H_{1}: \beta_{k}≠0$
- *rejection rule*: we reject the null hypothesis if the computed statistic falls below or grows larger than a certain critical value -> $t_{k}<-c$ or $t_{k}>c$ 
- *critical value*: it is the value of $t$ such that the probability of observing a value higher than $c$ or lower than $-c$ is equal to $\alpha/2$ for each case

### p-values
The $p$-value provides an alternative way to look at the result of an hypothesis test. It is defined as the *smallest significance level at which the null hypothesis would be rejected*. In other words, it is the probability of seeing a $t_{k}$ extreme as, or more than the one calculated assuming $H_{0}$ true.

So, if the probability (p-value) of seeing $t_{k}$ higher (or lower) than $c$ (or $-c$) is less than the chosen significance level $\alpha$, we reject $H_{0}$. This is because the probability of that particular case occurring when $H_{0}$ is true is too far low. 


## F-Test for joint hypothesis
Sometimes we wish to test wether some variables inside the model are actually useful to increase the model's fit. To do so, we can use the **F-test** to check for the null hypothesis where some of this variables are set to zero. The intuition is simple: we compare the full model and a restricted version of it, with less parameters, and if the difference between those two models in "explanatory power" is high enough, then it means those variable are actually useful. So:
- **unrestricted model**: our complete model, with all the parameters inside
- **restricted model**: our complete model with some parameters removed (set to zero) under the null hypothesis for which they are not important

First of all, notice that by construction we have $SSR_{R} \geq SSR_{U}$, because adding variables to the model may not change its explanation power at all.

Now, we compute the F-statistic as:
$$
F= \frac{(SSR_{R}-SSR_{U})/q}{SSR_{U}/(N-K-1)}
$$
where $N$ is the sample size, $q$ the number of restriction being tested and $K$ the number of regressors in the unrestricted model (excluded the intercept). Under the *Gauss-Markov assumption* this statistic follows an F-distribution:
$$
F\sim F_{q,\space N-K-1}
$$
Now, notice: the larger is $F$, the bigger is the difference between the explanation power of the two models. To check if it's the case to refuse or to accept the null hypothesis, we can do equivalently one of the following two options:

1. **compare with a critical value**: we compute the critical value $c$ in the point of the distribution where we have our significance level; then we compare $F$ and $c$, so
$$
F>c \implies \text{refuse null hypothesis}
$$
2. **compute the p-value**: we compute the probability of observing that $F$ given our distribution, then we compare it with our significance level $\alpha$; so:
$$
p <\alpha \implies \text{refuse null hypothesis}
$$

In other words, it's like saying: "the probability of observing this difference between the power explanation of this two models by chance is low, therefore I must conclude that the variables do contribute", or viceversa.