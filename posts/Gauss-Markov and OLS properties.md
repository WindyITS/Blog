

#### Gauss-Markov assumptions and theorem
The four assumptions that are sufficient to prove that OLS is *unbiased* and *consistent* are the following:
1. **(A1) - linear in parameters**: the model is written by using linear combinations of the parameters, wether for the variables they can be non-linear. Mathematically:
$$
    y = \beta_0 + \beta_1 x_1 + ... + \beta_K x_K + \varepsilon
$$
2. **(A2) - random sampling**: the data are part of a random sample from the population, causing that each observation follows the population model because not biased. Mathematically:
$$
    y_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_K x_{iK} + \varepsilon_i \quad \text{for } i = 1, ..., N
$$
3. **(A3) - no perfect collinearity**: none of the $X$s is constant ($Var(X_{i}>0)$) and there are no exact linear relationships among them; this ensures for the matrix $X^TX$ to be invertible, making the OLS estimable
4. **(A4) - zero conditional mean** (ZCM): also called "*exogeneity*", states that the expected value of the error given the regressors is exactly zero, implying that the average effect of the error on $Y$ is null. Mathematically:
$$
E[\varepsilon_{i}|X_{i}]=0 \quad \forall X
$$

The needed assumption for the BLUE property of OLS is the fifth Gauss-Markov assumption, which specify the variance of the error term. In particular:

5. **(A5) - Homoskedasticity**: the conditional variance of the error given the regressors must be constant. Mathematically:
$$
Var(\varepsilon_{i}|X_{i}) = \sigma^2 \quad \forall X
$$
If this property fails, then we have *heteroskedasticity*:
$$
V(\varepsilon_i|X_i) = \sigma_i^2 = \sigma^2(X_i)
$$
This last case can be a problem, because it means that the variance of the error changes as $X$ does.

Now, notice:
- **A1 - A4 ->** there is a *theorem* of Gauss-Markov stating that the OLS estimator is unbiased:
$$
E[\hat{\beta}] = \beta
$$
- **A1 + A4 ->** the conditional mean is linear
$$
E(y_i|X_i) = \beta_0 + \beta_1 x_{i1} + ... + \beta_K x_{iK}
$$
- **A5** -> with homoskedasticity we can prove that $V(y_i|X_i) = V(\varepsilon_i|X_i)$, so the conditional variance is constant:
$$
V(y_i|X_i) = \sigma^2
$$
This is not very hard to visualize: in a regression we are conditioning on $X$, meaning that it is the fundamental part that we know (has no variance), therefore, the only variance that can be left out in the model is the one coming from the error term.
- **A1 - A5**: the famous *Gauss-Markov Theorem* states that the OLS estimator is the Best Linear Unbiased Estimator (BLUE) of the regression coefficients.

#### Measure of variations
If we want to know how well does $X$ in explaining $Y$, we can use the measure of variations:
- **Total Sum of Squares**: it represents the total sample variation, given by classic formula:
$$
\text{STT} = \sum_{i=1}^N (y_{i}-\bar{y})^2
$$
- **Explained Sum of Squares**: it represents the part of sample's variation explained by the regressors $X$
$$
\text{SSE} = \sum_{i=1}^N (\hat{y}_{i}-\bar{y})^2
$$
- **Residual Sum of Squares**: it represents the variation of the residuals / errors, the part not explained by $X$
$$
\text{SSR} = \sum_{i=1}^N(\hat{\epsilon}_{i}-\bar{\epsilon})^2
$$
but given the fact that we assume for the average of the error to be equal zero because captured by the constant term, we can also write it as
$$
\text{SSR} = \sum_{i=1}^N \hat{\epsilon}_{i}^2
$$

Now, we can *decompose* the total variation as:
$$
\text{SST} = \text{SSE} + \text{SSR}
$$
And we can also define the $R^2$, which measures the fraction of the total variation that is explained by the model:
$$
0\leq \frac{SSE}{SST} = 1-\frac{SSR}{SST} \leq 1
$$
where $R^2 = 1$ means *perfect fit* and $R^2=0$ means *no fit* at all. When multiplied by 100 it gives the percentage. Notice: $R^2$ is *non-decreasing* in the number of regressors: no matter what information I add, if useful it will increase the fraction, if not then it won't change it at all.


#### Algebraic properties of OLS
OLS have some algebraic properties that hold regardless of wether the assumption *A4* and *A5* hold or not. They follow directly from the first-order conditions for OLS estimation:
1. **Residuals sums to zero**: the sum of the OLS residuals, so the error terms, is zero. Mathematically:
$$
\sum_{i=1}^N \hat{\varepsilon}_i = 0
$$
The intuitive reason is the following: the non-zero effect that they have is represented by the *intercept*.
2. **Residuals are orthogonal to regressors**: the covariance between the residuals and each regressor is zero. Mathematically:
$$
    \sum_{i=1}^N x_{ki} \cdot \hat{\varepsilon}_i = 0 \quad \text{for all } k = 1, ..., K
$$
3. **Sample average lies on the regression line**: the point defined by the sample averages of $y$ and the $x$'s lies exactly on the fitted regression line. Mathematically:
$$
\bar{y} = \hat{\beta}_0 + \hat{\beta}_1 \bar{x}_1 + ... + \hat{\beta}_K \bar{x}_K
$$
#### Statistical properties of OLS
One thing we can do is to look at OLS across different samples, like a random variable. In this case, the estimator has specific statistical properties dependent on the Gauss-Markov assumptions:
- **unbiasedness**: if the first four (A1-A4) assumptions are satisfied, then the expected value of the estimator is equal to the true value of the coefficient. Mathematically:
$$
E[\hat{\beta}_{k}] = \beta_{k}
$$
- **sampling variance of OLS**: if *all five* (A1-A5) conditions are met, then the variance of the estimator, conditional on the sample values, is (depending on the type of regression):

 *Simple regression*: in this case there is just one regressor, so we have
$$
V(\hat{\beta}_1^{OLS}) = \frac{\sigma^2}{\sum_{i=1}^N (x_i - \bar{x})^2}
$$
*Multivariate regression*: there is more than one regressor, so
$$
        V(\hat{\beta}_k^{OLS}) = \frac{\sigma^2}{SST_k(1 - R_k^2)}      
$$

where:
- $\sigma^2 = V(\varepsilon_i|X_i)$ -> unknown population error variance
- $SST_k = \sum_{i=1}^N (x_{ki} - \bar{x}_k)^2$ -> total sample variation in $x_{k}$
- $R^2_{k}$ -> R-squared from the auxiliary regression on all other regressors in $X$ but $x_{k}$

##### Estimating the standard errors
As we have seen, the variance of the estimator depends on the variance of the error's population. Therefore, to compute the formulas I have to estimate $\sigma^2$ first!

To estimate the variance we use sample residuals. Now, because $K+1$ parameters were estimated to get the $N$ residuals, we lose $K+1$ degrees of freedom. Therefore, the unbiased estimator of $\sigma^2$ is:
$$
\hat{\sigma}^2=\frac{\sum_{i}^N \hat{\varepsilon}_{i}^2}{N-K-1} \implies E[\sigma^2]=\sigma
$$
under A1 - A5.

Now we compute the variance and then we take the square root for the **standard errors**, but given that we are using the estimation of the variance, the formula is: 
$$
SE(\hat{\beta}_k^{OLS}) = \sqrt{\hat{V}(\hat{\beta}_k^{OLS})} = \begin{cases} \sqrt{\frac{\hat{\sigma}^2}{\sum_{i=1}^N (x_i - \bar{x})^2}} & k = K = 1 \\ \sqrt{\frac{\hat{\sigma}^2}{SST_k(1 - R_k^2)}} & k = 1, ..., K > 1 \end{cases}
$$

Notice, this are the standard errors *if and only if* the homoskedasticity (A5) holds! If it doesn't, we have to change approach by using **robust standard errors**. Given that sometimes we have no clue wether there is homoskedasticity or heteroskedasticity, is often preferred to just use the robust formulas.

