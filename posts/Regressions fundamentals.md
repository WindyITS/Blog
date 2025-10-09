
Welcome to the world of regression, the Swiss army knife needed in every analyst's toolkit! Regression is a very versatile tool that we can use for a lot of different tasks, especially for:
- **description**: when we want to describe patterns among data in terms of associations between observations
- **prediction**: when we use the patterns discovered in past data to guess about the future with the most possible accurate prediction model
- **causal inference**: when, by using strong assumptions, we use the model to answer the "what if" questions about the implication that a variable would have on another one

To fully understand regression the best thing to do is to first of all explore a world where everything is known, where there is perfect information.

# The world of perfect information 
Also called "**population land**", the world of perfect information is a theoretical paradise where we have all the data, for the entire population, we can think of. The beauty of this world is that *uncertainty does not exist*.

## Regression
We define the **regression** as a statistical method used to analyze the relationship between a *dependent variable* $Y$ and one or more *independent variables* $X$, also called "*regressors*". In other words, it can be also defined as "any feature of the conditional probability distribution of $Y$ given $X$".

Now, in practice regression is a simple line that we draw to connect all the data that have in common a specific and chosen feature $X=x$. Careful: a lot of regression types exist, because there are many features that we could choose; two of them are:
- **mean regression**: the most famous one, it shows with a line the relationship of the average of the dependent variable at different levels of the independent ones, therefore denoted as $E[Y|X]$.
- **median regression**: it shows with a line the relationship between the median of the dependent variable at different levels of the independent ones

So, regression isn't just one thing. It is a big family where each type can be used to describe, predict or, sometimes, even inference a characteristic / feature of a variable as we move across different variables called regressors.

## Specification
Once we've decided the feature to track, there is a second choice to make: what is the shape of our regression line? In other words, for **specification** we mean the choice of how we believe the relationship behave. 
The possible options are two:
- **parametric regression**: in this case, we assume that the relationship between $Y$ and $X$ can be described using a finite number of parameters. The most famous case is linear regression, which usually looks like this:
$$
Y = \beta_{0}+\beta_{1}X
$$
where $\beta$s are the parameters.
This type of regression has the great advantage of being really simple and efficient, also is not very data-hungry because few data are needed to identify the line of best fit when we know what the shape looks like.
- **non-parametric regression**: in this case we do not know what the relationship between the variables looks like, therefore we use a huge amount of data to let them show us what the actual shape really is. Our only assumption is that the relationship is at least *continuous*. Usually, it can be seen as:
$$
Y = g(X)
$$
where $g(\cdot)$ is a general continuous function. 
The strength of this regression's type is its capability of uncovering complex patterns that rigid model would miss.

## Best Predictors and Loss Functions
When we put our hands on the model, we have to make sure that our model actually is the best one to predict $Y$ given $X$. In other words, we have to know that the model maximizes the useful information from $X$ to explain $Y$. Saying that we want our information to be used as much as possible it's the same thing as saying that we want to **minimize** what is not used or, in other words, we want to minimize the average error between our prediction and the correct value.

To do so, we define a **loss function** $L(\cdot)$ as a tool that assigns a penalty based on the size of our prediction error $(Y-p)$ where $p$ stands for "predictor". It can be generally defined as *any function* such that:
$$
0<u<v \implies 0=L(0)\leq L(\pm u)\leq L(\pm v)
$$
The game now is to choose a predictor $p$ that minimizes the *expected error* given our information $X$:
$$
\min_{p} E[L(Y - p)|X]
$$
So, the predictor $p$ which successfully minimize the average loss is the **best predictor**. Now, given that the best predictor will generally change for different values of $X$, the best predictor itself will be a function of $X$!

Obviously, there are some common loss functions already well known that are associated with their best predictor:
- **square loss**: the penalty for an error is its square. Mathematically:
$$
L(u)=u^2 
$$
The characteristic of this function is that it should make you terrified of large errors, and to minimize this squared average error the best strategy is to always predict the **conditional mean** $E[Y|X]$. Why? Because the mean is the "center" of the data, therefore it is the closest to all other points density-wise.
- **absolute loss**: the penalty is the absolute value of the error. Mathematically:
$$
L(u)=|u|
$$
This loss function is much more forgiving, because it doesn't jump when large errors occur. In this case, to minimize our average penalty, the best thing to do is to predict the **conditional median**. Why? Because it is the value that splits the data in half, minimizing the total distance to all other points.
- **asymmetric absolute loss**: the penalty depends if we over-predict or under-predict. Mathematically:
$$
L(u)= 
\begin{cases}
\alpha|u|  & \text{if } \space u\geq 0 \\
(1-\alpha)|u| & \text{if } \space u\leq 0 \\
\end{cases}
\quad \quad \text{where} \space \space \alpha\in (0,1)
$$
The idea behind is that the function penalizes more one type of error than the other, and we can see that it is the generalized case of the absolute loss (which can be obtained with $\alpha=0.5$). To minimize the average penalty, the best thing to do is to predict the **conditional quantile**, because if under-predicting penalizes more than over-predicting then we will try to aim high and viceversa. 

These are only 3 possible types, but generally they are divided into two big categories: *symmetric* and *asymmetric* loss functions. As showed from the last case, the "asymmetric" stands for penalizing with different intensity the over and the under predicting cases.

One last thing because moving on: **how do we choose a loss function**? It depends on our objective: different loss functions emphasize different aspects of a model, where some harshly penalize outliers, others are more forgivable about certain aspects. Therefore, no universal answer exists.

### Cases where BPs behave similarly
There are special cases where the choice of loss function doesn't matter at all because different best predictors tell exactly the same thing:
- **if the probability conditional distribution $P(Y|X)$ is symmetric for all $X$**: when this is the case, the *mean* and the *median* will be exactly the same, making their regression equal
- **if the probability conditional distribution shifts as $X$ varies but doesn't change shape**: in this case all the regression lines of mean, median and quantiles will be parallel, because the distribution has the exact same shape but is just shifted along the axis. Therefore, their regression will have the same slope no matter which you look at.

## Extrapolation
**Extrapolation** is a statistical technique used to predict or estimate values for data points that are not in our realm or, in other words, that lie *outside* of the range of our observed data. It can be seen as extending patterns or trends from our existing data to make *inferences* about unseen scenarios.

A more rigorous way to see this is by using the best predictor concept. BP regression solves $\min_{p} E[L(Y-p)|X]$ for a general loss function $L(\cdot)$, and this gives us back a function $g(X)$. What extrapolation does is asking: "can we reuse $g(X)$ (or derive a new one) for a revised problem without having to re-estimate everything (if possible at all) from scratch?".

Usually, to make extrapolation possible we have to make **strong assumptions** about reality, so that we can take our model into the "unknown".

### 1 - Change of loss function
When we make a regression model with a given loss function, usually we can't extrapolate from this model another regression with a different loss function, and that's because the regression (remember it is a feature) is not the *distribution* of our data. In other words, we can't *usually* estimate another feature of our observations by building a regression over another regression, because the base one doesn't capture all the "particulars" of the entire data but instead only a feature of them, making the one built upon not able to truly grasp the informations of our data.

Mathematically, the big problem relies on the difference of best predictors for loss functions. Therefore, if two loss functions share the same best predictor because of some special case, then changing from one to another is not a problem at all.

(e.g.) - going from average to quantile
Imagine we have just created a mean regression model, thus we have used the square loss as loss function. Now we would like to use an asymmetric absolute loss as function because we changed our mind on error's penalization. The only way to do so (if not by making strong assumptions) is to go back to our data and build the regression from the start, because the mean regression only captures the average of the data conditioned on some variable, not the real informations that the whole data distribution can provide.

### 2 - Prediction off the support of $X$
When we want to build a regression we deal with real world data, and these data are called the *support* of $X$. Clearly, as long as we stay where we have informations everything's fine: we can predict and describe $Y$ given $X$ without any problem at all. Now, what if we push ourselves to the point where we don't have data for $X$ anymore? In this unknown space we are *off the support* of $X$.

 Formally, we define the concepts of on/off the support as:
 - **on the support**: this is the case when there is a positive probability of observing a value $x_{0}$ arbitrarily close to $x$. Mathematically:
$$
x_{0} \in S_{X} \quad \text{if and only if} \quad P(x≈x_{0})>0
$$
- **off the support**: this is the case when there is zero probability of observing a value $x_{0}$ arbitrarily close to $x$. Mathematically:
$$
x \not \in S_{X} \quad \text{if and only if} \quad P(x≈x_{0})=0
$$
In this case, the conditional distribution $P(Y|X=x_{0})$ is not defined, therefore the best predictor can't be directly computed. Also, we can't use the non-parametric models at all, simply because we literally don't have the data to build the line. Here, *parametric* models are the only solution and, even in this case, we need strong assumptions for it to work.

### 3 - Change in $P(X)$
There are cases where the distribution of predictors $P(X)$ changes due to some real world events—like a sudden economic shift or policy tweak shaking things up. First of all, let's get this straight: the best predictor $g(X)$, say $E[Y|X]$, stays the same whenever $P(X)$ shifts but $P(Y|X)$ holds, because it's built on that conditional relationship, not on the marginal one. This fact is at the base of the distinction between **structural BPs** (those BPs that capture stable causal links) and **non-structural BPs** (the one that just follows correlations). 

Now, more in depth:
- **$P(Y|X)$ remains unchanged**: if the conditional distribution stays the same, then our regression model remains usually valid because the regression function $g(X)$ depends **only on** the conditional distribution and not on the marginal distribution. Mathematically:
$$
P(X) \to P'(X) \text{ but } P(Y|X) = P'(Y|X) \implies g(X) \text{ unchanged}
$$
- **$P(Y|X)$ changes**: we can't use our model anymore because the fundamental relationship between $X$ and $Y$ has changed. Mathematically:
$$
P(X) \to P'(X) \implies P(Y|X) \to P'(Y|X) \implies g(X) \text{ changes}
$$

### 4 - Prediction of $g(Y)$ conditional on $X$
In the case where we have created our model with a given BP to work on our distribution $Y|X$, and then we want to move to model $g(Y)$ given $X$, we generally can't use the same best predictor we have used before. Why is that? Because usually the best predictor for $Y|X$ does enable us to go to (does not reveal) the BP of $g(Y)|X$!

As usual, there are exceptional cases:
- *if $g(\cdot)$ is linear*: if this is true, then $E[g(Y)|X]=g[E(Y|X)]$
- *if $g(\cdot)$ is increasing*: if this is true, then $Q_{\alpha}[g(Y)|X]=g[Q_{\alpha}(Y|X)]$

### 5 - Prediction of $Y$ conditional on $h(X)$
Imagine we want to predict $Y$ based on a *transformed version* of $X$ that we'll call $h(X)$. Well, there are situations where this is a problem and others where this is not:
- **one-to-one transformation**: if our $h(X)$ takes one value from the support and gives back just another value, we are completely fine. That's because we are not fundamentally changing who $X$ is: a one-to-one transformation has an inverse, therefore $X$ doesn't really lose its characteristics because it lives in the "shadow" of the transformation. Mathematically, the **entire conditional distribution** is preserved:
$$
P(Y | h(X)) = P(Y  | X)
$$
This means **all features** are preserved: mean, median, variance, quantiles, and any other statistical property of $Y|X$ equals the corresponding property of $Y|h(X)$.
- **many-to-one transformation**: if our $h(X)$ takes more than one value from the support and gives back a single value, we can't say that conditioning on $h(X)$ is equal to just conditioning on $X$. Why this happens? The idea is that we are fundamentally changing who the variable is, because many-to-one transformations do not have an inverse! Thus, information about $X$ is lost. Mathematically, **all distributional features** differ:
$$
P(Y | h(X)) \neq P(Y| X)
$$

If the last one is our case, we can't derive the new regression from the last one because $h(X)$ contains less information than $X$.

# Sample Land
The time in the magical world where we know everything, where we have all the possible data, is over. We are back on earth, so from population land we are in *sample land*. Why sample? Because reality is literally sample made: small, imperfect snapshots of greater mechanisms that we are not allowed to see and touch.

In this case, what type of regression do we choose? The core idea is that if in population land parametric specification was a great choice for having an efficient model that could've been used also for extrapolation, in sample land it's usually the only option we have. Why? Because to have a non-parametric model in reality we need to have an incredible amount of data to build the regression and this is not something common, but is reserved for special cases or applications where doing so is possible.

Now, what is the problem? The problem is that we cannot calculate anymore the true Best Predictors for our cases, because we have no more access to how the true distribution is made! The only thing we can do is to *estimate* them as best as we can starting from our samples. Here an important fact should become clear: taking the best possible sample is crucial for creating a great regression.

## Parametric specification
If we use the **parametric specification** we are assuming that the BP takes a rigid form $f(X;\beta)$ where $\beta$ is a *finite vector of unknown parameters*. Now, this vector is "**point identified**", meaning it can be computed *uniquely* from population data. The big problem is that we do not have the population anymore, therefore we have to estimate it with a statistical technique called **method of moments** (**MM**): solve the minimization problem with the sample analog of the population feature (e.g. sample average instead of population average).

We can see the difference in this way:
- **population land**: we are capable of finding the true parameter $\beta$ that best describe our distribution of $Y|X$ by minimizing the *population average* of the loss function:
$$
\min_{b} E\{L[y - f(x; \beta)]\}
$$
- **sample land**: we can't calculate the true expectation of the distribution, so instead we substitute it with the *sample average*:
$$
\min_{b} \frac{1}{N} \sum_{i=1}^{N} L[y_i - f(x_i; \beta)]
$$

This is the only thing we can do, because the sample average is the best guess for the population average! Also, notice this: identification is a requisite for us to do an estimation. If we can't solve for $\beta$ in population land, we surely can't estimate it in sample land!

What if we choose a specification that is not the best for showing how the distribution of $Y|X$ works? In this case, we say that we have made a **misspecification**. Choosing the correct specification for the regression is fundamental: you can't model as linear a relationship that is squared.

### Method of Moments for three loss functions
We have seen what the **method of moments** actually is: substituting the population feature with the sample feature in order to estimate the true parameters. Now, for the three types of loss functions that we have seen, we have three different BP:
- **square loss $\implies$ Ordinary Least Squares (OLS)**: by far the most known type of estimator, it naturally comes out when we try to minimize the square loss with respect to the vector $\beta$
- **absolute loss** $\implies$ **Least Absolute Deviations (LAD)**: minimizing the average absolute loss in the sample gives us the LAD estimator, best guess for the conditional median
- **asymmetric absolute loss** $\implies$ **Quantile Regression**: minimizing the average asymmetric absolute loss gives us the best guess for any conditional quantile

## Non-parametric specification
When we don't know the shape of the distribution of $Y|X$ and also we don't want to make assumptions about it, the only thing we can do is to let the data talk by assuming only continuity.

Now, remember: non-parametric specification works by averaging nearby data points "locally" around a specific value of $X$ called $\xi$. When we pass from population land to sample land, this can become a big problem. To fully understand why, we have to make a difference between two cases: when $X$ is discrete and when it's continuous.

### discrete $X$
If $X$ is discrete, then the probability of having $X$ exactly equal to $\xi$ is not zero: 
$$
P(X=\xi)>0
$$
This is very important, because it allow us to compute the average of all the data points where $X=\xi$. It's like splitting our dataset into groups and calculating group averages, because $X$ can actually assume that value $\xi$.
Now, what happens if my possible groups (possible value of $X$) are a lot? Well, this creates a little problem: each group get tinier, risking of having not accurate averages because there are too few observations. Here, the bigger the datasets the better it is.

### continuous $X$
If $X$ is continuous things are not that easy as they are in the discrete case, because the probability of $X=\xi$ is equal to zero:
$$
P(X=\xi)=0
$$
Imagine that $X$ refers to a group. If I have infinite possible groups, the probability of picking exactly $\xi$ among all the others is exactly zero. In this case, if I have infinite groups I need an infinite large sample in order to have all the data points for computing my average for each group. Here's the fact: an infinite large sample is the same thing as being in the population land.

Sadly, we are in the sample land and we have to find a solution to this problem. Having $N$ (sample size) finite prevents us from knowing all the possible value of $X$. A possible solution to this problem is, instead of looking to each single possible value, considering the *local average* of the values "near" the one we're interested with.

The neighborhood of our value must have a width, called **bandwidth**, that we have to choose in order to compute the average of the value inside it, but it's exactly this the real problem:
- **tiny bandwidth**: our average will be very accurate because we are only looking at values really close to our target one, thus the bias will be low; but the big problem is the *variance*: the closer we go to the target, the less values to compute the average will be available and, therefore, the result will have a very high variance because deeply unstable
- **huge bandwidth**: in this case is the opposite because our average, computed with a lot of data, will have much less variance and will be more stable BUT it will be less accurate, because we're including values that are not really close to ours

Of course the bigger $N$ is the better we can estimate our average in $X=\xi$, because we can restrict our bandwidth without increasing the variance.

This problem was thought unsolvable for a very long time, but in the era of machine learning a possible solution has come out: **kernel regression**. Basically, kernel regression is a more sophisticated version of a local averaging that weights every single value inside the local average depending on the distance between it and the value we are trying to estimate. This "local weighted average" produces a smoother result.

### Mathematics for both cases
Remember, non-parametric specification means letting the data speak about the best predictor for $Y$ given $X$ (for a specific value of $X=\xi$) which is $g(X)$, where $g(\cdot)$ is a continuous function determined by the minimization of the loss function (e.g. loss squared -> $g(\cdot) =E[Y|X=\xi]$). Now, given the fact that we are not in the population land, the only thing we can do is trying to estimate that function.

Given the fact that it is the most popular one, the discussion will considering as loss function the *squared loss*: the BP is the conditional mean.

#### Discrete case math
As discussed, this is the simplest case of all, because we can just grab all the data points where $X=\xi$ and average the $Y$ values for it. Mathematically:
$$
\hat{g}(\xi) = \frac{\sum_{i=1}^N y_{i}\cdot I(x_{i}=\xi)}{\sum_{i=1}^NI(x_{i}=\xi)}
$$
What is this formula saying? It's saying:"take all the $Y$ where $X=\xi$, sum them all up and then divide them for their number". This is nothing more than a standard sample average written to be conditional on $X=\xi$.
Why does it work? Because given the fact that we are in a discrete case, we expect our $X$ to be, in some cases, actually equal to $\xi$. Remember the definition of the discrete case: $P(X=\xi)>0$. If this event was not possible, then using that formula would just give us $0/0$ because there wouldn't be any case where $X=\xi$.

Also, notice the problem: if $X$ can take a lot of values, the probability for $X=\xi$ goes down, increasing the necessity of a larger sample size and also getting us closer to the continuous case. This is called "**curse of dimensionality**".

#### Continuous case math
When $X$ is continuous, the probability of having $P(X=\xi)=0$, so we can't use the average sample size to solve our problem. What we can do instead is average $Y$ values for $X$s that are "near" $\xi$.

To do that we need to pick a neighborhood around $\xi$, the *bandwidth* $\delta_{N}$, so that we have all the value that we decided were "near" to $X=\xi$. Now yes, now we can go back to the sample mean:
$$
\hat{g}(\xi)=\frac{\sum_{i=1}^Ny_{i}\cdot I[\rho(x_{i},\xi)<\delta_{N}]}{\sum_{i=1}^NI[\rho(x_{i},\xi)<\delta_{N}]}
$$
Let's understand what's inside the counter function $I[\cdot]$. The function $\rho(x_{i},\xi)$ is a *way to measure* the distance between $x_{i}$ and $\xi$. Do not stress about it, it's a just a way to quantify how much space there is between the two values! It can be miles, kilometers, whatever; usually the *Euclidian distance* is used. So, what the $I[\cdot]$ is doing in this case is assuming value equal 1 for all the cases where the distance between $x_{i}$ and $\xi$ is less than bandwidth we decided, meaning that $x_{i}$ is "near" $\xi$.

Now, this is powerful because the local average gets closer and closer to the true $g(\xi)$ (that we can't know) if the neighborhood is small bust still has *enough* points to compute the average and to compute it with not too much variance! Therefore, as $N$ grows the best thing to do is to shrink down $\delta_{N}$: our estimates of $g(\xi)$ will be more precise without increasing the variance of it. 

##### Kernel regression
The Kernel regression uses a more sophisticated way to compute estimate for the function in $X=\xi$, which is not cutting-off the points that are not in the bandwidth but, instead, it uses all points but weights them by the distance from $\xi$. Why? Because this way closer points matter more, farther ones less, and you still use all the available information that you have.
Mathematically:
$$
\hat{g}(\xi)=\frac{\sum_{i=1}^N y_{i}\cdot W_{N}(x_{i}-\xi)}{\sum_{i=1}^NW_{N}(x_{i}-\xi)}
$$
Here $W_{N}(x_{i}-\xi)$ is a function that weights $y_{i}$ smaller as $|x_{i}-\xi|$ increases; it's called "*kernel*" and it often takes the shape of a bell-shaped curve scaled by $\delta_{N}$ in order to personalize how much weighting a given distance with respect to another one.

##### LOOCV - Leave-One-Out Cross-Validation
Now, how do we decide *how well our non-parametric estimator is really performing*? If we simply check our estimator on the same data we used to fit it, we're at risk of "cheating": the model might look great not because it's truly flexible, but because it's literally memorizing the data! That's where **LOOCV (Leave-One-Out Cross-Validation)** comes in.

LOOCV is a statistical technique designed to test the out-of-sample performance of our regression estimator. Here’s the core idea:
- For each data point in our sample, we pretend that point never existed
- We build our non-parametric estimator using the remaining $N-1$ points
- We then predict the value of $Y$ at the left-out $X$
- We repeat this process for every single observation

Again, the idea behind is quite simple: to test how well our model works, we literally use it to predict every single data we have by estimating $Y$ without using an $X$. If we repeat this process for all our $X$, we will be able to see how well our model has predicted $Y$ without having the $X$ to do so. Mathematically, LOOCV is defined as:
$$
\text{LOOCV} = \frac{1}{N} \sum_{i=1}^N (y_{i}-\hat{g}_{-i}(x_{i}))^2
$$
where $\hat{g}_{-i}(x_{i})$ is our estimated regression function at $x_{i}$ **excluding** the $i$-th observation from the computation.

Using LOOCV can be very helpful for deciding a bandwidth, because what we can do is try different $\delta_{N}$ and choose the one that, among all the ones we have plugged in, has the lower LOOCV.


