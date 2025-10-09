
When studying causality having a framework to work with can be very helpful, because it enables us to try to quantify - systematically - a given situation. The **potential outcome framework** (POF) is such a tool and is widely used to study causality.

The POF considers two main scenarios: one where a treatment is applied to a unit and another one where it is not, then measures the outcomes from both cases and, by using the difference, it tries to quantify the impact that treatment had over the control unit.
So, we consider three key elements:
- **units**: elements of the population (individuals, firms, etc...) indexed with $i$
- **treatment**: variable being studied to assess its casual effect  
- **outcome**: measure after being treated or not treated at all

Especially, we say that:
- $D_{i}$ -> binary variable that determines wether a unit got the treatment or not
	- $D_{i} = 0$ -> no treatment
	- $D_{i}=1$ -> yes treatment
- $Y_i(D_{i})$ -> outcome of the i-th unit 
	- $Y_{i}(0)$ -> outcome of the unit that didn't get treated
	- $Y_{i}(1)$ -> outcome of the unit that did get treated

Before diving in the model, is important to understand that there are two worlds in which we have to think: a *potential* one and an *observational* one:
- **potential world**: it represents any logically possible value of $D$ or, in other words, the "what could it be" because nothing has been observed yet
- **observational world**: it represents the actual $D$ that we observe, so it's the "what it is" because is something we get to know

The key to fully understand the framework is to see the problem: if $D$ is a binary variable which can only assume value of 0 or 1 when it's observed, what about the case we don't observe? The observational world creates, just by existing, it's complement: the **counterfactual world**. In this space we have all the values that we have not observed because by  design our reality prevents us to do so. Mathematically, a counterfactual, the counterfactual outcome for unit $i$ is  $Y_{i}(1-D_{i})$.

This counterfactual world defines the **fundamental problem of causal inference** (FPCI), because the impossibility to observe the counterfactual outcome of a unit prevents us from comparing the case where the treatment has been administered and the case where it hasn't, so that we can't see the actual impact of it.

**(e.g.) - potential, observational and counterfactual worlds**
Imagine we have a scenario where a person will take a drug or not. In the potential world we have that $D=\{ 0,1 \}$. In the moment that the person takes the drug or doesn't, $D$ assumes a precise value $D_{i}=0$ or $D_{i}=1$. The fact that our person took or didn't take the drug creates a parallel universe where the opposite would have occurred: the *counterfactual world*, something we cannot observe because something else has been chosen.

## POM - Potential Outcomes Model
The **potential outcomes model** is a tool that we can use, based on the framework just described, to estimate the impact of a given treatment on a unit. It is described by the following equation:
$$
Y_{i} = D_{i}Y_{i}(1)+(1-D_{i})Y_{i}(0)
$$
Which can be also written as:
$$
Y_{i} = Y_{i}(0) +(Y_{i}(1)-Y_{i}(0))D_{i}
$$
Now, this equation is saying: "the outcome for the i-th unit is equal to $Y_{i}(1)$ when $D_{i}=1$ or to $Y_{i}(0)$ when $D_{i}=0$". So, when:
- $D=1$ -> $D=0$ is the counterfactual
- $D=0$ -> $D=1$ is the counterfactual

### Assumptions
When dealing with the model and with the version of it that we are going to discuss for the moment, is very important to know the assumption at the core of it. In this case, that assumption is called **SUTVA** (Stable Unit Treatment Value Assignment), which requires:
- **no interference**: treatments given to one unit do not affect others outcomes, implicating that there are no spillovers or networks effects -> $Y_{i}(D_{i},D_{-i}) = Y_{i}(D_{i})$.
- **no hidden variations of treatments**: the treatment is standardized for all units, so it is exactly the same for every $D_{i}=1$.

The importance of this assumptions is trivial:
- interference -> treatments affect others outcomes, meaning that they   are biased by indirect influences and not only from the treatment
- hidden variations -> different treatments make impossible to define a correct causal impact you are changing the treatment you are giving

At the core this assumption wants to make sure that the only factor that affect the outcome is a given treatment, so there are not other influences which make the estimation of the causal relation impossible (or too complex) to define.

### Defining the impact of the treatment over the outcomes
Now that we have our assumptions and a model that supports us, we can actually define the formula to quantify treatment's impact by computing the difference between the case where it has been administered and it has not:
$$
\Delta_{i} = Y_{i}(1)-Y_{i}(0)
$$
And by recalling the main equation of the POM:
$$
Y_{i} = Y_{i}(0)+\Delta_{i}\cdot D_{i}
$$
So, the idea is simple: if $D=1$ we observe $Y(1)$ and if $D=0$ we observe $Y(0)$, then if SUTVA assumption is satisfied we can say that the difference $\Delta$ between the two outcomes must be the impact of the treatment.

At a logical level it is very easy to visualize, but the problem arrives when we actually try to do so because, due to FPCI, we cannot observe for the same unit two different worlds in the same point in time and, if we think of observing one at a given $t$ and the other at $t+\delta$, it is not correct because those are two different situations.

What if we change our perspective and try to think of multiple units instead of always single ones?

### Multiple units and group-level effects
Given the fact that we can't figure out the effect of a treatment on a specific individual due to the unobservability of counterfactuals, we have in someway to reconstruct these "what if" scenarios. To do that, we look at *groups of individuals* and compare their outcomes: we lose some information about individual differences, but this allows us to estimate the average effects at the group level.

The idea now is to comprare what happens to the group under different treatment strategies called *regimes*: hypothetical scenarios where everyone in the group gets the same treatment, for example
- regime 1 -> everyone is treated
- regime 0 -> everyone is not treated

Now that we have groups, we have to ask ourselves: do we want differences in outcome distributions, or distributions of outcome differences to analyze treatment effects?

- **Comparing outcome distributions**: one way to analyze the impact of the treatment is to see the difference between the distributions (probability density functions)
	- $f[Y_{i}(1)]$: distribution if treated
	- $f[Y_{i}(0)]$: distribution if not treated
- **Analyzing the distribution of outcome differences**: another way is to look at the distribution of individual differences in outcomes $f[Y_{i}(1)-Y_{i}(0)]$ 

This difference is not interesting at all when we want to analyze the causal effect using the expected values, because what happens is that:
$$
E[Y_{i}(1)-Y_{i}(0)] = E[Y_{i}(1)]-E[Y_{i}(0)]
$$
What this equation is saying is that making the difference of average outcomes (right term) is equivalent to make the average of the difference of individual outcomes (left term). 

Things change when we use anything other than the mean (e.g. median) because this equivalence falls to pieces, then the way we want to confront the groups to analyze the causal effect starts mattering.

#### Choice of population
Another very important question is: "which population is relevant to my analysis?". In other words, what is that I'm truly interested in? Basically, we have three possible population to choose from:
- **population to be treated**: to be used when you are evaluating something that would affect all the units of a group, so that you consider all both the treated and the untreated 
- **actually treated**: to be used when you are interested on the effects of the treatment on the group that actually received the treatment
- **actually untreated**: to be used when you are interested on the effects of the treatment to units that have not been treated

This way of thinking can feel counterintuitive. You might wonder: "How can I be interested in the effect of the treatment for the group that already received it?"; the fact is you can indeed be interested in that case when you remember that there are two worlds: the real (observed world) and the potential (theoretical world). For every group (all units, just the treated, or just the untreated), what we're really asking is: "What would have happened to these specific units if things had gone differently?" So, for the actually treated, the real question becomes: "What difference did the treatment make for those who got it, compared to what would have happened to them if they hadn’t been treated?".

### Average Treatment Effects
The **average treatment effects** is the logic way that we would use to analyze, most of the times, the causality of the treatment. The idea is that if we can't have the impact for each individual, what about the mean of it? Mathematically:
$$
\delta^{ATE}=E[\Delta_{i}] = E[Y_{i}(1)-Y_{i}(0)] = E[Y_i(1)]-E[Y_{i}(0)]
$$
This can be ridden as: "the average effect of the treatment is equal to the differences between the averages of the case where everyone received treatment and the case when no one did". So, careful:
- $E[Y_{i}(1)]$ is the expected value of the outcome that would be observed *if everyone in the population got treated*
- $E[Y_{i}(0)]$ is the expected value of the outcome that would be observed *if everyone in the population didn't get treated*

Instead, when we want to refer to the mean of observed outcome of a given group (what we actually see), we write: $E[Y_{i}|D=\{ 0,1 \}]$. This difference is key to understand what comes next.

Now, two variants exist when we choose to analyze the other two populations: 
- **average treatment effect on the treated - (ATT)**: average effect of the treatment on the population that has been selected to be treated. Mathematically:
$$
\delta^{ATT} = E[\Delta_{i}|D_{i}=1] = E[Y_{i}(1)|D_{i}=1] - E[Y_{i}(0)|D_{i}=1]
$$
What we're saying is that the ATT is calculated as the difference between the average outcome we actually observe for the treated group and the average outcome they _would have had_ if they hadn't received the treatment ($E[Y_{i}(0)|D_{i}=1]$). In other words, ATT tells us how much the treatment changed things for those who got it compared to a parallel world where those same units didn’t get treated.
- **average treatment effect on the untreated - (ATU)**: average effect of the treatment on the population that has been chosen to not be treated. Mathematically:
$$
\delta^{ATU} = E[\Delta_{i}|D_{i}=0] = E[Y_{i}(1)|D_{i}=0] - E[Y_{i}(0)|D_{i}=0]
$$
Analogous thing here, the ATU is the difference between the average outcome the units *would have had* if they had received the treatment ($E[Y_{i}(1)|D_{i}=0]$) and what we actually observe for the untreated group. So, ATU measures how the outcomes would have changed for those who didn't get treated, if they got the treatment instead.

A key thing to know is that we can write the ATE as a weighted average of ATT and ATU:
$$
\delta^{ATE} = \delta^{ATT} \cdot P(D = 1) + \delta^{ATU} \cdot P(D=0) 
$$
$$
\delta^{ATE} = \delta^{ATT} \cdot P(D = 1) + \delta^{ATU} \cdot (1-P(D=1))
$$
This works for a simple reason: ATE is the population average, so it is the result of mixing treated and untreated cases by their proportion in each group.
#### Conditional Average Treatment Effects
Something that we can often do is to condition on an **observable characteristic** $X$ and analyze the average treatment effects only on that population. Mathematically, we write this as:
$$
CATE(X) = E[Y_{i}(1)-Y_{i}(0)|X] = E[Y_{i}(1)|X] - E[Y_{i}(0)|X]
$$
This is not something new, because is basically a special case of conditioning which let us refine our population on $X$ instead of $D$. 

(e.g.) - CATE based on location 
Suppose we observe a lot of individuals and we categorize them with respect to $D=0$ or $D=1$. Now we notice that there is also another variable we can observe: the location $X$, which may be city or campaign; then $X \in \{ \text{city, campaign} \}$. What we can do is to analyze the average treatment effect by selecting only one value of $X$ so, for example, for all the people living in a city.

### Naive Estimator
When we stop looking at the theory and start doing things, we end up in a situation where we surely can't compute $\delta^{ATE}$ because we have the FPCI that prevents us. Instead, we define the **naive estimator** (or NE) on the base of what we can actually see: 
$$
NE = E[Y_{i}|D_{i}=1]-E[Y_{i}|D_{i}=0]
$$
This is the difference in *observed means* between the treated group and the untreated group.

This estimator is called "naive" because it is not always right: we do not take into consideration the counterfactuals and we can't observe. To understand this, we have to actually rewrite $NE$ in another way:
$$
NE = E[Y_{i}|D_{i}=1]-E[Y_{i}|D_{i}=0]
$$
$$
= E[Y_{i}(1)|D_{i}=1]-E[Y_{i}(0)|D_{i}=0]
$$
Now let's add and subtract $E[Y_{i}(0)|D_{i}=1]$:
$$
(E[Y_{i}(1)|D_{i}=1] - E[Y_{i}(0)|D_{i}=1])

+ (E[Y_{i}(0)|D_{i}=1]-E[Y_{i}(0)|D_{i}=0])
$$
Watch the magic happen:
$$
= \delta^{ATT}+\{ \text{selection term} \}
$$
Let's develop some intuition for this by first analyzing what each term is:
- $\delta^{ATT}$ -> the average treatment effect on the group that has been selected to be treated
- $\text{selection term}$ -> also known as $\text{bias}$, it represents the difference in "what would have happened without treatment" between the treated and untreated group

Here the formula starts having its logic: the effect that I expect the treatment to have is given by the actual effect ($\delta^{ATT}$) plus some noise ($\text{bias}$) that I encounter when I consider the differences between the two groups. In reality this term is rarely zero, so the measure that we can have of treatment's impact is often biased because is very difficult to have two identical experimental groups.

#### NE further decomposition
Instead of stopping here, we can actually go further if we rewrite the ATT in terms of ATE and ATU:
$$
ATE = P(D=1) \cdot ATT + (1-P(D=1)) \cdot ATU
$$
Let's rewrite it in order to get $ATT$:
$$
ATT - ATE = ATT - [ATT \cdot P(D=1) + ATU \cdot (1-P(D=1))]
$$
$$
ATT = ATE + (1-P(D=0))\cdot (ATT-ATU)
$$
Notice this: we have the average treatment effect on the treated both on the left and right. Why is that? Because equation doesn't want to be a definition but, instead, an identity that we can use to include the $ATE$ and $ATU$ into $NE$.

Rewriting now this into our last decomposition, we obtain:
$$
NE = \delta^{ATE} + \text{ bias term} +\underbrace{(1-P(D=0)) (\delta^{ATT}-\delta^{ATU})}_{\text{ heterogeneity term}}


$$
The right term is called **heterogeneity** and what it does is essentially comparing $ATT$ and $ATU$. The intuition is that if these two are *different*, then it means that treatments effects are not the same of the two groups but, instead, their biased. Remembering that $NE$ is something we would like to use to estimate $ATE$, we can notice what we just said in plenty words also with mathematics:
- if $ATT > ATU$ -> $NE$ overstates $ATE$
- if $ATT < ATU$ -> $NE$ understates $ATE$

So, basically, heterogeneity is a term that quantify how much the treatment effects on a group differs from the treatment on the other by asking: "would the treatment effect itself be the same for treated and untreated?".

#### Intuition behind decompositions
We define the naive estimator in order to have an idea of what our $ATE$ is, but two problems comes to mind when we use it:
1. *are we sure that the treated and untreated have had the same baseline outcome if no one had been treated?* What we are doing here is considering the counterfactual of untreated outcomes of the treated group
2. *are we sure that the treatment effect itself would be the same both for the treated and untreated?* Here, instead, we consider the counterfactual of treated outcomes of the untreated group

Essentially, we want to be sure that everyone starts from the same line and that the treatment has an homogeneous causal impact on everybody.

The first point is covered by the *selection term*, while the second one from the *heterogeneity term*. Those are fundamentals to truly understand wether our $NE$ can considered a good tool to compute $ATE$ or not. 

## Perfect randomization 
The key problem when working with causality is that our only way to "naturally" estimate the ATE is by using the $NE$ but, as we saw, this tool can easily fall victim of the *selection* and *heterogeneity* terms. The only way to solve this problem when dealing with real world is by designing a **perfectly randomized experiment**, also called **randomized controlled trial** (**RCT**), in order to ensure our bias terms to be null.

### Randomization Vs Bias Terms
If we take a whole population and we start selecting in a completely random way (based on no observational characteristics) units to split in a **treated group** (**T**) and in a **control group** (**C**), we end up with two **statistically identical groups** in both *observed* and *unobserved* characteristics expect for *treatment status* $D$.

Why this happens? The idea is that if I completely randomize the selection then the groups that I create will be, on average, exactly one equal to the other. The implication for having identical groups except for the treatment status $D$ also means that the *baseline* from which they start and the *impact of the treatment* on them will be also the same, eliminating both the selection and heterogeneity terms. Mathematically:
$$
E[Y(0)|D=0] = E[Y(0)|D=1] \implies \text{Selection Term} =0
$$
$$
E[Y(1)|D=1] = E[Y(1)|D=0] \implies ATT = ATU \implies \text{Heterogeneity Term } = 0
$$

In other words: randomization works because it balances both the “*starting points*” (what outcomes would look like without treatment) and the “*responses*” (how much treatment changes those outcomes) across the two groups. The only thing left distinguishing them is whether they got the treatment, which is exactly what we want.

Now, careful: an RCT does **not** eliminate heterogeneity at the individual level, because different units may still respond differently to treatment. The RCT instead guarantees that heterogeneity is *balanced across the groups on average*, because each group will have the same mix of "high responders" and "low responders". Therefore, the ATT equals the ATU, making the heterogeneity term null. If it's true then that randomization doesn't actually kills heterogeneity, it surely kills bias.



