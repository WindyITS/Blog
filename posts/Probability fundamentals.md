
Probability is a branch of mathematics that is used to deal with *uncertainty*, so that we can understand and have information about the possible outcomes of an *experiment*.

## Fundamental terminology
### Sample space
We define the **sample space** $\Omega$ as the set of all possible **outcomes** $\omega$ of an experiment, which are also collet "elementary events". 
Mathematically it can be written as:

$$
\Omega =\{  w=(w_{1},w_{2},\dots):w_{i}=\dots \}
$$
This expression means that within the sample space there are all the possible elementary events $\omega$ such that those elementary events can take the value $\omega_{i}$.

(e.g.) - Tossing a coin
When we toss a coin we have two possible outcomes: head or tail. This allows us to say that the sample space of the experiment is:
$$
\Omega = \{ H,T\} = \{ w=(w_{1}):w_{i}=H,T \}
$$
If we toss the coin twice, our sample space changes:
$$
\Omega =\{ \omega=(\omega_{1},\omega_{2}): \omega_{i}=H,T \}
$$
Or we can also use another format which allow us to write:
$$
\Omega= \{HH,TT,HT,TH\} = \{H,T\} \times \{H,T\} = \{H,T\}^2
$$
This way of writing the set uses the *cartesian product*.

#### Finite vs Countable vs Uncountable
When we talk about sample spaces it's important to understand the space we have to deal with and how it is built:
- **finite**: a sample space is finite if it contains a limited and specific number of possible outcomes (e.g. rolling a six-sided die)
- **countable**: a sample space is countable when the elements can be put into a one-to-one correspondence with the natural numbers, meaning you can actually list them in a sequence even if they are infinite (e.g. all positive integers)
	- the *finite* case is therefore a special case of countable space
- **uncountable**: a sample space is uncountable when the elements cannot be put into a one-to-one correspondence with the natural numbers, meaning that you can't list those elements as a sequence (e.g. all the real numbers in $[0,1]$)

This result may be counter-intuitive: "in the end the infinite between 0 and 1 in real numbers is the same of the ones that occurs in a countable sample space", well this is not true at all. If we take into account *cardinality*, then we discover that the "size" of the infinity in an uncountable sample space like $[0,1]$ is much larger and different than the case of a countable - but infinite - sample space.

### Event
An **event** is a *subset* of the sample space $\Omega$:
$$
A \subset \Omega
$$
Therefore, an event is itself a set: 
$$
A = \{ \dots \}
$$
A special case is when an event contains only one point, then it is called a "*singleton*".
We say that "*A has occurred*" when:
$$
w \in A
$$

#### Event complement 
We call **complement of event A** the event that occurs if and only if A does not. Mathematically:
$$
A^c = \{ \omega \in \Omega : \omega \not\in A \}
$$
#### Impossible and certain events
We define an **impossible event** $\phi$ as an event that can't occur at all; instead, we consider the **certain event** $\Omega$ as the one that will happen for sure (so it must be that $A = \Omega$).

#### Events implications
When an event $A$ is a subset of another event $B$ (so $A \subset B$) we say that $A$ **implies** $B$ because if A occur then so does B.


### Union
The **union** is a type of operation which gives as result an event that occurs if and only if one of the events that belongs to the union occurs. Basically it groups events together into only one:
$$
\bigcup_{i} A_{i} = \{ \omega \in \Omega: \exists i: \omega \in A_{i} \}
$$
We can read this expression as follows: "the union of $A_{i}$ is a set made by all the elementary events $\omega$ belonging to the sample space $\Omega$ such that exists a case where at least one of this elementary events belongs to one of the events of the union".

### Intersection
The **intersection** is a type of operation which gives as result an event that occurs if and only if all the underlying events occur.
Mathematically:
$$
\bigcap_{i}A_{i} = \{ \omega \in \Omega: \omega \in A_{i} \space \space\forall i \}
$$
We can read this expression as: "the intersection of $A_{i}$ is a set made by only the elementary events $\omega$ that are shared by all the events object of the intersection, so that it occurs only when all the underlying events do".

### Factorials
A **factorial** is a mathematical function that multiplies non-negative integer by every positive integer between it and 1. 
Mathematically:
$$
n! = n \times (n-1) \times (n-2) \times 2 \times 1
$$
Also, by convention we have that $0! = 1$.

In probability factorials are really useful because they represent the all possible **permutations**, which are ways to arrange objects while caring about their order ($a,b$ is not the same of $b,a$).

We can get an intuitive understanding of factorials as follows: when arranging objects with have $n$ ways to choose the first one, $n-1$ to choose the second and so on.

(e.g.) - Permutation of 3 objects
If we're given three objects - say $a, b,c$ - we can compute the amount of possible ways to arrange them with the factorial: $3! = 6$.

### Binomial coefficients
A **binomial coefficient** is a function that describes in how many ways you can choose $k$ objects from a set of $n$ objects without caring of the order at all ($a,b$ is the same of $b,a$).
Mathematically:
$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$
It is ridden as: "*n chooses k*".
A good way to compute those binomial coefficients is to simplify the fraction, which actually leads us to:
$$
\binom{n}{k} = \frac{n(n-1)\dots (n-k+1)}{k!}
$$

and also is useful to notice that:
$$
\binom{n}{k} = \binom{n}{n-k}
$$

We can also develop some intuition behind the formula. Let's think about the components:
- $n!$ -> counts every permutations for all the $n$ objects
- $k!$ -> counts every permutations for the object I choose
- $(n-k)!$ -> counts every permutations for the objects I don't choose

When we compute $n!$ we are counting the same choice of objects more than once, because permutations care about order; therefore, we can divide by $k!$ to normalize and solve the problem. Same goes for $(n-k)!$. 
In the end, what we are actually doing is this:
$$
\binom{n}{k} = \frac{\text{all arrangements}}{\text{ways to arragne the chosen }k \text{ and unchosen }n-k}$$

(e.g.) - Combination of choosing 3 objects from 5
If we're given five objects and we want to choose randomly three of them, we can use the binomial coefficient to compute in how many ways this selection can go without caring about the order: $\binom{5}{3} = \frac{5 \times 4}{2!} = 10$.

## De Morgan's laws
**De Morgan's laws** are fundamental rules in set theory and probability, describing the relationship between unions, intersections and complements of sets or events.

There are two laws:

1. **First law**: the complement of the union is equal to the intersection of the complements. Mathematically:
$$
(A\cup B)^c = A^c \cap B^c
$$
	The idea behind the statement is if you take the union of two events you are actually creating a bigger set made from both $A$ and $B$ together; so, if you take the complement of this event you are taking something that does not belong neither to $A$ or $B$, which can be written as the intersection of something that does not belong to $A$ AND does not belong to $B$.
	We can generalize this law so that it can be more useful:
$$
\left( \bigcup_{i}A_{i} \right)^c = \bigcap_{i}A_{i}^c
$$
	Another way to say what the first law is stating is: "everything that is *not* in the union is exactly something that is *not in $A$* and *not in* $B$.
2. **Second law**: the complement of the intersection is equal to the union of the complements. Mathematically:
$$
(A\cap B)^c = A^c \cup B^c
$$
	The idea behind the second law is that if you take the complement of an intersection of two events (which contains all that is not in $A$ AND $B$), then it's the same as taking the what is not in A or in B! It's not different from saying: "if I take something that is not in A and B, then that thing surely is not in A or not in B".
	We can also generalize this law so that it can me more useful:
$$
\left( \bigcap_{i} A_{i} \right)^c = \bigcup_{i}A_{i}^c
$$

This laws are very important for logic and probability; one use of them will be shown in the derivation of the first sigma-algebra's property.

## Sigma-Algebra
A **sigma-algebra** (or $\sigma-\text{algebra}$) is a mathematical structure that is used in probability theory to define a collection of subsets of a given set to which a measure (in our case, probability) can be assigned to. In other words, given a set, a sigma-algebra is a set of subsets that defines the *measurable space* $(\Omega, \mathcal{F})$ in which we have certain characteristics that allow us to correctly assign probabilities to given events inside it.

More rigorously, a sigma-algebra is set defined with respect to the *sample space* $\Omega$ and denoted as $\mathcal{F}$ which satisfies the following properties:

1. **non-emptiness**: the entire sample space is in the sigma-algebra, ensuring that the whole $\Omega$ is measurable. Formally:
$$
\Omega \in \mathcal{F}
$$
2. **closure under complement**: if an event belongs to the sigma-algebra, then also does the complement of that event. Formally:
$$
A \in \mathcal{F} \implies A^c \in \mathcal{F}
$$
	The idea is that if we can measure a set, then we can also measure everything that is not in that set.
3. **closure under countable unions**: if we have a countable sequence of subsets in our sigma-algebra, then their union also belongs to the sigma-algebra. Formally:
$$
\{A_n\}_{n=1}^\infty \in \mathcal{F} \implies \bigcup_{n=1}^\infty A_n \in \mathcal{F}
$$
	The idea is that if we combine a countable number of measurable sets, the result still is a measurable set.

### Properties of the sigma-algebra
By the first three properties we can derive more of those that can be very useful to understand and to work with probability.

- **closure under countable intersection**: if we know that the sigma-algebra is closed under countable unions and under complements, we can say that the same goes for countable intersections. This happens - on a concept level - because we can actually describe intersections using unions and complements such that the complementary of the intersection of all the sets is the union of all the complementary sets (De Morgan Law):
$$
\left(\bigcap_{n=1}^{\infty}A_{n}\right)^c = \bigcup_{n=1}^{\infty}(A^c_{n})
$$
	This has it's logic: if something is not common to every event, then that something does not belong to some events. Given the fact that the right side captures all the elements that are not in any $A_{n}$, we are sure that those elements belong to what the complementary of the set containing all the elements common between all events.
	Now, given the fact that $\mathcal{F}$ is closed under complements (each $A_{n}^c \in \mathcal{F}$) and under countable unions ($\bigcup_{n=1}^{\infty} A_n^c \in \mathcal{F}$), we have that $\left(\bigcap_{n=1}^{\infty}A_{n}\right)^C \in \mathcal{F}$. Taking the complement again will give us:
$$
\left(\bigcap_{n=1}^{\infty}A_{n}\right)^C \in \mathcal{F} \implies 
\left(\left(\bigcap_{n=1}^{\infty}A_{n}\right)^{c}\right)^c = 
\bigcap_{n=1}^{\infty}A_{n} \in \mathcal{F}
$$
- **empty set inclusion**: the empty set $\phi$ is included in the sigma-algebra because one of the characteristic is closure under complements so $\Omega^c = \phi$. Formally:
$$
\phi \in \mathcal{F}
$$
- **closure under finite unions and intersections**: finite unions and intersections are special cases of countable unions and intersections (where the count is finite); therefore, our conditions naturally includes this one.

### Sigma-Algebra vs countable and uncountable sets
Building the sigma-algebra is not that easy, because there is a big difference wether the sample space is *countable* or *uncountable*. Why this happens? Because when we fall in the second case, so in the real ream, we encounter subsets to which we cannot actually assign a measure at all (e.g. Vivaldi set). Before diving to see how to solve the problem, let's discuss the countable case.

#### Power sigma-algebra
A **power sigma-algebra** includes every possible subset of outcomes in the sample space and arises naturally when the sample space $\Omega$ is *countable*. The idea is simple: if we can list the sample space, we can include every possible subset and assign a probability to it.

In this case, we take as sigma-algebra *all subsets of the sample space*.

#### Borel sigma-algebra
A **Borel sigma-algebra** includes only some subsets of outcomes and not every possible one. The idea is that it is smaller than it's full potential, because it takes out all the subsets that are too weird to measure in a consistent way. It can be wrote as:
$$
\mathcal{B}(\Omega)
$$

## Probability and Kolmogorov's axioms
The fundamentals axioms of probability are called **Kolmogorov's axioms** due to their thinker and those enable us to compute and measure the odds of a given event to occur. First of all, probability is is a *function* with the *sigma-algebra as domain* and the *interval $[0,1]$ as co-domain*:
$$
P:\mathcal{F} \to [0,1]
$$
A function, to be considered a probability function, has to satisfy the following axioms:
1. **non-negativity**: for every possible event $A$, the odds for it to happen must always be non negative. Mathematically:
$$
P(A)\geq 0
$$
2. **Normalization**: the probability for the certain event $\Omega$ to happen must be equal to 1. Mathematically:
$$
P(\Omega)=1
$$
3. **Infinite countable additivity**: for disjoint events it must be true that the probability of their union is equal to the sum of their probability. Mathematically:
$$
\text{If } A_{i} \bigcap A_{j} = \phi, \space \forall i≠j \implies
P\left( \bigcup_{i}^{\infty}A_{i} \right) = \sum_{i}^{\infty}P(A_{i})
$$
Arrived here, we define the **probability space** $(\Omega,\mathcal{F},P)$ as the mathematical structure used to model random experiments and analyze uncertainty.

### Probability's properties
Given the definition of probability and its axioms, we can derive some key properties out of it:

1. **Probability of the impossible event is also null**: intuitively, the odds for the impossible event to occur are zero. Mathematically:
$$
P(\phi)=0
$$
2. **Finite countable additivity**: given that the third axiom refers to the infinite case and finite is a subset of infinite, we conclude that the third axiom works even for finite cases. Mathematically:
$$
\text{If } A_{i}\bigcap A_{j} ≠ 0, \space \forall i≠j \implies P\left( \bigcup_{i}^nA_{i} \right) = \sum_{i}^n P(A_{i})
$$
3. **Monotonicity**: if an event $A$ implies an event $B$ (so $A \subset B$), we know that the odds of $A$ to occur are lower than $B$'s ones. Mathematically:
$$
\text{If }A \subset B \implies P(A)\leq P(B)
$$

4. **Probability of an event**: an event $A$ can be, as extremes, impossible or certain, nothing less or nothing more. Therefore:
$$
0\leq P(A)\leq 1
$$
5. **Complementary rule**: the odds for an event to not occur are equal to the odds of the certain event minus the odds for it to occur. Mathematically:
$$
P(A^c)= 1-P(A)
$$
6. **Union rule**: the probability of an union between two events is equal to the sum of each odds minus the intersection of those, because already counted in the sum. Mathematically:
$$
P(A\cup B) = P(A)+P(B)-P(A\cap B)
$$

Those properties are keys to access a more complexed realm, where conditional probabilities, cumulative distributions and so on live.

## Conditional probability
**Conditional probability** is a subfield probability theory which tries to answer the following question: "what are the odds for $A$ to occur if $B$ has occurred?". In this case, we literally *condition* the odds of one event by assuming that another one has happened.
Mathematically:
$$
P(A|B) = \frac{P(A\cap B)}{P(B)} \space \text{ with } P(B)>0
$$

An intuition can be built over this formula: the odds for $A$ to occur when $B$ can be seen as the odds that $A$ occurs when the sample space becomes $B$. In this case, our event of interest is $A\cap B$ and all the other possibilities are represented by $B$.

An interesting fact is that we can rearrange the definition in order to obtain a formula to compute the intersection between the two events that is widely used in statistics:
$$
P(A\cap B)=P(A|B)\times P(B)
$$

Now, before going further, knowing partitions can help.

### Partitions
A **partition** is a way to split the sample space $\Omega$ into non-overlapping events such that every possible outcome belongs to exactly one event. In a more rigorous way, we say that a partition of a sample space $\Omega$ is a set of events $E_{1},E_{2},..,E_{n}$ with the following characteristics:
- **non-empty**: each set in the partition must contain at least one outcome. Mathematically:
$$
E_{i} ≠ \phi
$$
- **mutually exclusive**: the events in the partition can't overlap, as no outcomes should appear in more than one event. Mathematically:
$$
E_{i}\cap E_{j}=\phi \space \text{ when } i≠j
$$
- **exhaustive**: the union of all sets in the partition covers the entire sample space $\Omega$. Mathematically:
$$
\bigcup_{i}E_{i}=\Omega
$$

A partition can also be formalized as follows:
$$
\forall \omega \in \Omega, \exists!i: \omega \in E_{i}
$$
It can be ridden as: "for every elementary event in the sample space, exists one and only one $i$ such that the event belongs to the i-th partition".

### Total Probability theorem
The **total probability theorem** (or *law of total probability*) allows us to compute the probability of an event considering how it might occur across partitions in the sample space. Formally, we can say that if $\Omega$ is partitioned, then:
$$
P(A) = \sum_{i}^n P(A|E_{i})P(E_{i})
$$
As we saw, this is not different from writing:
$$
P(A) = \sum_{i}^n P(A\cap E_{i})
$$
The intuition behind the formula is simple: imagine we have our $\Omega$ partitioned in $i$ events; then, we can compute the odds of $A$ to occur as the sum of the odds that $A$ occurs when each partition event occurs.

(e.g.)- The odds for someone to have a disease
Suppose in the world there are three regions ($E_{1},E_{2},E_{3}$), $P(E_{i})$ is the probability someone is from region $i$ and $P(A|E_{i})$ is the probability that the subject has the disease given the fact that they come from that region; we can compute the odds that a random person has the disease by:
$$
P(A)=P(A∣E_{1})P(E_1)+P(A∣E_{2})P(E_{2})+P(A∣E_{3})P(E_{3})
$$

### Bayes theorem
