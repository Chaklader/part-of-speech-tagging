# C-2: Probability Theory and Bayesian Methods

1. Fundamentals of Probability

    - Law of Conditional Probability
    - Law of Total Probability
    - Chain Rule of Probability
    - Bayes' Theorem and Its Applications
    - Independence and Conditional Independence

2. Working with Probability Distributions
    - Discrete Probability Distributions
    - Joint and Marginal Probabilities
    - Calculating Conditional Probabilities
    - Normalization Methods
    - Complex Probability Examples

#### Fundamentals of Probability

##### Law of Conditional Probability

Conditional probability is the foundation of Bayesian reasoning. It answers the question: "Given that event B has
occurred, what is the probability that event A will occur?" This is written as P(A|B).

The formal definition of conditional probability is:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Where:

- P(A|B) is the probability of event A given that event B has occurred
- P(A ∩ B) is the probability of both events A and B occurring (joint probability)
- P(B) is the probability of event B occurring (and must be greater than zero)

This can be rearranged to give us the multiplication rule:

$$P(A \cap B) = P(A|B) \times P(B)$$

This form tells us how to calculate the joint probability of two events.

For example, if we know the probability of rain tomorrow is 0.3, and the probability of heavy traffic given that it
rains is 0.8, then the probability of both rain and heavy traffic tomorrow is:

$$P(\text{Rain} \cap \text{Traffic}) = P(\text{Traffic}|\text{Rain}) \times P(\text{Rain}) = 0.8 \times 0.3 = 0.24$$

Conditional probability is often illustrated using tree diagrams or tables to visualize the branching possibilities:

```mermaid
graph TD
    A["Event Space"] --> B["B occurs: P(B)"]
    A --> C["B doesn't occur: P(¬B)"]
    B --> D["A occurs given B: P(A|B)"]
    B --> E["A doesn't occur given B: P(¬A|B)"]
    C --> F["A occurs given ¬B: P(A|¬B)"]
    C --> G["A doesn't occur given ¬B: P(¬A|¬B)"]
```

##### Law of Total Probability

The Law of Total Probability allows us to calculate the probability of an event by considering all the ways that event
can occur through mutually exclusive pathways. It's particularly useful when we have information about conditional
probabilities but need the overall probability.

For any partition {B₁, B₂, ..., Bₙ} of the sample space (where the Bᵢ are mutually exclusive and exhaustive):

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \times P(B_i)$$

In the simplest case with just two complementary events B and ¬B:

$$P(A) = P(A|B) \times P(B) + P(A|\neg B) \times P(\neg B)$$

This formula tells us that the total probability of A is the weighted sum of conditional probabilities, where the
weights are the probabilities of the conditioning events.

For example, to find the probability of having a fever, we might consider whether a person has an infection:

$$P(\text{Fever}) = P(\text{Fever}|\text{Infection}) \times P(\text{Infection}) + P(\text{Fever}|\text{No Infection}) \times P(\text{No Infection})$$

If P(Fever|Infection) = 0.9, P(Infection) = 0.1, P(Fever|No Infection) = 0.05, and P(No Infection) = 0.9, then:

$$P(\text{Fever}) = 0.9 \times 0.1 + 0.05 \times 0.9 = 0.09 + 0.045 = 0.135$$

So the overall probability of fever is 13.5%.

##### Chain Rule of Probability

The Chain Rule (or General Product Rule) extends the multiplication rule to multiple events. It provides a way to break
down joint probabilities into a product of conditional probabilities:

$$P(A_1, A_2, ..., A_n) = P(A_1) \times P(A_2|A_1) \times P(A_3|A_1, A_2) \times ... \times P(A_n|A_1, A_2, ..., A_{n-1})$$

This rule is crucial for Bayesian networks because it shows how a joint probability distribution can be decomposed into
simpler conditional probabilities.

For example, the joint probability of three events A, B, and C is:

$$P(A, B, C) = P(A) \times P(B|A) \times P(C|A, B)$$

A detailed proof can help solidify understanding:

For three events A, B, and C, start with the basic conditional probability formula:

$$P(A, B, C) = P(A|B, C) \times P(B, C)$$

Then apply the same formula to P(B, C):

$$P(B, C) = P(B|C) \times P(C)$$

Substituting this into the first equation:

$$P(A, B, C) = P(A|B, C) \times P(B|C) \times P(C)$$

The chain rule generalizes this pattern to n events.

In the context of our earlier "Wet Grass" example, we can use the chain rule to express the full joint probability:

$$P(\text{Cloudy}, \text{Rain}, \text{Sprinkler}, \text{Wet}) = P(\text{Cloudy}) \times P(\text{Rain}|\text{Cloudy}) \times P(\text{Sprinkler}|\text{Cloudy}) \times P(\text{Wet}|\text{Rain}, \text{Sprinkler})$$

Note that we exploited the conditional independence of "Rain" and "Sprinkler" given "Cloudy" to simplify the expression.

##### Bayes' Theorem and Its Applications

Bayes' Theorem is the cornerstone of Bayesian inference. It provides a way to update our beliefs based on new evidence.
The theorem is derived from the definition of conditional probability:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Where:

- P(A|B) is the posterior probability: what we want to know after seeing evidence B

- P(B|A) is the likelihood: how probable is the evidence if A is true

- P(A) is the prior probability: our initial belief in A

- P(B) is the evidence: the total probability of observing B

<div align="center">
<img src="images/alarm_bayes_network.png" width="600" height="auto">
<p style="color: #555;">Figure: Application of Bayes' Theorem in the Alarm Network</p>
</div>

Bayes' theorem is particularly valuable when we know P(B|A) but need P(A|B) - often the case in diagnostic reasoning.
For example, we might know the probability of a positive test result given a disease, but what we want is the
probability of the disease given a positive test.

A canonical example is medical testing:

- P(Disease) = 0.01 (1% of population has the disease)
- P(Positive|Disease) = 0.95 (test is 95% sensitive)
- P(Positive|No Disease) = 0.05 (test is 95% specific)

Using Bayes' theorem to find P(Disease|Positive):

$$P(\text{Disease}|\text{Positive}) = \frac{P(\text{Positive}|\text{Disease}) \times P(\text{Disease})}{P(\text{Positive})}$$

The denominator can be expanded using the law of total probability:

$$P(\text{Positive}) = P(\text{Positive}|\text{Disease}) \times P(\text{Disease}) + P(\text{Positive}|\text{No Disease}) \times P(\text{No Disease})$$
$$P(\text{Positive}) = 0.95 \times 0.01 + 0.05 \times 0.99 = 0.0095 + 0.0495 = 0.059$$

Now we can calculate:

$$P(\text{Disease}|\text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$

This means despite the 95% test accuracy, the probability of having the disease given a positive test is only about 16%.
This counter-intuitive result, known as the base rate fallacy, highlights the importance of considering prior
probabilities.

##### Independence and Conditional Independence

Two events A and B are independent if the occurrence of one doesn't affect the probability of the other:

$$P(A|B) = P(A)$$

Equivalently:

$$P(A \cap B) = P(A) \times P(B)$$

Independence is a powerful simplifying assumption in probabilistic models. When events are independent, we can simply
multiply their probabilities to find joint probabilities.

Conditional independence is a more nuanced concept. Events A and B are conditionally independent given event C if, once
we know C has occurred, information about B doesn't change our beliefs about A:

$$P(A|B,C) = P(A|C)$$

This is written as A ⊥ B | C (A is independent of B given C).

Conditional independence is central to Bayesian networks. In the network structure, a node is conditionally independent
of its non-descendants given its parents. This property enables the compact factorization of joint distributions.

Consider three types of conditional independence patterns in Bayesian networks:

1. **Causal chain (A → B → C)**:
    - A and C are dependent
    - A and C are conditionally independent given B
    - Example: Disease → Symptom → Treatment
2. **Common cause (A ← B → C)**:
    - A and C are dependent
    - A and C are conditionally independent given B
    - Example: Fever ← Infection → Cough
3. **Common effect (A → C ← B)**:
    - A and B are independent
    - A and B are conditionally dependent given C (explaining away)
    - Example: Rain → Wet Grass ← Sprinkler

```mermaid
graph TD
    subgraph "Causal Chain"
    A1[A] --> B1[B]
    B1 --> C1[C]
    end

    subgraph "Common Cause"
    B2[B] --> A2[A]
    B2 --> C2[C]
    end

    subgraph "Common Effect (v-structure)"
    A3[A] --> C3[C]
    B3[B] --> C3
    end
```

The v-structure (common effect) exhibits a particularly interesting property called "explaining away." If we observe the
effect C, then learning about cause A changes our beliefs about cause B, even though A and B were initially independent.
For example, if we know the grass is wet and we learn it rained, the probability of the sprinkler having been on
decreases.

Understanding these patterns of independence is crucial for both constructing Bayesian networks and performing efficient
inference in them. The conditional independence relationships encoded in the network structure allow us to reduce the
number of parameters needed and enable more efficient algorithms for probabilistic reasoning.

#### Working with Probability Distributions

##### Discrete Probability Distributions

Discrete probability distributions model random variables that can take only distinct, separate values. These are
fundamental building blocks for probabilistic graphical models, especially when modeling categorical variables like
disease states, weather conditions, or parts of speech.

A discrete probability distribution assigns a probability to each possible value of the random variable, with the total
probability summing to 1. For a random variable X with possible values {x₁, x₂, ..., xₙ}, the probability mass function
P(X) must satisfy:

1. Non-negativity: P(X = xᵢ) ≥ 0 for all i
2. Normalization: ∑ᵢ P(X = xᵢ) = 1

Common discrete distributions include:

- **Bernoulli distribution**: Models a binary outcome (success/failure) with probability p of success
    - P(X = 1) = p
    - P(X = 0) = 1-p
    - Example: Modeling whether a patient has a disease
- **Binomial distribution**: Models the number of successes in n independent Bernoulli trials
    - P(X = k) = (n choose k) × pᵏ × (1-p)ⁿ⁻ᵏ
    - Example: Number of patients who recover out of 10 treated
- **Categorical distribution**: Generalizes Bernoulli to more than two outcomes
    - P(X = xᵢ) = pᵢ where ∑ᵢ pᵢ = 1
    - Example: Modeling parts of speech (noun, verb, adjective, etc.)

In Bayesian networks, each node typically has a discrete probability distribution conditioned on its parents. For root
nodes, we specify a prior distribution; for child nodes, we specify a conditional distribution for each combination of
parent values.

##### Joint and Marginal Probabilities

Joint probability distributions model the probabilities of multiple random variables simultaneously. For two discrete
random variables X and Y, the joint distribution P(X,Y) gives the probability of each combination of values (x,y).

For example, in a medical diagnosis context, we might have:

```
P(Disease=present, Symptom=present) = 0.08
P(Disease=present, Symptom=absent) = 0.02
P(Disease=absent, Symptom=present) = 0.12
P(Disease=absent, Symptom=absent) = 0.78
```

This joint distribution contains complete information about the probabilistic relationship between Disease and Symptom.
From it, we can derive marginal and conditional distributions.

The marginal distribution of a single variable is obtained by summing over all values of the other variables. For
example:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

Using our example:

$$P(\text{Disease=present}) = P(\text{Disease=present, Symptom=present}) + P(\text{Disease=present, Symptom=absent}) = 0.08 + 0.02 = 0.10$$

Marginalization is a fundamental operation in probabilistic inference. When we have evidence about some variables in a
Bayesian network and want to infer others, we often need to sum out (marginalize) the hidden variables.

```
<div align="center">
<img src="images/alarm_bayes_network.png" width="600" height="auto">
<p style="color: #555;">Figure: Joint probability distribution factored according to a Bayesian network</p>
</div>
```

##### Calculating Conditional Probabilities

Conditional probabilities can be derived from joint probabilities using the definition:

$$P(X|Y) = \frac{P(X,Y)}{P(Y)}$$

For each value y of Y, P(X|Y=y) is a probability distribution over X. Continuing our medical example:

$$P(\text{Disease=present}|\text{Symptom=present}) = \frac{P(\text{Disease=present, Symptom=present})}{P(\text{Symptom=present})}$$

$$P(\text{Disease=present}|\text{Symptom=present}) = \frac{0.08}{0.08 + 0.12} = \frac{0.08}{0.20} = 0.40$$

This tells us that if a patient has the symptom, there's a 40% chance they have the disease.

In Bayesian networks, conditional probability tables (CPTs) explicitly represent these conditional distributions. For
each node, its CPT specifies the distribution over its values for each combination of parent values.

Note that while we can calculate conditional probabilities from joint probabilities, a key insight of Bayesian networks
is that we can reconstruct the full joint distribution if we know the conditional probabilities following the network
structure:

$$P(X_1, X_2, ..., X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i))$$

##### Normalization Methods

Normalization is a fundamental operation in Bayesian inference. When updating beliefs based on evidence, we often first
calculate proportional probabilities (unnormalized) and then normalize them to ensure they sum to 1.

For a discrete probability distribution P'(X) that is proportional to the true distribution P(X), we normalize by
dividing by the sum:

$$P(X = x_i) = \frac{P'(X = x_i)}{\sum_j P'(X = x_j)}$$

The normalization factor, often denoted by α or η, is just the reciprocal of this sum:

$$P(X = x_i) = \alpha \times P'(X = x_i) \text{ where } \alpha = \frac{1}{\sum_j P'(X = x_j)}$$

This process appears frequently in Bayesian inference. When applying Bayes' theorem:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

We can compute P(B|A)P(A) for each value of A, and then normalize to find P(A|B).

Example: Consider a simple diagnostic test for a disease with:

- Prior: P(Disease) = 0.01
- Likelihood: P(Positive|Disease) = 0.95, P(Positive|No Disease) = 0.05

For a positive test result:

1. Unnormalized posterior:
    - P'(Disease|Positive) = P(Positive|Disease) × P(Disease) = 0.95 × 0.01 = 0.0095
    - P'(No Disease|Positive) = P(Positive|No Disease) × P(No Disease) = 0.05 × 0.99 = 0.0495
2. Normalization:
    - Sum = 0.0095 + 0.0495 = 0.059
    - P(Disease|Positive) = 0.0095 / 0.059 ≈ 0.161
    - P(No Disease|Positive) = 0.0495 / 0.059 ≈ 0.839

##### Complex Probability Examples

Let's examine a more complex example that integrates multiple concepts: the famous "Monty Hall problem."

In this game show scenario:

- There are three doors (A, B, C)
- A car is behind one door, goats behind the others
- The contestant picks a door, e.g., door A
- The host (who knows where the car is) opens another door with a goat, e.g., door C
- The contestant can stick with door A or switch to door B
- The question is: Should they switch?

We can solve this using Bayes' theorem:

Let:

- Cₐ = Car is behind door A
- Cᵦ = Car is behind door B
- Cᶜ = Car is behind door C
- H₍c₎ = Host opens door C

We want to compare P(Cₐ|H₍c₎) versus P(Cᵦ|H₍c₎)

For P(Cₐ|H₍c₎): $$P(C_a|H_{(c)}) = \frac{P(H_{(c)}|C_a) \times P(C_a)}{P(H_{(c)})}$$

- P(Cₐ) = 1/3 (prior)
- P(H₍c₎|Cₐ) = 1/2 (host chooses randomly between B and C)
- P(H₍c₎) = normalization factor

For P(Cᵦ|H₍c₎): $$P(C_b|H_{(c)}) = \frac{P(H_{(c)}|C_b) \times P(C_b)}{P(H_{(c)})}$$

- P(Cᵦ) = 1/3 (prior)
- P(H₍c₎|Cᵦ) = 1 (host must choose C)
- P(H₍c₎) = same normalization factor

Comparing the unnormalized posteriors:

- P'(Cₐ|H₍c₎) = 1/2 × 1/3 = 1/6
- P'(Cᵦ|H₍c₎) = 1 × 1/3 = 1/3

After normalization:

- P(Cₐ|H₍c₎) = 1/3
- P(Cᵦ|H₍c₎) = 2/3

Therefore, switching doubles the probability of winning!

Another compelling example is the "Two-Test Cancer Scenario":

A patient takes two tests for a rare disease (1% prevalence). Each test is 90% sensitive (90% true positive rate) and
80% specific (80% true negative rate). Both tests come back positive.

What is the probability the patient has the disease?

Using Bayes' theorem with multiple pieces of evidence:

$$P(C|T_1=+,T_2=+) = \frac{P(T_1=+,T_2=+|C) \times P(C)}{P(T_1=+,T_2=+)}$$

Assuming the tests are conditionally independent given disease status:

$$P(T_1=+,T_2=+|C) = P(T_1=+|C) \times P(T_2=+|C) = 0.9 \times 0.9 = 0.81$$
$$P(T_1=+,T_2=+|¬C) = P(T_1=+|¬C) \times P(T_2=+|¬C) = 0.2 \times 0.2 = 0.04$$

Calculating the unnormalized posterior:

- P'(C|T₁=+,T₂=+) = 0.81 × 0.01 = 0.0081
- P'(¬C|T₁=+,T₂=+) = 0.04 × 0.99 = 0.0396

After normalization:

- P(C|T₁=+,T₂=+) = 0.0081 / (0.0081 + 0.0396) ≈ 0.17

Despite two positive tests, the post-test probability is only 17% due to the low prevalence of the disease. This
illustrates the challenge of diagnostic testing for rare conditions and the importance of considering both
sensitivity/specificity and base rates.

These examples demonstrate how probability theory provides a rigorous framework for reasoning under uncertainty,
especially when dealing with multiple sources of evidence or counterintuitive situations.
