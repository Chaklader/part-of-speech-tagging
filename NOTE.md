# Fundamentals ofProbabilistic Graphical Model


# 1. Introduction to Probabilistic Models


In this course, we'll talk about probabilistic models, which will allow knowledge-based AI agents to handle uncertainty in the real world with the concept of a "belief state" (a probability distribution over the possible world states). These ideas have been used in robot localization and other real-world tasks, including use in neural networks(opens in a new tab).

Knowledge-based AI agents utilize belief state to reason and plan under uncertainty.

In this course, we will learn Probabilistic Models and Bayes Networks as the tools for these AI agents to quantify and act under the uncertainty of the surrounding world. Unlike the logical assertions, which see the possible worlds in a set of strict rules, the probabilistic assertions quantify how probable the various worlds are.


### Bayesian Networks (Bayes Nets)

Bayesian Networks are probabilistic graphical models that represent dependencies among variables using a directed acyclic graph (DAG). Think of it as a family tree of probabilities, where each node's probability depends only on its parent nodes, making complex probability calculations much more manageable.

## Detailed Explanation

### Core Components
1. Network Structure
   - Directed Acyclic Graph (DAG)
   - Nodes represent random variables
   - Edges represent direct dependencies
   - Conditional Independence relationships

2. Probability Tables
   - Each node has CPT (Conditional Probability Table)
   - P(X|Parents(X)) for each node X
   - Root nodes have prior probabilities

### Mathematical Foundation
1. Chain Rule
```
P(X₁,...,Xₙ) = ∏ᵢ P(Xᵢ|Parents(Xᵢ))
```

2. Conditional Independence
```
P(X|Y,Z) = P(X|Z) if X ⊥ Y|Z
```

## Key Features
1. Compact Representation
   - Reduces complexity from O(2ⁿ) to O(2ᵏ)
   - k = maximum number of parents

2. Inference Types
   - Exact inference
   - Approximate inference
   - Most Probable Explanation (MPE)

## Common Applications
1. Medical Diagnosis
   - Symptoms → Diseases
   - Risk factors → Conditions

2. Decision Support
   - Expert systems
   - Risk assessment
   - Fault diagnosis

## Building Bayes Nets
1. Structure Learning
   - Expert knowledge
   - Data-driven approaches
   - Hybrid methods

2. Parameter Learning
   - Maximum Likelihood Estimation
   - Bayesian Estimation
   - EM Algorithm for hidden variables

## Inference Methods
1. Variable Elimination
   - Factor multiplication
   - Marginalization
   - Ordered elimination

2. Sampling Methods
   - Forward sampling
   - Likelihood weighting
   - Gibbs sampling

## Limitations
1. Computational Complexity
   - Exact inference is NP-hard
   - Structure learning is complex

2. Data Requirements
   - Large datasets needed
   - Quality of probability estimates

## Best Practices
1. Network Design
   - Keep structure simple
   - Use expert knowledge
   - Validate independence assumptions

2. Inference Choice
   - Consider problem size
   - Balance accuracy vs speed
   - Use approximations when appropriate


# 2. Probability



# 3. Spam Classifier with Naive Bayes
# 4. Bayes Nets
# 5. Inference in Bayes Nets
# 6. Part of Speech Tagging with HMMs
# 7. Dynamic Time Warping
# 8. Project: Part of Speech Tagging


In this lesson, we learned how to make inferences (query) from Bayes Nets based on the evidence variables and the conditional 
probabilities as configured in the Bayes Nets. of the evidence variables as defined in the network.

There are two algorithms that to compute exact inferences:

   1. Enumeration: the query’s conditional probability is computed by summing the terms from the full joint distribution.
   2. Variable Elimination: an algorithm to reduce the enumeration computation by doing the repeated calculations once and 
      store the results for later re-use.

However, it is computationally expensive to make exact inference from a large and highly connected Bayes Network. In these 
cases, we can approximate inferences by sampling. Sampling is a technique to select and count the occurances of the query 
and evidence variables to estimate the probability distributions in the network. We looked at four sampling techniques as 
follows:

   1. Direct sampling: the simplest form of samples generation from a known probability distribution. For example, to sample the 
      odds of Head or Tail in a coin flip, we can randomly generate the events based on uniform probability distribution (assuming 
      we use a non-bias coin).
   2. Rejection sampling: generates samples from the known distribution in the network and rejects the non-matching evidence.
   3. Likelihood sampling: is similar to rejection sampling but generating only events that are consistent with the evidence.
   4. Gibbs sampling: initiates an arbitrary state and generates the next state by randomly sampling a non-evidence variable, 
      while keeping all evidence variables fixed.

In the final lesson, we will learn the Hidden Markov Model (HMM) and its application in the Natural Language Processing task 
to tag Parts of Speech. HMM assumes unobservable states and computes the transition and emission probabilities from one state 
to another.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
The general formula for the joint probability of multiple events is known as the chain rule of probability or the general 
product rule. For n events A₁, A₂, ..., Aₙ, the general formula is:

`   P(A₁, A₂, ..., Aₙ) = P(A₁) * P(A₂|A₁) * P(A₃|A₁,A₂) * ... * P(Aₙ|A₁,A₂,...,Aₙ₋₁)
`

In our specific case with cancer (C) and two test results (T1 and T2), we have:

`   P(C, T1=+, T2=+) = P(C) * P(T1=+|C) * P(T2=+|C,T1=+)
`
However, we typically assume that the test results are conditionally independent given the disease status. This means that 
knowing the result of one test doesn't affect the probability of the result of the other test, given that we know whether 
the person has cancer or not. Under this assumption:

`   P(T2=+|C,T1=+) = P(T2=+|C)
`
Which is why we can simplify to:

`   P(C, T1=+, T2=+) = P(C) * P(T1=+|C) * P(T2=+|C)
`
This assumption of conditional independence is common in many probabilistic models, including Naive Bayes classifiers, but 
it's important to recognize that it's an assumption that may not always hold in real-world scenarios.


General form of Bayes' theorem:

`   P(A|B) = P(B|A) * P(A) / P(B)
`
In our specific case:

`   P(C|T1=+,T2=+) = [P(T1=+,T2=+|C) * P(C)] / P(T1=+,T2=+)
`

Where:
- A is C (having cancer)
- B is (T1=+,T2=+) (both tests being positive)

Breaking it down further:

1. P(T1=+,T2=+|C) * P(C) is equivalent to P(C,T1=+,T2=+), by the chain rule of probability:

   `P(C,T1=+,T2=+) = P(T1=+,T2=+|C) * P(C)`

2. P(T1=+,T2=+) in the denominator can be expanded using the law of total probability:
   
   `P(T1=+,T2=+) = P(T1=+,T2=+|C) * P(C) + P(T1=+,T2=+|¬C) * P(¬C)`

So, the full expansion of the formula in terms of the general Bayes' theorem would be:

   `P(C|T1=+,T2=+) = [P(T1=+,T2=+|C) * P(C)] / [P(T1=+,T2=+|C) * P(C) + P(T1=+,T2=+|¬C) * P(¬C)]`
   `P(C|T1=+,T2=+) = P(C,T1=+,T2=+) / P(C,T1=+,T2=+) + P(¬C,T1=+,T2=+)`
   `P(C|T1=+,T2=+) = P(C,T1=+,T2=+) / P(T1=+,T2=+)`

This form directly shows how we're updating our prior belief P(C) based on the likelihood of the test results given cancer 
P(T1=+,T2=+|C) and normalizing it by the total probability of getting these test results.


Bayes Rule
––––––––––

   P(A|B) = P(B|A) P(A) / P(B) 
   P(¬A|B) = P(B|¬A) P(¬A) / P(B)
   
   P(A|B) + P(¬A|B) = 1
   
   P'(A|B) = P(B|A) P(A)
   P'(¬A|B) = P(B|¬A) P(¬A)
   
   P(A|B) = η P'(A|B)
   P(¬A|B) = η P'(¬A|B)
   
   η = (P'(A|B) + P'(¬A|B))^-1


These formulas represent various forms and implications of Bayes' Rule in probability theory. The prime notation (P') is 
used to denote unnormalized probabilities, and η (eta) represents a normalization factor.


Summary: Two-Test Cancer Screening Problem


Theory
–––––––

   1. Bayesian Inference:
      - Uses Bayes' theorem to update probabilities based on new evidence.
      - P(A|B) = P(B|A) * P(A) / P(B)

   2. Probabilistic Graphical Models:
      - Represent dependencies between variables (Cancer -> Test1, Test2).
      - Allows for intuitive visualization of the problem structure.

   3. Conditional Independence:
      - Assume T1 and T2 are conditionally independent given C.
      - P(T1,T2|C) = P(T1|C) * P(T2|C)

   4. Chain Rule of Probability:
      - P(A,B,C) = P(A) * P(B|A) * P(C|A,B)

   5. Law of Total Probability:
      - P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A)


Problem Setup

- P(C) = 0.01 (prior probability of cancer)
- P(+|C) = 0.9 (test sensitivity)
- P(-|¬C) = 0.8 (test specificity)
- Two independent tests performed: T1 and T2


Solution Approach

   1. Calculate joint probabilities:
      P(C,T1=+,T2=+) = P(C) * P(T1=+|C) * P(T2=+|C)
      P(¬C,T1=+,T2=+) = P(¬C) * P(T1=+|¬C) * P(T2=+|¬C)

   2. Calculate total probability of two positive tests:
      P(T1=+,T2=+) = P(C,T1=+,T2=+) + P(¬C,T1=+,T2=+)

   3. Apply Bayes' theorem:
      P(C|T1=+,T2=+) = P(C,T1=+,T2=+) / P(T1=+,T2=+)

Calculations

   1. P(C,T1=+,T2=+) = 0.01 * 0.9 * 0.9 = 0.0081
   2. P(¬C,T1=+,T2=+) = 0.99 * 0.2 * 0.2 = 0.0396
   3. P(T1=+,T2=+) = 0.0081 + 0.0396 = 0.0477
   4. P(C|T1=+,T2=+) = 0.0081 / 0.0477 = 0.1698

Final Result:
P(C|T1=+,T2=+) ≈ 0.1698 or 16.98%

Interpretation:
- Despite two positive tests, the probability of having cancer is only about 17%.
- This counterintuitive result is due to the low prior probability of cancer (1%).
- Demonstrates the importance of considering base rates in diagnostic testing.
- Shows how multiple tests can increase diagnostic confidence, but not as dramatically as one might expect.

Key Takeaways:
1. Multiple positive tests increase the probability of the condition, but the increase may be less than intuitively expected.
2. Low-prevalence conditions can still have relatively low probabilities even after positive tests.
3. Bayesian reasoning is crucial for correctly interpreting medical test results.
4. The assumption of conditional independence between tests simplifies calculations but may not always hold in reality.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

To find P(T2=+ | T1=+), we need to use the concept of conditional independence and the law of total probability. Here's 
how we can calculate it:

1. Given:
   P(C) = 0.01
   P(+|C) = 0.9
   P(-|¬C) = 0.8

2. We can derive:
   P(¬C) = 1 - P(C) = 0.99
   P(+|¬C) = 1 - P(-|¬C) = 1 - 0.8 = 0.2

3. Using the law of total probability:
   P(T2=+ | T1=+) = P(T2=+ | T1=+, C) * P(C | T1=+) + P(T2=+ | T1=+, ¬C) * P(¬C | T1=+)

4. Due to conditional independence:
   P(T2=+ | T1=+, C) = P(T2=+ | C) = 0.9
   P(T2=+ | T1=+, ¬C) = P(T2=+ | ¬C) = 0.2

5. We need to calculate P(C | T1=+) using Bayes' theorem:
   P(C | T1=+) = [P(T1=+ | C) * P(C)] / [P(T1=+ | C) * P(C) + P(T1=+ | ¬C) * P(¬C)]
                = (0.9 * 0.01) / (0.9 * 0.01 + 0.2 * 0.99)
                ≈ 0.0435

6. P(¬C | T1=+) = 1 - P(C | T1=+) ≈ 0.9565

7. Now we can calculate:
   P(T2=+ | T1=+) = 0.9 * 0.0435 + 0.2 * 0.9565
                  ≈ 0.0391 + 0.1913
                  ≈ 0.2304

Therefore, P(T2=+ | T1=+) ≈ 0.2304 or about 23.04%.

This result shows that the probability of the second test being positive, given that the first test was positive, is about 23.04%. 
This is higher than the base rate of positive tests (which would be around 2.7% if we calculated P(T=+) directly), but lower than 
one might intuitively expect, demonstrating the importance of careful probability calculations in such scenarios.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

P(A | X, Y) is read as "the probability of A given X and Y" or "the probability of A in the presence of both X and Y."

More specifically:

1. P(A | X, Y) represents the conditional probability of event A occurring, given that both events X and Y have occurred.

2. It means we're considering the probability of A in the subset of scenarios where both X and Y are true or have happened.

3. This notation is used when the probability of A depends on or is influenced by the joint occurrence of X and Y.

4. In practical terms, it could represent situations like:
   - The probability of a certain medical condition (A) given two specific symptoms (X and Y)
   - The likelihood of a stock price increase (A) given both positive market trends (X) and good company earnings (Y)

5. It's important to note that P(A | X, Y) may be different from P(A | X) or P(A | Y) individually, as the combination of 
X and Y together might affect the probability of A in ways that X or Y alone do not.

6. In some cases, if X and Y are independent with respect to A, then P(A | X, Y) might equal P(A | X) or P(A | Y), but 
this is not generally assumed without evidence.

This concept is crucial in probability theory, especially in complex scenarios where multiple conditions or events can 
influence the outcome we're interested in.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

R = Raise (at work)
S = Sunny

Given this new context:

1. P(R) = 0.01: The probability of getting a raise is 1%.
2. P(S) = 0.7: The probability of a sunny day is 70%.
3. P(R|S) = 0.01: The probability of getting a raise given that it's sunny is also 1%.

Your explanation is correct:

P(R|S) is the same as P(R) because the events are independent. The weather (sunny or not) does not affect the probability 
of getting a raise at work, and vice versa.

This independence means:
P(R|S) = P(R) = 0.01

This makes perfect sense in the context of a Bayesian network modeling how external factors (weather and work events) might 
affect happiness. The equality P(R|S) = P(R) correctly represents that the chance of a raise doesn't change based on the weather.

Thank you for this clarification. It's an excellent example of how crucial context is in interpreting probability problems 
and how independence between events is represented in probabilistic models.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

1. Variables:
   S: Sunny (weather)
   R: Raise (at work)
   H: Happiness

2. Given probabilities:
   P(S) = 0.7 (probability of a sunny day)
   P(R) = 0.01 (probability of getting a raise)
   P(R|S) = 0.01 (probability of a raise given it's sunny, same as P(R) indicating independence)

3. Conditional probabilities for Happiness:
   P(H|S,R) = 1 (100% happy if sunny and got a raise)
   P(H|¬S,R) = 0.9 (90% happy if not sunny but got a raise)
   P(H|S,¬R) = 0.7 (70% happy if sunny but no raise)
   P(H|¬S,¬R) = 0.1 (10% happy if not sunny and no raise)

4. The main calculation shown is for P(R|H,S), which is the probability of getting a raise given that you're happy and it's sunny:

   P(R|H,S) = [P(H|R,S) * P(R|S)] / P(H|S)
             = [1 * 0.01] / [0.01 + 0.7*0.99]
             = 0.01 / 0.7033
             ≈ 0.0142 (or 1.42%)

5. The network graph shows that both S and R influence H, but S and R are independent of each other.

This example demonstrates how Bayesian networks can model the relationships between different variables and how to calculate 
conditional probabilities using Bayes' theorem. It shows that even if you're happy and it's sunny, the probability of having 
received a raise is still quite low (only slightly higher than the base probability of getting a raise).


Thank you for that hint. You're right to point this out, as it's a crucial part of the calculation shown in the image. Let's 
incorporate this into our explanation:

P(R|H,S) = [P(H|R,S) * P(R)] / [P(H|R,S)*P(R) + P(H|¬R,S)*P(¬R)]

The second formula is indeed correct and matches the hint you provided. This is an application of Bayes' theorem for calculating 
the probability of R given H and S.

Let's break down the calculation using this formula:

P(R|H,S) = [P(H|R,S) * P(R)] / [P(H|R,S)*P(R) + P(H|¬R,S)*P(¬R)]

Given:
P(H|R,S) = 1
P(R) = 0.01
P(H|¬R,S) = 0.7 (from the conditional probability table)
P(¬R) = 1 - P(R) = 0.99

Plugging in the values:

P(R|H,S) = (1 * 0.01) / [(1 * 0.01) + (0.7 * 0.99)]
         = 0.01 / (0.01 + 0.693)
         = 0.01 / 0.703
         ≈ 0.0142 (or about 1.42%)

This calculation shows that even if you're happy and it's sunny, the probability of having received a raise is still quite low, 
only slightly higher than the base probability of getting a raise (1%).

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

1. Given Probabilities:
   P(S) = 0.7 (probability of sunny weather)
   P(R) = 0.01 (probability of getting a raise)
   P(R|S) = 0.01 (probability of raise given sunny, indicating independence)

2. Conditional Probabilities for Happiness (H):
   P(H|S,R) = 1 (100% happy if sunny and got a raise)
   P(H|¬S,R) = 0.9 (90% happy if not sunny but got a raise)
   P(H|S,¬R) = 0.7 (70% happy if sunny but no raise)
   P(H|¬S,¬R) = 0.1 (10% happy if not sunny and no raise)

3. Calculated Probabilities:
   P(R|H,S) = 0.0142 (probability of raise given happy and sunny)
   P(R|H) = 0.0185 (probability of raise given happy)

4. Calculations Shown:

   a. P(R|H) calculation:
      P(R|H) = [P(H|R) * P(R)] / P(H)
              = (0.97 * 0.01) / 0.5245
              = 0.0185

   b. P(H|R) calculation:
      P(H|R) = P(H|R,S) * P(S) + P(H|R,¬S) * P(¬S)
              = 1 * 0.7 + 0.9 * 0.3
              = 0.97

   c. P(H) calculation (Total Probability):
      P(H) = P(H|S,R) * P(S,R) + P(H|¬S,R) * P(¬S,R)
           + P(H|S,¬R) * P(S,¬R) + P(H|¬S,¬R) * P(¬S,¬R)
           = 1 * 0.7 * 0.01 + 0.9 * 0.3 * 0.01
           + 0.7 * 0.7 * 0.99 + 0.1 * 0.3 * 0.99
           = 0.5245

5. Interpretation:
   - P(R|H,S) = 0.0142 means that if you're happy and it's sunny, there's a 1.42% chance you got a raise.
   - P(R|H) = 0.0185 means that if you're happy (regardless of weather), there's a 1.85% chance you got a raise.
   - P(H) = 0.5245 means that the overall probability of being happy is about 52.45%.

6. Key Insights:
   - Being happy slightly increases the probability of having received a raise (from 1% to 1.85%).
   - Sunny weather and happiness together only marginally increase the probability of a raise (to 1.42%).
   - The overall probability of happiness (52.45%) is influenced more by sunny weather than by getting a raise, due to the 
   low probability of getting a raise.

This Bayesian network demonstrates how different factors (weather and work events) can influence happiness, and how we can 
use probability theory to understand these relationships quantitatively.

Certainly. This image presents a Bayesian network problem involving Sunny weather (S), getting a Raise (R), and Happiness (H). 
Let's break it down:

1. Given probabilities:
   P(S) = 0.7 (probability of sunny weather)
   P(R) = 0.01 (probability of getting a raise)
   P(R|S) = 0.01 (probability of raise given sunny weather, indicating independence)

2. Conditional probabilities for Happiness:
   P(H|S,R) = 1 (100% happy if sunny and got a raise)
   P(H|¬S,R) = 0.9 (90% happy if not sunny but got a raise)
   P(H|S,¬R) = 0.7 (70% happy if sunny but no raise)
   P(H|¬S,¬R) = 0.1 (10% happy if not sunny and no raise)

3. The main calculation shown is for P(R|H,¬S), which is the probability of getting a raise given that you're happy and it's not sunny:

   P(R|H,¬S) = [P(H|R,¬S) * P(R|¬S)] / P(H|¬S)

   This is derived from Bayes' theorem.

4. The calculation is expanded as:

   P(R|H,¬S) = [0.9 * 0.01] / [P(H|¬S,R) * P(R) + P(H|¬S,¬R) * P(¬R)]
              = 0.009 / (0.9 * 0.01 + 0.1 * 0.99)
              = 0.009 / (0.009 + 0.099)
              = 0.009 / 0.108
              ≈ 0.0833 or about 8.33%

5. Interpretation:
   If you're happy on a day that's not sunny, there's about an 8.33% chance that you got a raise. This is significantly 
   higher than the base probability of getting a raise (1%), indicating that being happy on a non-sunny day is a strong 
   indicator of having received a raise.

6. The network graph shows that both S and R influence H, but S and R are independent of each other.

This problem demonstrates how Bayesian networks can model complex relationships between variables and how to use Bayes' 
theorem to calculate conditional probabilities based on observed evidence.

This image illustrates the concept of Conditional Dependence in a Bayesian Network involving three variables: S (Sunny), 
R (Raise), and H (Happiness). Let's break it down:

1. Given Probabilities:
   P(S) = 0.7 (probability of a sunny day)
   P(R) = 0.01 (probability of getting a raise)
   P(R|S) = 0.01 (probability of a raise given it's sunny, same as P(R), indicating independence)

2. Conditional Probabilities for Happiness:
   P(H|S,R) = 1 (100% happy if sunny and got a raise)
   P(H|¬S,R) = 0.9 (90% happy if not sunny but got a raise)
   P(H|S,¬R) = 0.7 (70% happy if sunny but no raise)
   P(H|¬S,¬R) = 0.1 (10% happy if not sunny and no raise)

3. Calculated Probabilities:
   P(R|H,S) = 0.0142 (probability of a raise given happy and sunny)
   P(R|H,¬S) = 0.0833 (probability of a raise given happy and not sunny)

4. The main calculation shown is for P(R|H,¬S):
   P(R|H,¬S) = [P(H|R,¬S) * P(R|¬S)] / P(H|¬S)
              = [0.9 * 0.01] / [P(H|¬S,R) * P(R) + P(H|¬S,¬R) * P(¬R)]
              = 0.009 / (0.9 * 0.01 + 0.1 * 0.99)
              = 0.009 / 0.108
              = 0.0833

5. Conditional Dependence:
   - While R and S are independent (P(R|S) = P(R) = 0.01), they become dependent when conditioned on H.
   - P(R|H,S) ≠ P(R|H,¬S), showing that R and S are dependent given H.
   - This is because H is a common effect of both R and S, creating a "explaining away" effect.

6. Interpretation:
   - The probability of getting a raise, given you're happy, is higher on a non-sunny day (8.33%) than on a sunny day (1.42%).
   - This counterintuitive result occurs because happiness on a non-sunny day is more likely to be explained by getting a raise, whereas 
   on a sunny day, the sunshine itself could explain the happiness.


A Bayesian Network (Bayes Net) is a probabilistic graphical model that represents a set of variables and their conditional 
dependencies via a directed acyclic graph (DAG). Based on the information you've shared and the image, let's explain the 
Bayes Net concept in more detail:

1. Components of a Bayes Net:
   - Random variables (In this case: S, R, H)
   - Conditional independence relationships (shown by the graph structure)
   - Probability distributions (given in the image)

2. Structure:
   The graph shows S (Sunny) and R (Raise) as parent nodes to H (Happiness), indicating that H is directly influenced by both S and R.

3. Conditional Independence:
   S and R are not connected, implying they are independent. However, they become conditionally dependent when we observe H (explaining away effect).

4. Probability Distributions:
   - Prior probabilities: P(S), P(R)
   - Conditional probabilities: P(H|S,R), P(H|¬S,R), P(H|S,¬R), P(H|¬S,¬R)

5. Inference:
   The image shows calculations of P(R|H,S) and P(R|H,¬S), demonstrating how we can infer the probability of a raise given happiness and weather conditions.

6. Normalization Constant:
   While not explicitly shown in the image, the concept of normalization constant (α) is used in Bayes Net calculations to simplify computations.

7. Explaining Away:
   The difference between P(R|H,S) and P(R|H,¬S) demonstrates the explaining away effect. The probability of a raise is higher when happy on a non-sunny day because the raise better explains the happiness in the absence of sun.

8. Value of the Network:
   This Bayes Net allows us to model complex relationships and make inferences about unobserved variables based on observed evidence.

9. D-Separation:
   While not directly addressed in the image, d-separation is a concept used in Bayes Nets to determine conditional independence relationships.

This Bayes Net example demonstrates how we can model real-world scenarios with multiple interacting factors, represent their relationships 
probabilistically, and make inferences based on observed evidence. It's a powerful tool for reasoning under uncertainty in AI and machine 
learning applications.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



Inference in Bayes Net
––––––––––––––––––––––

Imagine a smart home system that manages energy usage. We'll create a simple Bayesian Network for this scenario:

Nodes:
1. Time of Day (T): Morning, Afternoon, Evening
2. Occupancy (O): Occupied, Unoccupied
3. Outside Temperature (E): Hot, Mild, Cold
4. AC Usage (A): On, Off
5. Energy Consumption (C): High, Medium, Low

Network Structure:
Time of Day → Occupancy
Time of Day → Outside Temperature
Occupancy → AC Usage
Outside Temperature → AC Usage
AC Usage → Energy Consumption

Now, let's discuss inference in this Bayesian Network:

1. Types of Inference:

   a) Predictive Inference: Reasoning from causes to effects.
      Example: What's the probability of high energy consumption given that it's a hot afternoon?

   b) Diagnostic Inference: Reasoning from effects to causes.
      Example: If energy consumption is high, what's the probability that the AC is on?

   c) Intercausal Inference: Reasoning between causes of a common effect.
      Example: If energy consumption is high and it's occupied, how does this affect the probability of it being hot outside?

2. Inference Methods:

   a) Exact Inference:
      - Variable Elimination: Systematically "sum out" variables not involved in the query.
      - Junction Tree Algorithm: Create a tree structure for efficient exact inference.

   b) Approximate Inference:
      - Monte Carlo methods: Use random sampling to estimate probabilities.
      - Variational methods: Approximate complex distributions with simpler ones.

3. Example Inference Task:

   Query: P(A = On | T = Afternoon, C = High)
   "What's the probability the AC is on given it's afternoon and energy consumption is high?"

   Steps:
   1. Apply Bayes' Rule: P(A|T,C) = P(C|A,T) * P(A|T) / P(C|T)
   2. Expand using marginalization:
      P(A|T,C) = Σ[O,E] P(C|A) * P(A|O,E) * P(O|T) * P(E|T) / P(C|T)
   3. Use probability tables to compute each term.
   4. Sum over all possible values of O and E.

4. Challenges in Inference:

   - Computational Complexity: As the network grows, exact inference can become intractable.
   - Continuous Variables: Many algorithms are designed for discrete variables; continuous variables may require discretization 
     or special techniques.
   - Incomplete Data: Handling missing values in the evidence.

5. Applications of Inference:

   - Prediction: Estimating future energy consumption based on current conditions.
   - Diagnosis: Identifying potential causes of unexpected energy usage patterns.
   - Decision Making: Determining optimal AC settings to balance comfort and energy efficiency.

In practice, software libraries (like PyMC, PGMPy, or BUGS) are often used to perform these inferences, as they can handle 
the complex calculations required for larger, real-world Bayesian Networks.

This example demonstrates how Bayesian Networks can model complex systems with multiple interrelated variables, allowing 
for various types of probabilistic reasoning and inference.


I apologize for not explaining that step clearly. Let's break down Step 3 in more detail:

The Chain Rule of Probability is a fundamental rule that allows us to express a joint probability as a product of conditional probabilities. In its general form, for events A, B, and C, it states:

P(A,B,C) = P(A|B,C) * P(B|C) * P(C)

Now, let's apply this to our specific case of P(T,C|A):

1. We start with P(T,C|A), which is a joint probability of T and C, conditioned on A.

2. We can rewrite this using the chain rule as:
   P(T,C|A) = P(C|T,A) * P(T|A)

   This step breaks down the joint probability into:
   - P(C|T,A): The probability of C given both T and A
   - P(T|A): The probability of T given A

3. The order matters here. We chose to condition on T first, but we could have done it the other way:
   P(T,C|A) = P(T|C,A) * P(C|A)

   We typically choose the order that aligns with the causal structure of our Bayesian network or the available conditional probability tables.

4. In our case, P(C|T,A) aligns better with the causal structure (Time and AC usage influence Energy Consumption), so we stick with:
   P(T,C|A) = P(C|T,A) * P(T|A)

This decomposition is useful because:
- It breaks a complex joint probability into simpler conditional probabilities.
- These simpler probabilities often directly correspond to the conditional probability tables in our Bayesian network.
- It allows us to leverage the conditional independence assumptions encoded in the network structure.

In the context of Bayesian networks, this step is crucial as it helps us express probabilities in terms of the local conditional probabilities associated with each node, making computations more manageable and aligned with the network structure.

Break down these steps in more detail

1. Applying Bayes' Rule:
   P(A|T,C) = P(C|A,T) * P(A|T) / P(C|T)

   This is a direct application of Bayes' theorem. Here's why we use it:
   - We want to find P(A|T,C), but it's often easier to define P(C|A,T) in a Bayesian network.
   - P(A|T) represents our prior belief about A given T.
   - P(C|T) acts as a normalization constant.

2. Expanding using marginalization:
   P(A|T,C) = Σ[O,E] P(C|A) * P(A|O,E) * P(O|T) * P(E|T) / P(C|T)

   This step involves several key concepts:

   a) Marginalization: We sum over all possible values of O and E because they're not observed (hidden variables).

   b) Chain Rule of Probability: We break down the joint probability into a product of conditional probabilities.

   c) Conditional Independence: We use the structure of the Bayesian network to simplify relationships. For example:
      - P(C|A,T) becomes P(C|A) assuming C is independent of T given A.
      - P(A|T) is expanded to P(A|O,E) * P(O|T) * P(E|T), reflecting the network structure.

   Let's break it down further:

   - P(C|A): Probability of the observed energy consumption given the AC state.
   - P(A|O,E): Probability of the AC state given occupancy and temperature.
   - P(O|T): Probability of occupancy given time of day.
   - P(E|T): Probability of temperature given time of day.
   - The sum (Σ) over O and E accounts for all possible scenarios of occupancy and temperature.

3. Why this expansion is useful:
   - It allows us to use the conditional probability tables defined in the Bayesian network.
   - It breaks a complex probability into simpler, more manageable parts.
   - It explicitly shows how we account for unknown variables (O and E).

4. Computational process:
   - Calculate each term inside the sum for all combinations of O and E.
   - Sum these results.
   - Divide by P(C|T), which itself would be calculated using a similar marginalization process.

This approach, while computationally intensive, allows us to compute the desired probability using only the conditional 
probabilities defined in the Bayesian network structure. It's a fundamental method in understanding how inference works in 
these networks, though more efficient algorithms are often used in practice for larger networks.

Certainly! Let's break down how we go from P(A|T,C) to P(C|A,T) * P(A|T) / P(C|T) using Bayes' theorem. This is a fundamental transformation in probability theory and is crucial for many applications in Bayesian networks.

Step 1: Start with Bayes' theorem
Bayes' theorem in its general form is:
P(X|Y) = P(Y|X) * P(X) / P(Y)

Step 2: Apply Bayes' theorem to our specific case
In our case:
X = A (AC usage)
Y = (T,C) (Time of day and Energy Consumption together)

Applying the theorem:
P(A|T,C) = P(T,C|A) * P(A) / P(T,C)

Step 3: Use the chain rule of probability to expand P(T,C|A)
P(T,C|A) = P(C|A,T) * P(T|A)

Substituting this back:
The probability of Time of day (T) and Energy Consumption (C) occurring together, given that the AC usage (A) is known.

P(A|T,C) = [P(C|A,T) * P(T|A)] * P(A) / P(T,C)



Step 4: Rearrange using the definition of conditional probability
P(A|T) = P(T|A) * P(A) / P(T)

Rearranging this: P(T|A) * P(A) = P(A|T) * P(T)

Step 5: Substitute this into our equation
P(A|T,C) = P(C|A,T) * [P(A|T) * P(T)] / P(T,C)

Step 6: Cancel out P(T) in numerator and denominator
P(A|T,C) = P(C|A,T) * P(A|T) / [P(T,C) / P(T)]

Step 7: Recognize that P(T,C) / P(T) = P(C|T)
This is again by the definition of conditional probability.

Final result:
P(A|T,C) = P(C|A,T) * P(A|T) / P(C|T)

This derivation shows how we can transform a conditional probability that we want to calculate (left side) into a form 
that often matches the structure and available information in a Bayesian network (right side). It's a powerful tool for 
inference in these networks.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Variable elimination is an algorithm for inference in Bayesian networks. Based on the chart provided, let's explain how it would work for this specific network:

The network shows:
R (Rain) → T (Traffic) → L (Late)

We're calculating P(+l) (probability of being late) using the formula:
P(+l) = Σr Σt P(r) P(t|r) P(+l|t)

Variable elimination would proceed as follows:

1. Goal: Compute P(+l) by eliminating variables R and T.

2. Order of elimination: We'll eliminate R first, then T.

3. Eliminate R:
   - Compute f1(t) = Σr P(r) P(t|r)
   - This creates a factor f1(t) that no longer depends on R

4. Eliminate T:
   - Compute f2 = Σt f1(t) P(+l|t)
   - This results in the final probability P(+l)

5. Detailed steps:
   a. Calculate f1(+t) = P(+r)P(+t|+r) + P(-r)P(+t|-r)
                      = 0.1 * 0.8 + 0.9 * 0.1 = 0.17
   b. Calculate f1(-t) = P(+r)P(-t|+r) + P(-r)P(-t|-r)
                      = 0.1 * 0.2 + 0.9 * 0.9 = 0.83
   c. Final computation:
      P(+l) = f1(+t) * P(+l|+t) + f1(-t) * P(+l|-t)
            = 0.17 * 0.3 + 0.83 * 0.1
            = 0.051 + 0.083
            = 0.134

The key idea of variable elimination is to compute intermediate factors (like f1) that allow us to sum out variables one 
at a time, reducing the overall computation compared to enumerating all possible combinations.

Certainly! Likelihood Weighting is an approximate inference technique used in Bayesian networks. It's a type of importance sampling method that's particularly useful for networks with evidence. Let's break it down:

Key Concept:
Likelihood Weighting generates weighted samples from the network, where the weights are determined by how well each sample matches the observed evidence.

How it works:

   1. Evidence Fixing:
      - Set the evidence variables to their observed values.
      - These values remain fixed for all samples.

   2. Sampling:
      - For each non-evidence variable, in topological order:
        - If the variable is an ancestor of an evidence variable, sample it from its conditional probability distribution.
        - If not, set its value based on its parents (as per the network structure).

   3. Weight Calculation:
      - Start with a weight of 1.
      - For each evidence variable, multiply the weight by the probability of observing that evidence given its parents' values in the current sample.

   4. Repeat:
      - Generate multiple samples, each with its associated weight.

   5. Estimation:
      - Use the weighted samples to estimate probabilities of query variables.

Advantages:
   1. More efficient than rejection sampling, especially with unlikely evidence.
   2. Easy to implement.
   3. Handles multiple evidence variables well.

Disadvantages:

   1. Can be less accurate for unlikely evidence scenarios.
   2. May require many samples for good accuracy in complex networks.

Example (using the R→T→L network from the image):

Let's say we want to estimate P(R|L=+l) (probability of rain given we're late).

   1. Fix L=+l for all samples.
   2. For each sample:
      - Sample R from P(R)
      - Sample T from P(T|R)
      - Set L=+l
      - Calculate weight: w = P(L=+l|T)
   3. Repeat for many samples.
   4. Estimate P(R=+r|L=+l) as:
      (sum of weights where R=+r) / (total sum of weights)

Likelihood Weighting is particularly useful in this network because it ensures that every sample is consistent with the 
evidence (L=+l), making it more efficient than methods that might generate inconsistent samples and reject them.

Gibbs sampling is another method of approximate inference in Bayesian networks. It's a Markov Chain Monte Carlo (MCMC) technique that's particularly useful for high-dimensional problems. Let's break it down with an example.

Gibbs Sampling Concept:
The idea is to sample each variable in turn, conditioned on the current values of all other variables in the network.

Using our R (Rain) → T (Traffic) → L (Late) network as an example:

Step 1: Initialize
Start with arbitrary values for all variables. Let's say:
R = +r (Raining)
T = -t (No traffic)
L = +l (Late)

Step 2: Sampling Process
1. Sample R given T and L:
   P(R | T=-t, L=+l) ∝ P(R) * P(T=-t | R) * P(L=+l | T=-t)
   Calculate this for R=+r and R=-r, normalize, then sample.

2. Sample T given new R and L:
   P(T | R, L=+l) ∝ P(T | R) * P(L=+l | T)
   Calculate for T=+t and T=-t, normalize, then sample.

3. Sample L given new R and T:
   P(L | R, T) ∝ P(L | T)
   Calculate for L=+l and L=-l, normalize, then sample.

Step 3: Repeat
Repeat this process many times (e.g., 1000 iterations).

Step 4: Analysis
After the "burn-in" period (initial samples we discard), count the frequency of each state to estimate probabilities.

Example Iteration:
Let's say we want to estimate P(R | L=+l)

1. Start: R=+r, T=-t, L=+l

2. Sample R:
   P(R=+r | T=-t, L=+l) ∝ 0.1 * 0.2 * 0.1 = 0.002
   P(R=-r | T=-t, L=+l) ∝ 0.9 * 0.9 * 0.1 = 0.081
   Normalize: P(R=+r) ≈ 0.024, P(R=-r) ≈ 0.976
   Sample new R, let's say we get R=-r

3. Sample T:
   P(T=+t | R=-r, L=+l) ∝ 0.1 * 0.3 = 0.03
   P(T=-t | R=-r, L=+l) ∝ 0.9 * 0.1 = 0.09
   Normalize: P(T=+t) ≈ 0.25, P(T=-t) ≈ 0.75
   Sample new T, let's say we get T=-t

4. Sample L:
   Here, L is fixed to +l as it's our evidence.

5. New state: R=-r, T=-t, L=+l

Repeat this process many times. After the burn-in period, the proportion of samples where R=+r will approximate P(R=+r | L=+l).

Advantages of Gibbs Sampling:
1. Works well in high-dimensional spaces.
2. Can be more efficient than other sampling methods for certain types of networks.
3. Relatively easy to implement.

Challenges:
1. May converge slowly if variables are highly correlated.
2. Needs to run for many iterations to get accurate results.
3. Requires ability to sample from conditional distributions.

Gibbs sampling is particularly useful in complex networks where direct probability calculations are intractable, allowing 
us to approximate probabilities through this iterative sampling process.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Hidden Markov Models (HMMs) in Natural Language Processing

1. Definition:
   A Hidden Markov Model is a statistical model used to represent probability distributions over sequences of observations. It consists of hidden states that emit observable outputs.

2. Key Components:
   a. Hidden States: Unobservable states (e.g., Parts of Speech)
   b. Observations: Visible outputs (e.g., words in a sentence)
   c. Transition Probabilities: Likelihood of moving between states
   d. Emission Probabilities: Likelihood of an observation given a state
   e. Initial State Probabilities: Likelihood of starting in each state

3. Part-of-Speech (POS) Tagging Example:
   States: N (Noun), M (Modal), V (Verb)
   Observations: Words in a sentence
   
   Sample sentence: "Jane will spot Will."
   
   Transition Probabilities:
   - Start → N: 3/4
   - N → M: 1/3
   - M → V: 3/4
   - V → N: 1
   - N → End: 4/9

   Emission Probabilities:
   - N → "Jane": 2/9
   - M → "will": 3/4
   - V → "spot": 1/4
   - N → "Will": 1/9

4. HMM Process:
   a. Start in an initial state
   b. Emit an observation based on emission probability
   c. Transition to next state based on transition probability
   d. Repeat b and c until reaching end state

5. Markov Assumption:
   The probability of a state depends only on the previous state, not the entire history.

6. Applications:
   - Speech Recognition
   - Machine Translation
   - Named Entity Recognition
   - Gene Prediction in Bioinformatics

7. Algorithms for HMMs:
   a. Forward Algorithm: Computes probability of an observation sequence
   b. Viterbi Algorithm: Finds most likely sequence of hidden states
   c. Baum-Welch Algorithm: Learns model parameters from data

8. Example: Weather Prediction HMM
   States: Sunny, Rainy, Cloudy
   Observations: Dry, Wet

   Transition Probabilities:
   Sunny → Sunny: 0.7, Sunny → Rainy: 0.2, Sunny → Cloudy: 0.1
   Rainy → Rainy: 0.6, Rainy → Sunny: 0.3, Rainy → Cloudy: 0.1
   Cloudy → Cloudy: 0.5, Cloudy → Sunny: 0.3, Cloudy → Rainy: 0.2

   Emission Probabilities:
   Sunny → Dry: 0.9, Sunny → Wet: 0.1
   Rainy → Dry: 0.2, Rainy → Wet: 0.8
   Cloudy → Dry: 0.6, Cloudy → Wet: 0.4

9. Challenges:
   - Computational complexity for large state spaces
   - Handling long-range dependencies
   - Dealing with sparse data

10. Advanced Topics:
    - Higher-Order HMMs
    - Hierarchical HMMs
    - Continuous HMMs

Understanding HMMs is crucial for many NLP tasks, as they provide a powerful framework for modeling sequential data with 
underlying hidden states. They form the basis for more advanced models in machine learning and artificial intelligence.


Transition Probabilities and Emission Probabilities are two key components of Hidden Markov Models (HMMs). While they're 
both types of probabilities, they represent different aspects of the model. Let's break down the differences:

1. Transition Probabilities:

   Definition: The probability of moving from one hidden state to another.
   
   Characteristics:
   - Represent the likelihood of state changes over time.
   - Only involve hidden states, not observations.
   - Form a square matrix where rows sum to 1.
   
   Example (POS tagging):
   - P(Verb | Noun) = 0.3 (probability of a verb following a noun)
   - P(Noun | Verb) = 0.7 (probability of a noun following a verb)

2. Emission Probabilities:

   Definition: The probability of observing a particular output given a specific hidden state.
   
   Characteristics:
   - Represent the relationship between hidden states and observable outputs.
   - Involve both hidden states and observations.
   - Each row (corresponding to a state) sums to 1.
   
   Example (POS tagging):
   - P("run" | Verb) = 0.05 (probability of seeing "run" given the state is Verb)
   - P("cat" | Noun) = 0.02 (probability of seeing "cat" given the state is Noun)

Key Differences:

1. What they model:
   - Transition: State-to-state relationships
   - Emission: State-to-observation relationships

2. Matrix structure:
   - Transition: Square matrix (state x state)
   - Emission: Rectangular matrix (state x observation)

3. Usage in the model:
   - Transition: Used to predict the next state
   - Emission: Used to predict the observation given a state

4. In the HMM process:
   - Transition: Applied when moving between time steps
   - Emission: Applied at each time step to generate an observation

5. Information they capture:
   - Transition: Temporal dynamics of the hidden process
   - Emission: How hidden states manifest as observable data

Understanding the distinction between these two types of probabilities is crucial for effectively designing, implementing, 
and interpreting Hidden Markov Models in various applications.


Viterbi Algorithm
–––––––––––––––––

Suppose we have a Hidden Markov Model for predicting weather conditions (Sunny, Rainy) based on observed activities (Walk, Shop, Clean).

States: Sunny (S), Rainy (R)
Observations: Walk (W), Shop (H), Clean (C)

Given:
1. Initial Probabilities:
   P(S) = 0.6, P(R) = 0.4

2. Transition Probabilities:
   P(S|S) = 0.7, P(R|S) = 0.3
   P(S|R) = 0.4, P(R|R) = 0.6

3. Emission Probabilities:
   P(W|S) = 0.1, P(H|S) = 0.4, P(C|S) = 0.5
   P(W|R) = 0.6, P(H|R) = 0.3, P(C|R) = 0.1

Observed sequence: Walk, Shop, Clean

Viterbi Algorithm Steps:

1. Initialization:
   Create a matrix with states as rows and observations as columns.
   Initialize the first column with initial probabilities * emission probabilities:
   
   V(S,1) = 0.6 * 0.1 = 0.06
   V(R,1) = 0.4 * 0.6 = 0.24

2. Recursion:
   For each subsequent observation, calculate:
   V(s,t) = max[V(s',t-1) * P(s|s') * P(o|s)]
   where s is the current state, s' is the previous state, t is the time step, and o is the observation.

   For t=2 (Shop):
   V(S,2) = max(0.06 * 0.7 * 0.4, 0.24 * 0.4 * 0.4) = 0.0168
   V(R,2) = max(0.06 * 0.3 * 0.3, 0.24 * 0.6 * 0.3) = 0.0432

   For t=3 (Clean):
   V(S,3) = max(0.0168 * 0.7 * 0.5, 0.0432 * 0.4 * 0.5) = 0.00588
   V(R,3) = max(0.0168 * 0.3 * 0.1, 0.0432 * 0.6 * 0.1) = 0.00259

3. Path Tracking:
   Keep track of which previous state led to each current state.

4. Termination:
   The most likely final state is the one with the highest probability in the last column.
   Here, it's Sunny with 0.00588.

5. Backtracking:
   Trace back from the final state to find the most likely sequence of states.

Final Result:
Most likely weather sequence: Rainy, Rainy, Sunny

Explanation:
- Day 1: It was most likely Rainy when the person Walked.
- Day 2: It was most likely still Rainy when they went Shopping.
- Day 3: It most likely became Sunny when they Cleaned.

This example demonstrates how the Viterbi algorithm efficiently computes the most likely sequence of hidden states (weather 
conditions) given a sequence of observations (activities), taking into account both transition and emission probabilities 
at each step.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Dynamic Time Warping (DTW) is not typically associated with Probabilistic Graphical Models (PGMs), but it is an important 
algorithm in time series analysis and pattern recognition. Let me explain DTW and then clarify its relationship to PGMs:

Dynamic Time Warping (DTW):

1. Definition:
   DTW is an algorithm for measuring similarity between two temporal sequences which may vary in speed or length.

2. Purpose:
   To align two time series by warping the time axis iteratively until an optimal match between the two sequences is found.

3. Applications:
   - Speech recognition
   - Gesture recognition
   - Data mining
   - Time series classification

4. How it works:
   a. Create a distance matrix between all points of two sequences.
   b. Find an optimal path through this matrix that minimizes the total distance between the sequences.
   c. This path represents the best alignment of the two sequences.

5. Key features:
   - Can handle sequences of different lengths
   - Allows for non-linear alignment
   - More robust than Euclidean distance for comparing time series

Relationship to PGMs:

While DTW itself is not a PGM, it can be used in conjunction with PGMs in certain applications:

1. Feature Extraction:
   DTW can be used to extract features from time series data, which can then be used as inputs to PGMs like Hidden Markov 
   Models (HMMs).

2. Sequence Alignment:
   In some PGM applications involving time series, DTW might be used as a preprocessing step to align sequences before 
   modeling with PGMs.

3. Distance Metric:
   In probabilistic models that require a distance metric between sequences (e.g., some variants of Gaussian Process models), 
   DTW distance could be used instead of Euclidean distance.

4. HMM Alternatives:
   In some cases, DTW might be used as an alternative to HMMs for sequence classification tasks, especially when the temporal 
   warping of the signal is more important than its probabilistic generation process.

5. Hybrid Models:
   Some research has explored combining DTW with probabilistic models to create hybrid approaches that leverage the strengths 
   of both techniques.

Example in Speech Recognition:
In speech recognition, DTW might be used to align a spoken word with a template, and then the aligned features could be fed 
into an HMM for classification.

While DTW is a powerful tool for sequence alignment and comparison, it's important to note that it's deterministic and doesn't 
provide probabilistic interpretations like PGMs do. In many modern applications, especially those involving large datasets, 
probabilistic methods like HMMs or more advanced deep learning techniques have largely supplanted DTW. However, DTW remains 
useful in certain domains and can complement probabilistic approaches in some scenarios.
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––