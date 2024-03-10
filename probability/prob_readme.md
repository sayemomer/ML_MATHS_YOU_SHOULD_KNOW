### Bernoulli distribution
- A Bernoulli distribution is a distribution over a single binary random variable.
- It is a special case of the binomial distribution where a single trial is conducted (n=1).
- The probability mass function (PMF) of a Bernoulli distribution is given by:
    - P(X=1) = p
    - P(X=0) = 1-p
- The expected value of a Bernoulli distribution is given by:
    - E[X] = p
- The variance of a Bernoulli distribution is given by:
    - Var(X) = p(1-p)

#### Simple Example:

Consider flipping a fair coin. There are two possible outcomes: "heads" or "tails". If we define "heads" as a success (1) and "tails" as a failure (0), and since the coin is fair, the probability of getting "heads" (success) is 0.5, and the probability of getting "tails" (failure) is also 0.5.

So, the random variable XX representing the outcome of the coin flip follows a Bernoulli distribution with p=0.5p=0.5. The pmf would be:
$$
P(X=1)=0.5(probability of heads)
$$
$$
P(X=0)=0.5(probability of tails)
$$

The expected value of XX would be 
$
E[X]=0.5
$
and the variance would be $ Var(X)=0.5×(1−0.5)=0.25 $
