---
---
## <center>Bayesian Updating of Key Performance Indicators

Why is this important?

Bayesian statistics allows us to use our former knowledge (say of the distribution of the general population of players) - the prior -  and any gained information (say KPI's from just a few Football games) - our observed data - to better inform us about something we are interested in (here, a better estimation of the value of a KPI) - to form a posterior. <br>
It allows us to use even small samples of data to inform us and make inferences, when otherwise to the question, 'so what does the small sample tell us?' <br>
we could only say: ¯|_(ツ)_/¯


```python
import numpy as np
from scipy.stats import poisson, gamma, beta
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
```



As an example we will just take a function that somewhat fits to population values to model our per Game Values:


```python
prior_dist = gamma(1.5)
```


```python
x = np.linspace(0,5,501)
unnormalized_priors = prior_dist.pdf(x) #densities at each point in our x-grid
priors = unnormalized_priors / unnormalized_priors.sum() #normalize
```


```python
plt.bar(x, priors, width = (x[0]+x[1])/2)
plt.plot(x, priors)
plt.suptitle('Prior Knowledge',fontsize = 16);
plt.title('Normalized Gamma PDF ~ Player Population', fontsize = 12);
plt.gcf().subplots_adjust(top=0.85)
plt.xlabel('KPI Value', fontsize = 12)
plt.ylabel('Probability', fontsize = 12);
```


![png](/images/bayesupdate_files/output_6_0.png)


So most players values lie between 0.1 and 4


```python
priors[:10] #A few of the prior probabilities for each of the possible 'true' values of a KPI
```




    array([0.        , 0.00113846, 0.001594  , 0.00193282, 0.00220962,
           0.00244585, 0.00265264, 0.00283667, 0.00300235, 0.00315279])



Those priors values represent the probabilities for each of the respective values of a KPI shown on the x-axis.

To model per Game (or p90) values, each of those values can serve as input to a Poisson Distribution, this parameter being is its mean.<br>

To quote from Wikipedia: <br>
The <b>Poisson Distribution</b> is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event.

Let's look at how those distributions look like


```python
for lam in x[::5]:
    pG_dist = poisson(lam).pmf(range(10)) #probability mass function for a discrete function
    plt.plot(range(10), pG_dist, linewidth = 0.5, alpha = 0.75, color='#969696')
for lam in x[::75]:
    pG_dist = poisson(lam).pmf(range(10))
    plt.plot(range(10), pG_dist, linewidth = 3, label = lam, alpha = 1)
plt.legend(title = 'λ', ncol=4, markerscale = 0.5, fontsize = 10)
plt.suptitle('Poisson Distributions for some chosen lambdas', fontsize = 16);
plt.title(' -- Distributions of per 90 Values --', fontsize = 12);
plt.gcf().subplots_adjust(top=0.85)
plt.xlabel('KPI p90 Value', fontsize = 12)
plt.ylabel('Probability', fontsize = 12);
```


![png](/images/bayesupdate_files/output_11_0.png)


To look at the distribution of per 90 Values for a random player, we could just take the average of the above gamma distribution (1.5) and the respective Poisson distribution. <br> That is the quick and dirty and wrong way though. To be exact, we need to look at all the possible Poisson distributions and mix them together according to the probabilities of the respective lambdas.


```python
pG_dists = []
for lam, prior in zip(x, priors):
    pG_dists.append(poisson(lam).pmf(range(10)) * prior)
pG_dists = np.sum(pG_dists, axis = 0)

plt.plot(range(10), pG_dists, linewidth = 3, label = 'Mixed Distribution')
plt.plot(range(10), poisson(1.5).pmf(range(10)), linewidth = 3, label = 'λ = 1.5')

plt.suptitle('Mixed Poisson Distributions of all lambdas', fontsize = 16);
plt.title(' -- Distribution of per 90 Values --', fontsize = 12);
plt.legend(ncol=4, markerscale = 0.5, fontsize = 10)
plt.gcf().subplots_adjust(top=0.85)
plt.xlabel('KPI p90 Value', fontsize = 12)
plt.ylabel('Probability', fontsize = 12);
```


![png](/images/bayesupdate_files/output_13_0.png)


This gives us a distribution of per90 Values we can expect from the population or a random player. <br>
Here we see that the mixed distribution has wider tails.

So now we want to take some observed data, which we can exemplatorily sample from one of the Poisson Distributions. <br> We will use that data to update each of the prior probabilities by multiplying them with the respective likelihood under the observed data.
This gives us the posterior probabilities for all the possible lambda values.
<br> If that didn't make sense yet, that's ok, I will explain it in more detail.


```python
#let's say we observed a sample of N = 12 games, and let's pick 2 as the mean value
N = 12
sample = poisson(2).rvs(N)
```


```python
sample, sample.mean()
```




    (array([2, 2, 3, 1, 2, 3, 4, 2, 1, 2, 5, 5]), 2.6666666666666665)



Now that we have our sample, the question is: <br>
How do we update our belief about the player given the information we observed?<br>

What we then need to do is as follows: <br>
Each of the lambdas and the respective prior probability represents a hypothesis that our player has a true action rate of that value.
The prior probability is our prior belief of that hypothesis being true.

When we see the data we observe, we should ask ourselves one thing:
<b> How probable is the observation given a hypothesis i?</b> (And do so for all hypotheses)
In a very easy example you could count up the ways the data could happen given a hypothesis.


<i>
So here for example, I observe a player making 7 Shots a game. Well, his true value being smaller than 1 sounds rather unlikely right? But should we think, wow, this guy might be a player who really shoots this often?
Not so fast, as high values were pretty unlikely to begin with. We need to consider this. That is why we multiply the probability of the data under a hypothesis with the probability we believed that hypothesis to be true to begin with!
</i>

Posterior = Likelihood * Prior / Marginal Likelihood

The probability of the data under a hypothesis is the <b>likelihood</b> of a hypothesis i. Multiply this with the prior probability and we almost get what we want. The <b>Posterior Probability</b> of a hypothesis i.
Combining all hypotheses and their posterior probabilities gives us the <b>posterior distribution</b>. All we have to do now is to <i>normalize</i> the probabilities, so they again sum to one.
We do this by diving by the Marginal Likelihood, which is the sum of likelihoods of all hypotheses or all the ways the observed data can happen within the model.

Formally this can also be written as p(H|E) = p(E|H)p(H) / p(E)  -  Bayes Rule

The last part of the equation is Bayes Rule, H being the hypothesis, E the evidence, p(H|E) the posterior, p(E|H) the marginal likelihood.

Done. Sound easy?


```python
def update_probability(lams, priors, data) -> 'posteriors':
    #likes = [poisson(l).pmf(data) for l in lams]
    # you can use this and a loop to update after each observed data point
    likes = [np.prod(poisson(l).pmf(data)) for l in lams]
    #vectorized for an array of values - like_l1*like_l2*...*like_ln
    posteriors = priors * likes
    return posteriors/posteriors.sum() #normalize
```


```python
#posteriors = priors
#for s in sample:   
    #posteriors = update_probability(x, priors, s)

posteriors = update_probability(x, priors, sample)
```


```python
plt.bar(x, priors, width = (x[0]+x[1])/2)
plt.plot(x, priors, label = 'Prior')

plt.plot(x, posteriors, label = 'Posterior')
plt.bar(x, posteriors, width = (x[0]+x[1])/2);

plt.annotate('ø '+str(round((posteriors * x).sum(),2)), xy = (3.35, 0.5*np.max(posteriors)), color='r')
plt.annotate('ø '+str(round((priors * x).sum(),2)), xy = (3.35, 0.4*np.max(posteriors)), color='b')

plt.xlabel('Lambda', fontsize = 12)
plt.ylabel('Density', fontsize = 12)

plt.legend()
plt.suptitle('Posterior given the Sample');
```


![png](/images/bayesupdate_files/output_21_0.png)


So this gives us quite a lot of information.
In this case, we can be rather sure that the players true value lies between ~1.75 and ~3.5.

We can also have a look at the players updated p90 Distribution


```python
pG_dists = []
for lam, posterior in zip(x, posteriors):
    pG_dists.append(poisson(lam).pmf(range(10)) * posterior)
pG_dists = np.sum(pG_dists, axis = 0)

plt.plot(range(10), pG_dists, linewidth = 3, label = 'Mixed Distribution')
plt.plot(range(10), poisson((posteriors * x).sum()).pmf(range(10)),
                                 linewidth = 3, label = 'λ = {}'.\
                                 format(np.round((posteriors * x).sum(), 2)))

plt.suptitle('Mixed Posterior Poisson Distributions of all lambdas', fontsize = 16);
plt.title(' -- Posterior Distribution of per 90 Values --', fontsize = 12);
plt.legend(ncol=4, markerscale = 0.5, fontsize = 10)
plt.gcf().subplots_adjust(top=0.85)
plt.xlabel('KPI p90 Value', fontsize = 12)
plt.ylabel('Probability', fontsize = 12);
```


![png](/images/bayesupdate_files/output_24_0.png)


<br>The next step is quantifying the uncertainty with credible intervals, ie: <br>
With 95% degree of certainty we believe the player's true value lies in the range [x1, x2]

For this we need to figure out the parameter values of the posterior distribution, between which x% of the probability mass lies:<br>
To do this, we can sample from our parameters  according to their posterior probability.


```python
samples = np.random.choice(x, p=posteriors, size = int(1e4), replace=True)
```


```python
np.percentile(samples, [2.5, 97.5])
```




    array([1.8 , 3.52])




```python
np.percentile(samples, [17, 83])
```




    array([2.16, 3.01])




```python
np.percentile(samples, [25, 75])
```




    array([2.27, 2.87])



how likely does the parameter lie above 2?


```python
(samples > 2).sum()/(len(samples))
```




    0.9109




```python
#above 1.75?
(samples > 1.75).sum()/(len(samples))
```




    0.9816




```python
#under 1.25?
(samples < 1.25).sum()/(len(samples))
```




    0.0001



another thing we can do is, look at the highest posterior density interval (HDP) that covers a certain percentage of the probability mass


```python
import pymc3 as pm
pm.hpd(samples, alpha = 0.34)

#alpha is the probability of a Type I Error
#The probability of the parameter not being covered by our interval
```



    array([2.1 , 2.94])



That's it. That should cover the basics of Bayesian inference.
