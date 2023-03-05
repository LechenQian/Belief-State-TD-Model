import numpy as np
from scipy.stats import norm

# initialize
nTrials = 1000  # number of trials
x = []  # series of observations during trials - will be filled in later

# Gaussian probability distribution and cumulative probability distribution
ISIpdf = norm.pdf([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8], 2, 0.5) / \
    sum(norm.pdf([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8], 2, 0.5))
ISIcdf = [0.0477, 0.1312, 0.2558, 0.4142, 0.5858, 0.7442, 0.8688, 0.9523, 1.0]

# Uncomment 2 lines directly below for flat probability distribution -
# as seen in Supplementary Figure 9
# ISIpdf = [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111]
# ISIcdf = [0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 1.0]

# Create distribution of ISI's - 10% unassigned are omission trials
# Possible ISIs range from 5-13
ISIdistributionMatrix = np.zeros(nTrials)
for i in range(int(nTrials - nTrials / 10)):
    ISIdistributionMatrix[i] = np.sum(ISIcdf < np.random.rand()) + 5
ISIdistributionMatrix = np.append(
    ISIdistributionMatrix, np.full(int(nTrials / 10), np.nan))
np.random.shuffle(ISIdistributionMatrix)

# Calculate hazard rate of receiving reward after substates 5-14 (ISIhazard);
# Used later to create transition matrix
ISIhazard = np.zeros_like(ISIpdf)
ISIhazard[0] = ISIpdf[0]
for i in range(1, len(ISIpdf)):
    ISIhazard[i] = ISIpdf[i] / (1 - ISIcdf[i - 1])

# Set hazard rate of transitioning OUT of the ITI
# 1/65 is based on task parameters
ITIhazard = 1 / 65

# Generate sequence of observations that corresponds to trials
# Observations:
#   Null-->1
#   Odor ON-->2
#   Reward-->3
ITIdistributionMatrix = np.zeros(nTrials)
for i in range(nTrials):
    if not np.isnan(ISIdistributionMatrix[i]):  # reward delivery trials
        ISI = np.ones(int(ISIdistributionMatrix[i]))
        ITI = np.ones(int(np.random.geometric(ITIhazard)))
        trial = np.concatenate(([2], ISI, [3], ITI))
        x.append(trial)
    else:  # omission trials
        ITI = np.ones(int(np.random.geometric(ITIhazard)))
        trial = np.concatenate(([2], ITI))
        x.append(trial)
    ITIdistributionMatrix[i] = len(ITI)


# states
#ISI = np.arange(1, 15)
#ITI = 15

# Fill out the observation matrix O
# O[x, y, :] = [a, b, c]
# a is the probability that observation 1 (null) was observed given that a
# transition from sub-state x-->y just occurred
# b is the probability that observation 2 (odor ON) was observed given that
# a transition from sub-state x-->y just occurred
# c is the probability that observation 2 (reward) was observed given that
# a transition from sub-state x-->y just occurred
O = np.zeros((15, 15, 3))

# ISI
O[0:13, 1:14, 0] = 1
O[0:13, 1:14, 1] = 0
O[0:13, 1:14, 2] = 0

# obtaining reward
O[13:15, 14, 2] = 1
O[12:14, 14, 2] = 1
O[11:13, 14, 2] = 1
O[10:12, 14, 2] = 1
O[9:11, 14, 2] = 1
O[8:10, 14, 2] = 1
O[7:9, 14, 2] = 1
O[6:8, 14, 2] = 1
O[5:7, 14, 2] = 1

# stimulus onset
O[14, 0, 1] = 1  # rewarded trial
O[14, 14, 1] = 0
O[14, 14, 2] = ITIhazard * 0.1  # omission trial

# ITI
O[14, 14, 0] = 1 - (ITIhazard * 0.1)

# Fill out the transition matrix T
# T[x, y] is the probability of transitioning from sub-state x-->y
T = np.zeros((15, 15))

# odor ON from substates 1-6
# no probability of transitioning out of ISI while odor ON
T[0:5, 1:6] = 1

# T(ISIsubstate_i+6-->ISIsubstate_i+7) = ISIhazard(i)
# these substates span the variable ISI interval
# if reward is received, then transition into the ITI
for i in range(5, len(ISI) + 4):
    T[i, i + 1] = 1 - ISIhazard[i - 4]
    T[i, 14] = ISIhazard[i - 4]
T[13, 14] = 1

# ITI length is drawn from exponential distribution in task
# this is captured with single ITI substate with high self-transition
# probability
T[14, 14] = 1 - (ITIhazard * 0.9)
T[14, 0] = ITIhazard * 0.9

# Visualize the transition and observation matrices
# Code for Supplementary Figure 7
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].imshow(T)
axs[0].set_title('Transition Matrix')
axs[0].set_xlabel('Next Substate')
axs[0].set_ylabel('Current Substate')
axs[1].imshow(O)
axs[1].set_title('Observation Matrix')
axs[1].set_xlabel('Next Substate')
axs[1].set_ylabel('Current Substate')
plt.show()

# Run TD learning
results = TD(x, O, T)

# Plot RPE as a function of ISI; plot every trial
reward_indices = np.where(x == 3)[0]
reward_indices = reward_indices[int(len(reward_indices) * 0.4):]  # only look at trials after 2000 trials
ISI_distribution_rewarded_trials = ISI_distribution_matrix[~np.isnan(ISI_distribution_matrix)]
ISIs_for_plot = ISI_distribution_rewarded_trials[int(len(ISI_distribution_rewarded_trials) * 0.4):]  # only look at trials after 2000 trials

plt.plot(ISIs_for_plot, results['rpe'][reward_indices], 'k*')
plt.xlabel('ISI')
plt.ylabel('TD error')
plt.show()

# Plot average RPE for each ISI
# Code for Supplementary Figure 5
reward_rpe = results['rpe'][reward_indices]

# Average RPEs (and standard error) for each ISI length
average_RPE = []
error_RPE = []
for i in range(1, 10):
    idx = np.where(ISIs_for_plot == i + 4)[0]
    average_RPE.append(np.sum(reward_rpe[idx]) / len(idx))
    error_RPE.append(np.std(reward_rpe[idx]) / np.sqrt(len(idx)))

# Plotting average RPE and standard error for each ISI
plt.errorbar(range(1, 10), average_RPE, error_RPE, fmt='k')
plt.plot(range(1, 10), average_RPE, '.', color=[1 - i * 0.1, i * 0.1, 1], markersize=25)
plt.xlabel('time of reward delivery', fontSize=20)
plt.ylabel('Average TD error', fontSize=20)
plt.show()

# Value, valueprime, and RPE
# Code for value signals and RPEs shown in Figure 6
cue_onsets = np.where(x == 2)[0]
which_ISI = 13  # how long is the ISI for the trial type that you want to plot the value signal for? range: 5-13
cue_onsets = cue_onsets[ISI_distribution_matrix == which_ISI]
cue_onsets = cue_onsets[int(len(cue_onsets) * 0.4):-12]  # only look at trials after 2000 trials

value = np.zeros(20)
value_prime = np.zeros(20)
rpe = np.zeros(20)
for i in range(len(cue_onsets)):
    for j in range(20):
        value[j] += results['value'][cue_onsets[i] + j - 2]
        value_prime[j] += results['value'][cue_onsets[i] + j - 1]
        rpe[j] += results['rpe'][cue_onsets[i] + j - 2]


# Plot value, value(t+1), and RPE
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].plot(value/len(cueonsets), 'k')
axs[0].hold(True)
axs[0].plot(valueprime/len(cueonsets), color=[0.5, 0.5, 0.5])
axs[0].set_title('Value [black] and Value(t+1) [grey]')
axs[1].plot((valueprime-value)/len(cueonsets), 'k')
axs[1].set_title('Value(t+1)-Value(t)')
axs[2].plot(rpe/len(cueonsets), color=[1-(whichISI-4)/9, (whichISI-4)/9, 1])
axs[2].set_title('TD error')

# Plot substate weights
# Code for weights shown in Figure 6
weight = np.zeros(15)
for i in range(15):
    weight[i] = sum(results.w[round(len(results.w)*0.4):, i])
ax = plt.figure().add_subplot()
ax.bar(range(1, 16), weight/(round(len(results.w)*0.6)))
ax.set_ylim([0, 2])
ax.set_ylabel('Weight')
ax.set_xlabel('substate')

# Generate matrix of RPEs for plotting
# Code for RPEs shown in Figure 5
RPE = np.zeros((10, 30))
allOdorIndices = np.where(x == 2)[0]
odorIndicesforplot = allOdorIndices[int(len(allOdorIndices)*0.4):-12]
ISIsforplot = ISIdistributionMatrix[int(len(allOdorIndices)*0.4):-12]

for i in range(len(ISIsforplot)):
    if not np.isnan(ISIsforplot[i]):
        RPE[ISIsforplot[i]-4, :] = (results.rpe[odorIndicesforplot[i]-5:odorIndicesforplot[i]+24]) + RPE[ISIsforplot[i]-4, :]

# Divide by the number of trials to compute each ISI's average RPE
for i in range(RPE.shape[0]):
    RPE[i, :] = RPE[i, :]/np.sum(ISIsforplot == i+4)
plt.figure()
plt.plot(RPE[0:9, :].T)
plt.ylabel('TD Error')
plt.xlabel('time')

# Plot omission trials
# Code for Supplementary Figure 1
odorIndices = np.where(x == 2)[0]
Omissiontrials = odorIndices[np.isnan(ISIdistributionMatrix)]
Omissiontrials = Omissiontrials[int(0.4*len(Omissiontrials)):]
omissionRPEs = np.zeros(30)
for i in range(len(Omissiontrials)):
    omissionRPEs[0:30] += results.rpe[Omissiontrials[i]-4:Omissiontrials[i]+25]
omissionRPEs = omissionRPEs/len(Omissiontrials)
plt.figure()
plt.plot(np.arange(-0.8, 5.2, 0.2), omissionRPEs)
plt.ylim([-0.8, 0.8])
