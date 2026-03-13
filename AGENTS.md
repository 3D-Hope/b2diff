now i have 3 rewards for 3d case you see right

1. pentration
2. boundary
3. object_count

i call these all collectively universal rewards. because they are true for all tasks. now im converting this codebase to be my experimentation lab for testing custom rewards.

i need config.

when universal_rewards = true. all those 3 are calculated and added to create the sum reward.

on top of that i may or may not give other custom reward.
eg tv_bed reward that is a specific type of reward. when i have flag for universal_rewards and custom = not none then also calculate custom reward. add that reward to others. the sum of the rewards for each scene from all these rewards is the new reward for reest of the code. all rewards are computed without tracking gradients. come up with plan to do this now
put penetration reward in separate file and do this. i will put rewards like tv_bed in a dir ./core/custom_rewards

and we need to put universal rewards all in ./core/universal_rewards/

import them correctly. lets do this one by one at a time. first start with dir strucutre and import