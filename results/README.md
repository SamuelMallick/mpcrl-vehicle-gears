# Results

The results folder are organized as follows:
- `train_c3`: policy obtained from the first stage of training. Only one of the 10 trained policies is present in the repository, corresponding to the one used in the simulations for controller LC-1.
- `train_c4`: policy obtained from the second stage of training. Only one of the 10 trained policies is uploaded in the repository, corresponding to the one used in the simulations for controller LC-2. The policy has been obtained by running the second stage of training on the policy in the folder `train_c3`.
- `eval_single`: results for the single-agent case, i.e., N=15 and M=1.
- `eval_platoon`: results for a platoon with M=5 and both N=15 and N=30.

The additional training files, including those used to generate the plots with the training trajectories, are not included in the repository due to space reasons. They are available upon request.
