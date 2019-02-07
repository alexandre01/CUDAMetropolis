from lib.experiment import Experiment
from lib import scheduler, utils


experiment = Experiment(N=1000, M=5000, t_max=10000, beta_scheduler=scheduler.ConstantBetaScheduler(0.5),
                        algorithm="Metropolis", batch_size=None, use_gpu=False)
errors, energies, x = experiment.run()
utils.plot_errors_energies(errors, energies)
