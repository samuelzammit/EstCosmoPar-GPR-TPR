# import required packages/modules
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from CustomCauchyKernel import CustomCauchy  # kernel not defined in TensorFlow, so we define it manually

# define required parts of tensorflow_probability
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

# define list of priors, covariance functions, and datasets
priornames = ['NoPrior', 'Riess', 'TRGB', 'H0LiCOW', 'CM', 'Planck', 'DES']
priorvals = [None, 74.22, 69.8, 73.3, 75.35, 67.4, 67.4]
priorsdevs = [None, 1.82, 1.9, 1.7, 1.68, 0.5, 1.1]
covnames = ['Squex', 'DoubleSquex', 'RatQuadratic', 'Matern32', 'Matern52', 'Matern72', 'Matern92', 'Cauchy']
datanames = ['CC', 'CC+SN', 'CC+SN+BAO']

os.chdir("~/GaPP_27/Application/4_TP/files/")

for i in range(len(datanames)):

    dataname = datanames[i]

    for j in range(len(covnames)):

        covname = covnames[j]

        for k in range(len(priornames)):

            print()
            print(i, j, k)
            priorname = priornames[k]

            # prepare the data
            (z, H, sigmaH) = np.loadtxt("~/GaPP_27/Application/data/Hdata_" + dataname + ".txt", unpack=True)

            # append prior (at z=0) if applicable
            if priorname != 'NoPrior':
                z = np.insert(z, 0, 0, axis=0)
                H = np.insert(H, 0, priorvals[k], axis=0)
                sigmaH = np.insert(sigmaH, 0, priorsdevs[k], axis=0)

            # X = redshift value
            z = np.asarray(z).reshape(-1, 1)

            # y = corresponding value of H(z)
            H = np.asarray(H).astype(np.float32)


            # define kernel and list of hyperparameters
            # double square exponential only
            def build_tp_doublesquex(sigmaf1, l1, sigmaf2, l2, sigmaH, nu):

                k = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=sigmaf1,
                                                                length_scale=l1) + tfp.math.psd_kernels.ExponentiatedQuadratic(
                    amplitude=sigmaf2, length_scale=l2)
                return tfd.StudentTProcess(kernel=k, index_points=z, observation_noise_variance=sigmaH, df=nu)


            # rational quadratic only
            def build_tp_ratquad(sigmaf, l, alpha, sigmaH, nu):

                k = tfp.math.psd_kernels.RationalQuadratic(amplitude=sigmaf, length_scale=l, scale_mixture_rate=alpha)
                return tfd.StudentTProcess(kernel=k, index_points=z, observation_noise_variance=sigmaH, df=nu)


            # all other kernels
            def build_tp_others(sigmaf, l, sigmaH, nu):

                # define kernel
                if covname == "Squex":
                    k = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=sigmaf, length_scale=l)
                elif covname == "Matern32":
                    k = tfp.math.psd_kernels.MaternThreeHalves(amplitude=sigmaf, length_scale=l)
                elif covname == "Matern52":
                    k = tfp.math.psd_kernels.MaternFiveHalves(amplitude=sigmaf, length_scale=l)
                elif covname == "Matern72":
                    k = tfp.math.psd_kernels.GeneralizedMatern(df=3.5, amplitude=sigmaf, length_scale=l)
                elif covname == "Matern92":
                    k = tfp.math.psd_kernels.GeneralizedMatern(df=4.5, amplitude=sigmaf, length_scale=l)
                elif covname == "Cauchy":
                    k = CustomCauchy(amplitude=sigmaf, length_scale=l)

                # define Student's t-process prior
                return tfd.StudentTProcess(kernel=k, index_points=z, observation_noise_variance=sigmaH, df=nu)


            # function to return uniform tensor distribution
            def up(a, b):
                return tfd.Uniform(np.float64(a), np.float64(b))


            # define uninformative uniform priors on kernel hyperparameters
            if covname == "DoubleSquex":
                tp_joint_model = tfd.JointDistributionNamed({
                    'sigmaf1': up(0, 100),
                    'l1': up(0, 100),
                    'sigmaf2': up(0, 100),
                    'l2': up(0, 100),
                    'nu': up(30, 100),
                    'sigmaH': up(0, 0.01),
                    'obs': build_tp_doublesquex,
                })
            elif covname == "RatQuadratic":
                tp_joint_model = tfd.JointDistributionNamed({
                    'sigmaf': up(0, 100),
                    'l': up(0, 100),
                    'alpha': up(0, 1),
                    'nu': up(0, 100),
                    'sigmaH': up(0, 0.01),
                    'obs': build_tp_ratquad,
                })
            else:
                tp_joint_model = tfd.JointDistributionNamed({
                    'sigmaf': up(0, 100),
                    'l': up(0, 100),
                    'nu': up(30, 100),
                    'sigmaH': up(0, 0.01),
                    'obs': build_tp_others,
                })

            # constrain all parameters to positive and define initial values
            constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())


            # function to define parameters
            def defineParameter(init, name):
                return tfp.util.TransformedVariable(initial_value=init, bijector=constrain_positive, name=name,
                                                    dtype=np.float64)


            sigmafvar = defineParameter(50., 'sigmaf')  # not used for double squex
            lvar = defineParameter(1., 'l')  # not used for double squex
            sigmaf1var = defineParameter(50., 'sigmaf1')  # only used for double squex
            l1var = defineParameter(1., 'l1')  # only used for double squex
            sigmaf2var = defineParameter(50., 'sigmaf2')  # only used for double squex
            l2var = defineParameter(1., 'l2')  # only used for double squex
            alphavar = defineParameter(0.5, 'alpha')  # only used for rational quadratic
            nuvar = defineParameter(20., 'nu')  # degrees of freedom
            sigmaHvar = defineParameter(0.01, 'sigmaH')  # observation noise variance

            # list of hyperparameters
            if covname == "DoubleSquex":
                varlist = [sigmaf1var, l1var, sigmaf2var, l2var, nuvar, sigmaHvar]
            elif covname == "RatQuadratic":
                varlist = [sigmafvar, lvar, alphavar, nuvar, sigmaHvar]
            elif covname == "Squex" or covname == "Cauchy":
                varlist = [sigmafvar, lvar, nuvar, sigmaHvar]
            else:
                varlist = [lvar, nuvar, sigmaHvar]

            trainablevariables = [v.trainable_variables[0] for v in varlist]


            # define target log probability to be minimised
            def target_log_prob(sigmaf, l, sigmaf1, l1, sigmaf2, l2, alpha, nu, sigmaH):
                if covname == "DoubleSquex":
                    return tp_joint_model.log_prob(
                        {'sigmaf1': sigmaf1, 'l1': l1, 'sigmaf2': sigmaf2, 'l2': l2, 'nu': nu, 'sigmaH': sigmaH,
                         'obs': H})
                elif covname == "RatQuadratic":
                    return tp_joint_model.log_prob(
                        {'sigmaf': sigmaf, 'l': l, 'alpha': alpha, 'nu': nu, 'sigmaH': sigmaH, 'obs': H})
                else:
                    return tp_joint_model.log_prob({'sigmaf': sigmaf, 'l': l, 'nu': nu, 'sigmaH': sigmaH, 'obs': H})


            # define number of optimisers and learning rate
            num_iters = 10000
            optimizer = tf.optimizers.Adam(learning_rate=.001)


            # optimise loglikelihood
            @tf.function(autograph=False, jit_compile=False)
            def train_model():
                with tf.GradientTape() as tape:
                    logl = -target_log_prob(sigmafvar, lvar, sigmaf1var, l1var, sigmaf2var, l2var, alphavar, nuvar,
                                            sigmaHvar)
                grads = tape.gradient(logl, trainablevariables)
                optimizer.apply_gradients(zip(grads, trainablevariables))
                return logl


            # store value of loglikelihood at each iteration for plot
            lls_ = np.zeros(num_iters, np.float64)
            for ii in range(num_iters):
                logl = train_model()
                lls_[ii] = logl

            # print optimised parameter values
            with open("opt_para_" + dataname + "_" + covname + "_" + priorname + ".txt", "w") as f:
                f.write('Trained parameters:' + "\n")
                if covname == "Squex" or covname == "Cauchy" or covname == "RatQuadratic":
                    f.write('sigmaf: {}'.format(sigmafvar._value().numpy()) + "\n")
                if covname != "DoubleSquex":
                    f.write('l: {}'.format(lvar._value().numpy()) + "\n")
                if covname == "DoubleSquex":
                    f.write('sigmaf1: {}'.format(sigmaf1var._value().numpy()) + "\n")
                    f.write('l1: {}'.format(l1var._value().numpy()) + "\n")
                    f.write('sigmaf2: {}'.format(sigmaf2var._value().numpy()) + "\n")
                    f.write('l2: {}'.format(l2var._value().numpy()) + "\n")
                if covname == "RatQuadratic":
                    f.write('alpha: {}'.format(alphavar._value().numpy()) + "\n")
                f.write('nu: {}'.format(nuvar._value().numpy()) + "\n")
                f.write('sigmaH: {}'.format(sigmaHvar._value().numpy()) + "\n")

            zmin = 0.0
            zmax = 2.5 if dataname == "CC+SN+BAO" else 2.0
            nstar = 100

            # reshape to [100, 1] -- 1 is the dimensionality of the feature space.
            Xstar = np.linspace(zmin, zmax, nstar, dtype=np.float64)
            Xstar = Xstar[..., np.newaxis]

            # kernel function with optimised parameters
            if covname == "Squex":
                kopt = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=sigmafvar, length_scale=lvar)
            elif covname == "DoubleSquex":
                kopt = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=sigmaf1var,
                                                                   length_scale=l1var) + tfp.math.psd_kernels.ExponentiatedQuadratic(
                    amplitude=sigmaf2var, length_scale=l2var)
            elif covname == "RatQuadratic":
                kopt = tfp.math.psd_kernels.RationalQuadratic(amplitude=sigmafvar, length_scale=lvar,
                                                              scale_mixture_rate=alphavar)
            elif covname == "Matern32":
                kopt = tfp.math.psd_kernels.MaternThreeHalves(amplitude=sigmafvar, length_scale=lvar)
            elif covname == "Matern52":
                kopt = tfp.math.psd_kernels.MaternFiveHalves(amplitude=sigmafvar, length_scale=lvar)
            elif covname == "Matern72":
                kopt = tfp.math.psd_kernels.GeneralizedMatern(df=3.5, amplitude=sigmafvar, length_scale=lvar)
            elif covname == "Matern92":
                kopt = tfp.math.psd_kernels.GeneralizedMatern(df=4.5, amplitude=sigmafvar, length_scale=lvar)
            elif covname == "Cauchy":
                kopt = CustomCauchy(amplitude=sigmafvar, length_scale=lvar)

            # function to return TP posterior predictive function
            tprm = tfd.StudentTProcessRegressionModel(kernel=kopt, df=nuvar, index_points=np.float64(Xstar),
                                                      observation_index_points=np.float64(z),
                                                      observations=np.float64(H),
                                                      observation_noise_variance=np.float64(sigmaHvar),
                                                      predictive_noise_variance=np.float64(0.))

            # create 500 samples, each of size [nrow(Xstar), 1]
            num_samples = 500
            samples = tprm.sample(num_samples)

            # get sample mean and variance
            samplemean = tf.math.reduce_mean(samples, axis=0)
            samplevar = tf.math.reduce_variance(samples, axis=0)

            # reconstruction
            rec = np.zeros((len(Xstar), 3))
            rec[:, 0] = Xstar.ravel()
            rec[:, 1] = samplemean.numpy()
            rec[:, 2] = np.sqrt(samplevar.numpy())

            np.savetxt("H_" + dataname + "_" + covname + "_" + priorname + ".txt", rec)

            # plot the observations and posterior samples
            # similarly to what was done in GPR
            import plottingfunction

            plotname = "plot_" + dataname + "_" + covname + "_" + priorname
            plottitle = dataname + "/" + covname + "/" + priorname
            print("Plotting " + plottitle)
            plottingfunction.plot(z, H, sigmaH, rec, zmin, zmax, plotname, plottitle)
